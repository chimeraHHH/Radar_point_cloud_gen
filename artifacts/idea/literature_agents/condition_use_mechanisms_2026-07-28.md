# Full-RAED 条件绕过/条件坍缩专项文献与代码审计

> 日期：2026-07-28  
> 范围：只读审计；未运行科学计算；未修改模型或训练代码。  
> 当前症状：G1D cross-scene condition-shuffle Chamfer degradation 约
> `0.016%`；G1G 两步 smoke 也接近 `0`。后者只有两个优化步，不能当作
> G1G 的科学结论。  
> 目标：区分“把已有条件信号放大/注入得更深”和“训练目标直接惩罚模型忽略
> Cube 条件”，并为 Full-RAED -> exactly 10k points 冻结低成本 Stage-0。

## 1. 结论先行

1. **G1D 的失败是可定位的条件旁路，不是“没有 cross-attention”。**
   `cube_drae` 直接决定 proposals、局部 64-bin spectrum、energy 和最终 query
   state；`condition_cube_drae` 只替换全局 radar-token 分支。跨场景替换后
   Chamfer 几乎不变，说明局部分支已经足以完成当前训练目标，全局条件只是一条
   可选路径。
2. **G1G 已移除 center allocation 前的局部 Cube 旁路，但尚未从目标函数上
   排除常数解。** 2,500 个 learned query、固定 25 x 20 x 5 RAE template、
   residual FFN 都可以共同学习一个数据集平均场景。cross-attention 存在且梯度
   非零，只能证明路径可导，不能证明输出依赖条件。
3. **RaLD 值得继续借鉴的是生成骨架，不是“条件一定会被用”的证明。**
   RaLD 的 radar encoder、mixed set latent、逐层 radar cross-attention 和
   conditional EDM 是本项目最接近的结构来源；但其公开训练目标没有
   mismatched-condition ranking、输出反演条件的 cycle reward 或条件互信息项。
   因此“更像 RaLD”不能单独解决当前 `0.016%`。
4. **CFG、condition dropout、FiLM、AdaLN-Zero、zero-conv、gated
   cross-attention 和普通 MoE routing 都不是独立的 anti-collapse loss。**
   它们改善注入、稳定性、采样 guidance 或条件计算容量；若
   \(f(x,c)=f(x)\)，这些机制允许条件分支保持零贡献。
5. **真正直接惩罚忽略条件的低成本机制有四类：**
   mismatched-condition ranking、基于物理输出的 InfoNCE/MI 下界、
   output-to-condition consistency feedback，以及同一 noisy state 下的
   mismatched-condition contrastive flow matching。
6. **建议不改写当前 G1G 20-epoch 基线。** 先让它按冻结协议跑完；若最终
   shuffle `<1%`，并行启动本文 A/B/C/E 四条互斥路线。D（condition-routed
   MoE）保留为低优先级结构诊断，因为只有 76 个训练帧，expert specialization
   很容易变成 scene memorization。

## 2. 当前代码中的真实信息流

### 2.1 记号

- \(L_i\)：第 \(i\) 帧保持不变的 measured/local Full-RAED Cube；
- \(C_i\)：送入 learned global condition encoder 的 Full-RAED Cube；
- \(Y_i\)：该帧的 dense target；
- \(\pi(i)\)：固定、跨 scene、无自配对的 condition derangement；
- \(\hat Y_i^+=G_\theta(L_i,C_i)\)；
- \(\hat Y_i^-=G_\theta(L_i,C_{\pi(i)})\)。

现有 condition-shuffle 指标为

\[
\Delta_{\mathrm{shuffle}}
=
\frac{1}{N}\sum_i
\left[
\frac{
\operatorname{CD}(\hat Y_i^-,Y_i)
}{
\max(\operatorname{CD}(\hat Y_i^+,Y_i),\epsilon)
}
-1
\right].
\]

它回答的是一个反事实问题：**在局部测量完全不变时，换掉 learned global
condition 是否会改变几何质量。**

### 2.2 G1D：局部 query state 是强旁路

当前实现中：

- [`code/models/rald_query_field.py`](../../../code/models/rald_query_field.py)
  的 `stable_radar_proposals`、`query_tokens`、coarse/final
  `decode_query_field` 都读取 measured `cube_drae`；
- 只有 `radar_encoder(condition_cube_drae)` 使用可被 shuffle 的条件；
- 评估时保持 measured Cube、proposal cache 和 occupancy queries 不变，只替换
  `condition_cube_drae`。

因此 G1D 的 `~0.016%` 不是 shuffle 实现把整个输入换错后的假阴性；它恰好说明
局部 spectrum/energy/range 路径足以绕过全局 336-token condition。

### 2.3 G1G：移除了局部旁路，但保留了常数模板解

[`code/models/g1g_hierarchical_allocator.py`](../../../code/models/g1g_hierarchical_allocator.py)
的 center stage 只读取：

- 2,500 learned center queries；
- 固定 25 x 20 x 5 normalized RAE templates；
- 336 Full-RAED tokens。

局部 spectrum 只在 center allocation 后用于四子点 refinement。这是正确的
结构修正。不过 center block 是

\[
q^{\ell+1}
=
q^\ell
+\operatorname{CrossAttn}(q^\ell,E(C))
+\operatorname{FFN}(\cdot).
\]

当 cross-attention 学成零或 scene-invariant 输出时，learned queries、template
embedding 和 FFN 仍可产生固定平均几何。因此：

- `condition gradient > 0` 是必要条件，不是充分条件；
- 两步 smoke shuffle 近零主要验证流水线，不足以关闭 G1G；
- 20-epoch 终点若仍 `<1%`，才说明“移除局部旁路”仍不足以排除常数模板解。

### 2.4 RaLD：应借鉴什么，不应推断什么

[RaLD 论文](https://ojs.aaai.org/index.php/AAAI/article/view/38946)及
[官方代码](https://github.com/MetaIoT-WHU/RaLD)提供了四个可复用部件：

1. radar tensor -> spatial condition tokens；
2. order-invariant mixed point latents；
3. latent/query 对 radar tokens 的逐层 cross-attention；
4. radar-conditioned EDM/latent generation。

对应源码包括
[radar-conditioned Transformer](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/model/models_radar_generation.py#L133-L229)
和
[point latent autoencoder/query decoder](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/model/models_ae.py#L284-L432)。

但 RaLD 的公开主路径：

- 使用 intensity-only RAE condition；
- 输出 dense XYZ，不输出本项目的逐点 circular Doppler distribution 和
  independent confidence；
- 没有 condition-shuffle gate；
- 没有直接比较 matched/mismatched radar condition 的训练项。

所以本项目应该采用 **RaLD generation backbone + 显式 anti-ignore objective**，
而不是把更多 RaLD 式 cross-attention 层本身当作解决方案。

## 3. 文献与官方代码：哪些只是放大条件，哪些直接惩罚忽略条件

### 3.1 判定标准

若某机制在 \(G(x,c)=G(x)\) 时仍能达到其辅助目标的最优或稳定点，则它不是
anti-ignore guarantee。只有当条件被忽略会直接增加训练损失，或架构使非平凡
输出只能由条件路径产生时，才算对当前问题有因果约束。

| 机制 | 论文与官方/作者代码 | 真实作用 | 忽略条件会直接受罚？ | 对当前任务的判定 |
|---|---|---|---:|---|
| Classifier-free guidance / condition dropout | [CFG](https://arxiv.org/abs/2207.12598)；[DiT official `LabelEmbedder.token_drop` 与 `forward_with_cfg`](https://github.com/facebookresearch/DiT/blob/main/models.py) | 同一模型联合学习 conditional/unconditional score，推理时放大两者差值 | 否 | **放大器。** 若两分支已经相同，guidance 差值为零 |
| FiLM | [paper](https://arxiv.org/abs/1709.07871)；[official code](https://github.com/ethanjperez/film) | 条件产生 feature-wise scale/shift | 否 | 更强注入接口；\(\gamma,\beta\) 可退化为常数 |
| AdaLN-Zero | [DiT](https://arxiv.org/abs/2212.09748)；[official code](https://github.com/facebookresearch/DiT) | 条件产生 shift/scale/residual gates，零初始化改善深层生成训练 | 否 | 稳定但非 anti-collapse；官方实现的 gate 初始就是零 |
| Zero convolution | [ControlNet](https://arxiv.org/abs/2302.05543)；[official code](https://github.com/lllyasviel/ControlNet) | 在冻结 backbone 上渐进学习控制 residual | 否 | 保护预训练模型；训练初始条件贡献严格为零 |
| Gated cross-attention | [Flamingo](https://arxiv.org/abs/2204.14198)；[OpenFlamingo reference implementation](https://github.com/mlfoundations/open_flamingo) | 用可学习 gate 接入跨模态 tokens | 否 | gate 可关闭；没有 mismatch loss 时不保证使用视觉/雷达条件 |
| RaLD radar cross-attention | [paper](https://ojs.aaai.org/index.php/AAAI/article/view/38946)；[official code](https://github.com/MetaIoT-WHU/RaLD) | radar tokens 在 latent diffusion/decoder 中逐层注入 | 否 | 是本项目的生成 backbone 来源，但不是 anti-ignore 目标 |
| Hard information bottleneck | [Deep VIB](https://arxiv.org/abs/1612.00410)；[VQ-VAE](https://arxiv.org/abs/1711.00937) | 压缩中间表示；VQ-VAE 用离散 code 缓解 posterior collapse | 部分 | 只有 decoder 没有其他输入路径时才排除旁路；常数 code 仍可能存在 |
| MoE routing/load balance | [Switch Transformer](https://arxiv.org/abs/2101.03961)；[V-MoE](https://arxiv.org/abs/2106.05974)；[official V-MoE code](https://github.com/google-research/vmoe) | 条件计算、expert specialization、负载均衡 | 否 | load balance 防止 expert 闲置，不等于输出依赖 Cube |
| Mismatched-condition discriminator/ranking | [Reed et al.](https://proceedings.mlr.press/v48/reed16.html)；[official code](https://github.com/reedscot/icml2016)；[AttnGAN](https://openaccess.thecvf.com/content_cvpr_2018/html/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.html)；[official losses](https://github.com/taoxugit/AttnGAN/blob/master/code/miscc/losses.py) | 把 real+wrong-condition 作为负对，matched 必须优于 mismatched | **是** | 最直接、最低成本地对应现有 `condition_cube_drae` shuffle |
| InfoNCE / MI lower bound | [CPC/InfoNCE](https://arxiv.org/abs/1807.03748)；[InfoGAN](https://arxiv.org/abs/1606.03657)；[official InfoGAN code](https://github.com/openai/InfoGAN)；[CLIP](https://arxiv.org/abs/2103.00020)；[official CLIP code](https://github.com/openai/CLIP) | matched condition-output pair 在 batch negatives 中可辨识，提升可估计 MI 下界 | **是，带假设** | 输出表征必须是物理固定、不可藏信息的 geometry histogram |
| MINE | [paper](https://arxiv.org/abs/1801.04062)；[Microsoft Research page](https://www.microsoft.com/en-us/research/publication/mine-mutual-information-neural-estimation/) | 用 critic 估计并最大化高维变量 MI | 是，理论上 | 小数据下 critic 易过拟合，不如固定物理 InfoNCE 稳健 |
| Output-to-condition consistency | [ControlNet++](https://arxiv.org/abs/2404.07987)；[official code](https://github.com/liming-ai/ControlNet_Plus_Plus) | 从生成结果反演 condition，以 cycle reward 直接优化控制一致性 | **是** | 与本项目 point-to-RAED soft splat/Cube cycle 最契合 |
| Standard conditional flow matching | [Flow Matching](https://arxiv.org/abs/2210.02747)；[Meta official library](https://github.com/facebookresearch/flow_matching) | 条件化 velocity field | 不一定 | \(x_t\) 自身泄露 target 时，velocity 可忽略 condition |
| Contrastive flow matching | [paper](https://arxiv.org/abs/2506.05350)；[official DeltaFM code](https://github.com/gstoica27/DeltaFM) | 正确 flow target 外，推离其他样本 flow target，减少 conditional trajectory overlap | **部分/可改成是** | 必须使用同一 \(x_t\)、同一 noise，只替换 Cube condition，才能隔离条件因果作用 |

### 3.2 关键源码观察

1. **DiT/CFG。** 官方 `LabelEmbedder.token_drop` 随机将 class 替换为 null
   embedding；`forward_with_cfg` 在采样时组合 conditional/unconditional
   预测。它没有让两者必须不同的训练项。
2. **DiT/AdaLN-Zero。** 官方 `DiTBlock` 由 condition 产生
   `shift, scale, gate`，再用 gate 乘 attention/MLP residual。该接口表达力强，
   但 gate=0 是合法状态。
3. **ControlNet。** 官方 `make_zero_conv` 对 1 x 1 convolution 做
   `zero_module`，其目的就是初始时不影响 frozen backbone。因此 zero-conv
   解决“稳定接入”，不解决“必须接入”。
4. **CLIP/InfoNCE。** 官方实现归一化两模态 embedding，构造完整
   batch-by-batch cosine logit matrix。负对进入分母，因此所有输出 embedding
   完全相同不能最小化 loss。
5. **AttnGAN/wrong pair。** 官方 `discriminator_loss` 明确计算
   `real_features[:B-1]` 与移位 `conditions[1:B]` 的
   `cond_wrong_errD`。这是和当前 cross-scene condition derangement 最同构的
   先例。
6. **DeltaFM。** 官方 `TripletSILoss` 使用
   `positive_error - lambda * negative_error`，并在 class-conditioned 模式从
   不同 class 采样 negative target。对本项目还需要更严格的“同一个 noisy
   point state，只换 Cube”，否则 point state 本身可泄露目标。
7. **V-MoE。** router/load-balance 约束 token 到 expert 的使用分布；它不比较
   matched 与 mismatched condition，也不保证不同 Cube 导致不同几何。

## 4. 为什么现有常见修复可能无效

### 4.1 只加 condition dropout/CFG

训练目标仍是

\[
\mathcal L_{\mathrm{CFG}}
=
\mathbb E_{b\sim\operatorname{Bernoulli}(p)}
\left[
\ell(f_\theta(x,\tilde c_b),y)
\right],
\quad
\tilde c_b =
\begin{cases}
\varnothing,&b=1\\
c,&b=0.
\end{cases}
\]

如果 measured/local path 已经能预测 \(y\)，conditional 和 unconditional 两臂
都可收敛到同一函数。CFG 只能放大
\(f(x,c)-f(x,\varnothing)\)，无法凭空制造差值。

### 4.2 只把 cross-attention 换成 FiLM/AdaLN

这些模块扩大 \(c\) 调制 hidden state 的方式，但
\(\gamma(c)=1,\beta(c)=0\) 或 residual gate=0 都是合法解。当前问题不是“条件
进不去”，而是主损失没有要求条件必须改变 geometry。

### 4.3 只看 nonzero gradients/attention maps

一个分支可以有微小、非零、有限梯度，同时只影响 confidence、Doppler hidden
state 或小于 5 cm 的坐标扰动。必须以固定反事实评估：

- 保持 measured Cube 和 output count 不变；
- 跨 scene 替换 condition；
- 在 XYZ Chamfer/occupancy mass 上看到可复现退化；
- 同时满足 completeness、outlier、far-range 和 duplicate 门。

### 4.4 只加 MoE/load balance

load-balance 可以让四个 expert 都被调用，但 router 可能按 query index、range
template 或 scene-frequency 路由，而不是按 Cube。即使 routing 随 Cube 变化，
不同 expert 也可能实现相同函数。因此 MoE 必须附带 cross-scene routing
separation 和最终 geometry shuffle gate。

### 4.5 只加 learned MI critic

在 76 个训练帧上，critic 可从 scene ID、range histogram、hidden watermark
识别配对，而 geometry 仍基本不变。优先使用固定、无参数、低分辨率 RAE
occupancy/Doppler measurement operator；只有它通过后才考虑 MINE。

## 5. 所有新 Stage-0 的统一预注册合同

以下 A-E 是**互斥实验臂**。Stage-0 不组合机制，不用某一臂的 validation 结果
调整另一臂超参数。

### 5.1 数据与计算

- 当前冻结 `76 train / 24 validation` scene-held-out split；
- seed 固定 `20260716`；
- test access 保持 false；
- exactly `10,000` output points；
- H200 physical GPU 0 或 2；不用 GPU 1；
- BF16、EMA、每 5 epoch 固定评估；
- 所有预算是根据当前 G1G `41.68M` 参数、单帧 preflight peak
  `8.63 GB` 的**预估**，不是本次实测。

### 5.2 条件反事实

- `condition_cube_drae` 用固定 cross-scene derangement；
- `cube_drae`、target、proposal/cache、随机 noise、solver trajectory 全部保持
  不变；
- 负条件不得来自同一 sequence；
- matched/mismatched 两次推理使用相同精度和相同随机数。

### 5.3 Condition-use gate

主门保持当前冻结定义：

\[
\Delta_{\mathrm{shuffle}}^{\mathrm{frame\ first}}\ge 1.0\%.
\]

另加两个 anti-spike 报告门：

- scene-first median shuffle degradation `>=0.5%`；
- 8 个 validation scenes 中至少 6 个 degradation `>0`。

任何臂若只在 confidence、Doppler head 或少数异常帧上产生差异，而 XYZ
condition-shuffle `<1%`，都判为 condition-use 失败。

### 5.4 Geometry anti-gaming gate

Stage-0 matched-condition 输出必须同时满足：

| Metric | Stage-0 gate |
|---|---:|
| output count | exactly `10,000` |
| duplicate fraction at 5 cm | `<=15%` |
| median completeness | `<=2.50677 m` |
| mean outlier fraction at 2 m | `<=25%` |
| mean far completeness 60-120 m | `<=8.1239 m` |
| unique 2,500 center cells at 5 cm | `>=80%` |

其中 `2.50677 m = 0.7 x 3.5811 m`，沿用 G1G 相对 G1D epoch-15 的 30%
completeness 改善门。进入正式候选后仍需满足统一 final gate（包括
completeness `<=0.65 m`、duplicates `<=10%`），Stage-0 不放宽最终标准。

额外禁止：

- 通过 confidence threshold 丢点；
- 减少 output count；
- best-of-k 选取；
- 只让 mismatched 输出飞出物理边界来制造 rank margin；
- 在低于 5 cm 的不可用抖动中编码 condition。

## 6. 五条互斥、低成本 Stage-0

## A. G1G + cross-scene mismatched geometry ranking

**类别：** 无需新模型；直接训练目标惩罚条件忽略。  
**优先级：** 最高之一。

### A.1 机制与公式

复用现有 `G1GConditionExclusiveHierarchy` 和
`condition_cube_drae` 参数。对每个 matched/mismatched pair 定义

\[
d_i^+
=
D_{\mathrm{strat}}(G(L_i,C_i),Y_i),
\quad
d_i^-
=
D_{\mathrm{strat}}(G(L_i,C_{\pi(i)}),Y_i),
\]

\[
\mathcal L_{\mathrm{rank}}
=
\frac1N\sum_i
\left[
m\,\operatorname{sg}(d_i^+)
+d_i^+
-d_i^-
\right]_+,
\quad m=0.02.
\]

`sg` 表示 stop-gradient。训练 margin `2%` 比评估门 `1%` 稍强，避免边界噪声。
建议

\[
D_{\mathrm{strat}}
=
\operatorname{CD}
+0.5\,\operatorname{Completeness}
+0.25\,D_{\mathrm{range\ mass}},
\]

三项按 G1G control 的初始尺度归一化。`range_mass` 使用固定 0-20、20-40、
40-60、60-120 m bins，防止仅移动近距少数点满足 margin。

总损失：

\[
\mathcal L_A
=
\mathcal L_{\mathrm{G1G}}
+0.25\,\mathcal L_{\mathrm{rank}}.
\]

由于每个负 Cube 在其自己的样本上也是 matched positive，模型不能长期把某个
scene 的输出任意推到边界而不承担其正样本几何损失。

### A.2 代码接口

- 复用：
  `model(cube_drae=L_i, condition_cube_drae=C_i)`；
- 第二次 forward：
  `model(cube_drae=L_i, condition_cube_drae=C_pi)`；
- 新 loss 函数建议：
  `mismatched_condition_geometry_rank(positive_output, negative_output, target)`；
- 只需改 loss/trainer，不新增 model class；
- pair map 直接复用
  `cross_scene_condition_indices(...)`。

### A.3 Stage-0、预算与对照

- 10 epochs，760 matched + 760 mismatched forwards；
- 预估 `2.0-3.5 H200 GPU-hours`；
- 两个 graph 顺序 forward，预估 peak `17-22 GB`；
- control A0：原始 G1G，同 seed、10 epochs；
- control A1：`lambda_rank=0`，确认双 forward 本身不改变结果；
- 不做 margin/weight sweep。

### A.4 晋级与失败解释

除统一门外，要求 validation rank satisfaction `>=80%`。若训练 rank 已满足而
XYZ shuffle `<1%`，说明模型主要在不影响使用价值的几何方向上投机，A 关闭。
若 shuffle 通过但 matched geometry 失败，说明 condition 被使用了，但直接
regression representation/目标仍不足，不能把它解释为 anti-collapse 成功后
继续调权重。

## B. 固定物理 RAE histogram InfoNCE

**类别：** 无需新 backbone；MI/contrastive objective。  
**优先级：** 最高，计算最便宜。

### B.1 为什么不用 learned output encoder

learned point encoder 可以把 condition 藏进 confidence、Doppler hidden state
或微小坐标 watermark。这里定义固定、无参数、可微的物理表征：

- \(h_C(C)\)：`log1p` Cube 沿 Doppler 积分，再 average-pool 到
  `16 x 7 x 3 = 336` spatial bins，归一化为概率；
- \(h_Y(\hat Y)\)：10k XYZ 经 calibrated XYZ->continuous RAE、unit-mass
  trilinear soft splat 到相同 336 bins；**不使用 confidence 权重**；
- 两者取 square-root probability 后做 cosine similarity，相当于稳定的
  Hellinger geometry embedding。

### B.2 公式

对 queue 中 \(K=16\) 个跨 scene 条件：

\[
s_{ij}
=
\frac{
\langle \sqrt{h_Y(\hat Y_i)},\sqrt{h_C(C_j)}\rangle
}{\tau},
\quad \tau=0.07,
\]

\[
\mathcal L_{\mathrm{NCE}}
=
-\frac1N\sum_i
\log
\frac{\exp(s_{ii})}
{\sum_{j=1}^{K}\exp(s_{ij})}.
\]

在标准负采样假设下，
\(I(h_Y;h_C)\ge \log K-\mathcal L_{\mathrm{NCE}}\)。如果所有输出 histogram
相同，正对不能在分母中稳定胜过跨 scene 负对，因此忽略条件会直接受罚。

总损失：

\[
\mathcal L_B
=
\mathcal L_{\mathrm{G1G}}
+0.10\,\mathcal L_{\mathrm{NCE}}.
\]

Cube histogram 全部 `stop_gradient`；queue 只缓存 condition histogram，不缓存
learned output feature。

### B.3 代码接口

- 复用现有 calibrated axes 和 `soft_splat_raed`；
- 新函数建议：
  `fixed_cube_histogram(cube_drae, out_shape=(16,7,3))`；
- 新函数建议：
  `fixed_point_histogram(xyz_m, axes, out_shape=(16,7,3))`；
- trainer 保存 16 个跨 scene Cube histograms 的 deterministic FIFO；
- model forward 不变，不新增参数。

### B.4 Stage-0、预算与对照

- 10 epochs，只有 matched forward；
- 预估 `1.2-2.0 H200 GPU-hours`；
- peak memory 比 G1G 增加 `<1 GB`；
- control B0：原始 G1G；
- control B1：将 queue 中全部 condition histogram 替换为 batch mean，验证
  改善不是额外正则造成；
- 报告 `log(16)-L_NCE`、top-1 pair retrieval accuracy、RAE range-mass error。

### B.5 晋级与失败解释

要求 condition retrieval top-1 `>=25%`（chance `6.25%`），同时通过统一
shuffle/geometry 门。若 retrieval 上升而 XYZ shuffle 仍 `<1%`，说明 336-bin
表征仍允许小范围 mass watermark；B 关闭，不提高分辨率或换 learned critic。

## C. Condition-delta hard bottleneck allocator

**类别：** 架构级；移除 condition-independent coordinate residual。  
**优先级：** 中高。

### C.1 结构

保留固定 RAE positional queries \(P\)，但删除 learned query content 到
coordinate head 的 residual path。定义

\[
A_i(C)=\operatorname{CrossAttn}(P_i,E(C)),
\]

\[
\Delta A_i(C)=A_i(C)-A_i(C_\varnothing),
\]

\[
r_i(C)
=
b\cdot\tanh
\left(
W_o\,
\operatorname{AdaLN}
\left(
\Delta A_i(C);
\operatorname{pool}(E(C)-E(C_\varnothing))
\right)
\right),
\]

\[
x_i(C)=T_i+r_i(C).
\]

约束：

- \(P_i,T_i\) 固定，不训练；
- `coordinate_head` 无 bias；
- coordinate head 只读 \(\Delta A\)，不读 \(P_i\)、learned query 或 FFN
  residual；
- cross-attention residual gate 固定为 1，不使用可学零 gate；
- null Cube 下 \(r_i(C_\varnothing)=0\) 是结构恒等式；
- allocation 后仍用当前 G1G 的半-bin bounded four-child refinement。

这能证明**任何非零 center residual 都来自 Cube 条件**。它仍不能证明模型不会
选择全部零 residual，因此最终 shuffle gate 仍不可省略。

### C.2 与 FiLM/AdaLN/ControlNet 的关系

- 借用 FiLM/AdaLN 的条件调制；
- 借用 ControlNet 的“相对 null branch”思想；
- 关键区别是删除 condition-independent coordinate head 输入，而不是只把
  condition residual 初始化为零；
- 仍沿用 RaLD 的 radar-token cross-attention，但将 coordinate dependency
  收紧成可审计的 condition-delta 路径。

### C.3 代码接口

- 新 model：
  `G1GConditionDeltaHierarchy`；
- `allocate_centers(condition_cube_drae, null_cube_drae)`；
- `null_cube_drae` 为固定全零且经过同一 normalization/encoder；
- 输出额外记录：
  `null_center_residual_max_abs`,
  `condition_delta_norm_per_layer`,
  `condition_delta_coordinate_fraction`；
- preflight 必须断言 null residual `<=1e-7`。

### C.4 Stage-0、预算与对照

- 20 epochs，一 seed；
- 预估 `2.5-4.0 H200 GPU-hours`；
- peak `10-13 GB`；
- control C0：当前 G1G；
- control C1：同架构但恢复 learned-query residual；
- control C2：zero Cube，验证输出精确退回固定 templates；
- 不同时加入 ranking/InfoNCE。

### C.5 晋级与失败解释

统一门全部通过才晋级。若 null identity 通过而 condition shuffle `<1%`，说明
固定 template 已经足够接近数据集平均几何，或 Cube encoder 输出 scene-invariant；
C 关闭，不能通过增加 AdaLN 层数重开。

## D. Cube-routed sparse center experts

**类别：** 架构级 MoE；带 routing anti-collapse。  
**优先级：** 低于 A/B/C/E。

### D.1 机制与公式

将 center residual/patch growth 分成 4 个小 expert，Full-RAED condition 生成
top-2 routing：

\[
\pi_i(C)=
\operatorname{softmax}
\left(
R_i(\operatorname{pool}(E(C)))/\tau_r
\right),
\]

\[
r_i(C)=
\sum_{e\in\operatorname{Top2}(\pi_i)}
\pi_{i,e}(C)F_e(P_i,E(C)).
\]

使用 V-MoE/Switch 的 load-balance term，但明确它只防 expert collapse：

\[
\mathcal L_{\mathrm{lb}}
=
4\sum_e f_e p_e.
\]

再加入跨 scene router separation：

\[
\mathcal L_{\mathrm{route-cf}}
=
\left[
m_r
-\operatorname{JS}
\left(
\bar\pi(C_i),\bar\pi(C_{\pi(i)})
\right)
\right]_+,
\quad m_r=0.05.
\]

总损失：

\[
\mathcal L_D
=
\mathcal L_{\mathrm{G1G}}
+0.01\mathcal L_{\mathrm{lb}}
+0.05\mathcal L_{\mathrm{route-cf}}.
\]

router separation 只证明条件影响 expert assignment，不能证明 expert outputs
不同，因此仍以 XYZ shuffle 为主门。

### D.2 代码接口

- 新 model：
  `G1GConditionRoutedExperts`；
- 4 个 expert 只替换 center residual head + child patch FFN，不复制
  Full-RAED encoder；
- 输出 `router_probability`, `expert_load`, `expert_output_pairwise_distance`；
- hard negatives 优先从 target range-mass 差异最大的跨 scene frame 选择，但
  pair map 在训练前冻结。

### D.3 Stage-0、预算与对照

- 10 epochs；
- 总参数限制 `<55M`，top-2 active；
- 预估 `2.5-4.0 H200 GPU-hours`，peak `14-20 GB`；
- control D0：参数量匹配的 dense FFN；
- control D1：MoE + load balance，无 router separation；
- control D2：完整 D；
- 不与 A/B 的 output-level contrastive loss 联用。

### D.4 晋级与失败解释

除统一门外，要求每个 expert validation load 在 `[10%,40%]`，且不同 condition
的 expert output pairwise distance 非零。若 routing 指标通过而 geometry
shuffle 失败，说明 experts 功能等价；D 关闭。76 个训练帧下若出现按 scene
memorization，直接关闭，不增加 expert 数。

## E. RaLD-token-conditioned contrastive center flow matching

**类别：** 生成式；RaLD 主干 + mismatched-condition contrastive flow。  
**优先级：** 高于 MoE，低于 A/B；用于 direct regression family 不能同时解决
support 和 condition use 时。

### E.1 为什么只生成 2,500 centers

直接在 10k x 3 上做 diffusion/flow 会放大 H200 成本和 unordered
correspondence 噪声。Stage-0 只生成 2,500 RAE centers，再复用 G1G 的四子点
bounded patch decoder：

1. Full-RAED -> 336 RaLD-style radar tokens；
2. 2,500 full-frustum source centers \(z_0\)；
3. 6-layer condition-in-every-block flow Transformer；
4. 4 或 8 个 Heun steps；
5. 2,500 centers x 4 children -> exactly 10k。

这借用 RaLD 的条件生成骨架，但不重开已经 gate-failed 的 K-Radar occupancy
VAE/latent decoder family。

### E.2 公式

使用固定 point-set coupling 后的 target centers \(z_1\)：

\[
z_t=(1-t)z_0+t z_1,
\qquad
u_i=z_1-z_0.
\]

matched velocity：

\[
v_i^+
=
v_\theta(z_t,t,C_i),
\qquad
\mathcal L_i^+
=
\|v_i^+-u_i\|_2^2.
\]

保持同一个 \(z_t,t,z_0,z_1\)，只替换 condition：

\[
v_i^-
=
v_\theta(z_t,t,C_{\pi(i)}),
\qquad
\mathcal L_i^-
=
\|v_i^--u_i\|_2^2.
\]

\[
\mathcal L_{\mathrm{mismatch\ CFM}}
=
\frac1N\sum_i
\left[
m_f+\mathcal L_i^+-\mathcal L_i^-
\right]_+,
\quad
m_f=0.02\,\operatorname{sg}(\mathcal L_i^+).
\]

\[
\mathcal L_E
=
\mathcal L_{\mathrm{FM}}
+0.05\mathcal L_{\mathrm{mismatch\ CFM}}
+\mathcal L_{\mathrm{G1G\ patch}}.
\]

这比直接照搬 DeltaFM 更严格：DeltaFM 将预测 flow 推离其他样本 target；
本文让正负两次 forward 的 noisy point state 完全相同，唯一变化是 Cube，因而
condition 被忽略时
\(\mathcal L_i^+=\mathcal L_i^-\)，hinge 必然非零。

### E.3 CFG 的受限使用

训练可固定 `condition_dropout=0.1`，只为了得到 null branch。Stage-0 主结果必须
在 guidance scale `1.0` 通过 condition-shuffle；另报告 `1.5` 和 `2.0`，但不得
用后两者挽救 scale-1 的训练失败，也不得在 validation 上选择 guidance。

### E.4 代码接口

- 新 model：
  `FullRAEDContrastiveCenterFlow`；
- 复用 RaLD-style 336 radar tokens 和现有 time embedding/cross-attention；
- 建议复用
  [Meta flow_matching library](https://github.com/facebookresearch/flow_matching)
  的 probability path/solver API；
- loss 参考
  [DeltaFM `triplet_loss.py`](https://github.com/gstoica27/DeltaFM/blob/main/triplet_loss.py)，
  但 negative 改为 `same z_t + mismatched Cube`；
- `sample_centers(cube, noise, steps, guidance_scale)` 显式接收 fixed noise；
- 不做 best-of-k。

### E.5 Stage-0、预算与对照

- 10 epochs；4-step 为主评估，8-step 只作固定附加报告；
- 预估 `4-6 H200 GPU-hours`；
- peak `28-45 GB`；
- control E0：vanilla conditional FM；
- control E1：vanilla FM + CFG，仅 scale 1；
- control E2：mismatched-condition CFM，scale 1；
- 三臂相同 encoder、Transformer、source noise、coupling 和 solver。

### E.6 晋级与失败解释

除统一门外：

- matched velocity MSE 必须低于 mismatched MSE 至少 `2%`；
- 4-step 与 8-step 的 Chamfer 差异 `<5%`，避免只靠更多求解步；
- 同 noise、不同 Cube 的 endpoint separation 必须分布在至少 10% centers，
  不能集中在少数点。

若 velocity rank 通过而 endpoint XYZ shuffle `<1%`，说明 solver/patch decoder
重新抹掉了 condition；E 关闭。若 shuffle 通过但 geometry gate 失败，说明生成
轨迹确实使用 Cube，但 target coupling 或几何 representation 不够，不允许通过
CFG scale/best-of-k 掩盖。

## 7. Stage-0 选择顺序与并行策略

### 7.1 当前 G1G 尚未完成时

1. 原样完成当前 G1G 20 epochs，保留它作为无 anti-collapse loss 的独立基线；
2. 不因两步 smoke shuffle 近零提前改训练目标；
3. 若 G1G 最终 `>=1%` 且几何门通过，当前 bypass 假设已被 G1G 结构解决，
   A-E 不应作为必需组件；只做一条最便宜的 objective control。

### 7.2 G1G 最终 shuffle `<1%` 时

按两张允许的 H200 并行：

| Wave | GPU arm 1 | GPU arm 2 | 决策 |
|---|---|---|---|
| 1 | B: fixed physics InfoNCE | A: mismatch geometry rank | 最低工程成本，直接检验目标函数能否迫使条件使用 |
| 2 | C: condition-delta hard bottleneck | E: contrastive center flow | 分别检验架构强制和生成式强制 |
| 3 | D: condition-routed MoE | 复核前两波 survivor | 仅当前四臂都揭示条件容量不足而非 geometry failure 时启动 |

每条臂先跑自己的固定 Stage-0，不在 Stage-0 内融合。若多条通过，按以下顺序
选择：

1. 完整统一 geometry gate；
2. scene-first condition-shuffle；
3. H200 GPU-hours；
4. 参数量和推理步数；
5. 最后才考虑机制融合。

## 8. 推荐决策

### 8.1 最值得立即实现的两个方向

**B：固定物理 histogram InfoNCE** 是第一选择。它没有第二次 model forward，
不引入 learned critic，直接把 generated geometry 与 measured Full-RAED 的
coarse physical mass 配对，并用跨 scene negatives 排除统一平均输出。

**A：mismatched geometry ranking** 是最直接的反事实监督。它和现有
`condition_cube_drae` API 完全对齐，能把当前离线评估门变成训练信号，代价是
约两倍 forward。

### 8.2 最有解释力的架构方向

**C：condition-delta hard bottleneck** 比“再加 FiLM/AdaLN/cross-attention”
更严格。它让 null Cube 下 coordinate residual 精确为零，且 coordinate head
没有 learned-query 旁路。即使最终失败，也能明确区分：

- Cube encoder 没学到 scene-specific state；
- 固定 template 已足够拟合平均场景；
- local bounded refinement 抹掉了 global allocation。

### 8.3 最值得保留的 RaLD 生成方向

**E：RaLD-token-conditioned contrastive center flow**。它保留 RaLD 最有价值的
radar-token conditional generation，但把生成目标从失败的 latent occupancy
decoder 改为 2,500 full-frustum centers，并用 same-state mismatched Cube
hinge 直接排除 condition-free velocity。它是本文唯一应该进入生成式 Stage-0
的路线。

### 8.4 暂不推荐

- 只加 CFG/condition dropout；
- 只把 cross-attention 换成 FiLM/AdaLN-Zero；
- 只增加 cross-attention 深度；
- 只看 condition gradient 或 attention entropy；
- 只加 MoE load balance；
- 在 76 帧上训练自由 MINE critic；
- 用 guidance scale 或 best-of-k 在推理时补救训练期 condition collapse。

## 9. 可证伪的最终判断

本专项不是要证明某个 conditioning module “更先进”，而是要检验：

> 在保持 measured Cube、target、point count、noise 和 solver path 不变时，
> 跨场景替换 Full-RAED condition 是否会稳定、分布式地恶化 10k XYZ 几何，
> 且 matched 输出仍通过 completeness/outlier/far-range/duplicate 门。

若 A/B/C/E 均无法达到 `>=1%`，合理结论不是继续堆条件模块，而是：

1. 当前 76-frame supervision 对 condition-specific dense geometry
   不可辨识；或
2. measured/local Cube 与 global condition 的信息高度冗余，shuffle 门测的是
   一个不必要分支；或
3. target/cache 本身的跨 scene 几何差异不足以支持该因果主张。

此时应转向数据可辨识性 oracle：用固定物理 histogram 计算
\(I(C;Y)\)/pair retrieval upper bound，并暂停所有大模型条件注入实验。

## 10. 主要参考文献与代码

- RaLD: [paper](https://ojs.aaai.org/index.php/AAAI/article/view/38946),
  [official code](https://github.com/MetaIoT-WHU/RaLD)
- Classifier-Free Diffusion Guidance:
  [paper](https://arxiv.org/abs/2207.12598)
- DiT/AdaLN-Zero/CFG:
  [paper](https://arxiv.org/abs/2212.09748),
  [official code](https://github.com/facebookresearch/DiT)
- FiLM:
  [paper](https://arxiv.org/abs/1709.07871),
  [official code](https://github.com/ethanjperez/film)
- ControlNet:
  [paper](https://arxiv.org/abs/2302.05543),
  [official code](https://github.com/lllyasviel/ControlNet)
- ControlNet++:
  [paper](https://arxiv.org/abs/2404.07987),
  [official code](https://github.com/liming-ai/ControlNet_Plus_Plus)
- Flamingo:
  [paper](https://arxiv.org/abs/2204.14198),
  [OpenFlamingo reference code](https://github.com/mlfoundations/open_flamingo)
- CPC/InfoNCE:
  [paper](https://arxiv.org/abs/1807.03748)
- InfoGAN:
  [paper](https://arxiv.org/abs/1606.03657),
  [official code](https://github.com/openai/InfoGAN)
- CLIP:
  [paper](https://arxiv.org/abs/2103.00020),
  [official code](https://github.com/openai/CLIP)
- Generative Adversarial Text-to-Image Synthesis:
  [paper](https://proceedings.mlr.press/v48/reed16.html),
  [official code](https://github.com/reedscot/icml2016)
- AttnGAN/DAMSM:
  [paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.html),
  [official code](https://github.com/taoxugit/AttnGAN)
- Deep Variational Information Bottleneck:
  [paper](https://arxiv.org/abs/1612.00410)
- VQ-VAE:
  [paper](https://arxiv.org/abs/1711.00937)
- MINE:
  [paper](https://arxiv.org/abs/1801.04062)
- Switch Transformer:
  [paper](https://arxiv.org/abs/2101.03961)
- V-MoE:
  [paper](https://arxiv.org/abs/2106.05974),
  [official code](https://github.com/google-research/vmoe)
- Flow Matching:
  [paper](https://arxiv.org/abs/2210.02747),
  [Meta guide/code](https://github.com/facebookresearch/flow_matching)
- Contrastive Flow Matching:
  [paper](https://arxiv.org/abs/2506.05350),
  [official DeltaFM code](https://github.com/gstoica27/DeltaFM)

