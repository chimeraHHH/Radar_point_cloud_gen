# RaLD 深度迁移审计：从官方实现到 K-Radar Cube->10k dense point

> 审计完成日：2026-07-29
> RaLD 官方代码冻结版本：`ffec4b41241391734b1eda5c093de843c909eb8e`
> 任务边界：只做论文、代码、配置和依赖审计；本报告未运行任何科学计算。
> 当前项目证据边界：G1D/G1F 指标来自既有正式运行和归档，不以 RaLD 的 ColoRadar 数字替代 K-Radar 证据。

## 0. 结论先行

当前 G1D 并没有完整继承 RaLD 最可能解决平台期的两项机制：

1. **RaLD 的 occupancy decoder 是 coordinate-only decoder。**
   query 只提供空间坐标，场景内容必须经由 latent 传入。当前 G1D 则在每个 proposal、coarse query 和 final query 上直接读取原始 Cube 的 64-bin 局部谱、绝对能量和 range，因此即使 24 层全局 Full-RAED cross-attention 全部存在，模型仍能绕过它。`condition shuffle ~=0.016%` 与这个结构性旁路完全一致。
2. **RaLD 的推理查询域不是一个小型强能量 proposal pool。**
   官方配置先查询 `500k` 个全域随机点，再加入 `700k` 个低阈值 CFAR 邻域点，并对首轮正区域再查询 `500k` 个 refinement 点。当前 G1D 只有 `32k` 个强能量 NMS 邻域候选。G1F 的 GT oracle 已证明该 pool 的远距 2 m recall 只有 `17.80%`，所以 selector、top-k 或 loss 单独优化无法补出不存在的支撑。

因此，本审计最推荐的新路线是：

> **R-A：RaLD-wide condition-exclusive frustum occupancy field。**
> 保留现有 Full-RAED Cube encoder，但把 allocation decoder 恢复为 coordinate-only；把候选域恢复为 RaLD 式“全域随机 + 低阈值雷达邻域 + 二次 refinement”，并针对 120 m K-Radar 做固定 range-stratified 配额。先做不训练的 expanded-pool GT support oracle；只有 oracle 通过才训练 10 epoch occupancy field。

这条路线同时攻击 condition bypass、远距支撑不足和重复点三个根因，而且不重启已经被 G1E-D0 否决的完整 RaLD VAE/EDM。

## 1. 审计源与可复核链接

### 1.1 RaLD 原始来源

- 论文：[RaLD: Generating High-Resolution 3D Radar Point Clouds with Latent Diffusion, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/download/38946/42908)
- 官方仓库：[MetaIoT-WHU/RaLD](https://github.com/MetaIoT-WHU/RaLD)
- 本次逐行审计 commit：[ffec4b41241391734b1eda5c093de843c909eb8e](https://github.com/MetaIoT-WHU/RaLD/tree/ffec4b41241391734b1eda5c093de843c909eb8e)
- 官方 README 明确列出的主要来源依赖：[3DShape2VecSet、MAR、DiT、ColoRadar Toolkit 和 OpenPCDet](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/README.md#L73-L79)

### 1.2 关键依赖

- 3DShape2VecSet commit `8df9b7a55c42d4dcad152294755250a2ab1e34e5`：
  [set-latent encoder 与 coordinate-only occupancy decoder](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L338-L394)
- DiT commit `ed81ce2229091fd4ecc9a223645f95cf379d582b`：
  [condition dropout](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L67-L94) 和
  [classifier-free guidance](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L250-L266)
- EDM commit `008a4e5316c8e3bfe61a62f874bddba254295afb`：
  [18-step Heun sampler](https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/generate.py#L23-L60) 和
  [EDM weighted denoising loss](https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/loss.py#L61-L80)

### 1.3 当前项目证据

- G1D 协议与当前实现映射：`docs/g1d_rald_query_field_protocol.md`、`docs/g1d_rald_source_map.md`
- G1F oracle：`artifacts/g1/g1f_f0_ca60d76.json`
- G1E-D0 no-go 与 claim boundary：`paper/claim_evidence_ledger.md`
- G1G 条件独占候选：`artifacts/idea/pre_idea_drafts/g1g_condition_exclusive_hierarchy.md`

## 2. 当前平台期的代码级诊断

当前观测：

| 指标 | 当前值 | 主要含义 |
|---|---:|---|
| condition shuffle degradation | `~0.016%` | 全局条件几乎可被忽略 |
| Chamfer | `~5.56 m` | 整体几何仍远离目标 |
| completeness | `~3.59 m` | GT 大量区域没有生成支撑 |
| far completeness | `~8.78 m` | 远距支撑最弱 |
| duplicate fraction | `~29.9%` | center/child 分配和局部 offset 坍缩 |
| G1F far GT recall at 2 m | `17.80%` | 32k pool 自身不含足够远距候选 |

### 2.1 条件旁路不是“注意力层不够多”

当前 `RaLDQueryField.forward` 的信息流是：

```text
measured Cube
  -> top-energy NMS proposal coordinates
  -> proposal local 64-bin spectrum + absolute energy + range
  -> proposal tokens
  -> mixed latent

measured Cube
  -> coarse/final query local 64-bin spectrum + absolute energy + range
  -> decoder query tokens

condition Cube
  -> 336 global Full-RAED tokens
  -> 24 cross-attention blocks
```

代码证据：

- `query_tokens` 明确拼接 local spectrum、absolute energy 和 range：
  `code/models/rald_query_field.py:502-564`
- proposal tokens 来自 measured Cube：
  `code/models/rald_query_field.py:765-779`
- coarse 和 final query 再次读取 measured Cube：
  `code/models/rald_query_field.py:802-857`
- shuffled `condition_cube_drae` 只替换 global radar encoder 输入：
  `code/models/rald_query_field.py:728-791`

因此，现有 condition-shuffle 实验实际上在问：

> “当 proposal 坐标、proposal 局部谱、coarse query 局部谱和 final query 局部谱仍全部来自正确 Cube 时，额外打乱 336 个 global token 有多大影响？”

`0.016%` 不是意外，而是当前架构允许的最容易解。

RaLD 官方链路没有这个旁路：

- latent diffusion 通过 radar encoder 得到雷达条件；
- occupancy decoder 的 query 输入只做坐标 Fourier embedding；
- decoder query cross-attend latent，不再直接查询 radar spectrum。

对应官方代码：

- [RaLD `process_radar_cond`](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/model/models_radar_generation.py#L363-L407)
- [每个 latent block 的 radar cross-attention](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/model/models_radar_generation.py#L133-L169)
- [occupancy decoder 只使用 point-coordinate embedding 和 latent](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/model/models_ae.py#L408-L424)

### 2.2 G1F 已经否决“小 pool 内重排”假设

G1F 使用 GT 的不可达 oracle 在完整 `32k -> 10k` pool 内做最有利选择，仍得到：

| 指标 | G1F oracle | final gate |
|---|---:|---:|
| median Chamfer | `2.8863 m` | `<=2.50 m` |
| outlier | `24.853%` | `<=25%` |
| median completeness | `1.6513 m` | `<=0.65 m` |
| far completeness | `8.6533 m` | `<=8.0 m` |
| duplicate | `14.015%` | `<=10%` |

这意味着：

- hard top-k 不是唯一问题；
- balanced transport 不能创造 pool 外的点；
- 继续只调 selector、score、OT temperature 或 top-k threshold 不具备科学依据；
- 下一条 RaLD 迁移路线必须改变 **candidate support representation**。

### 2.3 重复点来自固定 parent-child 结构，而不是 RaLD 原生行为

当前 G1D 固定：

```text
2500 selected coarse centers x 4 tetrahedral children = 10k points
```

若多个 coarse center 邻近，或四个 child 的 learned offset 相互靠拢，重复会同时发生在 parent 间和 parent 内。现有 global repulsion 权重只有 `0.02`，而 geometry Chamfer 权重为 `1.0`：
`code/losses/rald_query_field.py:404-420,573-588`。

RaLD 官方并没有 `2500 x 4` decoder，也没有 duplicate loss。它在大量连续 query 上输出 occupancy logit，再以阈值筛选：
[RaLD inference query and threshold path](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/engine_generation.py#L249-L310)。

因此，可以借鉴的是“唯一 query 的 occupancy field”，不能声称 RaLD 已提供 fixed-10k 去重机制。

## 3. RaLD 的真实模块链

### 3.1 Frustum LiDAR autoencoder

论文 Sec. 3 将 LiDAR 点转换到 `(range, azimuth, elevation)`，以 polar frustum 而不是 Cartesian voxel 定义 occupancy。论文 Eq. (6)-(8) 和 Figure 3 的核心是：

- angular cell 与传感器采样方向一致；
- 近距密、远距疏的 LiDAR 采样不再被 Cartesian voxel 的均匀体积扭曲；
- 同一 ray 上的占用关系提供潜在遮挡结构。

官方配置使用：

- frustum size `[0.05 m, 0.25 deg, 0.5 deg]`
- `10,000` LiDAR points
- `6.25%` positive occupancy queries
- `512 x 32` latent

来源：

- [AE frustum、10k 和 6.25% 配置](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/configs/ae/ae_indoor_cfg_aniso_mix_view_cone.yml#L31-L45)
- [positive/empty query sampling](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/datasets/aligned_coloradar/Coloradar_dataset.py#L237-L294)

### 3.2 Order-invariant hybrid set latent

RaLD 在 3DShape2VecSet 基础上新增 `query_type='mix'`：

```text
Qs = learned static tokens
Qd = learned dynamic tokens
dynamic = CrossAttn(Qd, point embeddings)
Qenc = Proj(Qs + dynamic)
latent = CrossAttn(Qenc, point embeddings) + Qenc
```

官方符号：

- `KLAutoEncoder.s_latents`
- `KLAutoEncoder.d_latents`
- `KLAutoEncoder.mix_attn_layer`
- `KLAutoEncoder.query_proj`
- `KLAutoEncoder.encode`

精确代码：
[RaLD hybrid latent implementation](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/model/models_ae.py#L284-L405)。

要注意：这个 hybrid latent 原本编码的是 **目标 LiDAR point set**，并不是直接把 Cube 变成点。G1D 把 point embeddings 替换成 radar-proposal tokens，是合理改造，但不再是 source-faithful VAE。

### 3.3 Radar Cube encoder 与 RAE positional conditioning

官方生成器：

1. 对雷达 3D tensor 做 3D Conv/ResNet downsampling；
2. 将 feature 投影到 token channel；
3. 分别加入 learned range、azimuth、elevation embeddings；
4. flatten 为 condition tokens；
5. 在每个 DiT block 中做 latent-to-radar cross-attention。

代码：

- [3D radar encoder](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/model/models_radar_encoder.py#L137-L241)
- [RAE axis embeddings and token flatten](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/model/models_radar_generation.py#L351-L407)
- [radar condition at every Transformer block](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/model/models_radar_generation.py#L133-L169)

当前项目的 `FullRAEDRadarTokenEncoder` 已经实现：

- 64 Doppler bins 的 `1x1x1` spectral projection；
- native K-Radar 3D encoder；
- range/azimuth/elevation embeddings；
- `16 x 7 x 3 = 336` tokens。

对应：`code/models/rald_matched.py:369-411`。

结论：**Cube encoder 和 RAE embeddings 已经迁移，继续增加同类 encoder 深度不是首要杠杆。**

### 3.4 Radar-conditioned latent diffusion

RaLD 的 generator 不是直接 point decoder，而是：

```text
radar Cube -> radar tokens
Gaussian noise -> 512 x 32 latent
24-block radar-conditioned EDM denoiser
18-step Heun sampling -> denoised latent
coordinate-only occupancy decoder -> point field
```

官方代码：

- [EDM loss](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/model/models_radar_generation.py#L277-L295)
- [EDM preconditioner and radar condition](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/model/models_radar_generation.py#L314-L430)
- [18-step sampler invocation](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/model/models_radar_generation.py#L435-L449)

当前 G1D 是 deterministic `512 x 512` working tokens，不是 Gaussian `512 x 32` latent，也没有 EDM。这个差异必须继续如实保留。

### 3.5 Wide query domain 与 radar-guided initialization

RaLD 论文 Sec. 3 “Decoding with Radar-Guided Query Initialization” 明确使用：

- low-threshold CFAR 提供可能的 object region；
- 额外 random free-space queries 保持 completeness；
- 首轮 occupancy positive region 再 refinement。

官方 eval 配置实际冻结为：

- `500,000` random queries；
- `700,000` augmented CFAR helper queries；
- `500,000` refinement queries。

来源：

- [eval candidate counts](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/configs/generation/ge_indoor_cfg_aniso_mix_view_cone_unfreeze_enc_ints_only_eval.yml#L155-L169)
- [CFAR helper count](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/configs/generation/ge_indoor_cfg_aniso_mix_view_cone_unfreeze_enc_ints_only_eval.yml#L29-L36)
- [query/refinement execution](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/engine_generation.py#L249-L310)

论文 Table 3 也显示 radar encoder 与 CFAR query initialization 都带来独立收益。对当前项目而言，这是比“再加一层 attention”更直接的迁移证据。

### 3.6 Density、confidence 与训练目标

RaLD 官方输出没有独立 density head 或 calibrated confidence head：

- `KLAutoEncoder.to_outputs` 只输出一个 occupancy logit；
- inference 以 `logit > 0` 判定 occupied；
- 输出数量由候选密度与 threshold 联合决定，是 variable-count。

代码：

- [single occupancy output head](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/model/models_ae.py#L341-L424)
- [zero-threshold occupancy export](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/engine_generation.py#L274-L310)

所以：

- RaLD occupancy logit 可迁移为 allocation score；
- 它不能直接当作当前协议下的 calibrated confidence；
- fixed-10k confidence、Doppler distribution、PCE 和 Cube reprojection 都是本项目新增职责。

AE 训练目标为：

```text
L_AE = 0.1 * BCE_positive
     + 1.0 * BCE_empty
     + 1e-3 * KL
```

代码：

- [classwise BCE and KL](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/engine_ae.py#L48-L86)
- [loss weights](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/configs/ae/ae_indoor_cfg_aniso_mix_view_cone.yml#L54-L65)

Generator 只有 latent EDM weighted MSE，没有：

- Chamfer；
- far-range loss；
- duplicate/repulsion；
- condition shuffle margin；
- per-point confidence calibration；
- Doppler loss。

因此，RaLD objective 能作为 representation/generation objective，不能直接保证当前几何门。

## 4. 逐模块迁移判定

| RaLD 模块 | 判定 | 当前项目状态 | 迁移建议 |
|---|---|---|---|
| 3D Cube encoder | 可直接迁移，已完成 | `FullRAEDRadarTokenEncoder` 已使用 native Cube | 复用，不再单独加深作为主实验 |
| range/azimuth/elevation learned embeddings | 可直接迁移，已完成 | 336 tokens 已含三轴 embedding | 必须按 K-Radar shape 重新训练，不能加载 ColoRadar embedding |
| 每层 latent-to-radar cross-attention | 可直接迁移，已完成 | G1D 24 层均存在 | 单独存在不等于 condition use；必须移除 decoder local-Cube bypass |
| hybrid static/dynamic latent | 可迁移但语义需改 | G1D 用 proposal tokens 代替 LiDAR point set | 保留 set ordering 思想，不声称是官方 VAE |
| coordinate-only arbitrary occupancy decoder | **应直接迁移，当前缺失** | 当前 query decoder 额外读取 local spectrum/energy/range | 作为 R-A 核心，allocation 前禁止 local Cube |
| frustum occupancy | 可迁移，已部分完成 | 当前使用 RAE query/cell | 对 120 m 必须重设 range granularity 和分层采样 |
| `500k random + 700k CFAR + 500k refine` query domain | **应迁移并重标定** | 当前仅 32k energy-NMS pool | 先做 expanded-pool support oracle |
| 6.25% positive queries | 可作为 control | 当前已采用 | 不能把 15.8 m indoor 比例视为 120 m 最优 |
| `0.1 positive + 1.0 empty` BCE | 不应原样继续锁死 | 当前已采用 | 它偏向保守 occupancy，可能解释低 outlier、高 incompleteness；R-A 应冻结为 class-balanced mean BCE |
| `512 x 32` VAE latent | 原理可迁移，但当前完整路线已 no-go | G1E-D0 representation gate 失败 | 不重启相同 VAE；仅新 target representation 通过 upper-bound 后再议 |
| EDM/18-step Heun | 工程可迁移，当前不应启动 | G1E 下游锁定 | 它不能修复 candidate support 或 AE upper bound |
| occupancy logit | 可作为 selection score | 当前另有 occupancy/confidence | 必须在 fixed query prior 下校准 |
| density head | 不存在 | 无可迁移模块 | 若需要 density/count，必须作为本项目新模块 |
| confidence head | 不存在 | 当前 confidence 是本项目新增 | 仅在 post-selection 阶段保留 |
| Doppler input/output | 不能从官方迁移 | 官方 generator 实际 intensity-only | 保留本项目 Full-RAED 和 per-point Doppler 设计 |

## 5. 不能迁移的部分及原因

### 5.1 官方 checkpoint

不能作为 K-Radar matched baseline 或初始化：

- 训练域是 ColoRadar，主要是 `0-15.8 m` indoor；
- K-Radar 是 `0-120 m` outdoor driving；
- radar axes、FOV、resolution、强度尺度和 target density 均不同；
- 官方输出只有 XYZ occupancy，不含 per-point Doppler/confidence。

### 5.2 intensity-only condition

虽然 dataset preprocessing 保留 intensity 和 Doppler：
[RaLD radar preprocessing](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/datasets/aligned_coloradar/Coloradar_dataset.py#L432-L475)，
但 generator 在 `process_radar_cond` 中明确只取 channel 0：
[intensity-only slice](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/model/models_radar_generation.py#L363-L390)，
配置也固定 `use_radar_dopp: false`：
[generation config](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/configs/generation/ge_indoor_cfg_aniso_mix_view_cone_unfreeze_enc_ints_only_eval.yml#L116-L145)。

因此不能声称 RaLD 已经验证 Full-RAED 或 Doppler-conditioned geometry。

### 5.3 原始 frustum resolution

`0.05 m` range cell 在 15.8 m 场景合理，直接扩展到 120 m 会带来：

- 约 7.6 倍 range extent；
- 极高空占比；
- 远距 LiDAR target 稀疏；
- positive/empty prior 和 loss balance 变化。

可迁移的是 polar/frustum 表征，不是原始 cell size 和 class ratio。

### 5.4 variable-count threshold export

RaLD 没有 exact-10k constraint、duplicate gate 或 confidence calibration。当前任务不能复制 `logit > 0` 后直接输出全部点，因为这会破坏：

- 固定 10k 比较；
- duplicate fraction；
- downstream tensor contract；
- per-point Doppler/confidence配对。

### 5.5 完整 VAE/EDM

当前项目已通过 G1E-D0 证明：已有 target representation/proposal support 不足以授权完整 RaLD occupancy VAE/EDM。EDM 只能学习已有 latent distribution，不能修复：

- decoder upper bound；
- 远距 query support；
- duplicate allocation；
- condition bypass。

因此不允许把“训练更久的 EDM”作为 G1D/G1F 修复。

### 5.6 DiT classifier-free guidance

DiT 依赖提供 condition dropout 和 CFG，但 RaLD release 本身没有相应 radar-unconditional training/sample arm。RaLD `sample` 始终接受一个 condition：
[RaLD sample](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/model/models_radar_generation.py#L435-L449)。

因此 CFG 只能作为未来 diffusion route 的新增 control，不能冒充 RaLD 直接迁移结果，也不能解决 deterministic G1D 的 local-Cube bypass。

### 5.7 MAR

RaLD README 致谢 MAR，但发布代码没有 import MAR package，也没有采用 masked autoregressive decoding 作为 point generator。不能仅根据 acknowledgements 把 MAR 机制归因给 RaLD。

## 6. 针对四类失败的可迁移机制

### 6.1 强制 condition use

优先级从高到低：

1. **架构独占**：allocation decoder 只接收 coordinate query 和 global Cube-conditioned latent；local spectrum/energy 只能在 center/point 已选定后用于 Doppler/confidence。
2. **matched-vs-shuffled margin**：同一 batch 内构造跨场景 derangement，冻结：

   ```text
   L_cond = relu(margin - (L_occ_shuffled - L_occ_matched))
   ```

   这不是 RaLD 官方 loss，而是对当前 `0.016%` bypass 的项目特定约束。
3. **condition dropout control**：借鉴 DiT 的 label dropout，只允许在 latent route 使用；验证 matched、zero-condition、cross-scene-shuffle 三臂。
4. **双重 shuffle 归因**：
   - global-condition shuffle：只打乱 336 tokens；
   - all-learned-evidence shuffle：打乱 global tokens 和所有 learned local evidence，但保持 proposal/query coordinates 不变。

Stage-0 必须至少满足 matched 相对 cross-scene shuffle 的 Chamfer 或 occupancy degradation `>=1%`。

### 6.2 改善远距覆盖

直接来自 RaLD 的有效启发不是“更强 CFAR”，而是 **CFAR + random free-space union**。

对 K-Radar 必须改为：

- `0-40 m`、`40-60 m`、`60-120 m` 三段固定候选配额；
- 每段独立 low-threshold/quantile proposal，禁止全局 energy ranking 把预算全部占在近距；
- 全域 random query 也按 range strata 采样；
- 在 60-120 m 保留最少候选质量，不依赖当前 frame 的强反射数量；
- 训练和验证都报告 candidate-to-GT recall@2m by range。

expanded-pool F0 必须先证明 far recall 明显高于 G1F 的 `17.80%`。

### 6.3 降低重复

RaLD 可提供的只是 unique arbitrary query field。fixed-10k 下还需本项目约束：

- export 对 5 cm Cartesian cell 做 capacity-one；
- 或对 `(r,a,e)` frustum cell 做 capacity-one；
- score 相同时稳定按 flat index；
- 不够 10k 时从尚未使用的下一分数候选补齐；
- hierarchy 路线必须同时约束 center uniqueness 和 intra-parent child diversity；
- confidence 不得通过丢弃点降低 duplicate，输出始终 exact 10k。

### 6.4 改善 completeness 而不抬高 outlier

当前 source-faithful `0.1 positive + 1.0 empty` classwise BCE 倾向保守 occupancy。R-A/R-B 应使用：

```text
L_occ = 1.0 * mean(BCE_positive)
      + 1.0 * mean(BCE_empty)
```

并把 outlier 作为独立 hinge/gate，而不是继续依赖 negative-dominant occupancy loss间接压低 outlier。该变化必须和 source-faithful `0.1/1.0` control 成对报告一次，不能做 sweep。

## 7. 三条互斥候选路线

三条路线按 **输出表示** 互斥：

- R-A：continuous arbitrary-query occupancy field；
- R-B：ray-wise multi-hit frustum distribution；
- R-C：direct set hierarchy，`2500 centers x 4 children`。

任何两条都不得在 Stage-0 前融合。

## 7.1 R-A：RaLD-wide condition-exclusive frustum occupancy field

### 核心假设

恢复 RaLD 的 coordinate-only decoder 和 wide query domain，可以同时消除 global-condition bypass，并补足 G1F 证明缺失的远距候选支撑。

### 代码级接口

建议新增但本报告未实现：

```python
# code/models/rald_wide_query_field.py
class RaLDWideConditionEncoder(nn.Module):
    def forward(self, cube_drae) -> dict:
        # radar_tokens: [B, 336, 512]
        # scene_latent: [B, 512, 512]

class RaLDCoordinateOccupancyDecoder(nn.Module):
    def forward(self, scene_latent, normalized_rae) -> dict:
        # query input only: normalized RAE Fourier embedding
        # occupancy_logit: [B, Q]
        # offset_bins: [B, Q, 3]

def build_wide_query_domain(
    cube_drae,
    *,
    random_count=500_000,
    radar_count=700_000,
    range_strata=((0, 40), (40, 60), (60, 120)),
) -> dict:
    # continuous RAE candidates and immutable provenance

def capacity_one_exact_export(
    occupancy_logit,
    coordinates_rae,
    *,
    point_count=10_000,
    duplicate_cell_m=0.05,
) -> dict:
    # stable exact-10k export
```

```python
# code/losses/rald_wide_query_field.py
def range_balanced_occupancy_loss(
    matched_output,
    shuffled_output,
    query_labels,
    *,
    condition_margin=0.01,
) -> dict:
    # balanced positive/empty BCE
    # range-stratified positive coverage
    # matched-vs-shuffled condition margin
```

关键禁止项：

- allocation 前 query 不能读取 local Cube spectrum、energy、proposal score；
- selected 10k 后才查询 current Cube spectrum，用于 Doppler/confidence；
- candidate geometry 可来自 low-threshold Cube，但 learned occupancy score 必须只来自 coordinate + global condition latent。

### 预期改善

| 指标 | 机制性预期 | Stage-0 目标 |
|---|---|---:|
| condition shuffle | 去除 local-Cube bypass | `>=1%` |
| far candidate recall@2m | 1.2M range-stratified union | `>=50%` |
| duplicate | capacity-one export | `<=10%` |
| completeness | expanded support + balanced BCE | `<=2.50 m` screen |
| far completeness | 60-120 m 固定候选配额 | `<=8.0 m` |
| outlier | occupancy + 2m hinge | `<=25%` |

这些是冻结门，不是效果承诺。

### 最小 H200 Stage-0

**R-A0，不训练 support oracle**

1. 24 个 validation frames；
2. exact `500k random + 700k radar-guided` pool；
3. GT 仅用于不可达 capacity-one 10k oracle；
4. 报告 full geometry gate、pool recall@2m by range、candidate duplicate；
5. 不访问 test。

R-A0 通过后才允许：

**R-A1，一 seed 10 epochs**

- 训练仍只采样每帧 10k occupancy queries，不在训练中反传 1.2M queries；
- validation 才 chunk-decode full candidate pool；
- source-faithful `0.1/1.0` loss 只作为固定 control；
- 主臂使用 class-balanced `1.0/1.0`；
- exact-10k capacity-one export。

### 预算上限

| 阶段 | H200 预算上限 | 终止条件 |
|---|---:|---|
| R-A0 | `1.5 GPU-h` | pool oracle 任一完整几何门失败则停止 |
| R-A1 | `12 GPU-h` | epoch 5 shuffle `<0.5%` 或 far completeness 不优于 G1D epoch-15 则提前停止 |

### 冻结门与淘汰门

R-A0 必须同时满足：

- oracle Chamfer `<=2.50 m`
- outlier `<=25%`
- completeness `<=0.65 m`
- far completeness `<=8.0 m`
- duplicate `<=10%`
- far pool recall@2m `>=50%`
- exact `10,000`

R-A1 promotion：

- condition shuffle `>=1%`
- completeness 至少比 G1D epoch-15 改善 `30%`
- far completeness不差于 `8.1239 m`
- outlier `<=25%`
- duplicate `<=10%`
- exact `10,000`

任一失败即关闭 R-A，不允许调 random/CFAR 比例。

## 7.2 R-B：RaLD-frustum ray-wise multi-hit allocator

### 核心假设

RaLD 论文的 frustum/occlusion 观点比发布代码实现得更深：沿每个 `(azimuth,elevation)` ray 显式建模 range return distribution，可避免全局 top-energy proposal 的近距偏置，并以唯一 frustum cell 降低重复。

此路线是 **RaLD-derived extension**，不是官方已有模块。

### 代码级接口

```python
# code/models/rald_ray_allocator.py
class MultiHitRayFrustumAllocator(nn.Module):
    def forward(self, cube_drae) -> dict:
        # ray_hazard_logits: [B, H, A, E, R], H=4
        # range_residual_bins: [B, H, A, E]
        # return_confidence_logit: [B, H, A, E]

def multi_hit_hazard_probability(ray_hazard_logits) -> torch.Tensor:
    # p(r_k) = h_k * product_{j<k}(1-h_j)

def exact_range_balanced_cell_export(
    return_probability,
    *,
    point_count=10_000,
) -> dict:
    # unique frustum cell first, stable refill second
```

```python
# code/losses/rald_ray_allocator.py
def ray_allocator_loss(
    output,
    target_rae_confidence,
) -> dict:
    # multi-hit ray matching
    # balanced near/mid/far occupancy
    # Chamfer + outlier hinge
    # range residual + cell uniqueness
```

### 预期改善

| 指标 | 机制性预期 | Stage-0 目标 |
|---|---|---:|
| condition shuffle | 全部 geometry 由 Cube-conditioned ray logits 产生 | `>=1%` |
| far completeness | 每条 ray 显式分配 range return | `<=8.1239 m` |
| duplicate | unique frustum cells | `<=10%` |
| completeness | multi-hit distribution 而非 1000 seed | `<=2.50 m` screen |
| outlier | hazard confidence + 2m hinge | `<=25%` |

### 最小 H200 Stage-0

- one seed；
- 10 epochs；
- 76/24 split；
- `H=4` 固定；
- range strata 与 R-A 相同；
- controls：single-hit `H=1`、range-balance disabled、cross-scene Cube shuffle；
- exact 10k，不允许 confidence masking 降数量。

### 预算上限

- `8 H200 GPU-h`
- 若 epoch 5 出现以下任一情况，提前淘汰：
  - condition shuffle `<0.5%`
  - unique frustum cell `<90%`
  - far completeness 不优于 `8.7814 m`
  - outlier `>30%`

### promotion

- condition shuffle `>=1%`
- duplicate `<=10%`
- completeness 至少比 G1D epoch-15 改善 `30%`
- far completeness `<=8.1239 m`
- outlier `<=25%`
- multi-hit `H=4` 同时优于 `H=1` 的 completeness 和 far completeness

否则关闭 R-B。

## 7.3 R-C：RaLD-hybrid condition-exclusive center-child hierarchy

### 核心假设

把 RaLD 的 learned set queries 和 global radar cross-attention 用于直接分配 2,500 个 center；allocation 前完全禁止 local Cube，center 选定后才读取 local spectrum 并生成四个有界 child。

这是当前 G1G 的路线，代码骨架已经存在：

- `code/models/g1g_hierarchical_allocator.py`
- `artifacts/idea/pre_idea_drafts/g1g_condition_exclusive_hierarchy.md`

### 代码级接口

现有核心接口：

```python
class G1GConditionExclusiveHierarchy(nn.Module):
    def allocate_centers(self, condition_cube_drae) -> dict:
        # 336 global tokens -> 2500 centers
        # no local spectrum/energy/proposal score

    def refine_children(self, cube_drae, allocation) -> dict:
        # sample local spectrum after allocation
        # four bounded children per center

    def forward(self, cube_drae, *, condition_cube_drae=None) -> dict:
        # exact 10k output
```

代码中的关键 anti-bypass contract：

- `PRE_ALLOCATION_SOURCES`
- `FORBIDDEN_PRE_ALLOCATION_SOURCES`
- `allocate_centers`
- `refine_children`

对应：`code/models/g1g_hierarchical_allocator.py:73-98,357-507`。

### 预期改善

| 指标 | 机制性预期 | Stage-0 目标 |
|---|---|---:|
| condition shuffle | center 只能来自 global Cube tokens | `>=1%` |
| duplicate | center repulsion + child diversity | `<=15%` screen |
| completeness | 2500 global centers不受 32k pool 限制 | `<=2.50677 m` |
| far completeness | normalized RAE template coverage | `<=8.1239 m` |
| outlier | child offset 物理有界 | `<=25%` |

### 最小 H200 Stage-0

冻结协议已经定义：

- one seed `20260716`
- 20 epochs
- every 5 epochs validation
- controls：
  - cross-scene condition shuffle
  - zero-local-refinement
  - child-collapse
  - center occupancy/uniqueness
- exact `2500 x 4 = 10,000`

既有 H200 preflight 已验证：

- H200 identity；
- `41,684,584` parameters；
- 336 tokens；
- exact center/child count；
- measured Cube 对 allocation 无梯度；
- condition Cube 对 allocation 有梯度；
- peak CUDA memory `8,632,353,280` bytes。

这只是结构 preflight，不是科学结果。

### 预算上限

- `12 H200 GPU-h`
- 已有 preflight 不计入训练预算。

### promotion/abandonment

必须同时满足：

- condition shuffle `>=1%`
- duplicate `<=15%`
- completeness `<=2.50677 m`
- outlier `<=25%`
- far completeness `<=8.1239 m`
- unique 5 cm center cells `>=80%`

任何一项失败即关闭 R-C。即使 Stage-0 通过，正式最终门仍是 completeness `<=0.65 m`、duplicate `<=10%`。

## 8. 路线对比与决策

| 路线 | 解决 condition bypass | 改变 candidate support | 结构性去重 | 与 RaLD 官方距离 | 主要风险 |
|---|---|---|---|---|---|
| R-A wide occupancy | 是 | 是，1.2M union | capacity-one | 最近 | query decode 成本；K-Radar wide pool 仍可能无支撑 |
| R-B ray multi-hit | 是 | 不依赖 proposal pool | unique cell | 论文启发，代码外扩 | multi-return/ray target 构造复杂 |
| R-C center-child | 是 | 直接生成 center | center/child loss | 中等 | 2500 parent 可能继续坍缩，4-child 结构可能限制细节 |

### 推荐顺序

1. **先跑 R-A0 expanded-pool support oracle。**
   它成本最低，并能直接回答 RaLD 式 wide query domain 是否真的补足 G1F 缺失支撑。
2. R-A0 通过则优先 R-A1。
   它同时恢复 coordinate-only decoder 和 wide query domain，是本审计最有原始代码依据的迁移。
3. R-C 可作为并行、计算更紧凑的 direct-set control。
   它已经完成结构 preflight，但其 `2500 x 4` 不是 RaLD 原生输出机制。
4. R-B 只在 R-A0 失败或 R-C 证明 center hierarchy 仍受重复/远距问题限制时启动。

## 9. 最推荐路线及理由

**最推荐：R-A，RaLD-wide condition-exclusive frustum occupancy field。**

理由：

1. 它针对 G1D 的 `0.016%` condition shuffle 给出架构级修复，不再允许 local Cube query feature 绕过 global condition。
2. 它针对 G1F 的 `17.80%` far recall 给出 representation-level 修复，不再在失败的 32k pool 内继续重排。
3. 它直接对应 RaLD 论文和官方代码中都有独立消融支持的两个模块：
   - radar encoder conditioning；
   - CFAR-guided + random query initialization。
4. unique continuous query + capacity-one export 能在不减少 10k 数量的前提下降低重复。
5. 它不依赖完整 `512 x 32` VAE/EDM，因此不违反 G1E-D0 no-go。
6. R-A0 是一个有严格淘汰门的低成本 oracle。若 expanded pool 仍不能通过，能在最多 `1.5 H200 GPU-h` 内关闭整条路线，避免再次长期训练后才发现 support ceiling。

## 10. 审计边界

- 本报告没有运行 RaLD checkpoint、K-Radar 推理、H200 benchmark 或任何训练。
- 对 R-A/R-B/R-C 的指标改善均为机制性假设和冻结门，不是实验结果。
- RaLD 的 ColoRadar CD/EMD 不能与当前 K-Radar metric table 直接横向比较。
- 只有完成对应 Stage-0 且通过预注册门，才允许把相关机制写入方法主张。
