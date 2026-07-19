# A 模块相关论文与开源代码调研

> **后续证据更新：**本调研提出的 `RaLD-style-RAESum-matched` 候选已完成
> 官方规模实现、结构/梯度验证和一次预注册 hard-occupancy 修复，但仍未通过
> 单帧 AE Chamfer 门（`9.1444 m` vs `<= 5.0 m`）。因此“进入最终几何主表”
> 的早期建议已被否决；不继续训练 latent EDM。最终决定见
> [`rald_matched_baseline_decision_2026-07-19.md`](rald_matched_baseline_decision_2026-07-19.md)。

> 日期：2026-07-19
> 范围：4D Radar Cube 编码、Cube-to-dense 点云、Doppler 表征、可微 point-to-Cube、时序雷达增密
> 目的：核定 A 模块的 prior-art 边界，并给出可执行、可证伪、许可证清晰的借鉴路径
> 结论口径：本报告记录的是截至检索日的公开资料，不据此直接宣称“首次”

## 1. 结论摘要

1. **`完整 RAED Cube -> 稠密 XYZ` 不是空白。** DenserRadar 已将完整 Doppler 轴作为通道输入 3D U-Net，并输出稠密 3D occupancy。因此，当前 A 模块是必要基础能力和严格对照，但不能单独承担顶会主创新。
2. **`空间 + Doppler 联合建模` 也不能泛化地宣称首次。** SDDiff 已联合空间 occupancy 与峰值 Doppler 做生成和 ego-velocity refinement；RaUF 已联合空间不确定性与 Doppler consistency。我们的可防守边界必须更具体。
3. **当前最清晰的组合创新边界仍然存在：**当前完整 RAED Cube 条件下，生成 `XYZ + 点级 circular Doppler distribution + 独立 confidence`，再通过可微 point-to-RAED measurement cycle 约束局部频谱、全局 Doppler marginal 和空间能量；时序阶段由当前 Cube 校正历史 Doppler-warp prior。
4. **当前冻结的 G1 恢复不应因本轮调研改动。** 零初始化完整谱残差有明确的 ControlNet 式安全初始化依据，且已经通过本地不变量测试。所有新结构只能在 G1 关门后作为新分支，不能伪装成第二次恢复。
5. **若 G1 恢复失败，首选不是大模型或 4D Transformer。** 推荐按 `稀疏 key-voxel full spectrum -> 物理压缩谱 -> 低秩谱残差 -> RA-global Doppler branch` 的顺序做小规模 no-go 实验。
6. **G2/G3 存在两个比换 backbone 更关键的实现问题：**Doppler 频谱应在最终生成位置连续查询；cycle 的 local KL 不能只由预测覆盖域定义，否则存在支持域作弊。

## 2. 当前 A 模块实现审计

当前实现位于 [`code/models/cube_occupancy.py`](../code/models/cube_occupancy.py)：

- 输入为 `(B, 64, R, A, E)`，完整保留 K-Radar 的 64 个 Doppler bin。
- `RAE-Max`：沿 Doppler 取最大值，经 `1 -> 8` 的 `1x1x1 Conv3d` 投影。
- `RAE-Moments`：使用 peak、Doppler mean、Doppler standard deviation 三通道。
- `Full-RAED`：保留与 `RAE-Max` 完全相同的主路径，额外增加 `64 -> 8` 的零初始化 `1x1x1 Conv3d` 谱残差。
- 三种编码共享同一轻量 3D U-Net、occupancy head、目标、训练协议和点解码器。

### 2.1 当前恢复结构的性质

| 项目 | 数值/性质 |
|---|---|
| RAE-Max 参数量 | 125,769 |
| Full-RAED 新增参数 | 520 |
| 参数增幅 | 约 0.4135% |
| 初始化行为 | Full-RAED 与 RAE-Max 输出逐元素完全相同 |
| 初始谱分支梯度 | 非零，可从第一步开始学习 |
| 谱投影估算计算量 | `64 x 8 x 256 x 107 x 37`，约 0.519 GMAC/帧 |

这里要区分“参数小”和“计算小”：当前谱残差只有 520 个参数，但在约 101 万个 RAE cell 上逐点执行，计算量并不小。

### 2.2 当前证据状态

G1 首轮中，E1/E2 相比 CFAR 的几何指标有明显改善，但绝对 outlier 超过预注册的 25% 门；Full-RAED 相比 RAE-Max 的 Chamfer 和远距 completeness 显著恶化。当前只允许一次零初始化谱残差恢复，协议见 [`docs/g1_cube_occupancy_protocol.md`](g1_cube_occupancy_protocol.md)，证据口径见 [`paper/claim_evidence_ledger.md`](../paper/claim_evidence_ledger.md)。

因此当前只允许写：

> We test whether preserving the complete Doppler spectrum improves dense geometry under a matched spatial backbone.

不能写：

> The full Doppler spectrum improves dense geometry.

## 3. 直接竞争工作

| 工作 | 输入 | 输出 | Doppler 处理 | 时序 | 与本项目的关系 | 代码状态 |
|---|---|---|---|---|---|---|
| [DenserRadar, ITSC 2024](https://arxiv.org/abs/2405.05131) | 单帧完整 `D x R x E x A`，D 作为通道 | 稠密 3D occupancy / XYZ | 完整 D 轴进入网络，但不输出点级 Doppler | 无；多帧 LiDAR 仅构造 GT | **A 模块最直接竞品** | [仓库](https://github.com/hanzy21/DenserRadar)，无明确 LICENSE，不复制代码 |
| [SDDiff, IJCAI 2025](https://www.ijcai.org/proceedings/2025/979) | ADC 经 FFT 得到 RAED，随后沿 D 取峰值，形成 `RAE x 2` intensity + peak Doppler | 稠密 occupancy/点 + ego velocity | 峰值标量 Doppler、Doppler consistency、iterative refinement | 单帧 | 已覆盖“空间-Doppler 联合建模”的宽泛表述 | [仓库](https://github.com/StellarEsti/SDDiff) 仅 README/LICENSE，MIT，无模型实现可复用 |
| [RaLD, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38946) | 单帧 radar spectrum；论文方法实际使用 3D RAE intensity 表征 | 10k 稠密 XYZ | 官方主配置 `use_radar_dopp: false`，不输出 Doppler | 单帧 | 最强 `radar -> dense XYZ` 生成基线 | [官方实现](https://github.com/MetaIoT-WHU/RaLD)，Apache-2.0 |
| [RaUF, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_RaUF_Learning_the_Spatial_Uncertainty_Field_of_Radar_CVPR_2026_paper.html) | Radar 空间和 Doppler 特征 | 空间检测、各向异性不确定性 | Bidirectional Domain Attention 利用 Doppler consistency | 单帧 | 约束“首次物理联合/首次不确定性”表述 | [仓库](https://github.com/StellarEsti/rauf) 目前仅项目网页，无模型代码 |
| [RPDNet, T-RO 2022](https://github.com/thucyw/RPDNet) | Range-Doppler Matrix | 有效 RD cell 经 DOA 得 `(x,y,z,v)` | Doppler 来自检测 bin，不是生成分布 | 主干单帧；历史仅作后处理 | 传统学习式 point extraction 基线 | 代码完整，但仅限非商业使用 |
| [DoppDrive, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Haitman_DoppDrive_Doppler-Driven_Temporal_Aggregation_for_Improved_Radar_Object_Detection_ICCV_2025_paper.html) | 多帧稀疏 `XYZ + Doppler` | 径向移动、筛选后的聚合点 | 动态 Doppler 驱动径向修正和逐点历史时长 | 有 | 最强时序聚合基线；不是生成新点 | 方法代码未公开；仅 [LRR-Sim](https://github.com/yuvalHG/LRRSim) 数据仓库 |

### 3.1 对旧调研结论的修正

[`docs/competitor_rescan_2026-07.md`](competitor_rescan_2026-07.md) 是 2026-07-02 的历史快照，其中“完整 Cube-to-dense 仍为空白”和“SDDiff 有可用实现”等判断已不再可靠。本报告以论文正文、当前官方仓库和当前配置为准：

- DenserRadar 已直接覆盖完整 RAED-to-occupancy。
- SDDiff 仓库当前只有 README 和 LICENSE，不能当作可运行 baseline。
- RaLD 官方实现已公开，是最值得接入的生成基线。
- RaUF 已构成 Doppler-aware spatial uncertainty 的近邻工作，但当前没有模型代码。

## 4. 邻近模型中可借鉴的部分

### 4.1 A 模块谱编码

| 模型/机制 | 可借鉴点 | 不应直接照搬的原因 | 许可证 |
|---|---|---|---|
| [RADE-Net](https://arxiv.org/abs/2602.19994) | 将完整 RADE 投影为保留 Doppler/Elevation 的 3D 表征；论文报告相对完整张量减少 91.9% 数据量 | 其目标是 RA 2D 检测，并通过 max projection 丢失部分 D-E 联合关系 | [代码](https://github.com/chr-is-tof/RADE-Net)，MIT |
| [L2RLDB, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Shen_LiDAR-to-4DRadar_Diffusion_Bridge_via_Cross-Modal_Alignment_and_Translation_in_Latent_CVPR_2026_paper.html) | key-voxel-aware VAE，启发只在高信息空间位置保留完整谱 | 任务是 LiDAR-to-full-radar tensor 生成，不是 radar-to-point | 未发现官方代码 |
| [AdaRadar, CVPR 2026](https://arxiv.org/abs/2603.17979) | DCT 频率裁剪和 rate-accuracy 曲线，证明雷达特征存在可压缩谱结构 | DCT 低频假设不自动适用于 circular Doppler 和窄峰 | [代码](https://github.com/jp4327/adaradar)，Apache-2.0 |
| [GRT, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_Towards_Foundational_Models_for_Single-Chip_Radar_ICCV_2025_paper.pdf) | raw/full radar 相比有损表示的价值、数据规模与模型规模分析 | 单芯片雷达尺寸、数据量和 Transformer 规模与当前 K-Radar 小样本不匹配 | [代码](https://github.com/WiseLabCMU/grt)，MIT |
| [ControlNet, ICCV 2023](https://github.com/lllyasviel/ControlNet) | zero convolution 保证新分支初始化不扰动主干 | 只借鉴 zero-init 连接，不复制双分支大网络 | Apache-2.0 |
| [LoRA](https://github.com/microsoft/LoRA) | 将 `64 -> 8` 谱映射限制为 rank-1/2 低秩分解 | 没有预训练谱矩阵，不能套用“参数高效微调”的论文叙事 | MIT |
| [NeuralOperator/FNO](https://github.com/neuraloperator/neuraloperator) | 固定少量 Fourier mode 或低模谱算子 | 64 bins 很短，FNO 容易增加不必要复杂度 | MIT |

### 4.2 G2/G3 的可微物理模块

| 项目 | 可借鉴模块 | 本项目处置 | 许可证 |
|---|---|---|---|
| [PyTorch3D points-to-volumes](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/points_to_volumes.py) | 八邻域、质量加权、对位置可微的 trilinear splat | 当前 `point_to_cube.py` 已独立实现；借其不变量和接口语义做交叉测试，不必增加依赖 | BSD-3-Clause |
| [DART, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_DART_Implicit_Doppler_Tomography_for_Radar_Novel_View_Synthesis_CVPR_2024_paper.html) | 可微 range-Doppler 渲染；log/sqrt 强度域损失 | 借鉴 measurement loss，不复制其 JAX tomography backbone | [代码](https://github.com/WiseLabCMU/dart)，MIT |
| [NeuRadar, CVPRW 2025](https://openaccess.thecvf.com/content/CVPR2025W/WAD/html/Rafidashti_NeuRadar_Neural_Radiance_Fields_for_Automotive_Radar_Point_Clouds_CVPRW_2025_paper.html) | Multi-Bernoulli existence probability、Hungarian matching、matched/unmatched confidence NLL | 借鉴 proper confidence loss；仓库 README 仍标注 code release TODO，复用前逐文件核查 | [仓库](https://github.com/mrafidashti/neuradar)，Apache-2.0 |
| [RadarSplat, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Kung_RadarSplat_Radar_Gaussian_Splatting_for_High-Fidelity_Data_Synthesis_and_3D_ICCV_2025_paper.html) | 雷达极坐标 Gaussian footprint 和噪声建模 | 只读论文/实现思路，不并入代码 | [代码](https://github.com/umautobots/radarsplat)，CC BY-NC-SA 4.0 |
| [Radar-Diffusion, RA-L 2024](https://github.com/ZJU-FAST-Lab/Radar-Diffusion) | ColoRadar 预处理、快速 consistency sampling、评估脚本 | 可用于数据和生成 baseline 交叉验证；其输出是 2D/BEV，不是本项目目标 | MIT |

## 5. 对创新定位的影响

### 5.1 不能再使用的宽泛表述

- “首次使用完整 Radar Cube 生成稠密点云。”
- “首次利用 Doppler 改善点云生成。”
- “首次联合空间与 Doppler。”
- “首次做 Doppler 驱动多帧雷达增强。”
- “A 模块本身就是主要创新。”

### 5.2 当前可防守的精确定义

建议把主方法定义为：

> **A full-RAED-conditioned radar point field with per-point circular Doppler distributions and independent existence confidence, constrained by a differentiable point-to-RAED measurement cycle. A warped historical prediction is used only as a prior and is corrected by the current Cube.**

该定义同时要求以下四个组件成立，缺一项都容易退化为已有工作：

1. 完整 RAED 是当前观测，而不是历史稀疏点或 LiDAR 条件。
2. 输出是点级 64-bin circular Doppler distribution，不只是峰值标量或 ego velocity。
3. 生成点能够解释当前 Cube 的 measurement，而不是只受 LiDAR 几何监督。
4. 历史只作 prior，当前 Cube 必须生成新点、纠错并刷新 Doppler。

在本轮检索范围内，没有发现公开工作同时满足这四项。但论文投稿前仍需做一次正式检索和引用图复扫，不能把“没有检索到”直接写成“世界首次”。

## 6. A 模块后续结构优先级

### 6.1 当前 G1：保持冻结

继续完成当前 `RAE-Max main path + zero-initialized full-spectrum residual`。理由：

- 初始化时与 RAE-Max 完全等价，排除“新分支一开始破坏几何”的解释。
- 新增参数低于 1% 预注册门。
- zero convolution 有成熟先例。
- 当前实验已经启动，再修改会破坏唯一一次有界恢复的可解释性。

### 6.2 若 G1 恢复失败：另立 G1B，不重开 G1

| 优先级 | 候选 | 结构 | 预期收益 | 主要风险 |
|---|---|---|---|---|
| P0 | Sparse key-voxel full spectrum | 用 CFAR/SNR/range-balanced gate 选 1/3/5/10% RAE cell；仅对其 64-bin spectrum 做共享 MLP，scatter 回空间后 zero-init 融合 | 最大幅度减少谱分支计算，并保留局部完整谱 | gate 漏掉弱反射；需报告各距离段召回 |
| P0 | Physics-compressed spectrum | 固定 negative / near-zero / positive circular soft masks；每组提取 mass、mean、width，共 9 通道，再用 zero-init `9 -> 8` | 约 80 个参数，强物理归纳，易解释 | 可能不足以表达多峰和 Doppler-elevation 耦合 |
| P1 | Rank-1/2 spectral residual | `64 -> r -> 8`，`r=1/2`；最终投影 zero-init | 72/144 参数，计算量约为当前分支的 1/7 或 1/3.6 | 低秩可能只学到总能量或近似 moments |
| P1 | RA-global Doppler branch | `max_E(C)` 得 DRA，`Conv2d 64 -> 8` 后通过 elevation gate 广播回 RAE | 约 0.014 GMAC，约为当前密集谱投影的 1/37 | 丢失 D-E 联合关系，广播可能制造伪 elevation |
| P2 | Circular Fourier basis | 固定 sin/cos basis，保留 `K=4/8/16` mode，zero-init 投影 | 对 circular shift 有明确结构，可控压缩 | 低模会抹平窄峰；不能默认照搬 AdaRadar 的 DCT 结论 |
| P3 | 局部/轴向 attention、FNO、S4/Mamba | 在降采样后做谱或空间全局建模 | 表达力强 | 小数据、短谱轴和当前门限下风险最高 |

建议首先做 1 个种子、10-20 epoch 的 no-go 排序，只允许前两名进入三种子正式实验。

## 7. G2/G3 必须优先修正的实现问题

### 7.1 在最终生成位置查询 Doppler 频谱

当前 [`code/models/cube_cycle.py`](../code/models/cube_cycle.py) 先在整数 RAE index 取 feature 并预测 Doppler，再预测最多 0.5 bin 的位置 offset。这样 Doppler 分布并不真正由最终点位置决定。

应改为：

```text
u_i = project_RAE(p_i)
q_i(d) = normalize(
    epsilon + trilinear_query(log1p(C_t[d]), u_i)
)
z_i(d) = log(q_i(d) + epsilon) + delta_z_i(d)
p_i(d) = softmax(z_i(d))
```

其中 `delta_z` 的末层零初始化，使模型从“直接查询当前 Cube 局部谱”开始，只学习去噪、校准和必要残差。

关键消融：

- integer/nearest query vs final-position trilinear query；
- direct queried spectrum `Q0` vs scalar wrapped Gaussian `E3` vs residual distribution `E4`；
- position detach vs spectrum detach vs full gradient。

### 7.2 分离 existence confidence 与 spectral confidence

当前 cycle 使用 occupancy/top-k 置信作为 splat 能量权重，但“点是否存在”和“局部 Doppler 是否确定”不是同一随机变量。建议分为：

- `c_exist`：点存在概率，用于 geometry matching 和 splat energy；
- `c_spec`：频谱可观测性/确定性，用于 Doppler NLL、风险覆盖曲线和 calibration。

至少对 `c_exist` 使用 proper matched/unmatched Bernoulli loss，不能只依赖 confidence floor。

### 7.3 修复 prediction-defined cycle mask

当前 [`code/losses/cube_cycle.py`](../code/losses/cube_cycle.py) 的 local spectrum KL 只在 `rendered.covered` 上计算。模型可能把点移到容易解释的位置，绕开难重建的目标支持域。

必须对比：

- prediction-covered mask；
- 固定 target top-K support；
- target/prediction union；
- 额外报告 target-support recall。

主结果应使用固定 target support 或 union，prediction-only 只能作为失败对照。

## 8. 必做的可证伪实验

### 8.1 A 模块谱信息是否真实被使用

1. **Doppler-bin shuffle：**只打乱每个 cell 的 Doppler bin 顺序，不改变总能量。若增益不消失，模型利用的可能只是总功率或额外容量。
2. **Circular shift：**整体平移 Doppler bins，检查 feature/output 的等变性和损失稳定性。
3. **Matched moments：**构造 mean/variance 相同但峰形不同的单峰/双峰谱，检查 Full-RAED 是否区别于 RAE-Moments。
4. **Branch-off：**将谱残差显式置零，输出必须恢复 RAE-Max。
5. **Gradient test：**首步谱分支梯度必须非零，且 residual/trunk RMS 不应瞬间失控。
6. **Dynamic/far slices：**谱分支的收益必须集中在预注册的动态或远距 slice，而不是只改善近距静态背景。

### 8.2 Point-to-Cube 不变量

1. 点顺序置换不改变 rendered Cube。
2. 一个置信度 `c` 的点复制为两个相同位置、置信度 `c/2` 的点，能量应保持。
3. 边界内的 trilinear splat 总质量守恒。
4. 0.25/0.5 bin 亚像素位置扰动产生连续梯度。
5. Doppler circular shift 后，rendered spectrum 同步循环平移。
6. confidence 全高、全低、位置聚集和 offset saturation 都有独立监控。

## 9. 代码复用决策

### 9.1 可以直接纳入或局部移植

- **RaLD，Apache-2.0：**可借鉴 frustum point autoencoder、order-invariant latent、radar-conditioned EDM 和 implicit occupancy decoder；主基线必须在 K-Radar 从头训练并禁用 CFAR query helper，详见 [`rald_adapter_audit_2026-07-19.md`](rald_adapter_audit_2026-07-19.md)。
- **Radar-Diffusion，MIT：**ColoRadar 对齐、预处理、consistency sampling 和评估流程。
- **DART，MIT：**log/sqrt measurement loss 形式和雷达渲染测试思路。
- **PyTorch3D，BSD-3-Clause：**trilinear splat 语义与参考测试。
- **NeuRadar，Apache-2.0：**Multi-Bernoulli existence loss；复用前核查具体文件是否为正式发布实现。
- **RADE-Net，MIT：**K-Radar tensor projection 和数据处理模块，可用于 G1B 原型。
- **AdaRadar，Apache-2.0：**谱压缩实现和 rate-accuracy 评估方法。

### 9.2 只参考论文，不复制实现

- **DenserRadar：**没有明确 LICENSE，且与 A 模块高度重合；独立重写并作为 baseline。
- **RadarSplat：**CC BY-NC-SA 4.0，与宽松开源发布目标不兼容。
- **RPDNet：**仅限非商业使用，不合并到公开主代码。
- **RaUF：**当前仓库是项目网页，不是模型实现。
- **SDDiff：**当前仓库无训练/模型代码，只按论文重实现必要 baseline。
- **L2RLDB：**未发现官方代码，只借鉴 key-voxel 概念。

## 10. 执行决策

### 当前阶段

- 不修改正在运行的 G1 恢复。
- 将 DenserRadar 设为 A 模块最直接 related work 和必要对照。
- 原建议将 `RaLD-style-RAESum-matched` 列为独立几何 baseline；该建议已被后续 AE-B1 no-go 否决，不再进入主表或训练 EDM，且始终不阻塞或解锁 G2/G3。
- 将本报告作为 2026-07-02 竞品扫描的修订版，不删除旧文档，保留决策历史。

### G1 结束后

若通过：

- 保留当前 zero-init residual；
- 只补 Doppler shuffle/circular shift/matched-moments 机制证据；
- 进入 G2 前先实现 final-position trilinear spectrum query。

若失败：

- 按预注册规则关闭 G1、停止当前 G2/G3 队列；
- 新开 `G1B_physics_compressed_spectrum` 研究分支；
- 先比较 sparse key-voxel、physics-compressed、rank-1/2 三个小模型；
- 不把 G1B 结果写成原 G1 的第二次恢复。

## 11. 论文主张边界

当前可写：

> Existing approaches either densify radar geometry from a single radar representation, couple spatial occupancy with peak Doppler for perception, or aggregate historical radar points using Doppler. We instead study point-level circular Doppler distributions whose generated measurements are required to explain the current full RAED Cube, with history used only as a corrected prior.

在 G1-G4 关门前不能写：

- Full-RAED improves geometry.
- The cycle improves geometry and Doppler simultaneously.
- The generated confidence is calibrated.
- The temporal model predicts physically consistent future radar.
- The method is the first radar densification or Doppler-aware radar generation model.

## 12. 主要来源

- DenserRadar: https://arxiv.org/abs/2405.05131
- SDDiff: https://www.ijcai.org/proceedings/2025/979
- RaLD: https://ojs.aaai.org/index.php/AAAI/article/view/38946
- RaUF: https://openaccess.thecvf.com/content/CVPR2026/html/Wang_RaUF_Learning_the_Spatial_Uncertainty_Field_of_Radar_CVPR_2026_paper.html
- DoppDrive: https://openaccess.thecvf.com/content/ICCV2025/html/Haitman_DoppDrive_Doppler-Driven_Temporal_Aggregation_for_Improved_Radar_Object_Detection_ICCV_2025_paper.html
- RADE-Net: https://arxiv.org/abs/2602.19994
- L2RLDB: https://openaccess.thecvf.com/content/CVPR2026/html/Shen_LiDAR-to-4DRadar_Diffusion_Bridge_via_Cross-Modal_Alignment_and_Translation_in_Latent_CVPR_2026_paper.html
- AdaRadar: https://arxiv.org/abs/2603.17979
- GRT: https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_Towards_Foundational_Models_for_Single-Chip_Radar_ICCV_2025_paper.pdf
- DART: https://openaccess.thecvf.com/content/CVPR2024/html/Huang_DART_Implicit_Doppler_Tomography_for_Radar_Novel_View_Synthesis_CVPR_2024_paper.html
- NeuRadar: https://openaccess.thecvf.com/content/CVPR2025W/WAD/html/Rafidashti_NeuRadar_Neural_Radiance_Fields_for_Automotive_Radar_Point_Clouds_CVPRW_2025_paper.html
- PyTorch3D points-to-volumes: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/points_to_volumes.py
- ControlNet zero convolution: https://github.com/lllyasviel/ControlNet
