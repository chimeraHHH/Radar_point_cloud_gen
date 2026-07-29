# RaLD 与替代路线调研：面向 Full-RAED Cube 到稠密点云的几何恢复

日期：2026-07-29
范围：2022-2026 年论文原文、作者官方项目页、作者官方代码仓库
目标：针对当前 `Radar_point_cloud_gen` 长期几何效果不佳，给出能够直接进入实验协议的候选路线，而不是泛化的 related-work 列表。

## 0. 证据标签与结论边界

本文严格使用以下标签：

- **[P] 论文已证明**：只表示论文在其数据、任务、输出和评估协议内报告了该结论。
- **[C] 官方代码已实现**：已在论文作者或项目组织的官方仓库中看到对应实现；不等于已在 K-Radar 上复现。
- **[I] 本文迁移推断**：根据来源机制与当前项目故障证据提出的迁移方案，尚未得到本项目实验支持。
- **[E] 当前项目已验证**：来自本仓库冻结协议和归档结果；不外推到其他数据集。

对“没有官方代码”的表述均限定为：**截至 2026-07-29，本次从论文原文、作者项目页及其直接链接中未找到可运行的一方实现**。这不是对全互联网代码的绝对否定。

## 1. 执行摘要

### 1.1 当前主要故障不是候选空间不足，而是几何质量与置信度失配

当前 R-A1 RaLD-WCE 在冻结的 24-frame validation 上输出 exact-10k：

| 选择方式 | Mean CD | 2 m outlier | Far completeness |
|---|---:|---:|---:|
| 当前模型 confidence | 4.00895 m | 31.8129% | 11.12111 m |
| 同一 700k 候选池，用 validation GT 最近距离排序，仅作不可部署诊断 | 0.64102 m | 2.2804% | 0.70414 m |

同一候选池的 target-to-candidate mean distance 仅 `0.14858 m`，而 candidate-to-target mean distance 为 `13.3076 m`。当前 confidence 与负最近 GT 距离的 Pearson/Spearman 仅 `0.29389/0.54851`。[E]

因此：

1. 候选池包含足够好的位置，但大多数候选是坏点。
2. occupancy confidence 没有学成“这个候选最终会不会形成低误差点”的质量分数。
3. 当前硬 top-k 放大了分数与几何质量的失配。
4. exact-10k 本身不是主要原因：`2.5k/5k/7.5k/10k` 诊断中，减点虽降低尾部 outlier，却超出 completeness 容差。[E]

### 1.2 推荐顺序

**Priority 0：在现有 700k WCE 候选上做几何质量对齐排序。**

- 用 Rank-DETR/Align-DETR 的“分类分数对齐定位质量”思想，将 binary occupancy target 改为连续几何质量 target。
- 用 RaUF 的极坐标各向异性不确定性为每个候选预测 `sigma_r/sigma_a/sigma_e`，让高歧义候选即使 occupancy 高也不会排在前面。
- 这是最小改动、最直接命中已诊断故障的路线。

**Priority 1：继续 R-B2 `0.40 m x 4-slot` Cube-only voxel-slot。**

- 该表示的 GT-aided 结构 oracle 已达到 `CD=0.72268 m`、outlier `1.5417%`、far completeness `0.24981 m`，但它不是模型结果或严格上界。[E]
- 借 DenserRadar 的 Full-Doppler 3D occupancy encoder 和 AdaPoinTr 的 center-to-local-cluster decoder，直接生成固定数量的结构化 slots，避免从 700k 候选做全局 hard top-k。

**Priority 2：单帧几何过门后，再引入 RadarMP echo-flow + DoppDrive age gate。**

- RadarMP 在相邻 raw Cube 上学习低层 echo motion；DoppDrive 给每个历史点独立的可接受时间窗。
- 当前项目的解析 Doppler-history proposal 已 no-go，因此不能只重复“径向位移 + 合并”；需要让 echo-flow 负责非纯径向运动和可靠性，历史只提供 proposal，当前 Cube 决定最终点和 Doppler。

### 1.3 不建议现在重启的路线

- 不重启与已失败 matched RaLD 相同的 `512 x 32` VAE + 24-layer EDM 主线。
- 不先做 SDDiff/RadarSFD 式大扩散模型；它们的训练代价高，而且不能先验修复当前已定位的置信度排序错误。
- 不把 POCO/ALTO/NKSR 直接当主模型；它们需要已有的可靠表面点或法向，当前 raw Cube 不提供这些输入。
- 不以减少输出点数规避 exact-10k；当前 cardinality 诊断不支持该解释。

## 2. 当前项目接口与冻结证据

### 2.1 最小数据接口

`code/cube_dense/dataset.py::KRadarCubeDataset` 当前提供：

```text
cube_drae              float tensor [64, R, A, E]
occupancy              float tensor [R, A, E]
target_xyz_confidence  float tensor [N_gt, 4]
target_rae_index       int tensor   [N_gt, 3]
cfar_xyzd_power_snr    float tensor [N_cfar, ...]
ego_velocity_xyz_mps
ego_speed_mps
ego_yaw_rate_radps
```

主路线应继续遵守：

- 推理输入只使用当前 `cube_drae`、标定轴和明确授权的历史观测。
- `target_xyz_confidence` 只用于训练/validation 监督。
- validation GT 不得参与候选生成、排序、输出选择或超参数事后修改。
- 几何 parent 通过前，不解锁 Doppler head、cycle、temporal 或 test。

### 2.2 现有 WCE 接口

`code/models/rald_wce_field.py::RaLDWCEField`：

```text
forward(
    condition_cube_drae: [B, 64, R, A, E],
    normalized_rae:      [B, N_query, 3],
    wrong_condition_cube_drae: optional
)

outputs:
    occupancy_logit / confidence_logit
    confidence = sigmoid(occupancy_logit)
    residual_bins
    refined_normalized_rae
```

冻结推理：

- Q0：500k 全空间分层查询。
- Q1：200k matched-occupancy 驱动查询。
- 每个 range stratum 按 confidence 排序，并施加真实 5 cm 最小距离。
- exact-10k 配额：`8000/1700/300`。

这条接口只需新增质量/不确定性 heads，不需改 Cube encoder 或候选生成器。

### 2.3 已证伪与已授权边界

| 分支 | 当前结论 |
|---|---|
| R-A1 RaLD-WCE formal | 结构与条件依赖通过；CD/outlier/matched-win 未过门。作为 geometry parent 关闭。 |
| R-A1 frozen candidate diagnosis | 同一候选池的 GT 排序通过全部绝对几何检查，定位为 confidence-ranking bottleneck。 |
| R-A2 tiny/source-classwise | 8-frame memorization 可完成；任何新 5-epoch run 必须先绑定 cache/Cube provenance。不是 formal method。 |
| R-A2 current range-shell | 全量只读审计 no-go：9/76 帧缺失正距离类、67/67 可采样帧发生 range jitter crossing，当前 index shell 不具备干净物理 hard-negative 语义。不得启动。 |
| RAE-Max cardinality | exact-10k 不是主要或有界贡献因素。 |
| R-B1 direct GT-supported range echo | 稀疏帧无法满足 exact-10k/分段配额；该直接构造 no-go。 |
| R-B2 voxel-slot oracle | `0.40 m x 4 slots` 结构门通过；只授权 Cube-only one-frame memorization。 |
| G1T corrected temporal proposal | 各臂约 15.9 m CD/87.6% outlier，Doppler history 改变小于 0.05%；原解析 proposal no-go。 |

## 3. RaLD 官方论文与源码精读

### 3.1 一级来源

- [RaLD 论文，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38946)
- [RaLD arXiv](https://arxiv.org/abs/2511.07067)
- [RaLD 官方代码](https://github.com/MetaIoT-WHU/RaLD)
- 本次源码审计 commit：[`ffec4b41241391734b1eda5c093de843c909eb8e`](https://github.com/MetaIoT-WHU/RaLD/tree/ffec4b41241391734b1eda5c093de843c909eb8e)
- RaLD 直接继承的 latent-set 基础：[3DShape2VecSet 论文](https://arxiv.org/abs/2301.11445)、[官方代码](https://github.com/1zb/3DShape2VecSet)

### 3.2 Dense occupancy / point decoding

**[P] RaLD 的目标表示不是固定 Cartesian voxel grid，而是雷达极坐标 frustum occupancy field。**

- occupancy 由 LiDAR 点是否落入 `(range, azimuth, elevation)` frustum 决定。
- decoder 接受任意连续 3D query，输出 occupancy logit。
- 论文设置 frustum 粒度为 `0.05 m / 0.25 deg / 0.5 deg`。
- 10k LiDAR 点被编码为 `512 x 32` latent tokens；mixed query 由静态 learnable tokens 与从点特征 cross-attend 得到的动态 tokens 相加形成。

**[C] 官方 mixed latent 实现：**

- [`models_ae.py` mixed static/dynamic query](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/model/models_ae.py#L378-L386)
- [`models_ae.py` query-to-latent decoder](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/model/models_ae.py#L416-L424)

**关键边界：RaLD 不直接输出固定 10k points。**

**[C] 官方 inference 对所有 queries 做 `logit > 0` 阈值筛选：**

- [第一次 query threshold](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/engine_generation.py#L274-L290)
- [可选 positive-neighborhood refine 后再次 threshold](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/engine_generation.py#L292-L310)

因此 RaLD 原生行为是**变长正占据集合**，不是：

- DETR 式固定 slots；
- 700k 中 confidence top-10k；
- cardinality head；
- calibrated geometric quality ranking。

### 3.3 监督采样

**[P/C] 论文和代码一致的主设置：**

- 每帧下采样 10k LiDAR 点，并构造 10k decoder queries。
- 正查询占 `6.25%`，即约 625；负查询占 `93.75%`，即约 9375。
- 正查询：在随机选中的 occupied frustum 内均匀采样。
- 负查询：在随机选中的 empty frustum 内均匀采样。

官方实现：

- [positive/negative query sampling](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/datasets/aligned_coloradar/Coloradar_dataset.py#L237-L294)
- [query ratio `0.0625`](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/configs/ae/ae_indoor_cfg_aniso_mix_view_cone.yml#L34-L56)

**[C] loss 权重为：**

```text
0.1 * mean(BCE_positive) + 1.0 * mean(BCE_negative) + 1e-3 * KL
```

见 [official `engine_ae.py`](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/engine_ae.py#L48-L86)。

源码变量 `loss_vol` 对应前段 positive queries，`loss_near` 对应后段 empty queries；这里的 `near` 变量名**不代表 3DShape2VecSet 的 near-surface negatives**。当前 R-A2 `source_classwise` 已按这个实际代码口径复刻。

### 3.4 Radar-guided queries

**[P/C] inference query 组成：**

- 500k free-space random queries；
- 700k CFAR helper-region augmented queries；
- 可选围绕 positive prediction 再生成 500k refined queries。

官方配置：

- [500k random + 700k helper](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/configs/generation/ge_indoor_cfg_aniso_mix_view_cone_unfreeze_enc_ints_only_eval.yml#L155-L168)

这是 RaLD 最值得保留的部分之一：**把雷达峰附近的高分辨率查询和全空间查询并存**。当前 WCE 的 Q0+Q1 已经实现了同类思想，而且 frozen-candidate oracle 已证明 support 足够。[E]

### 3.5 置信度排序机制

**官方 RaLD 没有独立置信度排序机制。**

- decoder 只输出 occupancy logit；
- training 只把它当 binary occupancy classifier；
- inference 只做 `logit > 0`；
- 没有 localization-quality target；
- 没有 uncertainty/NLL；
- 没有 pairwise/listwise rank loss；
- 没有 top-k calibration；
- 没有固定 cardinality loss。

**[I] 对当前项目的结论：**

当前项目为了 exact-10k 将 occupancy probability 当成了 geometric quality score。这个额外假设不是 RaLD 论文或代码证明的。当前 `confidence vs -distance` 低相关性和 GT-ranking oracle 的巨大差距，正是该假设失效的直接证据。[E]

### 3.6 Doppler 边界

**[C] 官方 generation 配置为 `use_radar_dopp: false`。**

- [official generation config](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/configs/generation/ge_indoor_cfg_aniso_mix_view_cone_unfreeze_enc_ints_only.yml#L128-L136)

RaLD 的 “radar spectrum conditioning” 在发布配置中是 intensity-only；输出也是 XYZ occupancy points，不含逐点 Doppler distribution 或 confidence calibration。它不能作为当前 “Full-RAED -> XYZ + Doppler distribution + confidence” 已完成的证据。

### 3.7 哪些可借，哪些不应再做

| RaLD 机制 | 当前是否已借 | 结论 |
|---|---|---|
| Frustum occupancy supervision | 是 | 保留为 WCE/R-A2 control。 |
| Global + radar-guided wide queries | 是 | support 已过诊断；无需继续扩大候选数。 |
| Source-classwise 6.25/93.75 sampling | 是 | 作为 bounded control，不应替代质量对齐主路线。 |
| Mixed latent set | 已有 matched/anchor 路线 | 旧 matched AE 已 no-go；不原样重启。 |
| 24-layer EDM, 18-step sampler | 代码存在 | 训练昂贵；几何 parent 未过门前不启动。 |
| Thresholded variable-cardinality occupancy | 未作为主输出 | 可作 RaLD faithful control，但不满足当前 exact-10k contract。 |
| Confidence top-k | RaLD 不存在 | 当前必须另行设计。 |
| Per-point Doppler | RaLD 不存在 | 只能在 geometry parent 过门后由当前项目实现。 |

## 4. Radar Cube 到稠密几何：一级来源比较

### 4.1 直接相关方法

| 方法 | 输入 -> 输出 | 论文已证明 [P] | 官方代码 [C] | 对当前项目的边界 |
|---|---|---|---|---|
| [DenserRadar, ITSC 2024](https://arxiv.org/abs/2405.05131) | K-Radar `D x R x E x A` -> `2R x 2E x 2A` occupancy -> points | Doppler 作为 channel，经 3D U-Net/3D deconv 输出双分辨率 occupancy；多帧 stitched LiDAR 构造 dense GT；Dice+focal deep supervision。 | 本次一级来源未找到作者官方实现。 | 最接近 R-B2 的 Cube-to-grid backbone，但输出靠 threshold，仍有 class imbalance 和 cardinality 问题；不输出逐点 Doppler。 |
| [SDDiff, IJCAI 2025](https://www.ijcai.org/proceedings/2025/979) | raw ADC -> spatial-Doppler occupancy -> dense PCE + ego velocity | peak intensity 与 peak Doppler index 组成 SDDR；3D U-Net directional diffusion；每层 Doppler cross-attention；迭代 Doppler refinement。 | [官方仓库](https://github.com/StellarEsti/SDDiff)截至核查日只有 `.gitignore`、LICENSE、README，无模型实现。 | Doppler-consistency auxiliary 有价值；完整 diffusion 不是当前最小修复。输出不是逐点完整 Doppler distribution。 |
| [RaUF, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_RaUF_Learning_the_Spatial_Uncertainty_Field_of_Radar_CVPR_2026_paper.pdf) | `R x A x E x 2` intensity+Doppler -> spatial detections + anisotropic uncertainty | 极坐标 heteroscedastic Gaussian、Jacobian 转 Cartesian covariance、NLL；双向 spatial-Doppler attention；论文在其数据上报告 uncertainty calibration 和 geometry 改善。 | [官方项目页](https://shengpeng.wang/rauf/)标记开发中；其 code 链接指向的仓库截至核查日返回 404。 | 与当前“模糊映射 + confidence 不可靠”故障最对齐；需自行做最小 head，不能声称复现其完整 BDAF。 |
| [Radar-Diffusion, RA-L 2024](https://arxiv.org/abs/2403.08460) | single-frame RAH -> LiDAR-like BEV -> 2D points | EDM + consistency distillation；一阶段 inference。 | [官方代码](https://github.com/ZJU-FAST-Lab/Radar-Diffusion)含训练、蒸馏、推理、预处理；作者说明原 checkpoint 丢失，重训 checkpoint 数值不同。 | 可作 2D BEV diffusion control；论文明确未来才扩到 3D，不能当当前 full-3D 解法。 |
| [Radar-diffusion super-resolution, ICRA 2024](https://arxiv.org/abs/2404.06012) | sparse radar points/BEV -> dense LiDAR-like points | mean-reverting SDE，去 ghost 并增密；报告 registration downstream。 | 本次一级来源未找到作者官方代码。 | 输入已是稀疏点云而非 Full-RAED Cube；可作 point-SR related baseline，不命中当前候选排序根因。 |
| [RadarSFD, ICRA 2026](https://arxiv.org/abs/2509.18068) | single radar BEV -> LiDAR-like BEV/2D points | Marigold pretrained depth prior、latent concatenation、latent+pixel losses；single-frame reconstruction。 | [官方代码](https://github.com/phi-lab-rice/RadarSFD)与[官方权重](https://huggingface.co/Bin-0815/RadarSFD)已发布。 | 是完整可跑的单帧扩散参考，但表示为 2D BEV，不是 K-Radar full-3D Cube，也不保留逐点 Doppler。 |
| [RMap, IROS 2024](https://arxiv.org/abs/2310.13188) | trajectory-aggregated radar map patches -> completed volumetric map | 时序积累后用 AdaPoinTr 类 completion 做去噪/补全。 | [官方代码](https://github.com/arpg/RMap)已发布。 | 是 map completion，不是当前帧 Cube generation；只能作为“先积累再补全”的对照。 |
| [SD4R, ITSC 2025](https://arxiv.org/abs/2602.20653) | sparse radar points -> foreground virtual points -> detector | VoteHead 预测 object-center offset/logit/features；FPG 只从 foreground 生成 virtual points；LQE 改善 pillar features。 | [官方代码](https://github.com/lancelot0805/SD4R)已发布。 | 目标是 VoD 3D detection，不是 LiDAR-like scene geometry；foreground-only densification 可用于 downstream control，不应替代 geometry parent。 |

### 4.2 DenserRadar 对 R-B2 的直接启示

DenserRadar 的可迁移部分：

1. **[P] Full Doppler-as-channel。** 不先 `max_D` 丢掉谱，只在输入 3D conv 中把 D 当 channel。
2. **[P] spherical 3D U-Net。** 在 R/A/E 保持空间结构，避免把整个 Cube 变成无拓扑 token set。
3. **[P] 多尺度 occupancy supervision。** 在上采样层提供 Dice+focal，可缓解极端稀疏。
4. **[I] 输出改成 R-B2 voxel/slot heads。** 不照搬 thresholded occupancy；在 Cartesian candidate voxels 上输出 occupancy、4 slot logits、4 offsets。
5. **[I] 增加 continuous quality target。** DenserRadar 的 binary occupancy 仍不足以解决 exact-10k 排序。

最小接口：

```text
cube_drae
  -> 轻量 spherical 3D encoder
  -> gather candidate voxel features
  -> voxel occupancy + 4 slot quality + 4 local XYZ offsets
  -> fixed range quotas, whole-voxel export
```

## 5. Set prediction、quality ranking 与固定 cardinality

### 5.1 Rank-DETR：分数必须表示定位质量

- [Rank-DETR 论文，NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/34074479ee2186a9f236b8fd03635372-Paper-Conference.pdf)
- [官方代码](https://github.com/LeapLabTHU/Rank-DETR)
- 审计 commit：[`9b0f28f7ad6d4db4a43d048a09b6741975380a83`](https://github.com/LeapLabTHU/Rank-DETR/tree/9b0f28f7ad6d4db4a43d048a09b6741975380a83)

**[P]** Rank-DETR 诊断 DETR classification score 与 localization accuracy 不对齐，导致 top-ranked predictions 反而定位差。

**[C]** 对 matched prediction，它用 normalized GIoU `(GIoU+1)/2` 替代 binary class target，并 detach target：

- [official quality-aware criterion](https://github.com/LeapLabTHU/Rank-DETR/blob/9b0f28f7ad6d4db4a43d048a09b6741975380a83/projects/rank_detr/modeling/rankdetr_criterion.py#L127-L170)

**[I] 对当前点候选的等价映射：**

```text
d_i = nearest weighted target distance from refined candidate i
q_i = exp(-d_i / tau) * observability_weight_i
L_quality = quality-focal/BCE-with-soft-target(score_i, stopgrad(q_i))
```

这里的 `q_i` 是训练 target；推理时只使用模型 score，不能访问 GT。

Rank-DETR 证明的是 2D object detection，不证明该 target 在 dense radar points 上一定有效；迁移依据来自当前项目同构故障：**top-ranked confidence 与 localization quality 失配**。[I]

### 5.2 Align-DETR：many-to-one supervision 与局部排序

- [Align-DETR 论文](https://arxiv.org/abs/2304.07527)
- [官方代码](https://github.com/FelixCaae/AlignDETR)
- 审计 commit：[`4b125f7fe2ff8c3468cd71763fcf971d09ee38db`](https://github.com/FelixCaae/AlignDETR/tree/4b125f7fe2ff8c3468cd71763fcf971d09ee38db)

**[P/C]** Align-DETR 用预测概率与 IoU 的几何平均构造 quality target：

```text
t = p^alpha * IoU^(1-alpha)
```

并用 one-to-many matching 与 local-rank weight 让一个 GT 获得多个正 queries。

- [official aligned loss](https://github.com/FelixCaae/AlignDETR/blob/4b125f7fe2ff8c3468cd71763fcf971d09ee38db/aligndetr/losses/losses.py#L86-L108)

**[I] 当前最小迁移：**

- 每个 target 用空间 kNN 找 `k=4-16` 个候选，不做 700k x Ngt 全矩阵 Hungarian。
- candidate 可以服务最近 target；一个 target 可以监督多个候选。
- 在每个 target bag 内按 `exp(-d/tau)` 排序，给更近候选更高权重。
- 对远距 target 单独设置 `tau_r` 或按 range 归一化，避免 near-range 支配。

这比 binary occupied-cell 标签更接近当前 exact-10k 输出所需的几何目标。

### 5.3 RaUF：各向异性 uncertainty 作为可部署的风险分数

- [RaUF CVPR 2026 论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_RaUF_Learning_the_Spatial_Uncertainty_Field_of_Radar_CVPR_2026_paper.pdf)
- [官方项目页](https://shengpeng.wang/rauf/)

**[P]** RaUF 将雷达位置误差建模为极坐标各向异性 Gaussian：

```text
D = diag(sigma_r^2, sigma_a^2, sigma_e^2)
Sigma_xyz = J(r,a,e) D J(r,a,e)^T
L_nll = error^T Sigma_xyz^-1 error + log det(Sigma_xyz)
```

并用 spatial/Doppler 双向 attention 抑制 Doppler 不一致的 ghost returns。

**[I] 当前最小实现不应先复刻完整 BDAF，而应只新增：**

```text
quality_logit
log_sigma_r
log_sigma_a
log_sigma_e
```

候选排序分数可预注册为：

```text
score_i = sigmoid(quality_logit_i)
          * exp(-lambda_u * normalized_trace(Sigma_xyz_i))
```

或者只用 predicted expected geometric quality，将 uncertainty 作为 calibration/ablation，避免双重惩罚。两种方案必须在训练前冻结。

推荐增加：

- negative log likelihood；
- risk-coverage curve；
- expected calibration error；
- confidence 与 `-nearest-GT-distance` 的 Pearson/Spearman；
- top-k CD/outlier 随 coverage 的曲线。

### 5.4 AdaPoinTr：固定 cardinality 的 center-to-cluster decoder

- [AdaPoinTr 论文，ICCV 2023](https://arxiv.org/abs/2301.04545)
- [官方代码，PoinTr repository](https://github.com/yuxumin/PoinTr)
- 审计 commit：[`4603257ed3db9e7dad349b712e1b2fe0da207015`](https://github.com/yuxumin/PoinTr/tree/4603257ed3db9e7dad349b712e1b2fe0da207015)

**[P/C]** AdaPoinTr：

- 先预测固定数量 adaptive coarse queries/centers；
- 每个 center 用 FC/folding 重建固定数量 local points；
- 输出点数等于 `num_query * factor`；
- 用 coarse/fine Chamfer 和 denoising task 训练。

官方固定 cardinality 代码：

- [factor 与 fixed output assertion](https://github.com/yuxumin/PoinTr/blob/4603257ed3db9e7dad349b712e1b2fe0da207015/models/AdaPoinTr.py#L898-L928)
- [center-to-local-cluster rebuild](https://github.com/yuxumin/PoinTr/blob/4603257ed3db9e7dad349b712e1b2fe0da207015/models/AdaPoinTr.py#L951-L996)

**[I] 不建议直接搬完整 AdaPoinTr。** 它的输入是 partial object point cloud，当前输入是 scene-level raw Cube。应只借固定 center/child 解码方式：

```text
2500 selected Cartesian voxel centers x 4 immutable local slots = exact 10,000
```

这正好与 R-B2 已通过的结构 oracle 对齐。

### 5.5 Surface-aware sampling / implicit field

| 方法 | 论文/代码事实 | 当前可借部分 | 限制 |
|---|---|---|---|
| [3DShape2VecSet](https://arxiv.org/abs/2301.11445), [code](https://github.com/1zb/3DShape2VecSet) | [C] 1024 global-volume queries + 1024 near-surface queries；`loss_vol + 0.1 loss_near`；latent-set implicit decoder。 | 全空间与 surface-shell 双源采样，避免只有 easy empty negatives。 | object/surface setting；RaLD 已改变其采样定义，不能混称为 RaLD 原生。 |
| [POCO, CVPR 2022](https://arxiv.org/abs/2201.01831), [code](https://github.com/valeoai/POCO) | [P/C] 在每个输入表面点保存 local latent，用邻域插值回答 occupancy queries。 | 为 Cube peak/candidate 建 local evidence descriptor，而不是只靠 global latent。 | 需要相对可靠的输入点 anchors；raw Cube peaks 可能是 ghost。 |
| [ALTO, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_ALTO_Alternating_Latent_Topologies_for_Implicit_3D_Reconstruction_CVPR_2023_paper.html), [code](https://github.com/wzhen1/ALTO) | [P/C] 交替 point/grid latents，再以邻域 attention 解码 occupancy；论文报告在其 surface reconstruction 任务中更快。 | R-B2 的 Cartesian grid 与 candidate-point features 可交替融合。 | 工程复杂度高，且输入仍是点云，不应作为第一修复。 |
| [NKSR, CVPR 2023](https://arxiv.org/abs/2305.19590), [code](https://github.com/nv-tlabs/nksr) | [P/C] compact-support neural kernel field，支持大场景、稀疏 noisy oriented points。 | 可作后处理 surface regularizer 或 reconstruction reference。 | 依赖 oriented points/法向并输出 mesh；与雷达可见散射点目标不一致。 |

**[I] 对当前 R-A2 的结论：**

range-balanced global negatives + 2-4-cell surface-shell negatives是合理的 bounded control，但文献不能证明它会修复当前 top-k 排序。它最多回答“是否因为训练几乎只见 easy empty queries”，不应排在 geometry-quality alignment 之前。

## 6. Doppler 与多帧时序机制

### 6.1 RadarMP：从相邻 raw Cube 学 echo motion

- [RadarMP 论文，AAAI 2026](https://arxiv.org/abs/2511.12117)
- [官方代码](https://github.com/chengrui7/RadarMP)
- 审计 commit：[`bd03fd7f5e1da87f620f3b1775fa34e476d0a950`](https://github.com/chengrui7/RadarMP/tree/bd03fd7f5e1da87f620f3b1775fa34e476d0a950)

**[P]** RadarMP 输入相邻两帧低层 radar echo，联合预测 point detection 与 3D scene flow，并使用 Doppler shift 和 echo intensity 构造自监督时空一致性。

**[C] 官方实现包含：**

- RA/RE/AE 三个投影上的 PWC-style 2D flow；
- 64-D Doppler MLP；
- 3D ResNet 与 deformable correlation；
- echo-power warping loss；
- RA/RE/AE 2D warp loss权重 `0.6/0.35/0.05`。

代码入口：

- [RadarMP model](https://github.com/chengrui7/RadarMP/blob/bd03fd7f5e1da87f620f3b1775fa34e476d0a950/Models/RadarMP.py)
- [2D echo-flow loss](https://github.com/chengrui7/RadarMP/blob/bd03fd7f5e1da87f620f3b1775fa34e476d0a950/Losses/RadarMPLoss2d.py)
- [3D power/flow losses](https://github.com/chengrui7/RadarMP/blob/bd03fd7f5e1da87f620f3b1775fa34e476d0a950/Losses/RadarMPLoss.py)

**[I] 当前迁移不应直接训练完整 RadarMP detector。最小接口：**

```text
history_cube_drae, current_cube_drae
  -> RA/RE/AE projection flow
  -> history echo warp to current coordinates
  -> per-cell temporal consistency / reliability
  -> only add high-reliability history candidates to WCE or voxel-slot bank
```

当前 Cube 仍负责最终 geometry score、point position 和 Doppler refresh。

### 6.2 DoppDrive：每个历史点独立 age gate

- [DoppDrive 论文，ICCV 2025](https://arxiv.org/abs/2508.12330)
- [官方项目页](https://yuvalhg.github.io/DoppDrive/)
- 项目页的 code 链接实际指向 [LRR-Sim 数据仓库](https://github.com/yuvalHG/LRRSim)，截至核查日未发布方法实现。

**[P]** DoppDrive：

1. 去除 ego-induced Doppler；
2. 用 dynamic radial Doppler 将历史点沿视线方向平移；
3. 根据径向近似的 tangential error，为每个点设置独立最大聚合时长；
4. 超过误差容忍的老点不参与聚合。

**[I] 当前最有价值的不是再做一次 radial shift，而是加入 age/reliability gate：**

```text
keep(history point i) iff
    age_i <= max_age(v_dynamic_i, azimuth_i, error_budget)
    and echo_flow_consistency_i >= threshold
    and current_cube_support_i >= threshold
```

这可避免历史点在动态目标上形成切向拖尾。

### 6.3 点云 scene flow 备选

| 方法 | 一级来源 | 已实现机制 | 适用边界 |
|---|---|---|---|
| CMFlow, CVPR 2023 | [official code](https://github.com/Toytiny/CMFlow) | 两帧 radar points、cross-modal supervision、scene flow/motion segmentation/ego transform。 | 若已有可靠 sparse/dense radar points，可复用 point-flow head；raw Cube 路线优先 RadarMP。 |
| RaFlow, IROS/RA-L 2022 | [official code](https://github.com/Toytiny/RaFlow) | self-supervised 4D radar point scene flow。 | 可作无需 LiDAR flow label 的 control，但输入仍是点。 |
| RadarSFEMOS, RA-L 2025 | [official code](https://github.com/nubot-nudt/RadarSFEMOS) | diffusion scene flow + motion segmentation。 | 代价高；只有 RadarMP/CMFlow 失败后再考虑。 |

### 6.4 时序路线的证据边界

- 多帧 radar enhancement/aggregation 不是研究空白；DoppDrive、RadarMP、CMFlow、RaFlow 已覆盖相关任务。
- 当前项目的限定创新应是：**Full-RAED 当前 Cube 条件下联合生成 dense XYZ、pointwise Doppler distribution 和 confidence；历史只作为可拒绝的时序 prior，并由当前观测刷新。**
- 当前 G1T no-go 只关闭现有解析 proposal，不证明 learned echo-flow 无效。
- 但单帧 geometry parent 未过门时，任何“时序更平滑”都可能来自 copy/aggregation，不能当 geometry success。

## 7. 可执行路线矩阵

成本估计以单张 H200、当前 76 train + 24 validation Stage-0 为尺度，不包含全量数据正式训练。

| ID / 方法 | 能解决的已诊断瓶颈 | 与当前代码的最小接口 | 来源证据与推断边界 | 预期代价 | 预注册失败判据 | 推荐 |
|---|---|---|---|---|---|---|
| **Q1 Geometry-quality aligned WCE**：Rank-DETR/Align-DETR soft quality target + local many-to-one bags | 当前 confidence 与几何质量低相关；同一 700k pool 中有好点却选不出 | 保留 `RaLDWCEField` encoder/query/residual；新增 `quality_logit`。训练时用 kNN 最近 target distance 构造 `q=exp(-d/tau)`；export 改按 quality 排，Q0/Q1/配额/5 cm 全不变 | [P/C] Rank/Align 在 DETR 中对齐分数与 localization；[I] 将 IoU 换成 point-to-target distance | 1-2 天实现；8-frame 500 updates + 5-epoch pilot，约 1 个 H200 日 | Tiny 到 500 updates 未连续两次满足 `CD<=1.0/outlier<=10%/completeness<=0.75`；或 validation epoch-5 相对 R-A1 outlier 未改善 >=2 pp、far F1 未改善 >=25%、median completeness 恶化 >0.25 m | **P0，首选** |
| **Q2 RaUF-lite uncertainty WCE**：polar heteroscedastic NLL + calibrated risk ranking | 雷达角向歧义/ghost 使 binary occupancy 对高误差候选过度自信 | 在 Q1 head 上增加 `log_sigma_r/a/e`；用 Jacobian 得 `Sigma_xyz`；同一 refined candidate 做 NLL。可与 Q1 合并训练，但独立 ablation | [P] RaUF 在其任务中证明 anisotropic uncertainty；[I] 当前只实现 lite head，不声称 BDAF 复现 | Q1 之上约 0.5-1 天；显存增量小 | NLL 数值不稳；risk-coverage 不优于 plain quality；Spearman 对 `-distance` 不提高至少 `+0.10`；top10k CD 未改善至少 10% | **P0，与 Q1 同一分支** |
| **V1 Learned R-B2 voxel-slot**：Full-Doppler grid encoder + fixed 2500x4 slots | 绕开全局 700k hard top-k；用局部固定 slots 表达多点；oracle 已证明结构容量 | 继续现有 `cube_drae + candidate voxel IDs -> occupancy + 4 slot scores + 4 offsets`；第一版不引入 Transformer，只加 DenserRadar 式轻量 spherical features | [P] DenserRadar full-D 3D occupancy；[P/C] AdaPoinTr fixed center-cluster；[E] R-B2 oracle 过结构门；[I] Cube-to-slot generalization 未证 | 1-frame 500 updates 约数小时；通过后 76/24 pilot 1-2 H200 日 | 先过 candidate support `occupied recall>=20%/confidence coverage>=30%`；随后连续两次 exact10k、5 cm、`CD<=1.0/outlier<=10%/completeness<=0.75`；任一失败按现有协议 no-go | **P1，第二主线** |
| **S1 RaLD-faithful thresholded field control**：去 exact top-k，只保留 `logit>0`，报告 variable cardinality | 区分 field learning 失败与 fixed-count ranking 失败 | 复用 R-A1 checkpoint/field；只新增只读 evaluator，不改训练；报告 cardinality、precision/completeness、CD/F1 随 threshold 曲线 | [C] RaLD 官方 inference；[I] K-Radar control | 0.5 天，无长训练 | 所有 threshold 上 precision/completeness Pareto 均不优于 exact10k；或无法满足最小密度和远距覆盖 | **P1-control，不是最终输出** |
| **N1 Revised range + metric shell**：availability mask + boundary-safe jitter + metric/ray-aware shell | 排除 easy negatives 占绝对多数导致边界没学好的可能 | 当前 R-A2 range sampler 已被只读审计关闭；只有重新冻结缺失类口径、等量 control、物理 shell 和 cache provenance 后才可实现 | [C] RaLD occupied/empty classwise；[C] 3DShape2VecSet global/near mixture；[E] 当前 index-shell audit no-go；[I] metric/ray-aware replacement | 需新协议和预飞，不得直接启动现有实现 | 任一缺失 range 类被伪造/重分配、连续坐标跨段、metric shell 污染未冻结或 query budget 不匹配即失败 | **Blocked control；不与 Q1 并列** |
| **T1 RadarMP echo-flow proposals + DoppDrive age gate** | 现有 raw Doppler radial proposal no-go；动态物体存在切向散射与跨帧错误积累 | `history_cube_drae,current_cube_drae,pose -> flow/reliability`; 只将可靠历史候选加入 Q1/voxel bank；当前 Cube score 决定 export | [P/C] RadarMP raw echo motion；[P] DoppDrive pointwise age gate；[I] 组合到生成候选 | flow2D 预训练 + pilot 约 2-4 H200 日 | candidate support/far recall 不提高 >=5 pp；wrong-history 与 matched-history 无显著差；current-frame CD 恶化 >5%；或 history 占比升高但 current Cube support 降低 | **P2；单帧 parent 过门后做** |
| **L1 POCO/ALTO local implicit refinement** | global latent 无法利用 candidate 附近局部谱形；薄结构可能丢失 | 把 Cube peak/voxel local feature 作为 anchors，用 kNN attention 回答 queries；只 refine Q1 或 slot offsets | [P/C] local point/grid latent surface reconstruction；[I] radar peaks 作为 anchors | 3-5 天实现，邻域查询/稀疏算子风险中高 | Cube-only anchors 的 target coverage 未过门；one-frame 不优于 simpler Q1/V1；推理超过当前 WCE 2x 且 geometry 无显著改善 | **P3** |
| **D1 Latent diffusion / consistency refinement**：RaLD, SDDiff, Radar-Diffusion, RadarSFD | 表达多模态几何、生成完整结构 | 只在已通过的 Q1/V1 geometry parent 上做 latent/refinement；不得重开旧 matched AE | [P/C] 多个来源证明各自 2D/3D任务的 diffusion；[E] 当前 matched RaLD AE no-go | 至少 3-7 H200 日，调参和复现风险高 | parent geometry 未过门；refiner CD/outlier 无改善或 condition shuffle 不退化；成本 > deterministic parent 10x | **P4，最后考虑** |
| **F1 SD4R foreground virtual points** | downstream detection 需要目标区域密集点，而全场景 geometry 可能浪费容量 | 在已有 sparse radar points 上加 VoteHead/FPG；不接 raw Cube geometry parent | [P/C] VoD detection densification；[I] 仅作 downstream control | 2-4 天，需 VoD/downstream labels | geometry 指标不适用；若 detection 无提升则关闭 | **P3-control** |

## 8. Priority 0 的具体实验设计

### 8.1 Q1/Q2 合并模型

建议新增独立类而不是修改 R-A1 归档语义：

```text
RaLDWCEQualityField
  reuses:
    radar_encoder
    condition_latents
    coordinate_embedding
    decoder_attention
    residual_head

  adds:
    occupancy_logit       # 保留，作 occupancy control
    quality_logit         # 直接监督几何质量
    log_sigma_r/a/e       # optional RaUF-lite
```

训练 target：

```text
refined_xyz_i = polar_to_cartesian(query_i + predicted_residual_i)
d_i = nearest_target_distance(refined_xyz_i, target_xyz)
w_i = nearest_target_confidence
q_i = stopgrad(w_i * exp(-d_i / tau_range(i)))
```

防止训练初期 residual 与 quality 相互追逐：

1. 前 50 updates 对 `refined_xyz_i` 生成 quality target 时 detach residual。
2. 之后允许 residual target 每 10 updates 刷新一次，仍对 target detach。
3. 首轮只对 source-classwise positives/global negatives 分别归一化；只有
   revised metric/ray-aware shell 通过独立预飞后才增加 shell 类。
4. 同时保留 wrong-condition same-query intervention。

### 8.2 计算约束

不能对 700k queries 与全部 target 建全矩阵。训练 sampler 只有 10k/16k queries，可用：

- `torch.cdist` 分块；
- FAISS/kNN；
- target Cartesian hash grid。

正式 700k inference 不需要 GT 距离，只计算 model quality/uncertainty。

### 8.3 冻结 ablation

| Arm | Occupancy | Soft quality | Uncertainty | Sampler |
|---|---:|---:|---:|---|
| A0 R-A1 reference | yes | no | no | original |
| A1 R-A2 source-classwise | yes | no | no | RaLD source |
| A2 Q1 | yes | yes | no | source-classwise |
| A3 Q1 + revised shell | yes | yes | no | blocked pending a new sampler preflight |
| A4 Q1 + RaUF-lite | yes | yes | yes | source-classwise unless a revised sampler independently passes |

主比较必须使用：

- 完全相同 Q0/Q1 candidate IDs；
- 完全相同 residual；
- 完全相同 exact-10k 配额与 5 cm 选择；
- 只替换 ranking score。

这样能够直接回答“质量分数是否修复同一候选池的选择”。

### 8.4 新增诊断指标

除已有 CD/completeness/outlier/far F1 外，增加：

1. `Pearson(score, -distance)`；
2. `Spearman(score, -distance)`；
3. top `1k/2.5k/5k/7.5k/10k` 的 precision mean distance；
4. risk-coverage AUC；
5. uncertainty calibration error；
6. 每个 range 的 score-distance reliability diagram；
7. matched vs wrong Cube 的质量排序退化；
8. Q0 与 Q1 分来源的 top-k 占比和几何质量。

## 9. Priority 1 的具体实验设计

R-B2 第一版应保持现有 one-frame protocol，不在结构门前引入大模型：

```text
Full-RAED local spectrum
  + normalized Cartesian center
  + normalized RAE coordinate
  -> shared MLP / small sparse-grid feature
  -> voxel occupancy
  -> four slot quality scores
  -> four bounded local offsets
```

通过 one-frame gate 后才考虑：

1. DenserRadar 式 3D U-Net multiscale features；
2. AdaPoinTr 式 adaptive center features；
3. Q1 式 soft geometric quality；
4. RaUF-lite slot uncertainty；
5. slot-level Doppler distribution head。

避免：

- 用 GT 激活候选 voxel；
- 用 GT 排序 export；
- 复制/抖动补足 10k；
- 用一个自由中心生成 4 个近重复点；
- 在 validation 看到结果后改 slots 或 range quotas。

## 10. Priority 2 的具体实验设计

时序路线分三步，任何一步失败都停止：

### T0：只读 echo-flow support oracle

- 输入相邻 `history/current cube_drae`；
- 使用 RadarMP-style RA/RE/AE flow 或简化相关性；
- 只比较 warped history candidate bank 对 current target 的 support；
- 不训练最终 generator。

过门：

- far target recall@2m 相对 current-only 增加至少 5 pp；
- candidate-to-target outlier 不增加超过 2 pp；
- wrong-history arm 显著差于 matched-history。

### T1：proposal-only integration

- 历史只补充候选；
- current Cube 决定 quality、position、Doppler；
- DoppDrive age gate 丢弃高 tangential-risk 老点；
- 保持 exact-10k output。

### T2：temporal consistency

只在 T1 当前帧 geometry 不退化时加入：

- echo-power warp consistency；
- point displacement vs predicted Doppler consistency；
- current Cube Doppler-spectrum NLL；
- wrong-history intervention；
- rollout stability。

“更平滑”不是通过条件；必须同时保持 current geometry、coverage 和 Doppler refresh。

## 11. 来源清单与代码可用性

### Radar generation / enhancement

1. [RaLD paper](https://arxiv.org/abs/2511.07067) | [official code](https://github.com/MetaIoT-WHU/RaLD)
2. [DenserRadar paper](https://arxiv.org/abs/2405.05131) | 本次未发现一方代码
3. [SDDiff official IJCAI paper](https://www.ijcai.org/proceedings/2025/979) | [official placeholder repository](https://github.com/StellarEsti/SDDiff)
4. [RaUF official CVF paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_RaUF_Learning_the_Spatial_Uncertainty_Field_of_Radar_CVPR_2026_paper.pdf) | [official project](https://shengpeng.wang/rauf/) | code 未可取
5. [Radar-Diffusion paper](https://arxiv.org/abs/2403.08460) | [official code](https://github.com/ZJU-FAST-Lab/Radar-Diffusion)
6. [Diffusion radar point-cloud SR paper](https://arxiv.org/abs/2404.06012) | 本次未发现一方代码
7. [RadarSFD paper](https://arxiv.org/abs/2509.18068) | [official code](https://github.com/phi-lab-rice/RadarSFD) | [weights](https://huggingface.co/Bin-0815/RadarSFD)
8. [RMap paper](https://arxiv.org/abs/2310.13188) | [official code](https://github.com/arpg/RMap)
9. [SD4R paper](https://arxiv.org/abs/2602.20653) | [official code](https://github.com/lancelot0805/SD4R)

### Set prediction / fields / surface reconstruction

1. [Rank-DETR paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/34074479ee2186a9f236b8fd03635372-Paper-Conference.pdf) | [official code](https://github.com/LeapLabTHU/Rank-DETR)
2. [Align-DETR paper](https://arxiv.org/abs/2304.07527) | [official code](https://github.com/FelixCaae/AlignDETR)
3. [AdaPoinTr paper](https://arxiv.org/abs/2301.04545) | [official code](https://github.com/yuxumin/PoinTr)
4. [3DShape2VecSet paper](https://arxiv.org/abs/2301.11445) | [official code](https://github.com/1zb/3DShape2VecSet)
5. [POCO paper](https://arxiv.org/abs/2201.01831) | [official code](https://github.com/valeoai/POCO)
6. [ALTO paper](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_ALTO_Alternating_Latent_Topologies_for_Implicit_3D_Reconstruction_CVPR_2023_paper.html) | [official code](https://github.com/wzhen1/ALTO)
7. [NKSR paper](https://arxiv.org/abs/2305.19590) | [official code](https://github.com/nv-tlabs/nksr)

### Temporal / Doppler / scene flow

1. [RadarMP paper](https://arxiv.org/abs/2511.12117) | [official code](https://github.com/chengrui7/RadarMP)
2. [DoppDrive paper](https://arxiv.org/abs/2508.12330) | [official project](https://yuvalhg.github.io/DoppDrive/) | 只有 LRR-Sim 数据代码
3. [CMFlow official code and paper links](https://github.com/Toytiny/CMFlow)
4. [RaFlow official code and paper links](https://github.com/Toytiny/RaFlow)
5. [RadarSFEMOS official code and paper links](https://github.com/nubot-nudt/RadarSFEMOS)

## 12. 最终决策

当前最值得投入的不是“更大的生成模型”，而是把已经存在于候选池中的好几何可靠地选出来。

1. **立即做 Q1/Q2：geometry-quality aligned WCE + RaUF-lite uncertainty。**
   它直接针对已验证的 confidence-ranking bottleneck，复用所有冻结候选和 evaluator，能用最小成本得到明确的 go/no-go。

2. **并行保留 V1：learned R-B2 voxel-slot。**
   它提供一条不依赖 700k hard top-k 的独立表示路线，而且已有结构 oracle 支持；但必须先完成 Cube-only one-frame gate。

3. **时序 T1 延后到单帧 parent 过门。**
   届时使用 RadarMP 的 learned echo-flow 补足当前解析 Doppler warp 的非径向缺陷，并用 DoppDrive age gate 限制历史污染。它应是可拒绝的 proposal prior，不是复制历史点代替当前生成。

RaLD 仍然是重要架构来源，但本项目现在应借它的 **frustum field、wide radar-guided queries 和 latent-set inductive bias**，不应把它未提供的 fixed top-k confidence ranking 当作已解决问题。
