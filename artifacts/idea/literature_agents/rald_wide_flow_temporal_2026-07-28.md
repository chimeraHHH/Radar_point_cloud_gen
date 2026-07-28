# RaLD-wide、Polar Flow 与 Doppler 时序路线审计

> 日期：2026-07-28  
> 范围：论文与官方代码只读调研；未据此直接启动训练。  
> 目标：解释长期训练效果不佳，并冻结互斥、可证伪的小实验。

## 1. 当前故障不是单纯训练不足

现有证据指向三个上游限制：

1. 冻结 32k proposal pool 的 G1F GT-aided diagnostic 仍只有 17.80% 的
   60--120 m / 2 m support recall。selector、top-k 或 loss 无法恢复候选域中
   不存在的远距支持。
2. G1D 的局部频谱、能量和距离路径在全局 condition 之外直接读取正确 Cube，
   cross-scene condition shuffle 因而不能真正切断场景信息。
3. 固定 seed-template-parent-child 分配把“覆盖位置”和“局部点数”耦合，
   oracle 仍有明显重复，继续堆相同 refiner 不足以解决分配容量。

2026-07-28 的 corrected evaluator 又证明旧 far-completeness 存在删失偏差：
G1D epoch-15 的 23 个 far-target 验证帧全部计入后，mean far completeness
为 46.9407 m，而不是旧表的 8.1239 m。

## 2. 应该从 RaLD 借鉴什么

RaLD 最有价值的是生成表示与查询方式：

- 场景首先编码成无序 latent set；
- query decoder 只接收查询坐标，再 cross-attend 场景 latent；
- 推理使用宽域随机/helper queries，而不是只从固定高能 seed 展开；
- 首轮 occupancy 后，再围绕预测正区域做独立 refinement。

一手来源：

- [RaLD AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/38946)
- [Official point-latent decoder](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/model/models_ae.py#L375-L390)
- [Official two-stage wide-query inference](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/engine_generation.py#L224-L310)
- [Official radar-conditioned generator](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/model/models_radar_generation.py#L331-L369)
- [Official evaluation configuration](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/configs/generation/ge_indoor_cfg_aniso_mix_view_cone_unfreeze_enc_ints_only_eval.yml)

不能直接继承的主张：

- 官方发布路径主要使用 radar intensity condition；
- 输出为 XYZ，不是本项目的 exactly-10k XYZ+Doppler+confidence；
- 没有本项目的 corrected far gate、condition-shuffle gate 或 Cube-point
  Doppler 闭环。

因此 RaLD 在本项目中是“Cube-conditioned continuous occupancy-query
backbone”的结构来源，不是已经公平匹配的完整 baseline。

## 3. 相邻一手路线

### Continuous implicit field

- [UnO, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Agro_UnO_Unsupervised_Occupancy_Fields_for_Perception_and_Forecasting_CVPR_2024_paper.pdf)
- [DIO, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Diehl_DIO_Decomposable_Implicit_4D_Occupancy-Flow_World_Model_CVPR_2025_paper.pdf)
- [Official 4D occupancy code](https://github.com/tarashakhurana/4d-occ-forecasting)

可借鉴任意时空位置查询、正负区域平衡采样和 occupancy-flow 分解；不能原样
复制其 LiDAR renderer。

### Range/polar diffusion

- [Radar range-image diffusion](https://arxiv.org/abs/2503.02300)
- [RangeLDM](https://arxiv.org/abs/2403.10094)
- [RangeLDM official code](https://github.com/WoodwindHu/RangeLDM)
- [RadarGen](https://radargen.github.io/)

可借鉴 sensor-native range/polar representation、循环方位编码和 range-density
归一化。RadarGen 是条件仿真，不是 Cube-to-dense-point 方法。

### Flow matching

- [Flow Matching](https://arxiv.org/abs/2210.02747)
- [Meta official flow-matching library](https://github.com/facebookresearch/flow_matching)
- [PUFM official code](https://github.com/Holmes-Alan/PUFM)
- [DeltaFM official code](https://github.com/gstoica27/DeltaFM)

Rectified/conditional flow 可减少采样步数并直接运输 unordered points，但普通
conditional flow 并不保证使用 Cube condition。必须用固定 noisy state、仅交换
condition 的 wrong-condition objective 检验条件依赖。

### Temporal radar

- [DoppDrive, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Haitman_DoppDrive_Doppler-Driven_Temporal_Aggregation_for_Improved_Radar_Object_Detection_ICCV_2025_paper.html)
- [RadarMP official code](https://github.com/chengrui7/RadarMP)
- [RaFlow official code](https://github.com/Toytiny/RaFlow)

这些工作证明多帧 radar aggregation/scene flow 不是空白。项目差异必须限定在
raw 4D Cube condition、exactly-10k XYZ+Doppler+confidence、当前 Cube 刷新和
物理闭环。

### Wrong-condition consistency

- [GAN-CLS](https://proceedings.mlr.press/v48/reed16.html)
- [ControlNet++ official code](https://github.com/liming-ai/ControlNet_Plus_Plus)
- [CLIP official code](https://github.com/openai/CLIP)

real/right、real/wrong 和 output-to-condition consistency 可直接惩罚忽略
condition。FiLM、AdaLN、CFG 或更多 cross-attention 只能增强注入接口，不能单独
排除条件旁路。

## 4. 三条互斥小实验

### R-A1: RaLD-wide condition-exclusive implicit field

最小改动：

- 保留 Full-RAED token encoder；
- decoder 在几何选择前只读取 RAE/XYZ query coordinates 和全局 tokens；
- 先运行 1.2M initial-query support diagnostic；
- learned arm 才增加 occupancy-dependent second-pass refinement；
- 选定 exactly 10k 坐标后，再从当前 Cube 查询 Doppler distribution。

预注册信号：

- corrected 60--120 m / 2 m support recall 相对 32k 的 17.80% 明显提高；
- 10 epoch 后 completeness 相对 corrected G1D 至少改善 30%；
- far completeness 至少改善 20%；
- duplicate <=15%，condition shuffle degradation >=1%。

失败解释：

- initial-query support 不足：低阈值宽域仍没有可达支持；
- support 通过而 learned arm 失败：occupancy supervision 或条件容量不足；
- 几何改善而 shuffle 失败：模型学习了无条件场景先验。

预算：initial-query diagnostic 0.8--1.5 H200 GPU-hour；learned Stage-0
8--12 H200 GPU-hour。

### P-RF: polar exactly-10k rectified flow

最小改动：

- 不使用 32k pool 或固定 parent-child templates；
- 10k 分层 polar particles 的状态为 R/A/E/Doppler；
- 每个 flow block cross-attend Full-RAED tokens；
- 使用循环方位编码、range-density normalization 和 OT/Sinkhorn pairing；
- 比较 4-step 与 8-step。

预注册信号：

- 5 epoch screen 的 far completeness 相对 G1G 改善至少 15%；
- duplicate <=15%，condition shuffle degradation >=1%；
- 4-step 与 8-step Chamfer 差距小于 5%；
- range-mass distribution error 下降。

失败解释：

- flow loss 下降但 far 不改善：优化改善不能创造缺失支持；
- 8-step 显著优于 4-step：没有获得低成本生成；
- shuffle 失败：vanilla CFM 仍忽略 Cube。

预算：5--8 H200 GPU-hour，预计 30--45 GB。

### T-WC: Doppler history support plus wrong-condition ranking

最小改动：

- 复用 G1T current/ego/Doppler candidate construction；
- scorer 只读取 current global Cube tokens、候选位置、年龄和 warp residual；
- 几何选择前不读取正确 Cube 的局部频谱；
- 固定候选和时间，只交换其他 scene 的 Cube，加入 matched-vs-wrong rank loss。

预注册信号：

- corrected no-train T2 相对 T1 的 far completeness 或 2 m recall 改善至少 10%；
- 5 epoch learned scorer 再改善至少 10%；
- history fraction 保持 15%--60%；
- condition shuffle degradation >=1%；
- duplicate/outlier 相对控制不恶化超过 2 percentage points。

失败解释：

- T2 不通过：历史没有补到缺失的远距表面；
- T2 通过而 scorer 失败：当前 Cube 无法筛掉陈旧历史；
- 几何改善而 shuffle 失败：模型只在复制历史。

预算：no-train diagnostic 1--2 H200 GPU-hour；scorer 3--5 H200
GPU-hour，预计低于 24 GB。

## 5. 执行顺序

1. corrected G1T 先裁决时间支持；
2. 修正后的 R-A1 initial-query diagnostic 裁决空间宽域支持；
3. G1G formal 独立裁决 condition-exclusive hierarchy；
4. 只有空间支持或时间支持至少一条明确通过，才授权 P-RF 或 T-WC 训练；
5. 各机制先独立通过，不提前融合。
