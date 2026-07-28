# 时序雷达生成、预测与场景流文献和开源代码审计

> 检索冻结日：2026-07-29。文件名沿用 2026-07-28 候选冻结批次。
> 范围：2022--2026 年的 radar tesseract/cube temporal fusion、radar scene
> flow、Doppler-aware aggregation、radar odometry、point/occupancy forecasting
> 与 teacher-forcing/scheduled-sampling。
> 证据规则：优先论文原文、CVF/PMLR/ICLR 页面和作者官方仓库。没有找到代码时
> 只写“本次检索未找到官方实现”，不据此证明代码绝对不存在。

## 1. 面向当前失败的结论

本轮检索不支持“多帧雷达增强或雷达运动建模无人做”的表述。相反，
Radar-Mamba、RadarMP、DREAM-PCD、DoppDrive、RaFlow/CMFlow、
RadarSFEMOS 和 TARS 已分别覆盖多帧特征融合、tesseract 场景流、因果多帧
重建、Doppler 历史聚合和雷达场景流。

真正可吸收的机会不是再增加一个普通 temporal attention，而是以下四种机制：

1. **运动分解的历史 proposal**：静态点走 ego warp，动态点走
   Doppler 径向位移加有界切向残差，并显式输出 proposal uncertainty。
2. **radar-observable occupancy-flow field**：在连续时空场中联合建模占据和
   运动，但必须通过当前 Full-RAED Cube 的可微渲染闭环，而不是只拟合 LiDAR
   occupancy。
3. **masked-current Cube 时序瓶颈**：用历史预测被遮蔽的当前 RAED token，
   强迫 global condition 携带信息，直接针对 condition-shuffle 近零。
4. **直接多时距预测加 rollout curriculum**：先并行预测多个未来时距，再把
   free-running/self-conditioned rollout 作为训练课程；scheduled sampling
   只是消融，不是创新点。

这些机制分别针对当前三个已知病因：

| 当前证据 | 诊断 | 优先机制 |
|---|---|---|
| G1D 中期 condition shuffle 约 `0.016%` | direct local-Cube query 绕过 global condition | masked-current Cube 瓶颈 |
| G1F GT coverage oracle 的 60--120 m、2 m recall 仅 `17.80%` | 当前单帧 proposal support 本身不足，不只是 selector 不好 | 运动分解历史 proposal；occupancy-flow field |
| G1F duplicate `14.015%`，G1D 中期接近 `30%` | 固定 10k 预算在局部拥挤，缺少射线/时空质量守恒 | occupancy-flow field 加 ray-balanced rendering |
| G1T 尚无科学结果 | 不能预设历史一定有用 | 先完成无学习 proposal oracle，再决定是否训练时序模型 |

状态边界：

- **已验证**：G1F 单帧候选池 oracle 不足。
- **中期诊断**：G1D 长训练 plateau、condition bypass、重复率高；最终
  epoch-150 结论不能由本报告替代。
- **已实现但未验证**：G1G one-seed Stage-0 训练器。
- **准备中**：G1T temporal proposal 数据与 T0/T1/T2。
- **仅计划**：本报告提出的四个新 Stage-0，不得写成方法结果。

## 2. 强相关工作表

“保留 Doppler”区分“作为输入/损失使用”和“作为输出状态生成”。后者才与
Full-RAED 的逐点 circular Doppler distribution 主张直接相关。

| 工作 | 年份/任务 | 输入 | 输出 | 保留 Doppler | 预测未来 | 官方代码 |
|---|---:|---|---|---|---|---|
| [RaLD](https://arxiv.org/abs/2511.07067) | 2025/雷达生成 | 单帧 radar spectrum | 固定 10k `XYZ` | 否；公开主线为 intensity-only，输出无速度 | 否 | [是](https://github.com/MetaIoT-WHU/RaLD) |
| [Radar-Mamba](https://palm.seu.edu.cn/zhangml/files/MM%2725.pdf) | 2025/多帧增强 | 当前帧和前两帧；将 Full-RAED 压缩为 strongest reflection + Doppler，并保留 RAD map | 当前帧 dense occupancy/point cloud | 输入中保留速度特征；输出不生成逐点 Doppler 分布 | 否 | 本次未找到官方实现 |
| [RadarMP](https://arxiv.org/abs/2511.12117) | 2026/tesseract motion perception | 连续两帧 K-Radar tesseract | radar points/segmentation 与逐点 3D scene flow | 是，Doppler 进入编码和自监督损失；输出是 flow，不是 Doppler 分布 | 否；估计两帧位移，不生成未来观测 | [是](https://github.com/chengrui7/RadarMP) |
| [DREAM-PCD](https://arxiv.org/abs/2309.15374) | 2023/因果多帧重建 | 室内 raw radar 多帧 | 当前高质量 radar point cloud | Doppler 不作为逐点输出状态 | 否 | 本次未找到官方实现 |
| [DoppDrive](https://openaccess.thecvf.com/content/ICCV2025/html/Haitman_DoppDrive_Doppler-Driven_Temporal_Aggregation_for_Improved_Radar_Object_Detection_ICCV_2025_paper.html) | 2025/Doppler 聚合 | 历史 radar points、ego pose、measured Doppler | 对齐到当前时刻的测量点集合 | 是，动态 Doppler 决定径向 shift 和逐点历史长度 | 否 | [仓库仅有 README，未发布算法源码](https://github.com/yuvalHG/DoppDrive) |
| [RaFlow](https://arxiv.org/abs/2203.01137) | 2022/自监督 radar scene flow | 两帧 4D radar point clouds | 逐点 flow、motion segmentation | 是，radial displacement loss | 否 | [是](https://github.com/Toytiny/RaFlow) |
| [CMFlow](https://openaccess.thecvf.com/content/CVPR2023/html/Ding_Hidden_Gems_4D_Radar_Scene_Flow_Learning_Using_Cross-Modal_Supervision_CVPR_2023_paper.html) | 2023/跨模态 radar scene flow | 两帧 radar；训练期 LiDAR/camera/odometry supervision | flow、motion segmentation、ego motion | 是，保留 radar radial velocity 约束 | 否 | [是](https://github.com/Toytiny/CMFlow) |
| [RadarSFEMOS](https://miasgroup.tongji.edu.cn/_upload/tpl/06/23/1571/template1571/pdf/ral2025_liu.pdf) | 2025/自监督 scene flow + MOS | 两帧 4D radar points | pointwise flow 和 motion status | 是，soft Chamfer + smoothness + radial displacement | 否 | [是](https://github.com/nubot-nudt/RadarSFEMOS) |
| [TARS](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_TARS_Traffic-Aware_Radar_Scene_Flow_Estimation_ICCV_2025_paper.html) | 2025/traffic-aware scene flow | 两帧 radar points + detector feature map | pointwise scene flow | 是，RRV 为关键运动 cue | 否 | 本次未找到官方实现 |
| [EFEAR-4D](https://arxiv.org/abs/2405.09780) | 2024/learning-free radar odometry | 连续 4D radar points + Doppler | ego velocity、odometry、map | 是，用 Doppler 估计 ego velocity 并过滤动态点 | 否 | [是](https://github.com/CLASS-Lab/EFEAR-4D) |
| [RaLiFlow](https://arxiv.org/abs/2512.10376) | 2026/radar-LiDAR scene flow | 连续 radar + LiDAR point clouds | 两模态 scene flow | 是，用 radar dynamic cue 做双向融合 | 否 | [是](https://github.com/FuJingyun/RaLiFlow) |
| [IterFlow](https://arxiv.org/abs/2605.18507) | 2026/弱监督 radar scene flow | 两帧 radar；训练期 camera tracking + odometry | radar scene flow | 是，radar intrinsic motion + static rigid loss | 否 | [是](https://github.com/FuJingyun/IterFlow) |
| [Flow4D](https://arxiv.org/abs/2407.07995) | 2024/多帧 LiDAR scene flow | 五帧 LiDAR | 当前 source points 的 scene flow | 否 | 否；利用更长历史估计当前 flow | [是](https://github.com/dgist-cvlab/Flow4D) |
| [Self-supervised Point Cloud Prediction](https://proceedings.mlr.press/v164/mersch22a.html) | 2022/LiDAR forecasting | 多帧历史 LiDAR range images | 固定多个未来 range image/point clouds | 否 | 是；并行输出固定未来时距 | [是](https://github.com/PRBonn/point-cloud-prediction) |
| [S2Net](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870541.pdf) | 2022/随机点云预测 | 历史 LiDAR point sequence | 多样化未来点云 + uncertainty | 否 | 是 | 本次未找到官方实现 |
| [4D-OCC Forecasting](https://openaccess.thecvf.com/content/CVPR2023/html/Khurana_Point_Cloud_Forecasting_as_a_Proxy_for_4D_Occupancy_Forecasting_CVPR_2023_paper.html) | 2023/4D occupancy | 历史 LiDAR、未来 sensor pose/rays | future occupancy，渲染为 point cloud | 否 | 是，1 s/3 s | [是，含可微 voxel renderer 和 checkpoints](https://github.com/tarashakhurana/4d-occ-forecasting) |
| [UnO](https://openaccess.thecvf.com/content/CVPR2024/html/Agro_UnO_Unsupervised_Occupancy_Fields_for_Perception_and_Forecasting_CVPR_2024_paper.html) | 2024/continuous 4D occupancy | 历史 LiDAR | 连续时空 occupancy + rendered future points | 否 | 是 | 本次未找到官方训练代码 |
| [DIO](https://openaccess.thecvf.com/content/CVPR2025/html/Diehl_DIO_Decomposable_Implicit_4D_Occupancy-Flow_World_Model_CVPR_2025_paper.html) | 2025/decomposable occupancy-flow | sparse LiDAR + optional instance prompts | present completion + future occupancy-flow | 否 | 是 | 本次未找到官方训练代码；有[官方项目页](https://waabi.ai/research/dio) |
| [OccWorld](https://arxiv.org/abs/2311.16038) | 2024/occupancy world model | 历史 occupancy 和 ego tokens | future occupancy 和 ego trajectory | 否 | 是，自回归 scene tokens | [是](https://github.com/wzzheng/OccWorld) |
| [Copilot4D](https://proceedings.iclr.cc/paper_files/paper/2024/file/1d41343cf93afd6456db3d1820ce1a58-Paper-Conference.pdf) | 2024/离散扩散 world model | 历史 LiDAR observations/tokens | 多模态 future point observations | 否 | 是，1 s/3 s | 未公开训练代码；有[官方项目页](https://waabi.ai/copilot-4d/) |
| [Flipped Classroom](https://arxiv.org/abs/2210.08959) | 2022/rollout curriculum | 历史 time series | future sequence | 不适用 | 是 | [是](https://github.com/phit3/flipped_classroom) |

## 3. 开源代码审计

本节是静态源码审计，不代表已经复现任何外部结果。

### 3.1 可以直接借用的代码机制

1. **RaFlow/RadarSFEMOS 的径向位移损失**

   [RaFlow loss](https://github.com/Toytiny/RaFlow/blob/master/losses/loss.py)
   和
   [RadarSFEMOS radar loss](https://github.com/nubot-nudt/RadarSFEMOS/blob/main/losses/radar_loss.py)
   都把 soft Chamfer、spatial smoothness 与 radial displacement 分开。
   可迁移的是损失分解和评估口径，不是其 point-to-point backbone。

2. **RadarMP 的 tesseract motion stack**

   [模型源码](https://github.com/chengrui7/RadarMP/blob/main/Models/RadarMP.py)
   包含 Doppler MLP、3D ResNet、多尺度 3D correlation、2D flow initialization、
   segmentation attention 和 flow head；
   [损失源码](https://github.com/chengrui7/RadarMP/blob/main/Losses/RadarMPLoss.py)
   包含 power-warp、Doppler-derived motion label 和 local flow smoothness。

   不能照搬的部分是官方预处理会“截去前半段 range bins”。这与 Full-RAED
   当前最严重的远距覆盖失败方向相反。RadarMP 只能作为 scene-flow/temporal
   control，不能成为我们的数据裁剪模板。

3. **4D-OCC 的 measurement renderer**

   官方仓库的
   [differentiable voxel rendering](https://github.com/tarashakhurana/4d-occ-forecasting/blob/main/utils/layers/differentiable_voxel_rendering.py)
   和
   [model.py](https://github.com/tarashakhurana/4d-occ-forecasting/blob/main/model.py)
   已把 occupancy prediction 与 sensor-ray rendering 分开。最值得迁移的不是
   LiDAR voxel shape，而是“世界状态和传感器观测解耦，再通过可微传感器模型
   闭环”的接口。

4. **直接多未来帧预测**

   PRBonn 的
   [TCNet](https://github.com/PRBonn/point-cloud-prediction/blob/main/pcf/models/TCNet.py)
   用 3D spatio-temporal convolution，一次输出固定数量的未来 range 和
   visibility mask，并对方位维使用 circular padding。它提供了比纯
   autoregressive rollout 更干净的 Stage-0 对照。

5. **显式自回归 occupancy token**

   OccWorld 的
   [PlanUTransformer](https://github.com/wzzheng/OccWorld/blob/main/model/transformer/PlanUtransformer.py)
   同时实现 temporal attention 与 `forward_autoreg`。可借它的因果 mask 和
   direct-vs-autoregressive 对照，不应照搬 semantic occupancy tokenizer。

6. **rollout curriculum**

   Flipped Classroom 的
   [curriculum implementations](https://github.com/phit3/flipped_classroom/tree/main/models)
   将 teacher-forcing ratio 的 iteration-scale 与 training-scale 调度分开。
   这比固定 `p=0.4` 的 scheduled sampling 更适合做可证伪消融。

### 3.2 代码可用性和迁移风险

| 仓库 | 可用性 | 对当前路线的风险 |
|---|---|---|
| RaLD | 完整 AE/generation 训练与 checkpoint | ColoRadar、intensity-only；官方 checkpoint 不能作为 K-Radar 公平主表 |
| RadarMP | 训练、评估、tesseract 转换与 flow label 生成齐全 | 依赖 PWC/VoxFormer 自定义算子；官方 range 截断不可迁移 |
| RaFlow/CMFlow | 训练与 VoD 模型齐全 | 输入是稀疏 radar points，不是 Full-RAED Cube；LiDAR/camera supervision 改变任务 |
| RadarSFEMOS | 主模型、损失和多个 inherited baselines 齐全 | 工程树较重；scene-flow diffusion 不能等同于 point-generation diffusion |
| DoppDrive | 官方仓库当前只有 README | 只能按论文公式重实现，并做 sign/time/unit 测试 |
| EFEAR-4D | C++/ROS preprocessing 与 scan matching 可用 | 是 odometry 系统，不是生成器；适合做 ego/static control |
| RaLiFlow/IterFlow | 训练链已公开 | 引入 LiDAR 或 camera tracking，若进入主方法会削弱 radar-only 主张 |
| 4D-OCC | renderer、训练代码和 1 s/3 s checkpoints 齐全 | LiDAR ray model 不能直接替代 RAED spectrum rendering |
| OccWorld | tokenizer、temporal transformer、自回归代码齐全 | semantic occupancy token 与固定 10k radar state 不同 |
| UnO/DIO/Copilot4D | 论文/项目页完整，训练代码未公开 | 只能借抽象，不能声称 source-faithful reproduction |
| Radar-Mamba/DREAM-PCD/TARS | 本次未找到官方实现 | 只能做论文级结构对照，不做未经验证的代码复现声明 |

## 4. 最可迁移的四个机制

### M1. 运动分解的历史 proposal

**来源。** DoppDrive 的逐点径向 shift 和 adaptive history duration；
RaFlow/RadarSFEMOS 的 radial displacement；TARS 的 traffic-level tangential
motion；EFEAR-4D 的 ego/static filtering。

**定义。**

对历史点 `p_(t-k)` 先做 ego compensation，再分解为：

```text
p_t^proposal =
    T_(t-k -> t) p_(t-k)
  + dt * v_r * ray_direction
  + dt * delta_v_tangent
```

其中 `delta_v_tangent` 是有界、低容量、带置信度的 residual head。历史仅产生
proposal；最终 point budget、Doppler 分布和 confidence 仍由当前 Cube
rescore/生成。每个 proposal 保留 source age、warp component 和 uncertainty。

**为何可能有效。**

- ego-only 不能处理动态点；
- pure radial warp 会漏掉大切向运动；
- uncertainty 和 per-point age gate 可防止长历史产生拖影；
- history 可以补足 G1F 当前候选池中不存在的远距表面。

**与近邻的差异。**

- 不是 DoppDrive 式“把测量点直接聚合成最终点云”；
- 不是 TARS/RadarMP 式“scene flow 就是最终输出”；
- 所有历史点必须通过当前 Cube 的 measurement consistency 才能占用 10k
  输出预算。

#### M1 H200 Stage-0

| 项目 | 冻结设计 |
|---|---|
| 数据 | G1T 的 train/validation temporal manifest；test 不读取；四个 history frames |
| T0 | current-Cube proposals |
| T1 | current + ego warp |
| T2 | current + DoppDrive radial warp |
| T3 | T2 + bounded tangential residual + uncertainty/age gate |
| 训练 | one seed，10 epochs；只训练 residual/uncertainty head；LiDAR GT 不进入训练 |
| 损失 | current-Cube support、radial displacement、forward/backward cycle、residual magnitude prior |
| 预算 | 单张 H200，预计 4--6 GPU-hours，峰值显存目标 `<24 GB` |

**先验 oracle 门。** 在训练 T3 前，先检查所有 current+history candidates 的
GT coverage oracle。若 60--120 m、2 m recall 相对 T0 不能提高至少 `20%`
relative，M1 直接淘汰，不训练 residual head。

**晋级门。**

- T3 相对 `min(T0,T1,T2)` 的 far completeness 改善至少 `10%`；
- 60--120 m、2 m GT recall 相对最佳 control 提高至少 `20% relative`；
- overall completeness 改善至少 `10%`；
- outlier 和 duplicate 均不得恶化超过 `2 percentage points`；
- dynamic-slice radial displacement error 不高于 T2；
- history supplied fraction 不得通过只保留近距/静态点伪造收益。

任一硬门失败即关闭 M1 learned residual。T2 若通过而 T3 不通过，则仅保留
DoppDrive-style proposal control。

### M2. Radar-observable occupancy-flow field

**来源。** 4D-OCC/UnO 的 continuous occupancy 与 sensor rendering；DIO/DFIT
类方法的 static/dynamic flow decomposition；RaLD 的 global latent condition。

**定义。**

学习连续场：

```text
F(r, a, e, t) -> {
  occupancy_logit,
  circular_doppler_distribution,
  confidence,
  residual_3d_flow
}
```

将静态部分由 ego pose 解析 warp，动态部分由 flow 演化。通过一个
radar-specific differentiable renderer 把场投影回当前/未来 RAED spectrum，
再按射线和 range band 做质量守恒采样，输出固定 10k points。

关键不是“使用 occupancy”，而是：

1. renderer 必须重建 RAED energy 和 Doppler spectrum，而非 LiDAR depth；
2. circular Doppler distribution 是生成状态，不是输入 scalar 的复制；
3. current Cube 必须能够否决历史 hallucination；
4. fixed-count decoder 必须显示 near/far mass accounting，抑制 duplicate。

#### M2 H200 Stage-0

| 项目 | 冻结设计 |
|---|---|
| 表示 | `1/4 R x 1/4 A x 1/2 E` coarse field；time `{t-4,...,t,t+1}` |
| 数据 | 既有 temporal train/validation split；test 锁定 |
| O0 | G1G single-frame hierarchical allocator |
| O1 | history concat + direct occupancy，无 flow decomposition |
| O2 | static pose warp + dynamic residual occupancy-flow |
| O3 | O2 + differentiable RAED renderer + ray-balanced 10k export |
| 训练 | one seed，15 epochs，每 5 epoch evaluation |
| 预算 | 单张 H200，预计 12--16 GPU-hours，峰值显存目标 `<60 GB` |

**晋级门。**

- condition-shuffle geometry delta `>=1%`；
- median completeness `<=2.5068 m`；
- mean outlier `<=25%`；
- mean far completeness `<=8.1239 m`；
- duplicate `<=15%`；
- held-out current-Cube spectrum NLL 相对 O2 改善 `>=10%`；
- circular Doppler NLL 或 displacement-Doppler error 相对 O2 改善 `>=15%`；
- cross-scene wrong history 下，current-Cube reprojection 退化不超过 `5%`。

若只改善 LiDAR geometry 而不改善 Cube/Doppler 闭环，M2 淘汰为“LiDAR
occupancy world model transplant”，不得晋级。

### M3. Masked-current Cube 时序瓶颈

**来源。** RaLD 的 latent-only conditional decoding；Copilot4D/OccWorld 的
token prediction；masked modeling；当前 G1D 的 direct local-query bypass
反例。

**定义。**

在 geometry allocator 之前增加 pretext：

1. 随机遮蔽当前 Cube 的局部 RAED bins；
2. global current tokens + history tokens 预测被遮蔽的 energy/Doppler
   spectrum；
3. cross-scene current/history pair 作为 hard negative；
4. allocator 先读 global predicted field，再允许 bounded local refinement。

这不是简单 history concatenation。其目标是让 condition 对输出成为可测的
必要变量，同时保持当前 Cube 对历史的否决权。

#### M3 H200 Stage-0

| 项目 | 冻结设计 |
|---|---|
| parent | G1G condition-exclusive hierarchy；不得回到 G1D direct query path |
| C0 | 原 G1G |
| C1 | mask `{50%,70%,90%}`，仅 RAED reconstruction |
| C2 | C1 + cross-scene contrastive pairing |
| C3 | C2 + current-Cube override consistency |
| 训练 | one seed，20 epochs，与 G1G 相同数据/optimizer/geometry losses |
| 预算 | 单张 H200，预计 8--10 GPU-hours，峰值显存目标 `<45 GB` |

**晋级门。**

- exact cross-scene condition-shuffle geometry delta `>=1%`；
- zero-history 和 correct-history 必须有可测差异，但 cross-scene wrong history
  不得改善 geometry；
- masked RAED reconstruction NLL 相对 energy-only constant predictor 改善
  `>=20%`；
- 继续满足 G1G 的 completeness/outlier/far/duplicate 五门；
- wrong-history current-Cube reprojection 退化不超过 `5%`。

若 condition shuffle 仍 `<1%`，或者模型依赖历史到无法被当前 Cube 纠正，
M3 淘汰。单独提高 auxiliary reconstruction 但不改变 geometry condition
sensitivity 也视为失败。

### M4. 直接多时距预测加 rollout curriculum

**来源。** PRBonn TCNet 和 4D-OCC 的 direct multi-horizon output；
UnO/DIO 的 continuous time；OccWorld 的 autoregressive token path；
Flipped Classroom 的 iteration/training-scale curriculum。

**定义。**

不从纯 one-step teacher forcing 开始。共享时空 latent 一次输出
`{t+1,t+2,t+4}` 三个 anchor horizons；仅在 anchor 预测通过后，用上一步
generated state 做轻量 refinement。训练时比较：

```text
teacher forcing only
fixed scheduled sampling
direct multi-horizon
direct multi-horizon + increasing free-running curriculum
```

这样可把“表示是否能预测未来”和“自回归是否积累误差”分开，不会把长期失败
全部归咎于 scheduled sampling。

#### M4 H200 Stage-0

| 项目 | 冻结设计 |
|---|---|
| 前置门 | 只有通过完整 geometry gate 的 parent 才允许启动 |
| P0 | one-step teacher-forced autoregressive |
| P1 | fixed `p=0.4` scheduled sampling |
| P2 | direct `{1,2,4}` horizon heads，无 autoregressive feedback |
| P3 | P2 + iteration-scale increasing free-running curriculum |
| 训练 | one seed，10 epochs；同一 parent、同一 point budget、同一 temporal split |
| 预算 | 单张 H200，预计 10--14 GPU-hours，峰值显存目标 `<60 GB` |

**晋级门。**

- `t+1` Chamfer 不得比最佳 single-step parent 恶化超过 `5%`；
- `t+4` Chamfer 相对 P0 改善至少 `15%`；
- 从 `t+1` 到 `t+4` 的 Chamfer 增长斜率相对 P0 降低至少 `30%`；
- 每个 horizon 的 displacement-Doppler error 相对 P0 不得恶化；
- condition-shuffle delta 在所有 horizons 均 `>=1%`；
- duplicate/outlier 在任一 horizon 不得突破几何硬门。

P1 若仅靠 scheduled sampling 改善而 P2/P3 不通过，只能作为工程训练结果，
不能形成论文创新。P2 若胜出，则主叙事应是“direct radar-state forecasting”，
而不是“scheduled sampling 修复”。

## 5. Novelty claim 重新划界

### 5.1 现在可以安全使用的定位

> We study history-aware generation of a radar-observable state from Full-RAED
> measurements: a fixed-count dense 3D set, a calibrated per-point circular
> Doppler distribution, and confidence. Historical observations provide motion-
> compensated proposals or a temporal prior, while the current Cube remains the
> authority through Cube reprojection and displacement-Doppler consistency.

中文：

> 我们研究由 Full-RAED 和历史观测共同条件化的雷达可观测状态生成：固定数量
> 的稠密几何、逐点圆周 Doppler 分布与置信度联合输出；历史只提供运动补偿
> proposal 或时序先验，当前 Cube 通过重投影和位移-Doppler 一致性保持最终
> 测量约束。

这一定位与以下工作区分明确：

| 近邻 | 已覆盖 | Full-RAED 必须额外证明的差异 |
|---|---|---|
| Radar-Mamba | 三帧特征融合、当前点云增强、Doppler input feature | 逐点 circular Doppler distribution/confidence、current-Cube closure、fixed-count state |
| RadarMP | 两帧 tesseract、点生成 + pointwise scene flow、Doppler-guided losses | future/current radar observation generation、Doppler distribution calibration、current measurement override |
| DREAM-PCD | causal multiframe accumulation/reconstruction | automotive Full-RAED state、逐点 Doppler output、fixed-count and cycle |
| DoppDrive | Doppler-warped measured-point aggregation | 生成新点、当前 Cube rescore、uncertainty、distributional Doppler |
| RaLD | single-frame spectrum-conditioned latent diffusion to 10k XYZ | temporal prior、per-point Doppler/confidence、Cube and displacement cycle |
| 4D-OCC/UnO/DIO | future occupancy/flow and sensor rendering | radar RAED renderer、Doppler observability、fixed 10k radar state |

### 5.2 必须等待实验才能使用的主张

以下主张当前均为 **locked**：

- “Full-RAED history improves fixed-count geometry”需要 G1T/M1/M2 通过；
- “生成的 Doppler 与跨帧位移一致”需要 G2/G3 与 multi-horizon gate；
- “能预测未来 radar state”需要 M4 至少一个 direct horizon arm 通过；
- “current Cube can reject stale history”需要 wrong-history override test；
- “比 RadarMP 更完整的 radar state”需要逐点 Doppler distribution、
  confidence、Cube NLL 和 temporal consistency 都有定量结果。

### 5.3 禁止使用的 broad claims

- “现有雷达增强都是单帧”；
- “没有工作利用前一帧雷达生成/增强当前点云”；
- “首次利用 Doppler 做跨帧一致性”；
- “首次从两帧雷达估计 scene flow”；
- “首次用 occupancy 做未来点云预测”；
- “scheduled sampling 是本方法的创新”；
- “把速度积分成位移本身是创新”；
- “RaLD checkpoint 是 K-Radar 的公平 matched baseline”。

“首次 Full-RAED future radar-observable state generation”也不能在当前
检索后直接宣称。投稿前仍需重扫 2026 年 concurrent work，并以“within the
reviewed scope”限定。

## 6. 会导致同质化的路线

| 路线 | 会同质化为 | 原因 | 处理 |
|---|---|---|---|
| 对 current + previous two frames 加 Mamba/SSM 后输出当前 occupancy | Radar-Mamba | 输入、时序融合方式和目标都高度重合 | 只能作为 baseline，不作为主方法 |
| 两帧 tesseract 直接联合 segmentation + 3D flow | RadarMP | task/interface/loss 已接近重合 | flow 仅作 proposal teacher 或 control |
| 历史点 ego/Doppler warp 后直接作为最终稠密点云 | DoppDrive/DREAM-PCD | 本质是 aggregation/reconstruction，不是状态生成 | 强制 current-Cube rescore 和 generated-state output |
| 单帧 latent diffusion 只输出 10k XYZ | RaLD | 缺少 Doppler 和 temporal closure | 保留为相关工作/结构 parent |
| 把 LiDAR 4D occupancy 模型直接换成 radar input | 4D-OCC/UnO/DIO | 仍在预测几何世界，不建模 radar observability | 必须加入 RAED renderer 和 circular Doppler |
| scalar Doppler L1 + Chamfer | RaFlow/RadarMP | 已是标准 radial consistency | 主方法必须是 distribution calibration + Cube cycle |
| 依赖 LiDAR pseudo-flow 或 camera tracker 形成主训练信号 | CMFlow/RaLiFlow/IterFlow | 变成 cross-modal scene-flow method | 只作 train-time diagnostic/upper bound |
| 固定概率 scheduled sampling | 通用序列训练技巧 | 不改变任务和物理状态定义 | 仅作 P1 对照 |
| 继续增大 G1D direct-query Transformer | 当前失败 family | 不解决 condition bypass 或 proposal support | 不再作为时序路线 |

## 7. 推荐执行顺序

1. **先完成 G1T T0/T1/T2**。没有 history support 证据时不训练任何 temporal
   generator。
2. **并行完成 G1G 正式 Stage-0**。它先回答 condition-exclusive hierarchy
   是否能解决单帧 condition bypass。
3. 若 G1T history union oracle 通过，启动 **M1**；若不通过，关闭 point-level
   temporal proposal，直接评估 **M2** 的 field-level support。
4. 若 G1G condition shuffle 仍失败，启动 **M3**；若 G1G 已通过，M3 降为
   auxiliary ablation，不抢主线。
5. 只有 geometry parent、Doppler head 和 Cube cycle 均通过后，才启动
   **M4**。先 direct multi-horizon，后 rollout curriculum。
6. 所有 Stage-0 只用 H200 GPU 0/2、单 seed、validation；test 在 family
   冻结和三种子通过前保持锁定。

## 8. 审计结论

广泛检索后的研究判断是：

- **雷达多帧增强、Doppler warp 和 radar scene flow 都不是空白。**
- **真正尚可形成差异的是 radar-observable state interface 和闭环**：
  Full-RAED -> fixed-count geometry + circular Doppler distribution +
  confidence -> RAED reprojection，并把历史限制为可被当前测量否决的 proposal。
- 对当前长期效果最不应该做的是继续堆 temporal Transformer 或把 scheduled
  sampling 当万能修复。
- 最先值得计算的是 G1T support oracle 和 G1G condition-exclusive gate；
  它们分别回答“历史有没有新 support”和“global condition 能否真正被使用”。

本报告仅完成文献和源码只读审计，未运行任何科学计算。
