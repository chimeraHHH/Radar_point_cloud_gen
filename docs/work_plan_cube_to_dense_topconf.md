# 顶会工作计划：从 4D Radar Cube 生成物理一致的稠密 4D 点云

> 版本：2026-07-16  
> 目标：CVPR / ICCV / NeurIPS 等视觉与机器学习顶会  
> 论文主线：**当前帧 4D Radar Cube -> 稠密 `XYZ + Doppler` 点云**  
> 第二阶段扩展：历史点云经门控 Doppler-warp 后，作为当前 Cube 生成的时序先验  
> 关联材料：[proposal.md](proposal.md) · [draft_method.md](../paper/draft_method.md) · [技术路线图](assets/cube_to_dense_technical_roadmap.png)

![4D Radar Cube 到物理一致稠密点云技术路线](assets/cube_to_dense_technical_roadmap.png)

---

## 1. 论文目标与核心判断

### 1.1 研究问题

给定当前帧完整 4D Radar Cube

```text
C_t ∈ R^(R × A × E × V)
```

其中四个坐标分别表示 range、azimuth、elevation 和 Doppler，生成雷达可观测的稠密点云

```text
P_t = {(x_i, y_i, z_i, v_r,i, c_i)}_(i=1)^N,
```

其中 `v_r,i` 为逐点径向速度，`c_i` 为雷达可见性或预测置信度。输出既要具有接近同步 LiDAR 的空间完整性，又要与输入 Cube 的速度谱、自车运动和跨帧径向位移保持物理一致。

### 1.2 核心论文命题

> 稠密雷达点云生成不能只恢复空间几何。完整 4D Radar Cube 中的 Doppler 频谱提供了与运动直接相关的观测，生成点的位置、速度和可见性应在统一模型中联合推断，并通过 Cube-to-point 与 point-to-Cube 的双向一致性进行约束。

### 1.3 最可防守的创新边界

1. **完整 RAED Cube 条件生成**：保留 Doppler 轴，不将输入提前压缩为普通 RAE 强度张量。
2. **稠密 `XYZ + Doppler + confidence` 联合输出**：在恢复高质量几何的同时，为每个生成点估计速度分布及可靠性。
3. **Cube-point 双向物理闭环**：由 Cube 生成点云，再将生成点可微重投影回 Cube，约束空间位置和 Doppler 频谱共同自洽。
4. **解析静态项与学习动态项分解**：静态背景 Doppler 由 ego motion 解析计算，动态残差由 Cube 频谱学习，避免把所有速度作为自由属性回归。
5. **当前观测主导的时序生成**：历史点云仅提供 Doppler-warp 先验，当前 Cube 负责补点、纠错和刷新 Doppler，区别于简单历史点聚合。

### 1.4 不应作为核心创新的表述

- “首次从雷达生成稠密点云”：已有相关生成方法。
- “首次使用 Doppler”：检测、场景流和时序聚合工作已经使用 Doppler。
- “给点云增加一个速度通道”：若没有频谱监督和双向闭环，只是增量式结构修改。
- “历史点云经 Doppler 补偿后聚合”：DoppDrive 已覆盖这一任务形态。

---

## 2. 与现有工作的关系

| 方法类别 | 输入 | 输出 | 已有能力 | 本工作的新增部分 |
|---|---|---|---|---|
| RaLD 类方法 | 单帧 RAE radar spectrum | 稠密 `XYZ` | 高质量空间生成 | 保留 Doppler 轴，联合输出速度，加入 Cube-point 闭环 |
| DoppDrive 类方法 | 多帧稀疏 `XYZ+Doppler` | 移动和筛选后的聚合点 | Doppler 驱动时序增密 | 由当前 Cube 生成新点并刷新速度，而非只复用历史点 |
| 当前仓库单帧线 | LiDAR | 384 点 `XYZ+Doppler+RCS` | 点生成、ego 条件、物理损失 | 输入方向和稠密目标需要重做 |
| 当前仓库时序线 | 上一帧稀疏雷达点 | 下一帧稀疏雷达点 | Doppler-warp、桥式生成、scheduled sampling | 作为第二阶段时序先验复用 |

---

## 3. 方法设计

### 3.1 模块 A：4D Cube 数据表征与编码

输入保持完整 `R × A × E × V` 结构。首轮至少实现三种编码对照：

- `RAE-Max`：沿 Doppler 轴取最大值，作为不保留速度谱的基线。
- `RAE-Moments`：保留强度、速度均值和速度方差等低阶矩。
- `Full-RAED`：显式编码完整 Doppler 频谱，作为主模型。

Cube Encoder 输出空间对齐的多尺度特征 `F_t`。若显存不可接受，按优先级尝试 Doppler 低秩分解、稀疏峰值 token、局部窗口注意力，不在第一版直接使用全局四维注意力。

### 3.2 模块 B：稠密几何生成

采用 RaLD 风格的点潜空间或极坐标 frustum occupancy 表征，先建立稳定的 `Cube -> XYZ` 基线：

```text
F_t -> latent diffusion / flow -> occupancy decoder -> {p_i}
```

目标点应定义为 **radar-observable dense points**，而不是无条件复制全部 LiDAR 表面。同步 LiDAR 提供几何监督，Cube 能量、视场、遮挡和距离共同构造可见性 mask。

主要损失：

```text
L_geo = λ_cd L_CD + λ_occ L_occupancy + λ_conf L_confidence
```

### 3.3 模块 C：逐点 Doppler 分布生成

将生成位置 `p_i` 投影到雷达极坐标 `π(p_i)`，在对应 Cube 空间邻域查询 Doppler 频谱：

```text
q_i(v) = Softmax(H(F_t, π(p_i)))
v_hat_i = Σ_v v · q_i(v)
```

模型同时预测 `confidence_i`，对多峰、弱反射和不可观测位置显式表达不确定性。优先使用速度分布 NLL 或交叉熵，而不是只做标量 L1 回归。

### 3.4 模块 D：解析静态项与动态残差

静态背景满足：

```text
v_r^static(p) = -v_plat(p) · r_hat
v_plat(p) = v_ego + ω × (p + t_s)
```

最终速度分解为：

```text
v_r = v_r^static + v_r^dynamic
```

静态点由解析项约束，动态残差由 Doppler 频谱、目标运动和上下文学习。继续复用当前仓库的自门控静态损失、动态软约束和反事实 ego-speed 测试，但必须增加“全静态塌缩”监控。

### 3.5 模块 E：可微 point-to-Cube 重投影

构造可微渲染器：

```text
C_hat_t = R({p_i, v_r,i, confidence_i})
L_cycle = D(C_hat_t, C_t)
```

渲染器至少在 `range-azimuth-elevation-Doppler` 网格上进行软 splatting，并使用预测 confidence 作为能量权重。第一版只要求重建归一化局部频谱或稀疏峰值分布，不强行恢复原始复数 IQ 信号。

该模块是论文最关键的区别点，必须通过消融证明它同时改善以下至少两项：

- Doppler 频谱匹配；
- 逐点速度物理一致性；
- 几何位置准确性；
- 下游检测或速度估计。

### 3.6 模块 F：时序扩展

单帧主干稳定后，再加入：

```text
P_(t-1)
  -> 门控 Doppler-warp
  -> 当前帧几何先验
  -> 与 C_t 特征融合
  -> 当前帧稠密 P_t
```

跨帧约束：

```text
L_temp = |Δrange - v_bar_r Δt|
```

保留单帧 Cube 模型、ego-only warp、DoppDrive 式聚合和当前仓库 `copy_dopp` 作为对照。历史帧不能替代当前 Cube，避免任务退化为确定性点云聚合。

### 3.7 总目标

```text
L = L_geo
  + λ_spec L_doppler-spectrum
  + λ_static L_static
  + λ_dyn L_dynamic
  + λ_cycle L_cube-cycle
  + λ_temp L_temporal
```

采用分阶段训练，禁止从第一天同时打开全部损失。每新增一个模块都必须在固定基线上做独立消融。

---

## 4. 数据与任务协议

### 4.1 必需数据字段

- 同步或可精确配准的 4D Radar Cube 与 LiDAR；
- 雷达内外参、时间戳、ego pose、线速度和角速度；
- Doppler bin 到 m/s 的标定关系及符号口径；
- 可选：3D 框、类别、目标速度、跟踪 ID；
- 可选：已有 CFAR 点云，用于验证 Cube 读取和 Doppler 映射。

### 4.2 数据审计必须回答的问题

1. Cube 的真实维度顺序、数值类型和物理单位是什么？
2. Doppler 是否存在混叠、静态杂波抑制或预补偿？
3. Cube 与 LiDAR 的时间差和空间标定误差是多少？
4. 一个 LiDAR 点在 Cube 中是否有稳定可查询的局部频谱？
5. 哪些 LiDAR 表面属于雷达可观测区域，如何构造置信 mask？
6. 数据划分能否按场景隔离，避免相邻帧泄漏？

### 4.3 输出规模

首轮建议固定 `N=10,000` 便于与 RaLD 类方法对齐；后续比较 `N=5k/10k/20k`。同时报告真实有效点数和 confidence 校准，避免仅靠增加低质量点改善覆盖率。

---

## 5. 实验矩阵

### 5.1 主实验组

| ID | 方法 | 研究问题 | 论文位置 |
|---|---|---|---|
| E0 | CFAR / 原始稀疏点 | 原始传感器下限 | 主表 |
| E1 | RAE-Max -> dense XYZ | 不使用 Doppler 轴的几何基线 | 主表 |
| E2 | Full-RAED -> dense XYZ | Doppler 频谱是否帮助几何 | 主表、消融 |
| E3 | E2 + Doppler scalar head | 简单增加速度头的效果 | 消融 |
| E4 | E2 + Doppler distribution head | 分布预测是否优于标量回归 | 主表、消融 |
| E5 | E4 + static/dynamic physics | 物理分解是否减少不一致 | 主表 |
| E6 | E5 + Cube cycle | 双向闭环是否带来核心增益 | 核心消融 |
| E7 | E6 + temporal prior | 历史先验是否提升稳定性 | 时序表 |

### 5.2 必须包含的基线

- CFAR 或数据集官方点云提取结果；
- RaLD 风格 `radar spectrum -> dense XYZ` 重实现或官方实现；
- SDDiff / RPDNet 等可获得的雷达稠密化基线；
- 标量 Doppler 回归基线；
- DoppDrive 式多帧聚合；
- ego-only aggregation 与当前仓库的 gated Doppler-warp；
- Oracle：使用 GT 目标速度或 GT 运动 mask，测量物理模块上限。

### 5.3 指标

**几何质量**

- Chamfer Distance、EMD、F-score；
- precision、recall、completeness；
- 按距离区间统计的完整度和定位误差；
- 有效点数、重复点率、离群点率。

**Doppler 与物理一致性**

- PCE、CD-Doppler、W1(`v_r`)；
- Cube Doppler spectrum NLL / KL；
- 静态和动态子集分别统计速度误差；
- confidence calibration：ECE、NLL 或 reliability curve；
- 反事实 ego-speed 剂量响应。

**时序质量**

- `|Δrange - v_bar_r Δt|`；
- 多步 rollout CD 与 PCE；
- 点云 flicker、轨迹连续性、速度刷新率；
- 与单帧、ego-only、DoppDrive 聚合比较。

**下游价值**

- 3D object detection；
- 径向速度或目标速度估计；
- localization / mapping，若数据和时间允许。

### 5.4 统计协议

- 按场景划分 train/val/test；
- 至少 3 个随机种子用于主模型与关键消融；
- 报告均值、标准差和配对显著性检验；
- 所有阈值在验证集确定，测试集只评估一次；
- 每张主表同时报告几何和 Doppler，禁止只展示单一有利指标。

---

## 6. 分阶段执行计划

### P0：任务与数据可行性审计（W1-W2）

**任务**

- 完成 Cube schema、标定、时间同步和 Doppler 口径核查；
- 建立 `Cube -> CFAR point` 可视化与数值自检；
- 将 CFAR 点投回 Cube，验证空间 bin 和 Doppler bin 对齐；
- 定义 radar-observable LiDAR target 与 confidence mask；
- 固化场景级数据划分。

**产出**

- Cube loader、字段报告、同步误差报告；
- 100-500 帧可视化审计集；
- 数据协议文档和最小缓存格式。

**G0**

- Cube、LiDAR、ego motion 能稳定同步；
- CFAR 点回查 Cube 后空间与 Doppler 对应关系正确；
- radar-observable target 可以稳定构造。

若 G0 不通过，不进入大模型训练。优先修数据口径；若数据本身缺少完整 Doppler 轴或同步 LiDAR，则必须缩小论文命题。

### P1：单帧稠密几何基线（W3-W5）

**任务**

- 实现 `RAE-Max -> dense XYZ`；
- 实现 Full-RAED Cube Encoder；
- 建立 frustum occupancy 或 point-latent decoder；
- 对齐 CFAR、RaLD 风格基线和 LiDAR GT；
- 完成距离分层几何评估。

**产出**

- 可复现 Cube-to-XYZ 基线；
- 主表中的 E0-E2；
- 第一版定性结果和失败案例。

**G1**

- 稠密输出在 CD/F-score/completeness 中稳定优于 CFAR；
- Full-RAED 至少在一个几何指标或远距离子集上优于 RAE-Max；
- 输出不是通过大量重复点或离群点获得虚假覆盖率。

### P2：Doppler 联合生成（W6-W8）

**任务**

- 实现位置条件 Doppler spectrum query；
- 完成 scalar head 与 distribution head 对照；
- 复用 ego-conditioned static loss；
- 加入动态残差和 confidence 校准；
- 运行速度反事实实验。

**产出**

- E3-E5 消融；
- Doppler 主表、可靠性曲线、静动态分解图。

**G2**

- distribution head 稳定优于 scalar head；
- 物理约束改善 PCE 和频谱指标，几何 CD 相对退化不超过预注册容忍区间；
- 动态速度分布不发生“全静态”塌缩。

### P3：Cube-point 双向闭环（W9-W11）

**任务**

- 实现可微 RAED soft splatting renderer；
- 设计局部峰值、边际分布和完整 Cube 三种 cycle loss；
- 检查模型是否通过降低 confidence 逃避重建；
- 做频谱噪声、Doppler 混叠和标定偏差鲁棒性实验。

**产出**

- E6 核心消融；
- 生成点与输入 Cube 对应关系可视化；
- 核心方法图和机制分析图。

**G3：论文关键门**

Cube cycle 必须在 Doppler 频谱匹配、PCE、几何或下游任务中至少改善两类指标，并且不能通过置信度坍缩获得。若未达到，论文退化为“RaLD + Doppler head”，不具备足够强的顶会差异化，应停止扩规模并重新设计闭环。

### P4：时序先验扩展（W12-W14）

**任务**

- 将当前仓库 gated Doppler-warp 接入 Cube 模型；
- 比较 concat、cross-attention、draft refinement 三种融合方式；
- 复用 scheduled sampling 处理 rollout 分布偏移；
- 与 DoppDrive、ego-only 和单帧 Cube 模型比较。

**产出**

- E7 时序结果；
- 长序列可视化；
- 时序稳定性和失败边界分析。

**G4**

时序模型必须同时优于单帧模型的跨帧一致性和 DoppDrive 的几何/下游表现。若只改善平滑度而损伤当前帧准确性，则将时序模块降为 appendix，不影响单帧主线投稿。

### P5：规模化、下游与泛化（W15-W17）

**任务**

- 全量训练和至少 3 个随机种子；
- 跨天气、距离、速度、目标类别和场景分析；
- 检测、速度估计和可选 mapping；
- 运行效率、显存和推理速度统计；
- 完成失败案例分类。

**产出**

- 最终主表、鲁棒性表、效率表；
- 定性主图和补充视频素材；
- 完整 checkpoint 与评估脚本。

### P6：论文整合与内部评审（W18-W20）

**任务**

- 先完成 method、experiments、analysis，再写 introduction 和 abstract；
- 完成至少两轮独立内部审稿；
- 将主要 reviewer objection 映射到实验或限制；
- 主文控制在一个核心命题，时序结果不抢占 Cube-point 闭环主线；
- 整理代码、配置、数据协议和 supplementary。

**冻结标准**

- 每个贡献点至少有一个主表或主图支撑；
- 主结果包含强基线、完整消融和统计波动；
- 所有“首次”表述完成投稿前文献复扫；
- 没有尚未解决的数据泄漏、单位口径或评估不公平问题。

---

## 7. 当前仓库的复用与重构

### 7.1 直接复用

- `losses/physics.py`：静态自门控、动态一致性；
- `eval/physics.py`：PCE；
- `eval/gen_metrics.py`：CD-Doppler、MMD、JSD，可扩展到 dense protocol；
- `temporal_pairs.py`：门控 Doppler-warp 逻辑；
- `train_bridge_ss.py`：scheduled sampling 思路；
- 反事实 ego-speed 评估和 rollout 评估框架。

### 7.2 需要替换

- PCD-only loader -> 4D Cube loader；
- LiDAR condition encoder -> Cube Encoder；
- 固定 384 点稀疏目标 -> 5k-20k radar-observable dense target；
- LiDAR-to-radar 单帧任务 -> Cube-to-dense-4D 任务；
- 稀疏 radar-vs-radar 指标 -> dense geometry + spectral Doppler 协议。

### 7.3 需要新增

- Cube/LiDAR 配准与 observability mask；
- Full-RAED 多尺度编码器；
- 稠密点或 frustum occupancy 解码器；
- Doppler distribution head；
- differentiable point-to-Cube renderer；
- confidence calibration 与 spectrum-level metrics。

---

## 8. 顶会审稿风险与应对

| 审稿质疑 | 风险 | 必须准备的证据 |
|---|---|---|
| “只是 RaLD 加速度头” | 高 | Full-RAED vs RAE、scalar vs distribution、Cube cycle 核心消融 |
| “只是 DoppDrive 加生成器” | 高 | 当前 Cube 条件、生成新点、速度刷新和优于聚合的下游结果 |
| “LiDAR 点没有真实 Doppler” | 高 | radar-observable target、Cube 频谱监督、confidence 和可见性分析 |
| “物理约束让模型全部预测静态” | 高 | 静动态分开指标、速度分布、动态残差和反事实实验 |
| “Cycle loss 可以靠低 confidence 作弊” | 中 | confidence 正则、覆盖率约束、校准曲线和可视化 |
| “完整 Cube 太耗算力” | 中 | 参数量、FLOPs、显存、速度及低秩/稀疏编码消融 |
| “只在单一数据集有效” | 中 | 跨场景/天气/距离分析；有条件时增加第二数据集 |
| “几何更密但对任务无用” | 高 | 检测、速度估计或 mapping 至少一项稳定增益 |

---

## 9. 论文展示计划

### 主图

1. **Figure 1**：4D Cube -> dense XYZ+Doppler 总览与 Cube-point 双向闭环。
2. **Figure 2**：生成位置查询 Doppler 频谱和 point-to-Cube renderer。
3. **Figure 3**：几何、Doppler、confidence 的定性结果与失败案例。
4. **Figure 4**：静态解析、动态残差、时序一致性的机制分析。

### 主表

1. **Table 1**：与 CFAR、RaLD 类、SDDiff 类方法的几何和 Doppler 主结果。
2. **Table 2**：Full-RAED、Doppler head、physics、cycle 的完整消融。
3. **Table 3**：时序扩展与 DoppDrive/ego-only/single-frame 对比。
4. **Table 4**：下游任务、效率和泛化。

### Appendix

- 数据标定与 observability target；
- 网络结构、训练超参数和完整指标定义；
- 更多距离/类别/天气切片；
- Doppler 混叠、噪声与 confidence 分析；
- 失败案例和伦理/数据限制。

---

## 10. 最小可发表路径与止损规则

### 最小顶会主线

```text
Full-RAED Cube Encoder
  + dense XYZ generator
  + Doppler distribution head
  + static/dynamic physical decomposition
  + differentiable Cube-point cycle
```

时序模块不是最小主线的必要条件。若单帧闭环足够强，可以独立形成完整投稿；时序部分只在提供额外稳定收益时进入主文。

### 止损规则

- **G0 失败**：数据不支持完整 Cube、同步 LiDAR 或可靠 Doppler 标定，立即缩小命题，不投入大规模训练。
- **G1 失败**：Cube-to-XYZ 无法超过 CFAR 或合理生成基线，先解决几何，不进入 Doppler 联合训练。
- **G2 失败**：Doppler head 不优于简单回归，重新检查频谱查询和标签定义。
- **G3 失败**：Cube cycle 没有独立贡献，停止“顶会创新已成立”的表述，重设计闭环或转为应用型工作。
- **G4 失败**：时序模块降为 appendix，不拖累单帧主线。

---

## 11. 每周执行规范

- 每周只设一个主问题和一个可证伪假设；
- 每次训练绑定 config、commit、seed、数据 split 和指标输出；
- 新模块先在小规模数据上过拟合和单元验证，再进入全量；
- 每阶段结束更新 claim-evidence 表，禁止把未完成实验写成论文结论；
- 每两周重扫一次直接竞品和引用网络；
- 主结果图表使用固定 test split，测试集不参与调参；
- 任何异常优异的物理一致性结果都必须同时检查动态占比和速度方差，防止静态塌缩。

---

## 12. 下一步行动清单

- [ ] 获取并核对当前 4D Radar Cube 文件格式、维度、单位和 Doppler 口径。
- [ ] 确认同步 LiDAR、ego motion、标定和场景划分是否齐全。
- [ ] 实现最小 Cube loader 与单帧可视化。
- [ ] 验证 CFAR 点 `XYZ+Doppler` 能否准确回查到 Cube 峰值。
- [ ] 定义 radar-observable LiDAR target 和 confidence mask。
- [ ] 建立 `RAE-Max -> dense XYZ` 最小基线。
- [ ] 再实现 Full-RAED Encoder，正式启动 E1/E2 对照。

> 当前最高优先级不是继续优化旧时序模型，而是完成 **G0 数据闭环**，并打通第一个可复现的 `4D Cube -> dense XYZ` 基线。
