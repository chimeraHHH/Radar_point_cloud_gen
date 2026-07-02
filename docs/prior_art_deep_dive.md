# Prior Art 精读：4D-RaDiff vs RadarGen —— 多普勒到底怎么处理、留了什么坑

> 对应 PDF：`docs/papers/2512.14235_4D-RaDiff.pdf`、`docs/papers/2512.17897_RadarGen.pdf`
> 这两篇是与本课题最接近、且都"生成多普勒"的工作（均为 2025-12 arXiv 预印本）。本文拆解它们的多普勒机制，定位可攻击的空白。

---

## 0. 一句话对比

| 维度 | **4D-RaDiff** | **RadarGen** |
|------|---------------|--------------|
| 任务 | 框/LiDAR → 雷达点云 | 多视相机 → 雷达点云(BEV) |
| 表示 | 点级 (x,y,z,补偿Doppler,RCS) **5维** | BEV图 + 点级 (x,y,RCS,Doppler) **无z** |
| 范式 | Point-VAE + 双隐扩散(前景/背景) | SANA 图像隐扩散 + DiT |
| 多普勒来源 | **框速度条件**(box velocity) | **光流→径向投影**构造的"径向速度条件图" |
| 是否用 v_r=v·r̂ | ❌ 否(框速度直接条件) | ⚠️ **构造条件输入时用了径向投影**，但仅作输入 |
| 多普勒物理损失 | ❌ 无(仅 L2 特征回归) | ❌ 无 |
| 多普勒指标 | **CD_Doppler** | **MMD-Doppler** + DA(|Δv|<2.5m/s) |
| 数据集 | View-of-Delft / TruckScenes | MAN TruckScenes |
| 关键增益 | VoD mAP 46.0→53.3 | Hit Rate 0.37→0.66 |

---

## 1. 4D-RaDiff（arXiv:2512.14235, TU Delft / Perciv AI）

### 表示与架构
- 点云 `x ∈ ℝ^(N×5)` = (x, y, z, 补偿Doppler, RCS)。原文："The input to the VAE is a radar point cloud, represented as a set of N points with xyz-coordinates, Doppler and RCS as its features."
- **Point-based VAE**（密度感知压缩）编码到隐空间 `z ∈ ℝ^(M×d_z)`，VAE 冻结后训两个隐扩散：
  - **前景 LDM**：交叉注意力条件于 3D 框 `b_i ∈ [0,1]^9`（位置/尺寸/yaw/**2D 速度**）
  - **背景 LDM**：PointPillars 编码的 **LiDAR** 条件
- 直接在点级隐空间扩散，不转图像。

### 多普勒怎么来的
- **靠框的 2D 速度条件**。原文明确："We incorporate additional features such as the yaw angle and bounding box velocities, because they provide crucial information for the synthesis of radar point clouds with accurate Doppler values."
- 用**补偿多普勒**（去掉自车运动影响的真实径向速度）。
- ⚠️ **背景的 LiDAR 条件不含任何速度信息** → 背景多普勒纯靠扩散先验"猜"。
- ❌ **无任何运动学/物理约束**——Doppler 只是扩散框架里的一个被学习的特征通道。

### 损失
```
L_VAE = L_rec + λ_den·L_den + λ_card·L_card + λ_reg·L_reg
```
- 坐标用 Chamfer；**Doppler/RCS 用最近邻匹配后的 L2 特征损失** `L_feat`（Eq.12）——纯回归，无物理项。
- 扩散用标准去噪 `L_LDM = E||ε - ε_θ(z_t,t,τ(y))||²`，**多普勒无单独加权**。

### 评估与结果
- 提出 **CD_Doppler**（生成点到最近真值点的 Doppler 绝对差均值）。
- 消融（Table 4）：VAE+LDM 把 CD_Doppler 从 4.80 降到 **1.24**，CD_RCS 25.04→6.75。
- 下游：VoD 上 CenterPoint real+synthetic 把 mAP 从 46.0 提到 **53.3**（cyclist 65.2→75.0，pedestrian 34.9→44.4）。

### 自述局限（=可攻击点）
1. 固定点数，密度与真实有差异。
2. **前景不建模动态物体跨多帧的运动拖尾**；"This could be addressed by also compensating the motion of dynamic objects based on Doppler information"——**他们自己说多普勒-运动耦合没做好**。
3. 背景依赖 LiDAR → 传感器依赖、实用受限。

---

## 2. RadarGen（arXiv:2512.17897, NVIDIA-affiliated）

### 表示与架构
- 每点 `(x_i, y_i, r_i, d_i)`：平面坐标 + RCS + "Doppler velocity, the measurement of relative radial velocity"。**注意是 BEV 平面，无 z**。
- 三张单通道 BEV 图：点密度图(高斯核卷积)、RCS 图、Doppler 图（后两者用 **Voronoi 镶嵌**，每像素取最近检测的属性值）。
- 骨干 **SANA**（32× 压缩 + 线性注意力的图像隐扩散），三张图 latent 拼成 token 序列**联合自注意力去噪**，learnable modality embedding 区分通道。

### 多普勒怎么来的 —— ⚠️ 这里和你的想法最接近
- 构造"**径向速度条件图**"：用帧间**光流** + 深度反投影得到 3D 速度 `v(x)≈[p^{t+Δt}−p^t]/Δt`，然后——
  > **"retain only its component along the radial direction from the ego-vehicle to obtain a Doppler-like value."**
  - 即 **`v_r = v · r̂`**。**RadarGen 已经用了径向投影这个物理关系！**
- **但关键区别**：这个投影**只用来构造扩散的条件输入 `c`**，模型学的是 `p(z_p,z_r,z_d | c)`——**投影没有作为对生成输出的约束/损失**。生成的 Doppler 与生成的几何之间**没有强制自洽**。
- ❌ 无显式 Doppler 物理损失。

### 评估与结果
- **MMD-Doppler**（多尺度 RBF）：entire area 0.65→**0.31**。
- **DA(Distance-Attribute)** 匹配三条件：空间<1m、|ΔRCS|<8dBsm、**|ΔDoppler|<2.5m/s**；F1/Recall/Precision。
- 前景 Hit Rate 0.37→**0.66**；下游 VoxelNeXt NDS：真值0.48、RadarGen 0.30、baseline≈0。
- 数据集 **MAN TruckScenes**（不是 VoD）。

### 自述局限
- 受上游基础模型限制（夜间/强反射/遮挡失效）。
- 相机不可见区域会**幻觉**生成点。
- 检测质量仍明显低于真实数据，"subtle differences"未深究。

---

## 3. 综合：它们留下的、可被你攻击的空白

把两篇放一起，多普勒处理的共同短板非常清晰：

| 空白点 | 4D-RaDiff | RadarGen | 你的机会 |
|--------|-----------|----------|----------|
| **生成输出的多普勒-几何自洽** | ❌ 仅 L2 回归 | ❌ 仅联合注意力隐式 | ✅ 加**可微物理一致性损失**强制 `v_r` 与生成几何+运动自洽 |
| **静态背景多普勒解析约束** `v_r=−v_ego·r̂` | ❌ 背景靠扩散猜 | ❌ 靠光流条件 | ✅ 静态点多普勒**可解析**，作硬约束/强先验 |
| **LiDAR→Radar 的多普勒来源** | ❌ 单帧LiDAR无速度,背景多普勒无依据 | N/A(相机) | ✅ 用**时序LiDAR场景流**导出多普勒先验+物理约束 |
| **动态物体运动-多普勒耦合** | ❌ 作者自述未做 | ⚠️ 光流粗略 | ✅ 显式建模 `v_r=(v_obj−v_ego)·r̂` |
| **物理一致性作为评估指标** | 只有 CD_Doppler(精度) | 只有 MMD/DA(分布) | ✅ 新增"**多普勒物理一致性误差**"(借 arXiv:2605.00018 速度干预测试) |

### 核心论点（写论文 intro 可用）
> RadarGen 在**构造条件输入**时用了径向投影 `v_r=v·r̂`，4D-RaDiff 用框速度条件——两者都**让网络隐式/经验式地学习多普勒**，但**都没有把"多普勒必须与点的几何位置和场景运动严格自洽"这一物理定律，作为对生成分布的显式可微约束**。尤其是**静态背景**这部分多普勒完全由自车运动解析决定（`v_r=−v_ego·r̂`），现有工作均未利用。本课题将这一物理一致性嵌入 LiDAR→Radar 扩散生成的损失与条件，作为核心贡献。

### 投稿前必做的尽职核查
- 精读两篇 PDF 全文，确认"无物理一致性损失"判断（本笔记基于 HTML 全文，但二者均为未评审预印本）。
- 扫 RadarGen / 4D-RaDiff 的 Semantic Scholar 引用网络，确认 2026 上半年无同期竞品已补上物理约束。
