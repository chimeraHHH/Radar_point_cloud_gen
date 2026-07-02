# Proposal：物理一致的多普勒雷达点云生成

> 工作名：**DopplerConsist / FlowRadar-4D** · 版本 2026-07-02
> 依据调研：[related_work_doppler.md](related_work_doppler.md) · [prior_art_deep_dive.md](prior_art_deep_dive.md) · [survey_temporal_doppler.md](survey_temporal_doppler.md)

---

## 1. 背景与动机
4D 毫米波雷达在恶劣天气下鲁棒,且是**唯一逐点提供径向速度(Doppler)**的车载传感器,但点云稀疏、噪声大、标注昂贵——生成/增强雷达点云可缓解数据瓶颈。2025 年底,雷达生成开始输出 Doppler(RadarGen、4D-RaDiff、Song et al.、SDDiff),但**没有一篇把 Doppler 的物理定律作为对生成分布的显式约束**。

## 2. 精准研究空白
Doppler 不是可自由回归的属性通道,它由几何与运动**解析决定**:
```
v_r = (v_target − v_ego)·r̂ ,  r̂ = p/‖p‖
静态点:  v_r = −v_ego·r̂        (仅自车运动决定,可解析)
时序:    v_r·Δt ≈ Δrange        (帧间径向位移)
```
现状:RadarGen 用光流构造条件却**不约束输出**;4D-RaDiff 用框速度条件 + L2 回归;两者都让网络**隐式/经验式**学 Doppler,生成的 Doppler 与几何/运动**不保证自洽**,而静态背景这块本可解析的部分完全靠"猜"。4D-RaDiff 更自述"动态物体多帧拖尾/Doppler-运动耦合"未做。

## 3. 核心思想
把"**Doppler 必须与几何、自车运动、帧间位移严格自洽**"这条物理定律做成**可微一致性约束**,嵌入 LiDAR→Radar 扩散生成的损失与条件。一句话:**从"学习 Doppler"升级为"物理约束下生成 Doppler"。**

## 4. 方法:两条可叠加主线(共享同一 Doppler 物理内核)

**主线 A — 单帧物理一致性(Static-Analytic + Consistency Loss)**
- 骨架:LiDAR→Radar 条件扩散(对标 4D-RaDiff),每点输出 `(x,y,z,v_r,RCS)`。
- **静态解析硬约束**:对判为静态的点,`v_r` 由 `−v_ego·r̂` 直接给定/强监督,消除背景 Doppler 的猜测(现有全部缺失)。
- **动态一致性软损失**:前景点 `L_dop = |v_r_pred − (v_obj−v_ego)·r̂|`,`v_obj` 取自框/场景运动场。
- **反事实速度可控**:干预 `v_ego / v_obj` 应可预测地改变生成 `v_r`,借 MoCap-to-Radar(2605.00018)速度干预测试量化。

**主线 B — 时序 Doppler 驱动生成(FlowRadar-4D)**
- 任务升级为**多帧一致**雷达序列生成。
- **Doppler 驱动帧间 warp**:用实测/生成 `v_r` 沿径向把点推进到下一帧,替代 LiDAR 时序工作注入的模拟轨迹——雷达独有、物理更原理化。
- **双向一致性损失**(改造 RaFlow `L_rd`):`L_temp = |Δrange − v_r·Δt|`,让时序位移与 Doppler 互相监督。
- 借 LiDARCrafter 首帧 warp 抗漂移,保留"仅 ego-warp"稳健下限。

> A 可先行验证物理内核(单帧、门槛低);B 是更强、更空白的主线。二者可分别投稿,或合并为完整故事。

## 5. 数据与评估
- **数据**:单帧/配对 → **View-of-Delft**(raw+补偿 `v_r`,含 LiDAR);时序高帧率 → **MAN TruckScenes**(20Hz 逐点 Doppler+LiDAR)。
- **保真**:Chamfer / MMD / JSD;Doppler:CD_Doppler、MMD-Doppler。
- **新指标(贡献)**:多普勒物理一致性误差、速度场一致性 `‖v̂_r·Δt − Δrange‖`。
- **下游**:VoD 检测 mAP、速度估计增益(对标 4D-RaDiff 46.0→53.3 范式)。

## 6. 预期贡献
1. 首个把 Doppler 物理一致性作为**显式可微约束**的雷达生成框架。
2. 静态背景 Doppler 的**解析硬约束**(现有工作均缺)。
3. **Doppler 驱动的时序一致**多帧雷达生成 + 双向一致性损失。
4. 新的**物理一致性评估协议** + 反事实速度可控性。

## 7. 里程碑(粗排)
| 阶段 | 内容 | 产出 |
|------|------|------|
| P1 (2–3wk) | 复现 4D-RaDiff 类基线 + VoD 数据管线 | 可跑基线 + Doppler 指标 |
| P2 (3–4wk) | 主线 A:静态硬约束 + 动态一致性损失 | 消融显示一致性↑、mAP↑ |
| P3 (4–6wk) | 主线 B:TruckScenes 多帧 + Doppler-warp + 时序损失 | 时序一致性指标 + demo |
| P4 (2wk) | 反事实评估 + 论文 | 投稿稿 |

## 8. 风险与规避
- prior art 多为 **2025.12 未评审预印本**、领域推进快 → 投稿前重扫 RadarGen/4D-RaDiff 引用网络确认无同期竞品。
- 静/动分割误差污染约束 → 静态约束设**软加权**,并保留纯 ego-warp 稳健下限。
- VoD 雷达为多扫累积 → 时序实验以 **TruckScenes** 为主,VoD 主打单帧。
