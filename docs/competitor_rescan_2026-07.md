# 竞品重扫备忘（P0 · 立项核查）

> 日期：2026-07-02 · 对应 [work_plan.md](work_plan.md) P0 阶段与风险 R1/R2
> 方法：多角度 Web 检索（物理约束雷达生成 / Doppler 一致性损失 / 时序·多帧雷达生成 / 序列世界模型 / 基线代码可得性）+ 逐篇核验
> 目的：确认 [proposal.md](proposal.md) 两条主线的空白在 2026 上半年**仍成立**，并核查核心 prior art 代码可得性。

---

## 0. 结论（TL;DR）

- ✅ **两条主线空白仍成立**（截至 2026-07）。
- **主线 A（显式可微 Doppler 物理一致性损失，用于生成模型）**：仍无人做。**最近距离 = RadarMP**（感知侧，tesseract→点云+scene flow，含 **Doppler 引导的时序/运动一致性自监督损失**）——**须引用并明确区分「感知 vs 生成」**。它反而佐证了该损失机制有效，利于我们的定位。
- **主线 B（多帧一致雷达序列生成 + Doppler 驱动一致性）**：仍无人做。所有序列/世界模型仍是 **LiDAR**（LiSTAR、LiDARCrafter、Copilot4D、"Learning to Generate 4D LiDAR Sequences" 2509.11959）；雷达生成器（RadarGen、4D-RaDiff）仍单帧。
- ⚠️ **代码可得性（确认风险 R1）**：4D-RaDiff ❌、RadarGen ❌（仅项目页 radargen.github.io）、**SDDiff ✅**（github.com/StellarEsti/SDDiff）、R3D ✅。→ **核心基线需自建**，SDDiff 可作「Doppler 进扩散」的参考实现。
- 📌 **无同期直接竞品**，但新增几篇**几何增强类**雷达生成/增强论文（R3D、2606.26743、Radar-Mamba），需在 related work 覆盖，且它们都**不涉及 Doppler 生成/物理约束/时序**，不冲击空白。

置信度：中高。Web 检索覆盖良好；建议按计划**月度**用 Semantic Scholar / Google Scholar 复扫 RadarGen、4D-RaDiff、RadarMP 的引用图（本轮未做正式引用图遍历）。

---

## 1. 2026 上半年新增/相关工作一览

| 工作 | 日期 | 域/任务 | 生成 Doppler | 物理约束 | 时序 | 是否竞品 | 处置 |
|------|------|---------|:---:|:---:|:---:|:---:|------|
| **RadarMP** (2511.12117) | 2025.11 | tesseract→点云+**scene flow**（**感知**） | ⚠️点云非Doppler属性 | ✅ **Doppler 时序自监督损失** | ✅ 相邻帧 | ❗**最近邻，非生成** | **重点引用+区分任务** |
| **R3D** (2601.06465) | 2026.01 | LiDAR-radar 残差扩散**增强**（ColoRadar） | ❌ | ❌ | 未明示 | 否（几何增强） | related work 覆盖；代码✅可参考 |
| **Depth-Semantic Align.** (2606.26743) | 2026.06 | 视觉-雷达融合**补全** | ❌ | ❌（仅语义结构约束） | ❌ | 否（几何补全） | related work 覆盖 |
| **Radar-Mamba** (MM'25) | 2025 | SSM **Doppler-aware 增强** | ❌（Doppler 作输入特征） | ❌ | ❌ | 否（几何增强） | related work 覆盖 |
| **DRO** (2504.20339) | 2025 | Doppler-aware **里程计** | ❌ | ✅（Doppler↔ego 约束） | — | 否（非生成） | 佐证「物理约束在感知/里程计成熟、生成侧空白」 |
| Learning to Gen 4D LiDAR Seq (2509.11959) | 2025.09 | **LiDAR** 序列生成 | ❌（LiDAR无Doppler） | ❌ | ✅ | 否（非雷达） | 方法学借鉴（B 线） |
| LiSTAR / LiDARCrafter / Copilot4D | 2025 | **LiDAR** 世界模型 | ❌ | ❌ | ✅ | 否（非雷达） | 方法学借鉴（B 线，已在调研） |

> 已知 prior art（RadarGen 2512.17897 / 4D-RaDiff 2512.14235 / SDDiff 2506.16936）状态不变：均**单帧**、Doppler 靠条件或联合注意力**隐式/经验式**学、**无显式物理一致性损失**。

---

## 2. 对两条主线的判定

### 主线 A —— 空白成立（最近邻已识别）
- 无「生成式雷达点云模型 + 显式可微 Doppler 物理一致性损失（`v_r=(v_target−v_ego)·r̂`）」的工作。
- **RadarMP 是最强相邻工作**：它在**感知**管线里用 Doppler 引导的自监督损失约束**时序/运动一致性**——机制与我们提的损失同源，但**任务是从原始信号感知点云与 scene flow，不是生成/增强雷达数据**。
  - **定位策略**：把 RadarMP 作为「Doppler-一致性损失在感知侧被验证有效」的证据，我们**首次把它作为对生成分布的约束**引入 LiDAR→Radar 扩散；并保留**静态背景解析硬约束** `v_r=−v_ego·r̂` 这一 RadarMP/RadarGen/4D-RaDiff 均未用的差异点。

### 主线 B —— 空白成立
- 雷达侧仍全单帧；序列生成/世界模型仍 LiDAR 专属；RadarMP 虽跨帧但为感知、非序列**生成**。
- FlowRadar-4D（Doppler 驱动帧间 warp + `v_r·Δt↔Δrange` 双向一致性）无同期竞品。

---

## 3. 代码可得性（直接影响 P1 基线策略）

| Prior art | 代码 | 说明 |
|-----------|:---:|------|
| 4D-RaDiff (2512.14235) | ❌ | 未见官方仓库（2025.12 预印本） |
| RadarGen (2512.17897) | ❌ | 仅项目页 radargen.github.io，未见代码 |
| **SDDiff** (2506.16936) | ✅ | github.com/StellarEsti/SDDiff —「Doppler 进扩散」参考 |
| R3D (2601.06465) | ✅ | 增强类基线，可作对照 |

→ **触发风险 R1 应急**：P1 **自建最小 point-latent-diffusion 基线**（明确标注「类 4D-RaDiff 重实现」），并复用 SDDiff 的 Doppler 表示/处理作参考。P1 预留 +1 周。

---

## 4. 持续监控（并入 work_plan §5 月度核查）
- 每月用 Semantic Scholar / Google Scholar 遍历 **RadarGen、4D-RaDiff、RadarMP** 的引用图，盯「物理约束的生成」「时序雷达生成」新出现者。
- 读全文核实：**RadarMP**（损失形式与"生成"边界）、**2606.26743**（是否触及 Doppler/时序）。
- 关注 TU Delft/Perciv AI（4D-RaDiff 团队自述把「Doppler-运动耦合多帧」列为 future work，最可能补上时序）与 RadarGen 团队后续。

## 5. 参考链接
- RadarGen https://arxiv.org/abs/2512.17897 · https://radargen.github.io/
- 4D-RaDiff https://arxiv.org/abs/2512.14235
- SDDiff https://arxiv.org/abs/2506.16936 · https://github.com/StellarEsti/SDDiff
- RadarMP https://arxiv.org/abs/2511.12117
- R3D https://arxiv.org/abs/2601.06465
- Depth-Semantic Alignment https://arxiv.org/abs/2606.26743
- Radar-Mamba (ACM MM'25) https://palm.seu.edu.cn/zhangml/files/MM'25.pdf
- DRO https://arxiv.org/abs/2504.20339
- Learning to Generate 4D LiDAR Sequences https://arxiv.org/abs/2509.11959
