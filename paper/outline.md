# 论文骨架与素材映射(CVPR 2027 · 截稿 ≈2026-11-13)

> 工作名候选:**DopplerConsist** / **FlowRadar** / *Physically-Consistent Doppler Radar Point Cloud Generation*
> 状态图例:🟢 素材已终稿质量 · 🟡 结构定型待规模数字 · 🔴 依赖外部(VoD/规模实验)

## 结构与素材映射

| 章节 | 状态 | 素材来源 |
|---|---|---|
| Abstract | 🟡 | 待主表数字 |
| 1 Introduction | 🟡 | proposal §1-3;三条核心论点(调研);陈旧性发现(rollout) |
| 2 Related Work | 🟢 | related_work_doppler / prior_art_deep_dive / survey_temporal / competitor_rescan |
| 3 Physical Analysis of Radar Doppler(观察研究,贡献之一) | 🟢 | p0(口径判定/自洽性)、p1(静/动分离 95%@0.5、ω×r、R3)、rollout 陈旧性 |
| 4 Method | 🟢 | draft_method.md(本目录,公式全定型) |
| 5 Evaluation Protocol(贡献之一) | 🟢 | PCE/W1/动态样占比/复制锚 + 解读规范 |
| 6 Experiments | 🔴 | 主表=全量桥式(跑动中);消融=三轮六臂(结构定型,规模重跑);下游=等 VoD |
| 7 Limitations | 🟢 | 各轮负结果(SDEdit/白噪增广/G3 几何) |

## 主实验表设计(全量后填数)

- 表1 单帧生成:{条件扩散, +ego, +phys, +L_dop} × {CD, CD_Dopp, MMD, JSD, PCE, W1, 动态样}
- 表2 时序生成:{copy_ego, copy_dopp, 条件扩散, SDEdit, bridge(±L_temp)} 同指标 + 复制锚
- 表3 rollout 漂移:CD/PCE 随 t 曲线(0.5–2.5s),四臂
- 表4 下游(VoD 后):检测 mAP / 速度估计(对标 4D-RaDiff 46.0→53.3)
- 图:BEV 对比(统一 vmin/vmax!)、残差直方图(p0/p1 已有)、帕累托(一致性-多样性)
- 消融:ego 条件 / 静态硬约束 / L_dop / L_temp / 门控(自门控 vs 无) / dopp-vs-ego 草稿×视距

## 贡献点(对照 proposal,均已有支撑)

1. 首个显式可微 Doppler 物理一致性约束的雷达生成框架(静态解析硬约束+动态软损失)——P2 三轮实证
2. 桥式 draft→GT 时序生成 + Doppler-warp 草稿 + L_temp——P3 五轮实证
3. 物理一致性评估协议(PCE 及解读规范)——已固化
4. 反事实速度可控性——🔴 实验待做(ego 条件就位,纯推理)
5. (新增候选)雷达 Doppler 的实证观察研究(§3)——审稿人友好的"为什么"章节

## 风险与兜底

- R7 兜底:主线 A 单独成稿(§3+§4A+表1+表4)投 CVPR,B 线(§4B+表2/3)顺延 ICCV
- 竞品月度重扫:8 月初执行(RadarGen/4D-RaDiff/RadarMP 引用图)
