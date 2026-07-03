# Radar_point_cloud_gen

毫米波雷达点云生成方向的文献调研与研究想法整理。

## 文档索引

- [docs/proposal.md](docs/proposal.md)：**一页正式立项书**，把「单帧物理一致性约束」与「时序 Doppler 驱动生成」两条主线整合为统一课题（含方法、数据、评估、里程碑、风险）。
- [docs/work_plan.md](docs/work_plan.md)：**工作计划**（CVPR 2027 锚定，~19 周），含分阶段任务、团队分工、决策门 Go/No-Go、风险应急。
- [docs/competitor_rescan_2026-07.md](docs/competitor_rescan_2026-07.md)：**竞品重扫备忘**（P0），确认两条主线空白仍成立、核心基线无开源代码。
- [docs/p0_setup_checklist.md](docs/p0_setup_checklist.md)：**P0 就绪清单**——WHU 服务器环境搭建 + VoD/TruckScenes 数据字段核对，含 G0 通过判据与需人工先行的阻塞项。
- [docs/p0_progress_2026-07-03.md](docs/p0_progress_2026-07-03.md)：**P0 进度纪要**——环境落地、TruckScenes mini 字段实测、**自洽性校验通过**（Doppler=RAW，`v_r≈−v_ego·r̂` 残差 MAD 0.24 m/s）。
- [docs/p1_progress_2026-07-04.md](docs/p1_progress_2026-07-04.md)：**P1 进度纪要**——数据管线代码（`code/`）+ 静/动分离物理验证：框内静止目标 **95% 内点@0.5 m/s**、ω×r 修正有实测收益、动态软约束可行、发现框速度伪运动陷阱（R3 实证）。
- [docs/related_work_doppler.md](docs/related_work_doppler.md)：自动驾驶毫米波雷达点云生成中的 Doppler 文献调研，梳理哪些工作已经生成/回归 Doppler，以及真正剩下的研究空白。
- [docs/prior_art_deep_dive.md](docs/prior_art_deep_dive.md)：对最接近的 4D-RaDiff 与 RadarGen 做精读对比，重点分析它们如何处理 Doppler、缺少什么物理约束。
- [docs/survey_temporal_doppler.md](docs/survey_temporal_doppler.md)：多帧雷达点云生成与 Doppler 时序一致性的专项调研，论证 `v_r * dt ≈ drange` 这类约束用于生成/增强任务的可行性。
- [docs/papers/README.md](docs/papers/README.md)：本项目收集的论文 PDF 索引，按“生成 Doppler”“几何增强”“时序-Doppler 耦合”等主题归类。

## 当前方向

基于现有调研，较有潜力的方向是：**学习式多帧雷达点云增强/生成，同时联合生成 Doppler，并利用 Doppler 与帧间径向位移之间的物理关系建立时序一致性约束**。
