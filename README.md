# Radar_point_cloud_gen

毫米波雷达点云生成方向的文献调研与研究想法整理。

## 文档索引

- [docs/proposal.md](docs/proposal.md)：**一页正式立项书**，把「单帧物理一致性约束」与「时序 Doppler 驱动生成」两条主线整合为统一课题（含方法、数据、评估、里程碑、风险）。
- [docs/work_plan.md](docs/work_plan.md)：**工作计划**（CVPR 2027 锚定，~19 周），含分阶段任务、团队分工、决策门 Go/No-Go、风险应急。
- [docs/work_plan_cube_to_dense_topconf.md](docs/work_plan_cube_to_dense_topconf.md)：**修正后的顶会工作计划**，以完整 4D Radar Cube 生成稠密 `XYZ+Doppler` 为主线，覆盖 Cube-point 双向闭环、实验矩阵、G0-G4 决策门与投稿展示方案。
- [docs/competitor_rescan_2026-07.md](docs/competitor_rescan_2026-07.md)：**竞品重扫备忘**（P0），确认两条主线空白仍成立、核心基线无开源代码。
- [docs/p0_setup_checklist.md](docs/p0_setup_checklist.md)：**P0 就绪清单**——WHU 服务器环境搭建 + VoD/TruckScenes 数据字段核对，含 G0 通过判据与需人工先行的阻塞项。
- [docs/p0_progress_2026-07-03.md](docs/p0_progress_2026-07-03.md)：**P0 进度纪要**——环境落地、TruckScenes mini 字段实测、**自洽性校验通过**（Doppler=RAW，`v_r≈−v_ego·r̂` 残差 MAD 0.24 m/s）。
- [docs/p2_progress_2026-07-05.md](docs/p2_progress_2026-07-05.md)：**P2 进度纪要**——可微自门控物理约束 + 消融：三轮六臂消融完成：**ego 条件使 PCE↓89%**；静态约束推一致性至 90%@0.5；**动态分支 L_dop 恢复多样性(动态样 3.5%→32.3%)**——主线 A 三组件机制全部实证。
- [docs/p3_progress_2026-07-05.md](docs/p3_progress_2026-07-05.md)：**P3 预研纪要**——时序条件 ≫ LiDAR 条件(CD 8.5→7.2)；0.25s 视距 dopp-warp 条件无增益(负结果)；复制条件基线完爆从头生成；SDEdit 精修反伤草稿(模型瓶颈)；**dopp-warp 草稿 0.5s 起占优(视距翻转)→ **桥式扩散(RF+OT配对)验证成功:CD 7.7→4.0、MMD↓79%、修复复制的物理陈旧(PCE 19.7%→40.4%≈GT)**；rollout 2.5s:几何漂移暂未胜 ego-only 下限(G3 待战)，但物理新鲜度大胜(25.1% vs 陈旧链 8.6%)——"必须联合生成 Doppler"的定量证据；复赛证伪白噪增广、确认 L_temp(W1 0.706 最佳)，**G3 主杠杆收敛为规模(trainval)**。
- [docs/p3_fullscale_2026-07-06.md](docs/p3_fullscale_2026-07-06.md)：**G3 规模战**——trainval 全量(326k对)+40M 桥式：**G3 机制层通过(Doppler-warp 链几何优于 ego-only −31%@2.5s)**；dopp 草稿使生成 −24%(首次大分离)；分布形态贴合 GT；桥式几何税待 scheduled sampling。
- [docs/p3_ss_2026-07-07.md](docs/p3_ss_2026-07-07.md)：**G3 终局**——规模化 rollout:**copy_dopp 完胜 ego-only(−31%@2.5s,warp 机制维度通过)**;SS 训练改善长时距稳定并反超 ego-only;方法定型"warp 骨架 + 生成刷新物理"。
- [docs/p1_progress_2026-07-04.md](docs/p1_progress_2026-07-04.md)：**P1 进度纪要**——数据管线代码（`code/`）+ 静/动分离物理验证：框内静止目标 **95% 内点@0.5 m/s**、ω×r 修正有实测收益、动态软约束可行、发现框速度伪运动陷阱（R3 实证）。
- [docs/related_work_doppler.md](docs/related_work_doppler.md)：自动驾驶毫米波雷达点云生成中的 Doppler 文献调研，梳理哪些工作已经生成/回归 Doppler，以及真正剩下的研究空白。
- [docs/prior_art_deep_dive.md](docs/prior_art_deep_dive.md)：对最接近的 4D-RaDiff 与 RadarGen 做精读对比，重点分析它们如何处理 Doppler、缺少什么物理约束。
- [docs/survey_temporal_doppler.md](docs/survey_temporal_doppler.md)：多帧雷达点云生成与 Doppler 时序一致性的专项调研，论证 `v_r * dt ≈ drange` 这类约束用于生成/增强任务的可行性。
- [docs/papers/README.md](docs/papers/README.md)：本项目收集的论文 PDF 索引，按“生成 Doppler”“几何增强”“时序-Doppler 耦合”等主题归类。

- [paper/outline.md](paper/outline.md)：**论文骨架**——章节-素材映射、主表设计、贡献点、兜底方案。
- [paper/draft_method.md](paper/draft_method.md)：**Method 初稿蓝本**——观察研究要点 + 全部公式(已定型) + 已证伪路线清单。

## 当前方向

基于现有调研，较有潜力的方向是：**学习式多帧雷达点云增强/生成，同时联合生成 Doppler，并利用 Doppler 与帧间径向位移之间的物理关系建立时序一致性约束**。
