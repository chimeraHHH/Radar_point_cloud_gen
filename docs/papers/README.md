# 论文 PDF 索引

调研涉及的论文 PDF（arXiv 可获取者），与各调研文档对应。下载日期 2026-06-08。

## 与本课题最接近（生成多普勒）
| 文件 | 标题 | 年份 | 多普勒 |
|------|------|------|--------|
| `2512.14235_4D-RaDiff.pdf` | 4D-RaDiff: Latent Diffusion for 4D Radar Point Cloud Generation | 2025.12 | ✅ 生成(框速度条件) |
| `2512.17897_RadarGen.pdf` | RadarGen: Automotive Radar Point Cloud Generation from Cameras | 2025.12 | ✅ 生成(光流径向投影条件) |
| `2503.08068_Song2025_SimRadar_Lidar_Camera.pdf` | Simulating Automotive Radar with Lidar and Camera Inputs | 2025 | ✅ CNN回归,用投影 v_r=v·r̂ |
| `2506.16936_SDDiff.pdf` | SDDiff: Spatial-Doppler Diffusion (raw ADC) | 2025 IJCAI | ✅ 含自车速度估计 |

## 几何为主 / 多普勒留作 future work
| 文件 | 标题 | 多普勒 |
|------|------|--------|
| `2503.03637_L2RDaS.pdf` | L2RDaS: Synthesizing 4D Radar Tensors from LiDAR | ❌ 明确排除 |
| `2503.17097_R2LDM.pdf` | R2LDM: 4D Radar Super-Resolution via Diffusion | ❌ 留 future work |
| `2511.07067_RaLD.pdf` | RaLD: High-Res 3D Radar Point Clouds with Latent Diffusion | ❌ |
| `2503.02300_RangeImageDiffusion_Wu.pdf` | Diffusion-Based mmWave Radar PC Enhancement | ❌ |
| `2504.00859_NeuRadar.pdf` | NeuRadar: NeRF for Automotive Radar Point Clouds | ❌ |
| `2502.05550_4DR-P2T.pdf` | 4DR P2T: 4D Radar Tensor Synthesis with Point Clouds | ⚠️ 张量轴 |
| `2403.03896_DART.pdf` | DART: Implicit Doppler Tomography (Radar NVS) | ⚠️ RD图像轴 |
| `2410.13526_Nawaz_PointNetpp_GAN.pdf` | Generative Adversarial Synthesis of Radar PC Scenes | ❓ 未披露 |

## 物理一致性约束模板（相邻领域）
| 文件 | 标题 | 说明 |
|------|------|------|
| `2001.08582_microDoppler_ACGAN_sifting.pdf` | Kinematically Sifted ACGAN Micro-Doppler | 运动学后验过滤 |
| `2503.02242_PhiGAN_SAR.pdf` | Φ-GAN: Physics-consistency loss for SAR | SAR 物理损失模板 |
| `2605.00018_MoCap2Radar_physics.pdf` | What Physics do MoCap-to-Radar Models Learn? | 速度干预评估框架 |

## 物理仿真器（多普勒为白送）
| 文件 | 标题 | 多普勒 |
|------|------|--------|
| `2405.17030_SCaRL_Fraunhofer.pdf` | SCaRL: CARLA-based MIMO FMCW dataset | ✅ |
| `2310.03505_RadaRays.pdf` | RadaRays: Real-time rotating FMCW radar sim | ❌(唯一例外,仅强度) |

## 未在 arXiv（仅记录，未下载）
- L2R GAN (ACCV 2020) — https://openaccess.thecvf.com/content/ACCV2020/papers/Wang_L2R_GAN_LiDAR-to-Radar_Translation_ACCV_2020_paper.pdf
- Physics-Aware Multi-Branch GAN (RadarConf 2021) — https://ieeexplore.ieee.org/document/9455194/

---

## 时序 × Doppler 调研新增（对应 `survey_temporal_doppler.md`）

### Doppler↔时序耦合（感知侧，可借损失/机制）
| 文件 | 标题 | 关键点 |
|------|------|--------|
| `2203.01137_RaFlow_radial_displacement_loss.pdf` | RaFlow: Self-Supervised Scene Flow with 4D Radar (RA-L 2022) | **径向位移损失** `L_rd`，可改造为生成时序-Doppler 约束 |
| `2508.12330_DoppDrive_temporal_aggregation.pdf` | DoppDrive: Doppler-Driven Temporal Aggregation (2025) | 用 Doppler 补偿动态点跨帧运动做多扫聚合 |
| `2508.18506_DoGFlow_crossmodal.pdf` | DoGFlow: LiDAR Scene Flow via Cross-Modal Doppler Guidance | Doppler→LiDAR 场景流伪标签 |

### 时序点云/4D 世界模型方法学（LiDAR/通用，无 Doppler，可借机制）
| 文件 | 标题 | 时序机制 |
|------|------|---------|
| `2508.03692_LiDARCrafter_4D.pdf` | LiDARCrafter: Dynamic 4D World Modeling | 自回归 warp + 残差扩散（最可移植）|
| `2511.21256_LaGen_autoregressive_lidar.pdf` | LaGen: Autoregressive LiDAR Scene Generation | 逐帧自回归 + 噪声调制 |
| `2511.13309_DriveLiDAR4D_AAAI26.pdf` | DriveLiDAR4D (AAAI 2026) | 联合时空扩散 EST-Conv/EST-Trans |
| `2404.02903_LidarDM.pdf` | LidarDM: Generative LiDAR in Generated World | 组合式 4D 世界 + 渲染 |
| `2311.01017_Copilot4D_ICLR24.pdf` | Copilot4D (ICLR 2024) | token 自回归离散扩散世界模型 |
| `2511.16049_LiSTAR_raycentric.pdf` | LiSTAR: Ray-Centric World Models for 4D LiDAR | 射线中心建模 |
| `2512.02982_U4D_uncertainty.pdf` | U4D: Uncertainty-Aware 4D World Modeling | 不确定性感知 |

### 雷达生成现状 / 数据集
| 文件 | 标题 | 关键点 |
|------|------|--------|
| `2509.18068_RadarSFD_singleframe.pdf` | RadarSFD: Single-Frame Diffusion for Radar | 确认单帧趋势 |
| `2407.07462_MAN_TruckScenes_dataset.pdf` | MAN TruckScenes (NeurIPS 2024) | 20Hz 4D雷达逐点 Doppler，时序生成最佳数据集 |

