# RaLD 代码审计与 K-Radar 适配决策

> 日期：2026-07-19  
> 上游仓库：[MetaIoT-WHU/RaLD](https://github.com/MetaIoT-WHU/RaLD)  
> 审计 commit：`ffec4b41241391734b1eda5c093de843c909eb8e`，H200 上工作树干净  
> 目的：判断 RaLD 能否作为模块 A 的可运行基线，以及哪些结构可以公平借鉴

## 1. 结论

1. **不直接接入官方 checkpoint 作为主基线。** RaLD 的传感器尺度、视场、空间网格、位置 embedding、LiDAR AE latent 分布与 K-Radar 不同；只给 RaLD 使用 ColoRadar/HUSTRadar 外部预训练也会引入数据优势。
2. **实现 `RaLD-style-RAESum-matched` 作为几何生成基线。** 复用其隐式 occupancy autoencoder、radar-conditioned latent EDM 和 cross-attention 结构，但在 K-Radar 上从头训练，使用相同 scene split、目标、固定 10k 输出和评价器。
3. **RaLD 不是本项目 Doppler 输出或时序模块的直接竞品。** 上游预处理提取 peak Doppler，但生成路径强制只取 intensity 通道，官方生成配置也设置 `use_radar_dopp: false`；输出只有 XYZ occupancy，不包含逐点 Doppler 或 confidence。
4. **主基线禁用 CFAR query helper。** 否则生成器额外获得由 CFAR 点构造的查询位置，输入信息多于本项目的 Cube-only 主线。可另列 `+CFAR-helper` 为非公平上界。

## 2. 官方实现的真实数据流

```text
ADC
  -> Range/Doppler/Angle/Elevation FFT
  -> RAE intensity + peak velocity + validity
  -> generation path keeps intensity only
  -> 3D radar encoder
  -> 64 radar tokens, each 512-D
  -> 24-layer radar-conditioned EDM Transformer
  -> 512 x 32 latent
  -> implicit occupancy decoder queried at candidate coordinates
  -> all positive queries become a variable-size XYZ point cloud
```

代码证据：

- [`RAEIVVmap`](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/dataset_preprocessor/utils/radar_preprocessing.py#L6-L62) 沿 Doppler 求和得到 intensity，并用峰值 bin 得到单个 velocity，不保留完整频谱。
- [`process_radar_cond`](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/model/models_radar_generation.py#L363-L405) 通过 `radar_cube[..., 0]` 强制只取 intensity。
- [官方生成配置](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/configs/generation/ge_indoor_cfg_aniso_mix_view_cone_unfreeze_enc_ints_only.yml#L105-L147) 设置 `use_radar_dopp: false`。
- [`Encoder`](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/model/models_radar_encoder.py#L137-L240) 使用 3D Conv/ResNet 将雷达张量降采样后编码为 token。
- [`KLAutoEncoder`](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/model/models_ae.py#L284-L432) 将点云编码到 `512 x 32` latent，并在任意三维 query 上解码 occupancy。
- [`engine_generation.py`](https://github.com/MetaIoT-WHU/RaLD/blob/ffec4b41241391734b1eda5c093de843c909eb8e/engine_generation.py#L249-L338) 对大量随机 query 解码，并可追加 CFAR helper 和 refinement；阈值以上 query 全部输出，因此点数可变。

## 3. 与当前模块 A 的差异

| 维度 | RaLD | 当前模块 A / 后续主方法 |
|---|---|---|
| 输入 | RAE intensity，Doppler 关闭 | 完整 64-bin RAED Cube |
| 几何生成 | latent EDM + implicit occupancy | 当前轻量 frustum occupancy；主方法可比较 latent 生成 |
| 输出 | 可变点数 XYZ | 固定 10k `XYZ + Doppler distribution + existence confidence` |
| Doppler | 预处理中有 peak velocity，生成器不使用 | 最终点位置连续查询完整局部频谱 |
| 物理闭环 | 无 point-to-RAED cycle | 可微 point-to-RAED measurement cycle |
| 时序 | 单帧 | 历史 Doppler-warp prior，由当前 Cube 校正 |

因此 RaLD 能回答“更强的生成 backbone 是否改善单帧 XYZ”，但不能回答完整频谱、逐点 Doppler、cycle 或时序先验的贡献。

## 4. K-Radar 最小公平适配

### 4.1 数据和条件

1. 复用当前 manifest 和 scene split，直接读取 `cube_drae`、`target_xyz_confidence` 和 frame ID。
2. 构造 `RAE-Sum` 条件：沿 64 个 Doppler bin 求和，再取 `log1p`，归一化统计只由 train partition 计算。
3. 保留 K-Radar 原生物理网格和视场。禁止将 0-120 m 的 K-Radar range 机械插值到 RaLD 约 0-16 m 的位置语义后声称复现成功。
4. 调整 radar encoder 的降采样尺寸和位置 embedding，使 token 对应 K-Radar 原生 R/A/E 坐标。

### 4.2 Target 和 decoder

1. 将当前 radar-observable target 转为 `(range, azimuth, elevation)`，按 confidence 做确定性采样或补齐到 10k。
2. occupied/empty query 直接由当前 target 和固定 frustum query 构造，不重建 RaLD 的 LiDAR voxel cache。
3. 推理时在同一个 K-Radar query 集上分块解码，按 occupancy logit 取 exactly top-10k，再调用当前统一几何评价器。
4. 主结果禁用 query helper；`+CFAR-helper` 只作为单独标注的输入增强上界。

### 4.3 训练和环境

- 主表：K-Radar 从头训练，`external pretraining = no`。
- 附加行：允许 ColoRadar/HUSTRadar 权重微调，但必须标记 `external pretraining = yes`，不能与本项目同列作公平结论。
- 建立独立环境 `hym_rald`。上游依赖 PyTorch 2.5.1/CUDA 12.4、`torch_cluster`、`spconv` 和 PCDet；当前 `hym_radar` 已核验为 PyTorch 2.12.1/CUDA 13.0，禁止污染。
- 上游训练 YAML 当前含 `epochs: 1 #100`，正式复现必须显式固定训练步数并写入 provenance，不能依赖默认值。

## 5. 可借鉴与不借鉴

### 5.1 推荐借鉴

- radar token encoder 和空间位置 embedding；
- order-invariant `512 x 32` 点云 latent；
- radar cross-attention conditioned EDM；
- implicit occupancy decoder，可统一固定 query 后的输出规模；
- Apache-2.0 许可证下的结构适配，保留 LICENSE 与 attribution。

### 5.2 不直接借鉴

- 官方 sensor-specific checkpoint 作为公平主结果；
- 把 peak Doppler 通道误写成完整 Doppler spectrum；
- CFAR helper 参与 Cube-only 主结果；
- 可变输出点数与当前 exactly-10k 指标直接比较；
- 将 RaLD 的短距归一化坐标直接解释为 K-Radar 0-120 m 物理坐标。

## 6. 成本和风险

| 项目 | 粗估 |
|---|---|
| K-Radar 条件 cache | 约 0.2-0.4 GiB，源 Cube 仍复用共享存储 |
| adapter、测试、一帧 overfit | 3-5 人日 |
| 三种子 AE + EDM + 推理 | 约 15-45 H200 GPU-hours，误差可达 2 倍 |
| 最大风险 | 当前只有 76 个训练帧，24 层 latent diffusion 可能高成本过拟合 |

最先做的 no-go 不是完整三种子训练，而是：一帧 overfit、10-15 epoch 单种子验证、输出点数和 query 公平性检查。若单种子 Chamfer 不优于轻量 RAE-Max，或训练/推理成本超过主方法一个数量级，则保留其为 related work，不进入主表。

## 7. 决策状态

```text
Official RaLD checkpoint on K-Radar: NO-GO for the fair main baseline
RaLD-style-RAESum-matched from scratch: GO after the active G1 decision
CFAR helper: disabled in the main row; optional upper-bound row only
Environment: new hym_rald, H200 only
```

该基线不解锁 G2/G3，也不改变当前 G1 门槛。它是论文最终几何主表的独立 baseline 工作包。
