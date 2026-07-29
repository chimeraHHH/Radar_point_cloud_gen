# R-B1 极坐标多回波 Range-Heatmap：Stage-0 表示预飞协议

## 1. 目的与边界

R-B1 将单帧几何输出改写为固定极角射线上的多回波预测：

```text
Full-RAED encoder
  -> 107 x 37 azimuth/elevation rays
  -> each ray predicts at most K ordered continuous range peaks
  -> each peak: range-bin + bounded sub-bin offset + confidence
  -> frozen range quotas + true 5 cm Euclidean exclusion
  -> exact 10,000 XYZ points or hard failure
```

本阶段**不训练模型、不读取 Radar Cube、不读取 CFAR、不预测 Doppler**。它只回答：

1. `K=4/6` 的 ray-peak 表示是否有足够容量；
2. 固定 `8,000/1,700/300` range quotas 和 5 cm 间距下能否导出 exact-10k；
3. 在直接使用 validation GT 拟合和排序 peak 的不可部署条件下，几何端点能达到什么水平。

该诊断统一标记为：

```text
unattainable_gt_aided_structural_diagnostic
```

它不是可部署方法，也不是完整表示空间上的严格数学 upper bound。每 ray 的 peak 拟合和全局
exact-10k 选择均为确定性启发式，因此只能称为 **GT-aided best-found structural
ceiling diagnostic**。

## 2. 冻结输入

- Manifest：冻结的 `76 train + 24 validation` development manifest；
- Scene split：冻结的 scene-disjoint split；
- Target：cache 中唯一读取的 `target_xyz_confidence`；
- Axes：`info_arr.mat` 和 `arr_doppler.mat`；后者只用于复用严格 axes loader，
  本协议不使用 Doppler 值；
- 禁止：test record、Radar Cube、future Cube、CFAR、模型 checkpoint。

冻结哈希：

| 输入 | SHA256 |
|---|---|
| Manifest | `645307a8bae351db51b55128043dae69bce5b928169d5fa25c1b9c55083de4e4` |
| Scene split | `61596bd50ce0bdab633c9ff0ec5ab5148c2c78a10dfc06a71d0d6f0a6c9427cc` |

输出 JSON 绑定 source commit、四个 R-B1 源文件、manifest、scene split、axes 文件、
逐帧 target cache 及 target array hash。JSON 通过 `.tmp -> .json` 原子替换写入。

## 3. 表示定义

K-Radar 角轴固定为 `107 azimuth x 37 elevation = 3,959 rays`。每条 ray 最多输出
`K` 个 range peaks：

\[
r = r_b + \Delta r,\qquad
\Delta r \in [\ell_b-r_b,\ u_b-r_b],
\]

其中 `r_b` 是最近 range-bin center，`\ell_b,u_b` 是相邻 bin center 中点定义的
Voronoi 边界。每个 peak 同时携带一个 confidence。

原始 GT 点先按最近 A/E axis 分配到 ray；同 ray 内径向间隔小于 5 cm、因最终间距约束
无法同时存在的样本先聚为一个 echo group。若某 ray 的 echo group 数超过 K，则使用
确定性的加权一维 Lloyd 拟合压缩为 K 个连续 peak。该步骤使用 GT，且不等价于模型输出。

需要分别审计：

- `K=4`：最大原始 slot 容量 `3,959 x 4 = 15,836`；
- `K=6`：最大原始 slot 容量 `3,959 x 6 = 23,754`。

每帧报告：

- occupied ray 数；
- 含 zero-ray 的 echo-count histogram、occupied-ray median/P95/max；
- 超过 K 的 ray 数；
- raw echo group 数和 K 后 peak 数；
- K 截断后的 weighted radial MAE/P95、0.05 m/0.5 m 覆盖；
- 总候选容量及三个 range strata 的容量；
- sub-bin offset 边界检查和数组 hash。

## 4. exact-10k 与 5 cm

冻结输出：

| Range | Quota |
|---|---:|
| `[0, 30)` m | 8,000 |
| `[30, 60)` m | 1,700 |
| `[60, 120)` m | 300 |
| Total | 10,000 |

候选以 GT 最近距离、confidence、support weight 和稳定 ray/range 索引排序。为降低固定
range 处理顺序导致的假阴性，程序审计三个 strata 的全部 `3! = 6` 种确定性顺序，并在
成功顺序中选择 GT 最近距离和最小者。该多顺序 greedy 仍不是组合优化证明。

每个顺序使用真实三维欧氏空间 hash，拒绝与已选点距离小于 5 cm 的候选，并报告：

- raw/considered/selected candidates by range；
- 5 cm rejection count by range；
- 被拒候选到冲突点的 minimum/mean/maximum distance；
- quota deficit；
- 最终 observed minimum pair distance。

任一要求的 K 在任一帧无法填满配额时，写出完整失败 JSON 后以非零状态退出。禁止：

- copy；
- padding；
- jitter；
- duplicate；
- 降低 5 cm；
- 改写 range quotas。

## 5. 三种预飞

### 5.1 两帧 preflight

- 从 validation 中确定性选择两个不同 scene 的 frame；
- 运行完整候选、exact selection 和 geometry endpoint；
- 只检查代码路径和失败语义，不构成科学结果。

### 5.2 100-frame capacity audit

- 覆盖全部 `76 train + 24 validation`；
- train target 只用于表示容量审计，不用于模型选择或科学 geometry endpoint；
- 任一帧容量不足即对应当前 `direct_gt_supported_peak_construction` 的 K hard
  failure。

该候选构造每个 GT echo group 最多产生一个 peak，不实现曲面插值或 sparse-target
lifting。因此当某帧 GT 本身少于 10k，或某个 range stratum 的 GT-supported peaks
少于固定 quota 时，hard failure 只能关闭“直接 GT-supported peak 拟合”方案，不能
关闭 R-B1 多回波表示 family。任何后续 lifting 必须另立协议并重新检查 radar
observability、5 cm 间距和 exact-10k，不能把本预飞改写为表示空间的 upper bound。

### 5.3 24-val geometry

- 只在全部 24 个 validation frame 上报告几何；
- 同时报告两个端点：
  - variable-count `capped_ray_peak_pool`；
  - fixed `exact_10k_quota_5cm_selection`；
- 两者都使用 GT，均不可部署。

几何指标包括 Chamfer、precision/completeness、2 m outlier、F-score 和分 range
completeness。

## 6. 23 个 far-frame 的语义

只有 target 中至少存在一个 `60 m <= range < 120 m` 点的 validation frame 才计入
far-range metric；没有 far target 的 frame 不补零、不设为通过。冻结 24-val 应得到
`23` 个 far-target frames，且 far completeness 的 `sample_count` 必须为 23。

两帧 preflight 不要求恰好覆盖 23 帧；100-frame audit 不产生科学 geometry aggregate。

## 7. 本地非科学测试

```bash
export PYTHONPATH=code:code/scripts
python -m pytest -q code/tests/test_rb1_range_echo_oracle.py
```

测试必须覆盖：

- `K >= 4` 和每 ray `<= K`；
- 输入顺序置乱后的候选与 exact selection 稳定性；
- bounded sub-bin offset；
- exact count、固定 quotas 和真 5 cm；
- 容量不足 hard failure，且无 fill fallback；
- GT diagnostic、non-deployable、non-strict-upper-bound 标签；
- far-frame 语义；
- target loader 不访问 Cube/CFAR arrays。

## 8. 服务器运行模板

以下命令是提交后在 H200 服务器 CPU 上运行的模板；本代码子任务不启动 GPU：

```bash
export PYTHONPATH=code:code/scripts
python -u code/scripts/preflight_rb1_range_echo.py \
  --resources /path/to/K-Radar/resources \
  --cache-root /path/to/dense_cache \
  --manifest /path/to/manifest.json \
  --scene-split /path/to/scene_split.json \
  --source-commit "$(git rev-parse HEAD)" \
  --mode preflight \
  --k-values 4 6 \
  --output /path/to/rb1_preflight.json
```

依次将 `--mode` 改为 `capacity-audit` 和 `validation-geometry`。默认一次审计 K=4/6；
只要本次命令请求的任一 K hard fail，进程即以状态码 2 退出。若需要把 K=6 作为独立
候选，必须单独运行 `--k-values 6` 并在后续决策中明确重新冻结，不得用 K=6 静默覆盖
K=4 的失败。所有失败 JSON 均固定写入
`representation_family_closure_eligible=false`。
