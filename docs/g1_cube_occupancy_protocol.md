# G1 Cube-to-Dense Geometry Protocol

## Scope

G1 tests whether the complete Doppler spectrum improves single-frame dense
geometry before any Doppler output head, temporal prior, or cycle constraint is
introduced. The test uses the frozen K-Radar scene split and the 100-frame
train/validation cohort. The test partition remains untouched.

## Compared Systems

| ID | Input | Decoder | Output |
|---|---|---|---|
| E0 | Official Cube CFAR | None | Sparse `XYZ+Doppler` points |
| E1 | Doppler-axis maximum (`RAE-Max`) | Matched 3D occupancy U-Net | 10,000 unique RAE cell centers |
| E2 | Complete 64-bin RAED spectrum | Same 3D occupancy U-Net | 10,000 unique RAE cell centers |

E1 and E2 differ only in the spectral projection. Their spatial backbone,
occupancy target, loss, optimizer, schedule, point decoder, and evaluation code
are identical. The Full-RAED parameter increase must remain below 1%.

## Data and Normalization

- Use only frames marked `train` for optimization and only frames marked
  `validation` for model selection and G1 evidence.
- Build soft RAE occupancy from the cached radar-observable LiDAR target by
  retaining the maximum confidence in each occupied cell.
- Compute `log10(power + 1)` center and scale over every voxel in the complete
  training partition on the GPU server.
- The normalization artifact must contain the exact ordered training-frame
  list and matching manifest and split hashes. Training rejects partial or
  mismatched statistics.
- Do not use test frames for normalization, tuning, visualization selection, or
  failure analysis before G1 is frozen.

## Fixed Training Configuration

- Seeds: `20260716`, `20260717`, `20260718`.
- Epochs: 50.
- Optimizer: AdamW, learning rate `3e-4`, weight decay `1e-4`.
- Scheduler: cosine annealing over all 50 epochs.
- Base width: 8 channels.
- Loss: separately normalized soft focal loss plus `0.25` soft Dice loss.
- Gradient clipping: global norm 5.0.
- Model selection: lowest validation median confidence-weighted Chamfer distance.
- Numeric mode: BF16 autocast with FP32 model parameters.

Before the three-seed run, each encoding must pass a one-frame overfit check:
the training loss must decrease, the output must contain 10,000 finite unique
cell centers, and a checkpoint reload must reproduce the same logits.

## Metrics

Primary metrics are computed against the same radar-observable target for all
systems:

- confidence-weighted symmetric Chamfer distance;
- precision, recall, and F-score at 0.5 m, 1 m, and 2 m;
- confidence-weighted completeness distance;
- prediction-to-target outlier fraction beyond 2 m;
- distance-stratified completeness and F-score in 0-30 m, 30-60 m, and
  60-120 m.

The decoder always returns exactly 10,000 distinct RAE cells, so duplicate-point
inflation is structurally impossible. Reports retain per-frame prediction and
target counts and the effective target confidence mass.

## G1 Decision Rule

Use paired validation frames and aggregate all three seeds. Estimate 95%
bootstrap confidence intervals by resampling scenes first and seeds second.

G1 passes only when both conditions hold:

1. E1 or E2 improves over E0 in overall Chamfer distance and 1 m F-score,
   while keeping its absolute 2 m outlier fraction at or below 25%. This fixed
   threshold avoids requiring a 10,000-point dense output to match the naturally
   low outlier rate of sparse, high-precision CFAR detections.
2. E2 improves over E1 with a confidence interval excluding zero on at least
   one preregistered Doppler-sensitive geometry endpoint: overall Chamfer,
   60-120 m completeness, or 60-120 m 1 m F-score. E2 overall Chamfer must not
   degrade by more than 2%.

If dense occupancy beats CFAR but E2 does not beat E1, stop before P2 and test
whether the spectral projection is bottlenecked by the 1x1 projection or by
insufficient dynamic/far-range coverage. If neither dense model beats CFAR,
repair the target, loss, or decoder before adding Doppler heads.

### 2026-07-19 bounded recovery amendment

The first formal run failed: both dense arms improved Chamfer and 1 m F-score
over CFAR, but their absolute outlier fractions were approximately 26.5%, above
the fixed 25% gate. Full-RAED also significantly regressed versus RAE-Max,
especially at 60-120 m. The gate and threshold remain unchanged.

Exactly one representation recovery is allowed. Full-RAED retains the exact
RAE-Max projected path and adds a zero-initialized learned residual projection
of all 64 Doppler bins. E1 and E2 are rerun from scratch under the same new
source commit, seeds, data, schedule, and unchanged comparison rule. If this
recovery fails, G1 closes as failed and G2 does not start.

## Required Artifacts

- normalization JSON with hashes and exact frame list;
- configuration and provenance JSON for every seed;
- best and last checkpoints;
- per-frame and aggregate E0-E2 metric JSON;
- paired bootstrap report;
- fixed-frame qualitative panels and worst-five failure cases;
- one G1 decision note that records pass, fail, or branch without changing the
  rule above.
