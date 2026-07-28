# G1D formal endpoint and failure-factor decision

> Date: 2026-07-29
>
> Training source: `4c6150cdd86ec1298f4b056569e3780020b4d8af`
>
> Diagnostic source: `22f4aa65fb3dc7529bc0ba269728147174b9b153`
>
> Test accessed: false

## Decision

G1D v2 completed the frozen 150-epoch H200 Stage A. Its selected checkpoint
remained epoch 15, and continued optimization degraded geometry. The endpoint,
bounded-offset variants, same-pool ranking variants, and a validation-GT
ranking oracle all failed to authorize a repair. G1D is therefore closed as a
geometry parent without extending training or tuning the frozen gates.

## Formal trajectory

Both rows use the corrected evaluator over all 24 validation frames and all
23 far-target frames.

| Metric | Selected epoch 15 | Endpoint epoch 150 | Endpoint change |
|---|---:|---:|---:|
| Mean Chamfer | `6.0103 m` | `6.3969 m` | `+6.43%` |
| Median Chamfer | `4.4609 m` | `5.4221 m` | `+21.55%` |
| Median completeness | `3.5637 m` | `3.6331 m` | `+1.95%` |
| Mean outlier at 2 m | `17.8875%` | `24.8396%` | `+6.95 pp` |
| Mean far completeness | `46.9407 m` | `48.5173 m` | `+3.36%` |
| Mean duplicate at 5 cm | `26.9721%` | `30.7279%` | `+3.76 pp` |

The best checkpoint is epoch 15 with SHA-256
`3fb1781562d6d21dddd3ca5a7dda81a0f695e59ca3224ccded617f18a5f2f5bf`.
The epoch-150 checkpoint SHA-256 is
`88c90756a28e675284842933f9c830e8f8cce02292461ff139c84020e9436c40`.

## D1: residual-offset scale

The same top-k indices were decoded with joint coarse and final residual scales
`0`, `0.25`, `0.5`, `0.75`, and `1.0`.

- Every scale below `1.0` significantly worsened mean Chamfer and
  completeness.
- Setting offsets to zero reduced outliers by `7.66 pp` and duplicates by
  `8.22 pp`, but increased Chamfer by `29.11%` and completeness by `43.88%`.
- No bounded-offset repair passed the preregistered checks.

The learned offsets are compensating for missing support rather than merely
overshooting a correct allocation. Clipping them cannot repair the route.

## D2: measured path and global condition

The fixed cross-scene 2 x 2 intervention independently swapped the measured
local path and global condition Cube.

- With the measured path fixed, a wrong global Cube changed mean Chamfer by
  `+1.05%`; the scene-first 95% interval was `-0.28%` to `+3.51%`.
- With the global condition fixed, a wrong measured path changed mean Chamfer
  by `+11.74%`; the wide scene-first interval crossed zero.
- The diagnostic's preregistered local-bypass boolean was false because the
  absolute mean global-condition change exceeded `1%`.

This does not establish a reliable global-condition effect: the interval
includes zero, while the stronger measured-path trend remains heterogeneous.
The result is evidence of weak, unstable condition use rather than proof of a
complete bypass.

## D3: ranking source

All variants used the same frozen 32k proposal pool.

| Ranking arm | Mean Chamfer | Mean completeness | Mean outlier | Mean far completeness | Mean duplicate |
|---|---:|---:|---:|---:|---:|
| Occupancy + confidence | `6.3969 m` | `4.0499 m` | `24.8396%` | `48.5173 m` | `30.7279%` |
| Occupancy only | `5.6271 m` | `3.1319 m` | `27.8387%` | `38.8265 m` | `29.0583%` |
| Validation-GT distance oracle | `4.4854 m` | `2.3668 m` | `27.2567%` | `31.0307 m` | `25.0896%` |

Occupancy-only ranking improved Chamfer by `12.03%` and completeness by
`22.67%`, but increased outliers by `3.00 pp`. Even the non-deployable
validation-GT oracle failed the absolute geometry gate and worsened outliers by
`2.42 pp`. The preregistered learned-ranking repair decision is false.

## Consequence

The evidence closes three bounded repairs:

1. no more epochs on G1D;
2. no residual-offset clipping or weight sweep;
3. no learned selector on the same 32k proposal pool.

The next experiments must change the representation or transport mechanism:
RaLD-wide condition-exclusive occupancy and refinement, fixed-cardinality
polar rectified flow, or an independently defined direct multi-horizon
Doppler model.

## Evidence

- Raw diagnostic:
  `artifacts/g1/g1d_epoch150_d1_d2_d3_22f4aa6.json`
- Raw diagnostic SHA-256:
  `bae5d699a4c3d7be14a446c79aa4293b37509d1467b7b00516b48a3e32066e8c`
- Corrected evaluator SHA-256:
  `e1f970c1e827c7801c7c9018378603b89a7709b0b4bf261e740eab1575631e68`
- Validation coverage: 24/24 frames and 23/23 far-target frames
