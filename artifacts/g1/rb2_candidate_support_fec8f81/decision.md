# R-B2 Candidate-support Sweep Decision

Date: 2026-08-07

## Frozen execution

- Protocol: `rb2_cube_only_candidate_support_bounded_sweep_v1`
- Diagnosed parent source: `8fe928069169624f2db40fad3d5daba86cb6740f`
- Execution source: `fec8f81e0c40f2523615ea414efd0268e58c1279`
- Device: physical H200 GPU0 (`NVIDIA H200 NVL`)
- Inputs: current DRAE Cube and frozen range/azimuth/elevation axes only
- Report-only target: `target_xyz_confidence`, loaded after candidate construction
- Sweep: Doppler score `{max_d,sum_d,log_sum_d}` x bank `{20k,40k,80k}`
- Frozen neighborhood radius: four Cartesian lattice cells
- Training, validation selection, and test access: none

The execution snapshot passed 24 H200 CPU regression tests before the scientific
run. The frozen 20k `max_d` candidate SHA-256 was reproduced exactly:
`247a3ea5bf3ce5175a40818aae1f8f47d72d8d647bddd39ed8210bfe660b0544`.
Every arm on every audited frame emitted the requested unique candidate count
and exact `80%/17%/3%` range quotas.

## One-frame preflight

On frozen train frame `seq01/radar00232`, the smallest passing bank was 80k:

| Arm | Occupied-voxel recall | Confidence coverage | Gate |
|---|---:|---:|---|
| `max_d_bank80k_r4` | 22.9815% | 49.1062% | Pass |
| `sum_d_bank80k_r4` | 20.1583% | 44.5866% | Pass |
| `log_sum_d_bank80k_r4` | 15.3826% | 35.9347% | Fail recall |

The preflight authorized only the read-only 76-train-frame support audit. It
did not authorize model training.

## Full train audit

No arm passed the frozen requirement that every one of the 76 train frames
reach both occupied-voxel recall `>=20%` and confidence coverage `>=30%`.

| Arm | Min recall | Min coverage | Frames failing either gate |
|---|---:|---:|---:|
| `max_d_bank20k_r4` | 2.5424% | 3.6782% | 28/76 |
| `sum_d_bank20k_r4` | 0.8299% | 0.7098% | 30/76 |
| `log_sum_d_bank20k_r4` | 0.0000% | 0.0000% | 43/76 |
| `max_d_bank40k_r4` | 8.4746% | 8.7219% | 14/76 |
| `sum_d_bank40k_r4` | 2.0747% | 2.3878% | 19/76 |
| `log_sum_d_bank40k_r4` | 0.0000% | 0.0000% | 30/76 |
| `max_d_bank80k_r4` | 21.6066% | 14.6397% | 3/76 |
| `sum_d_bank80k_r4` | 4.9793% | 7.2231% | 6/76 |
| `log_sum_d_bank80k_r4` | 0.0000% | 0.0000% | 17/76 |

The strongest arm, `max_d_bank80k_r4`, passed occupied-voxel recall on all 76
frames and passed both gates on 73/76 frames. Its three failures were all due
to confidence coverage:

| Frame | Recall | Coverage |
|---|---:|---:|
| `seq57/radar00205` | 24.8062% | 14.6397% |
| `seq57/radar00404` | 50.3226% | 21.9326% |
| `seq58/radar00205` | 22.0339% | 28.8994% |

## Decision

The frozen decision is `close_current_cube_activation_family`. Increasing the
bank to 80k largely repairs occupied-voxel support, but the current
score-plus-fixed-neighborhood rule does not provide reliable high-confidence
target support on every train frame. No R-B2 memorization or formal training is
authorized from this sweep.

This no-go closes only this Cube-only activation family. It does not invalidate
the earlier GT-aided `0.40 m x 4-slot` structural-capacity result, and it does
not close future Cube-only representations with a materially different learned
or adaptive activation mechanism. It also does not unlock Doppler, cycle,
temporal, validation-selection, or test claims.

## Integrity

- `one_frame.json` SHA-256:
  `97f1f7b3714d4c6f864990c09a645a52a63e930cb2567c0e24aea0b65be7ffe9`
- `full_train.json` SHA-256:
  `c5c851da7553fb07e51bb1d2077cf42cb0bb08b538a6071b18a0e777f6056665`
- Manifest SHA-256:
  `645307a8bae351db51b55128043dae69bce5b928169d5fa25c1b9c55083de4e4`
- Scene split SHA-256:
  `61596bd50ce0bdab633c9ff0ec5ab5148c2c78a10dfc06a71d0d6f0a6c9427cc`
- Normalization SHA-256:
  `4d0bca7d027a1a9f457c526f21a034a406ce4973e2dccee55ddd2d7019b41b77`
