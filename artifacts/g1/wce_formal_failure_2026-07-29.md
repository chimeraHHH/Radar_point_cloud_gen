# R-A1 RaLD-WCE formal Stage-0 decision

Date: 2026-07-29 CST

## Decision

R-A1 completed the frozen 20-epoch, one-seed H200 Stage-0 and failed
promotion. The selected endpoint is epoch 20. It passed every structural
contract and learned a measurable Full-RAED condition effect, but it failed
three of the six scientific checks: mean Chamfer, mean outlier fraction, and
matched-condition win rate.

R-A1 is therefore not a geometry parent and does not unlock a Doppler head,
Cube cycle, temporal training, or test access. The run is closed without an
epoch extension.

## Formal trajectory

| Epoch | Mean Chamfer (m) | Median completeness (m) | Mean outlier | Far completeness (m) | Wrong-condition delta | Matched wins |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 4.4170 | 1.7931 | 32.0100% | 12.3964 | +0.1128% | 45.8333% |
| 10 | 4.7939 | 2.3325 | 30.8521% | 14.7098 | +2.7091% | 62.5000% |
| 15 | 4.2181 | 1.8135 | 30.9387% | 14.1524 | +10.9895% | 75.0000% |
| 20 | **4.0090** | **1.5081** | 31.8129% | **11.1211** | **+12.9874%** | 66.6667% |

The epoch-20 selection score is `4.14521091307203`. Training took
`8182.4741 s`; peak H200 memory was `8.5145 GiB` allocated and `9.0605 GiB`
reserved.

## Frozen endpoint

| Check | Result | Requirement | Passed |
| --- | ---: | ---: | :---: |
| Mean Chamfer | `4.00895 m` | `<=2.50 m` | No |
| Median completeness | `1.50805 m` | `<=2.4946 m` | Yes |
| Mean 2 m outlier | `31.8129%` | `<=25%` | No |
| 60--120 m completeness | `11.12111 m` over 23 frames | `<=46.9407 m` | Yes |
| Wrong-condition Chamfer degradation | `12.9874%` | `>=1%` | Yes |
| Matched-condition wins | `66.6667%` | `>=75%` | No |
| Exact export | `10,000` points in both arms | exact `10,000` | Yes |
| Minimum pair distance | `0.0500008 m` | `>=0.05 m` | Yes |

The matched prediction-to-target mean distance is `2.45498 m`, while the
target-to-prediction mean distance is `1.55397 m`. Together with the strong
condition intervention and low completeness error, this localizes the failure
to prediction precision and the selected low-quality tail rather than a total
absence of target support or a Cube-condition bypass.

This remains a diagnosis, not authorization for a repair. The next bounded
checks are inference-only candidate/ranking, positive-capacity, residual-off,
range-quota, and cardinality diagnostics. Independently named R-B1 range-echo
and R-B2 Cartesian voxel-slot representations must pass their structural
oracles before any training.

## Evidence

- Source commit:
  `f2a9489d40323d1ef45d85de958f4aea8126e1c8`
- Server run:
  `/home/wangning/Shared/l40s_wangning_radar/cube_dense_runs/formal_f2a9489_ra1_wce_seed20260716`
- Selected checkpoint:
  `checkpoint_epoch020.pt`
- Selected checkpoint SHA-256:
  `5be30e0f1ca23ea3b603abb0f5e330efd3599167362a8e23ab3a5967c411a2a0`
- Archived endpoint:
  `artifacts/g1/wce_formal_f2a9489/best_metrics.json`
- Endpoint JSON SHA-256:
  `ae5bca6273fd3e8c18672b5e3704536a43bb897b81b389f07862ead7922df5a6`
- Archive transport SHA-256:
  `ff05556cb64d0e958c28649213517e4995b0ec0460311df743c8bf6932887327`
- Test accessed: `false`
- Doppler head implemented/evaluated: `false/false`
