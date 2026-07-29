# R-A2 Tiny Memorization No-Go

Date: 2026-07-29

## Frozen Run

- Protocol: `g1_ra2_rald_wce_pilot_v1`
- Mode: `tiny_memorization`
- Source: `a7f0335a8eca86f9f36686a5d0ae86046dc0b88d`
- Data: frozen eight-frame train subset selected by the protocol
- Budget: 500 updates, exact-10k evaluation every 100 updates
- Device: physical H200 GPU2
- Runtime: `2673.57 s`
- Peak allocated/reserved: `8.33/8.73 GiB`
- Test, future Cube, auxiliary radar-point arrays, and Doppler head: not
  accessed or evaluated

## Exact-10k Trajectory

| Update | Mean Chamfer (m) | Mean completeness (m) | Mean outlier | Far recall@1m | Matched-condition wins |
|---:|---:|---:|---:|---:|---:|
| 100 | 8.5843 | 3.9691 | 58.2837% | 0.5637% | 62.5% |
| 200 | 7.3273 | 3.1492 | 50.0850% | 0.6379% | 87.5% |
| 300 | 5.1223 | 1.8816 | 40.8387% | 12.4282% | 87.5% |
| 400 | 5.0818 | 1.1486 | 46.7912% | 25.6604% | 87.5% |
| 500 | 5.1892 | 1.8302 | 45.2600% | 32.5214% | 62.5% |

The frozen per-evaluation pass required:

- mean Chamfer `<= 1.0 m`;
- mean completeness `<= 0.75 m`;
- mean outlier fraction `<= 10%`.

No evaluation passed any complete gate and the consecutive-pass count remained
zero. The terminal status is
`architecture_optimization_no_go_at_500_updates`.

## Interpretation

The training objective decreased and far recall increased, but the exact-10k
geometry entered a coverage-versus-precision failure: median completeness at
update 400 was `0.3979 m`, while mean completeness remained `1.1486 m` and
outliers rose to `46.79%`. At update 500 all three absolute checks still failed.

This run alone cannot distinguish architecture capacity from objective or
optimization failure. Combined with the independent frozen-pool diagnosis,
however, the evidence is consistent:

1. the unchanged 700k candidate pool contains a strong exact-10k subset;
2. binary occupancy confidence is poorly aligned with candidate geometry;
3. lower pointwise occupancy loss does not yield reliable within-range
   geometric ranking.

The range/surface-shell arm was independently found ineligible before training
and was not launched.

## Decision

1. Close the current R-A2 binary occupancy recipe. Do not extend the tiny
   budget and do not launch its five-epoch `source_classwise` or old
   `range_class_sampler` arms.
2. Keep Q1/Q2 as a separately named candidate: it introduces a continuous
   target-distance quality score and optional polar uncertainty rather than
   extending the failed binary objective.
3. Continue the independent R-B2 Cube-only voxel-slot one-frame gate.
4. Doppler, cycle, temporal, formal validation, and test remain locked until a
   geometry parent passes its own frozen gate.

## Integrity

- Server `last.pt`: `585691466` bytes,
  SHA-256 `e1938652f332e891518cb7d86ce7b7db5dbbc3ef031157a3b36dbfaba560d538`
- `metrics_update0500.json`:
  SHA-256 `f60f62b53ea9299f5501a2353b0d236b545b89ab60b958940c1c6c55305dc9c3`
- `summary.json`:
  SHA-256 `794fdbbfe1fef7f91bde10b5c5e92d06b40d3955d5104198baf1570b609a4195`
- Full JSON trajectory, progress, and run manifest are archived under
  `artifacts/g1/ra2_tiny_a7f0335/`.
