# Parallel Geometry Diagnostics Decision

Date: 2026-07-29

## Scope

This record consolidates three read-only or GT-aided diagnostics run after the
formal R-A1 RaLD-WCE failure. None of the reported oracle metrics is eligible as
a learned model result, and none unlocks Doppler, cycle, temporal, or test
evaluation.

## RAE-Max Exact-10k Cardinality

- Source: `2183e4828738cb0a21997144e6f0683c196eafd2`
- Scope: three archived seeds, 24 validation frames per seed, 23 far-target
  frames per seed, and frozen logits.
- The legacy exact-10k hashes and metrics were reproduced on all 72 frame-seed
  pairs.

| Selected points | Mean Chamfer (m) | Mean completeness (m) | Mean outlier | Far completeness (m) |
|---:|---:|---:|---:|---:|
| 2,500 | 3.45915 | 1.73960 | 16.0811% | 15.5760 |
| 5,000 | 2.99225 | 1.09570 | 19.9092% | 9.37905 |
| 7,500 | 2.93113 | 0.86567 | 23.0726% | 8.01826 |
| 10,000 | 2.93058 | 0.70248 | 25.6971% | 7.25032 |

Decision: the frozen rubric identified neither a strong major factor nor a
bounded contributing factor. Reducing cardinality removes some tail outliers,
but every reduced-count arm loses completeness and is not an exact-10k dense
output. Exact-10k is therefore not the primary geometry failure.

## R-B1 Direct Range-Echo Construction

- Source: `988eb113a394fb8845d62b769220c83aff70fe6f`
- Scope: two validation frames, GT-aided ray/echo grouping, fixed `K=4` and
  `K=6`, no copy, padding, or jitter.
- `K=4` yielded only 6,798 and 4,567 peaks on the two frames.
- `K=6` yielded only 7,939 and 5,104 peaks.
- Every arm missed at least one frozen `8,000/1,700/300` range quota, so exact
  10k was structurally unreachable.

Decision: no training is authorized for the direct GT-supported peak
construction. The failure scope is deliberately bounded; it does not close an
R-B1 family that introduces a separately specified sparse-target lifting
mechanism.

## R-B2 Cartesian Voxel Slots

- Source: `14b3e65e7f3e1848ece73d6ec16f8f1c9f9b22c0`
- Scope: a GT-aided structural heuristic, not a deployable method and not a
  strict mathematical upper bound.
- Both frozen configurations passed exact-10k, range quotas, fixed-cell/slot
  identity, and true 5 cm minimum-distance checks on 100/100 train-validation
  frames.
- The minimum observed pair distance was `0.05000033 m`; no copy, padding,
  jitter, duplicate slot identifier, or free center was used.

| Configuration | Mean Chamfer (m) | Median completeness (m) | Mean outlier | Far completeness (m) |
|---|---:|---:|---:|---:|
| 0.40 m, 4 slots | 0.72268 | 0.05963 | 1.5417% | 0.24981 |
| 0.60 m, 8 slots | 0.82494 | 0.07394 | 1.5617% | 0.56609 |

Decision: R-B2 passes the structural capacity gate. The `0.40 m x 4-slot`
configuration is preferred because it has the better complete-validation
geometry diagnostic. This authorizes only a Cube-only one-frame memorization
implementation with predicted occupancy/confidence and bounded local offsets.
GT-dependent activation, ranking, or selection is forbidden at inference.

## Route Decision

1. Do not extend the failed R-A1 recipe and do not treat reduced output
   cardinality as its repair.
2. Run the bounded R-A2 supervision pilots to test whether confidence learning
   can exploit the already adequate wide-query candidate support.
3. In parallel, implement the independent R-B2 `0.40 m x 4-slot` one-frame
   overfit gate. Promotion to a multi-frame or Doppler model requires a
   separately frozen Cube-only validation protocol.
4. Keep R-B1 closed at its current direct-construction scope unless a legal
   sparse-target lifting contract is proposed and preflighted.

## Artifact Integrity

| Artifact | SHA-256 |
|---|---|
| `rae_max_cardinality_2183e48/full.json` | `41c217d14e61b55a6b6f8a7e5c399e1957cb82e26e8f14e4b613f1006e1f8647` |
| `rae_max_cardinality_2183e48/preflight2.json` | `e659cef1826cb5bfe3e04f6f62072c6cbf408db4bac60617df2883500acbbb8d` |
| `rb1_range_echo_988eb11/preflight2.json` | `5e444ee0cb294cec321338b5e3719acaf254a5c05a18a25048a43f849b993a8c` |
| `rb2_voxel_slot_14b3e65/capacity100.json` | `ab064b12a973b6c637bcd3969503819d109bf9d192554f77f9716482d921a43d` |
| `rb2_voxel_slot_14b3e65/geometry24.json` | `69ab5dd0938f0b5bb0fc1a082a8268bb31d5c5abe2938a0618db17b8e3ecc3f2` |
| `rb2_voxel_slot_14b3e65/preflight2.json` | `92b7968a0727af266da6121897018d0b1e773382fd4307180a896bb88a8cb6b4` |
