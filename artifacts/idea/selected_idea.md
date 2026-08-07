# Selected geometry idea

> Successor revised on 2026-08-07 after the source-`380f3ea` VRH-F0
> scientific no-go. Test access is false.

## Closed predecessors

Q-Local-F0R removed the contradictory per-frame range quotas from the unchanged
Fresh-WCE 700k field. Its formal H200 oracle passed exact 10k, true 5 cm,
Chamfer, and outlier gates on all 12 train-only frames, but retained every
target-bearing range stratum on only 2/12. This closed pointwise global ranking
on that field before scorer training.

VRH-F0 then replaced pointwise exposure with a variable-return radial renewal
process on one frozen `512 x 214 x 74` lattice. Its complete H200 paired run was
implementation- and resource-valid. Renewal activity was real: both arms shared
the canonical stream, selected-cell hashes differed, and sequential exposure
accepted depth-at-least-two events. Renewal utility nevertheless failed:

- sequential/flat mean CD: `0.5494/0.1095 m`;
- CD gate: `9/12` versus `12/12`;
- exact-count and structural gate: `9/12` in both arms;
- all target-bearing strata retained: `3/12` versus `4/12`;
- complete first/later gate: `0/12` in both arms.

Terminal status is `vrh_f0_capacity_no_go`. This closes the frozen fitted-mark
renewal/frontier recipe and does not authorize learned renewal.

## Live route: sparse ray-range partial transport

The remaining geometry problem is set allocation: choose an exact-count,
well-spaced subset that preserves target-bearing radial structure instead of
optimizing every candidate independently. Sparse ray-range partial transport
is the next eligible mechanism because it couples candidate mass and range
coverage directly.

No transport network is authorized yet. The next artifact must first freeze a
zero-training hard-rounding oracle with:

- a target-independent candidate or ray-range support commitment;
- a sparse edge graph, never a dense `700k x target` matrix;
- explicit source/sink mass, exact-count, and unmatched-mass semantics;
- deterministic hard rounding with true minimum spacing at least 5 cm;
- per-frame Chamfer, outlier, target-stratum, and structural gates;
- comparison against the same unattainable pointwise oracle;
- target-free decoder/exporter APIs and physical GT-sidecar separation;
- a wall-time limit of 2 s/frame and a memory ceiling of 60 GiB for the final
  sparse implementation, unless the frozen protocol records a stricter bound.

The candidate hard-rounding result must remain within `0.15 m` Chamfer and
`2 percentage points` outlier of the pointwise oracle while restoring the
required strata and exact-count contracts. These thresholds are provisional
until the protocol receives an independent pre-implementation audit.

## Decision boundary

- hard-rounding capacity/resource pass on every frozen train-only frame:
  authorize a separately frozen Cube-conditioned sparse transport learnability
  gate;
- scientific capacity failure: close only the frozen sparse graph/rounding
  recipe and return to the representation board;
- implementation- or resource-invalid: repair and rerun the identical gate;
- no Doppler, cycle, temporal, validation, or test work starts until a learned
  76/24 geometry parent passes the complete frozen gate.

## Closed mechanisms not to fuse

The new route cannot recover fixed `8000/1700/300` quotas, fixed per-ray
`K=4/6`, Q-Local pointwise global export, VRH-F0 sequential frontier, or a dense
candidate-by-target matrix. No fusion is considered before sparse transport
has independent capacity evidence.
