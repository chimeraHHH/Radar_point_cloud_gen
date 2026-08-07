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

## Live route: sparse target-demand assignment

The remaining geometry question is set allocation: can a fixed 10,000-point,
well-spaced subset preserve target-bearing radial and first/later structure
when independent pointwise ranking and sequential renewal both fail?

Four independent pre-freeze auditors rejected the original "sparse ray-range
partial transport" label. Its graph and cost were Euclidean XYZ, so ray/range
was metadata rather than mechanism; the solver was direct integer assignment,
not hard rounding. The corrected route is **STDA-F0: parity-packed sparse
target-demand bipartite assignment**. A subsequent hostile revision audit
reached `FREEZE` in round 6 for protocol SHA-256
`a1bacd619ab462b92e8b0fbfccf142e9999ab18130996762f6964a3ccd2edf84`.
Assignment/partial-transport primitives are prior art and are not a project
novelty claim.

The revised challenge memo and protocol are:

- `artifacts/idea/pre_idea_drafts/stda_f0_sparse_target_demand_assignment.md`;
- `docs/stda_f0_sparse_target_demand_assignment_protocol.md`.

The external freeze record is
`artifacts/idea/stda_f0_freeze_record.json`; its approved audit is
`artifacts/idea/stda_f0_prefreeze_audit_round6.md`. Implementation of the
zero-training oracle is now authorized, but no training or scientific result is
implied. The frozen oracle requires:

- a target-free support process that commits all 76 train-frame supports before
  a physically separate target-reading process starts;
- one fixed largest-color 5 cm parity support and complete exact float32
  spatial-hash spacing enumeration;
- permutation-invariant target mass discretization into exactly 10,000 slots;
- a fixed `K=256` graph-constrained Euclidean assignment with positive int64
  costs, dual maximum-cardinality certificates, and fixed-version replay;
- packed-pointwise and target-atom round-robin greedy matched controls;
- absolute per-frame Chamfer, outlier, range-stratum, and all six
  representation-neutral `range x {first,later}` gates on all 76 train frames;
- explicit target-conditioning/deployment booleans, atomic content-addressed
  evidence, a 30 s/frame allocation-core ceiling, 120 s/frame total ceiling, and
  60 GiB host/CUDA ceilings.

The archived F0R full-700k pointwise result is descriptive history only. No
relative oracle tolerance can rescue a failed absolute gate.

## Decision boundary

- assignment passes every absolute gate on 76/76 frames while packed pointwise
  and greedy each fail a corresponding gate: authorize a separately frozen
  Cube-conditioned learnability gate;
- largest-color parity support below 10,000 points on any frame: close only the
  frozen parity-support construction; draw no graph-capacity conclusion;
- graph cardinality failure: close only the frozen parity-support/KNN graph;
- full-cardinality geometry failure: close only the frozen assignment recipe;
- pointwise or greedy success: retain only the simpler mechanism supported by
  that matched control;
- implementation- or resource-invalid: repair and rerun the identical gate;
- no Doppler, cycle, temporal, validation, or test work starts until a learned
  76/24 geometry parent passes the complete frozen gate.

## Closed mechanisms not to fuse

The new route cannot recover fixed `8000/1700/300` quotas, fixed per-ray
`K=4/6`, Q-Local pointwise global export, VRH-F0 sequential frontier, or a dense
candidate-by-target matrix. It cannot be presented as new optimal transport,
ray-range transport, or deployable selection. No fusion is considered before
STDA has independently audited capacity evidence.
