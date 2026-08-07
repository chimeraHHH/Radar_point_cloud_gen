# Cube-to-dense objective contract

> Refreshed 2026-08-08 after the VRH-F0 terminal decision and external
> activation of the STDA-F0 protocol. Test access remains false.

## Real objective

Given the current K-Radar Full-RAED Cube and, only after a single-frame family
passes, a bounded radar history plus ego motion, generate a fixed-count dense
radar state:

```text
10,000 x (x, y, z, circular Doppler distribution, confidence)
```

The final method must improve spatial utility while preserving measured motion
evidence. A complete claim requires current-frame geometry, generated per-point
Doppler, Cube-to-point-to-Cube closure, temporal consistency, and an untouched
test evaluation.

## Current executable question

There is no authorized geometry parent. The next question is therefore:

> Can a target-demand-coupled set assignment expose an exact-10k, strict-5-cm
> subset that preserves target-bearing range and first/later structure after
> pointwise ranking and sequential renewal both failed?

The deleted R-A1 checkpoint cannot be recovered. The new replay may be used
only as an explicitly fresh baseline or candidate generator; it cannot inherit
the original-parent identity or the failed Q1-R authorization.

## Decisive evidence

- RAE-Max reduced CFAR Chamfer from `10.6982 m` to `2.9306 m`, but missed the
  outlier gate at `25.697%`.
- Early Full-RAED fusion regressed Chamfer by `5.86%` relative to RAE-Max.
- G1D/G1G show that arbitrary queries and global hierarchy alone do not solve
  fixed-count allocation; condition bypass, duplicates, or outliers remain.
- Both the archived R-A1 pool and a fresh replay pool contain excellent
  exact-10k subsets under an unattainable GT-nearest ranking. Replay geometry
  changes from `4.0261 m/31.7854%` Chamfer/outlier to
  `0.6375 m/2.2771%` without changing the 700k candidate coordinates.
- Binary occupancy confidence is therefore not a reliable geometric-utility
  score. A continuous quality head has not failed; it was never trained because
  the old-parent equivalence prerequisite failed.
- R-B2 proves voxel-slot representational capacity, but its current
  score-plus-fixed-neighborhood activation family fails per-frame confidence
  coverage and is closed.
- Q-Local-F0 passed all source, gradient, intervention, and structural checks,
  but its fixed-quota GT-nearest oracle passed only 10/12 frames; its scorer was
  never trained.
- The 76-frame strict radial audit proves the hard `8000/1700/300` allocation
  impossible on train frames `47:94` and `58:404`, independently of network,
  candidate field, angular error, spacing, and completeness.
- Q-Local-F0R removed those quotas and passed per-frame CD, outlier, count, and
  spacing on 12/12, but retained every target-bearing range stratum on only
  2/12. Pointwise GT-nearest global ranking therefore has strong aggregate
  support but an explicit near-range allocation collapse.
- VRH-F0 certified real renewal activity on one shared stream, but sequential
  exposure was worse than flat exposure: mean CD `0.5494/0.1095 m`, strata
  retention `3/12` versus `4/12`, and complete first/later success `0/12` in
  both arms. The frozen renewal recipe is closed.
- Four independent auditors rejected the proposed ray-range/partial-transport
  wording because the actual graph is Euclidean XYZ assignment and the
  primitive is prior art. After six hostile revision rounds, the corrected
  STDA-F0 protocol received `FREEZE` for SHA-256 `a1bacd619ab462b92e8b0fbfccf142e9999ab18130996762f6964a3ccd2edf84`.
  It remains a non-deployable graph-constrained capacity oracle.

## Frozen development gate

The accepted 76-train/24-development scene split and corrected evaluator stay
fixed. A learned parent must satisfy all of:

- median Chamfer `<= 2.50 m`;
- mean outlier fraction at 2 m `<= 25%`;
- median completeness `<= 0.65 m`;
- mean 60--120 m completeness `<= 8.0 m` over all 23 far-target frames;
- mean duplicate fraction within 5 cm `<= 10%`;
- exact output count `10,000`, with no per-frame hard range quota;
- true minimum spacing `>= 5 cm`;
- explicit range-stratified completeness/recall anti-collapse checks whenever
  the frame contains targets in that stratum;
- preregistered matched-vs-cross-scene condition degradation `>= 1%`.

The first learned run must begin with a train-only memorization gate. The
24-frame development cohort may select the route but is not an untouched
generalization set; final generalization requires the still-locked test cohort.

## Trusted proxies

- exact per-frame geometry reconstructed from immutable frame reports;
- candidate-support and representation-capacity diagnostics that are marked
  non-deployable;
- matched/wrong-condition controls on identical candidate coordinates;
- range-stratified coverage, precision, duplicates, and exact-count checks;
- source, data, checkpoint, output, and runtime hashes from a clean H200
  snapshot.

## False-progress signals

The following do not authorize promotion:

- lower training loss without exact-10k geometry improvement;
- GT-aided ranking or capacity oracles presented as model results;
- improved precision obtained by discarding coverage or reducing point count;
- restoring the retired `8000/1700/300` per-frame range quotas;
- duplicate points, padding, or post-selection jitter used to satisfy count;
- nonzero gradients without measurable cross-scene condition dependence;
- reusing the Q1-R certificate or calling the replay the deleted checkpoint;
- another binary-occupancy score, fixed-neighborhood activation, deterministic
  query refiner, or latent occupancy model under a new label;
- attaching measured Doppler after geometry and calling it generated Doppler;
- best-of-k sample selection;
- accessing test before the single-frame, Doppler, and temporal families freeze.

## Hard constraints

- all scientific computation runs as `wangning` on the H200 server;
- only physical H200 GPU 0 or 2 may be used; GPU 1/RTX 5000 is forbidden;
- Conda environments use the `hym_*` namespace;
- unrelated jobs are not stopped or modified;
- no CFAR or LiDAR-derived inference helper may silently carry the main method;
- any new continuous-quality branch must first close the archived Q1-R static
  audit requirements: raw-input diagnostic binding, exact frozen-frame
  contracts, complete evaluation-chain validation, atomic failure commits,
  physical GPU provenance, and adversarial tests.

## Search and experiment budget

The completed idea pass compared direct radar generation, RaLD, fixed-count set
prediction, ray/range allocation, quality/ranking, and renewal mechanisms.
Q-Local-F0R and VRH-F0 are terminal scientific no-gos for their frozen recipes.
The live falsification is the externally frozen `STDA-F0` protocol: all 76
training frames, a target-free parity-packed support commitment, physically
separate GT-demand fitting, fixed `K=256` graph-constrained assignment, matched
pointwise/greedy controls, and absolute geometry/range/first-later gates.

STDA is an internal capacity test, not a novel transport claim. Its zero-training
implementation is authorized only from the freeze commit; learned allocation
remains prohibited until the complete capacity gate passes.

Before a training route is authorized, it needs a source-bound falsification
that completes in at most two H200 GPU-hours and states:

- the one primary mechanism being changed;
- the closest closed local route and why this is not a relabeling;
- the anti-win condition;
- the exact pass and abandonment rules;
- the artifact that unlocks the next stage.
