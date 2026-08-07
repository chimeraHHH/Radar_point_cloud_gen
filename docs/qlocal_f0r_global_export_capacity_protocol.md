# Q-Local-F0R global-export capacity protocol

Status: frozen before implementation and execution

## Decision question

After removing the mathematically contradictory hard per-frame range quotas,
does the unchanged Fresh-WCE-20 700k candidate field contain an exact 10,000
point, true-5-cm subset that passes the original per-frame geometry gates
without abandoning target-bearing middle or far strata?

This is a zero-training, train-only candidate/exporter capacity falsification.
It is not a learned Q-Local result, deployable exporter result, Doppler output,
cycle, temporal, development-validation, or test experiment.

## Why this is a new route

Q-Local-F0 is terminally closed under its frozen `8000/1700/300` exporter. The
76-frame source-bound audit proved that exporter contract impossible on train
frames `47:94` and `58:404`. F0R does not modify or reinterpret that result.

F0R changes exactly one mechanism: selection is global across `[0,120)` m
instead of enforcing three per-frame output quotas. It keeps all of the
following unchanged:

- Fresh-WCE-20 checkpoint and frozen network state;
- Q0 `166667/166667/166666` and Q1 anchors `20000/4250/750 x 8`;
- all 700,000 refined candidate coordinates;
- base confidence, residuals, candidate IDs, and stable score tie-breaking;
- exact 10,000 unique outputs;
- true Euclidean minimum spacing of 5 cm;
- the eight fit and four unseen-train frame identities and order;
- the original per-frame `CD <=0.8 m` and `outlier <=5%` capacity gates.

No model or threshold is trained or tuned.

## Source and byte binding

The run must bind:

- archived Q-Local preflight
  `artifacts/g1/qlocal_f0_preflight_bf1a166/preflight.json`, SHA-256
  `b1b740dff2515df274dfbf6fdb15ece53c1c3129d7847c6e803abc568c97fef4`;
- the same raw Cube/cache, manifest, split, normalization, checkpoint, metrics,
  and parent-manifest bytes;
- Git blob identity of the archived Q-Local scorer and both geometry-risk loss
  modules at source `bf1a166`;
- a clean source commit, physical H200 index/UUID/PCI identity, and atomic
  terminal output.

For every frame, the reconstructed candidate XYZ, Q0, Q1, and base-confidence
hashes must exactly match the archived Q-Local preflight. Any mismatch is
implementation-invalid and produces no scientific decision.

## Global exporter

The exporter sorts all 700k candidates by descending score and ascending stable
candidate ID, then greedily accepts a row only if it remains at least 5 cm from
all accepted points. It stops at exactly 10,000 points. It has no range quota,
range-mass input, GT input, padding, copying, jitter, duplicate fallback,
best-of-k, or threshold sweep.

The non-deployable capacity score is negative nearest-current-target distance.
Target geometry is opened only after the target-free candidate field and all
candidate hashes exist. The same exporter is also run with frozen parent
confidence to record a target-free baseline, but that baseline cannot pass or
fail the capacity question. If its score ordering cannot fill exact 10k, that
capacity error is recorded descriptively and the GT-nearest oracle still
answers the frozen candidate-capacity question.

## Anti-collapse gate

Every one of the 12 frames must satisfy:

- Chamfer `<=0.8 m`;
- outlier fraction at 2 m `<=5%`;
- exact 10,000 unique candidate rows;
- observed minimum pair distance `>=0.05 m`;
- no copy, padding, jitter, duplicate fallback, or range quota.

For each target-bearing range stratum, compare the global oracle with the
archived fixed-quota oracle on the same frame:

```text
global completeness <= fixed completeness + 1e-6 m
global recall@1m >= fixed recall@1m - 1e-6
```

This prevents removal of hard quotas from winning by silently abandoning real
middle/far targets. A stratum with no target has no retention gate. Selected
range counts remain descriptive and cannot be used to tune a replacement
quota.

## Decision

If all source/hash/structure, 12 per-frame geometry, and target-stratum
retention checks pass, status is `qlocal_f0r_capacity_passed`. This authorizes
only a separately frozen Q-Local 500-update run using the same global exporter;
all learned baselines must be recomputed with that exporter.

Any scientific gate failure gives `qlocal_f0r_capacity_no_go`. Then Q-Local
scoring on the Fresh-WCE field is closed, and the next representation family is
a variable multi-return renewal-hazard field. Fixed `K=4/6` measured peaks,
hard per-frame quotas, threshold sweeps, frame deletion, aggregate-only gates,
and cardinality reduction remain prohibited.

An input/source/GPU/hash/transaction/API failure or mutation of the frozen
parent state is implementation-invalid and must be repaired rather than
interpreted scientifically. An implementation-invalid terminal never
authorizes the ray-hazard successor.
