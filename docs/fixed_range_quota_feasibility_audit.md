# Fixed per-frame range-quota feasibility audit

Status: frozen before implementation and execution

## Decision question

Are the immutable per-frame output quotas `8000/1700/300` for the radial
strata `[0,30)`, `[30,60)`, and `[60,120)` m compatible with the frozen 2 m
outlier gate on every frame in the 76-frame train partition?

This audit is triggered by the source-bound Q-Local-F0 capacity no-go. It is a
task-contract audit, not a model, candidate-field, Doppler, temporal,
development-validation, or test experiment.

## Frozen input and access boundary

- manifest: `artifacts/g0/g0_audit_100_manifest.json`;
- scene split: `artifacts/g0/g0_scene_split.json`;
- target cache: the frozen `kradar_g0_cross_scene_cache_a7d06db` bytes;
- partition: all and only the 76 manifest records marked `train`;
- target array: only `target_xyz_confidence[:, :3]`;
- validation target caches, test records, Cubes, CFAR, model checkpoints, and
  predictions are forbidden.

Every cache file and target tensor is SHA-256 bound. The run must bind a clean
source commit and use staging plus atomic rename for its terminal JSON.

## Strict radial lower bound

For target ranges `T = {||y_j||_2}` and output stratum `I_k = [l_k, u_k)`,
define

```text
g_k = inf_{r in I_k, t in T} |r - t|.
```

Euclidean distance obeys `||x-y||_2 >= abs(||x||_2-||y||_2)`. Therefore, if
`g_k > 2 m`, every prediction forced into stratum `k` is necessarily a 2 m
outlier, regardless of angle, network, candidate field, score, renderer, or
ray parameterization.

The per-frame contract lower bound is

```text
forced_outlier_count = sum_k quota_k * 1[g_k > 2 m]
forced_outlier_fraction = forced_outlier_count / 10000.
```

This bound is intentionally conservative. A reachable stratum contributes
zero to the lower bound even if 5 cm packing, angular support, or the candidate
field would make some of its quota unusable.

## Frozen decision

The current hard-quota contract passes only if every one of the 76 train frames
has `forced_outlier_fraction <= 5%`.

If any frame exceeds 5%, terminal status is
`fixed_range_quota_contract_no_go`. Then:

1. Q-Local-F0 remains a valid capacity no-go under its frozen protocol, and its
   training remains prohibited;
2. the failure must not be generalized to all geometric-risk scorers;
3. no variable multi-return ray-hazard oracle may inherit the same hard
   per-frame quotas;
4. the successor contract must be separately frozen with target-free,
   Cube-predicted adaptive range mass or variable cardinality while preserving
   explicit far-range evaluation and anti-collapse controls.

If all 76 frames satisfy the bound, terminal status is
`fixed_range_quota_contract_radially_feasible`; this does not prove full
geometric feasibility and only allows the originally planned ray-hazard
capacity oracle to retain the quotas.

## Evidence boundary

This audit may establish that a benchmark constraint is contradictory. It may
not establish model quality, support a deployable allocation policy, relax the
Q-Local result retrospectively, or unlock Doppler, cycle, temporal, downstream,
or test stages.
