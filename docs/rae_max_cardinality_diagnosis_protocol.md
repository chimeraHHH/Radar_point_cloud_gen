# Frozen RAE-Max Cardinality Diagnosis

## Question

This read-only experiment tests whether forcing the frozen RAE-Max geometry
parent to emit exactly 10,000 cells is a major cause of its failed outlier
gate. It does not retrain the model and does not turn a smaller output into a
10k dense prediction.

The diagnosis is isolated from all formal training runs. It reads the three
bounded-recovery RAE-Max checkpoints and the same 24 validation frames, runs
one forward pass per checkpoint/frame, and decodes several prefixes from the
same occupancy logits.

## Frozen inputs

- Checkpoint source commit:
  `0e5fe8430892d57996ed26fa18f233cfa5e0c79b`.
- Seeds: `20260716`, `20260717`, and `20260718`.
- Run names:
  - `g1_rae_max_seed20260716_0e5fe843`
  - `g1_rae_max_seed20260717_0e5fe843`
  - `g1_rae_max_seed20260718_0e5fe843`
- Data: the frozen G1 manifest, scene split, train-only normalization, 76 train
  frames, and 24 validation frames.
- Test partition access is forbidden.
- The full evaluator must identify 23/23 validation frames containing a
  60--120 m target.

The output binds the diagnostic source, every checkpoint/config/archived
metric file, manifest, split, normalization, axes, Cube, dense cache, and both
evaluator implementations by SHA-256.

## Arms

The mandatory fixed arms are:

| Arm | Selected cells | Interpretation |
|---|---:|---|
| `fixed_k_2500` | 2,500 | Reduced-cardinality diagnostic |
| `fixed_k_5000` | 5,000 | Reduced-cardinality diagnostic |
| `fixed_k_7500` | 7,500 | Reduced-cardinality diagnostic |
| `fixed_k_10000` | 10,000 | Frozen dense-output contract |

All selections use sorted `torch.topk` on sigmoid occupancy logits. A flat RAE
cell can appear at most once, and each selected cell is decoded at its RAE
center. Validation GT is not an argument to the selection function.

The exact-10k arm must match the original `occupancy_to_points` decoder in
indices, confidence, XYZ, and output SHA-256. On a full run, its uncensored
overall per-frame and aggregate metrics must also reproduce the archived
values within `1e-5`. Old far metrics are not compared because they used the
censored evaluator.

### Optional train-frozen adaptive arm

`--fit-train-adaptive-threshold` enables an additional diagnostic. For each
train frame, the target effective count defines a clipped rank in
`[2500,10000]`; the median confidence at that rank becomes a global threshold.
The threshold is then frozen before validation. Validation GT is never used
to fit or apply the rule.

A preflight fits only two train frames and is marked `preflight_only`. A full
adaptive result must use all 76 train frames.

## Evaluation

Each seed and the combined three-seed result report:

- Chamfer, precision, completeness, F-scores, and 2 m outlier fraction;
- selected count, confidence-sum effective count, and count above confidence
  0.5;
- prediction and target coverage in 0--30, 30--60, and 60--120 m bins;
- validation target count and confidence-effective target count as
  diagnostics only;
- quality-only and density-quality Pareto sets;
- deltas from the exact-10k arm.

Corrected far completeness bins targets by target range and uses each target's
nearest prediction over the full prediction set. A frame remains in the far
metric even when it has no prediction in the far bin.

## Frozen interpretation rubric

For a reduced-count arm to support **exact-10k as a major outlier factor**, all
of the following must hold on the combined full result:

1. mean outlier fraction is at or below the frozen 25% gate;
2. outlier fraction falls by at least 3 percentage points versus exact-10k;
3. Chamfer degradation is at most 2%;
4. completeness degradation is at most 0.10 m;
5. 60--120 m completeness degradation is at most 1.0 m.

A weaker `bounded_contributing_factor` label requires crossing the 25% outlier
gate with some outlier reduction while satisfying the Chamfer and completeness
bounds. Otherwise exact-10k is not supported as the primary explanation.

These labels diagnose a frozen decoder intervention. They do not establish
that a model retrained for fewer points would have the same Pareto frontier.

## Execution

Run only from a committed snapshot on an allowed H200. Do not launch this
diagnostic while a formal process needs that GPU.

Two-frame preflight:

```bash
python -u code/scripts/diagnose_rae_max_cardinality.py \
  --data-root /home/wangning/Shared/l40s_wangning_radar/kradar_g0 \
  --cache-root /home/wangning/Shared/l40s_wangning_radar/kradar_g0_cache \
  --manifest /path/to/g0_audit_100_manifest.json \
  --scene-split /path/to/g0_scene_split.json \
  --normalization /home/wangning/Shared/l40s_wangning_radar/cube_dense_runs/formal_0e5fe84_g1_recovery/g1_cube_normalization_train.json \
  --rae-max-runs \
    /home/wangning/Shared/l40s_wangning_radar/cube_dense_runs/formal_0e5fe84_g1_recovery/g1_rae_max_seed20260716_0e5fe843 \
    /home/wangning/Shared/l40s_wangning_radar/cube_dense_runs/formal_0e5fe84_g1_recovery/g1_rae_max_seed20260717_0e5fe843 \
    /home/wangning/Shared/l40s_wangning_radar/cube_dense_runs/formal_0e5fe84_g1_recovery/g1_rae_max_seed20260718_0e5fe843 \
  --scope preflight \
  --source-commit "$(git rev-parse HEAD)" \
  --output /path/to/rae_max_cardinality_preflight.json \
  --device cuda:0
```

After the preflight passes, change `--scope preflight` to `--scope full` and
use a new output path. Add `--fit-train-adaptive-threshold` only when the
train-frozen adaptive arm is required. Output JSON is written through a
same-directory temporary file followed by an atomic rename.
