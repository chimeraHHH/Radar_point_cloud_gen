# G1 R-A2 RaLD-WCE Pilot Protocol

## Scope

R-A2 is a diagnosis-only continuation of the failed R-A1 geometry gate. It
reuses `RaLDWCEField` and the formal exact-10k evaluator without changing the
R-A1 model, training script, loss, or evaluator. The three authorized modes are:

1. `tiny_memorization`: test whether the current architecture and optimizer can
   memorize a frozen eight-frame train subset.
2. `source_classwise`: isolate the classwise occupancy loss audited from RaLD.
3. `range_class_sampler`: test range-balanced positives and hard negatives while
   retaining classwise normalization.

All modes remain XYZ plus confidence only. No Doppler head, EDM, future frame,
test frame, or new scientific claim is authorized.

> **2026-07-29 eligibility addendum:** the read-only full-cohort audit in
> `docs/ra2_range_sampler_audit.md` found that the frozen
> `range_class_sampler` is not launch-eligible. Nine of 76 train frames lack a
> required positive range class, every one of the 67 otherwise sampleable
> frames has continuous jitter crossing a frozen range boundary, and the
> index-space shell is not a clean metric hard-negative shell. The
> `range_class_sampler` arm is therefore cancelled under this protocol. The
> already-started non-resumed `tiny_memorization` run may complete because it
> uses the source-classwise sampler; its evidence must be accompanied by the
> independently audited ordered cache digest. Any later `source_classwise` or
> replacement range pilot must first bind cache/Cube provenance in its resume
> contract.

## Frozen data and access boundary

- Development manifest: the existing 76 train and 24 validation frames.
- Tiny subset: exactly eight train frames, including the two globally sparsest
  target frames, at least four frames with 60-120 m targets, and at least two
  scenes.
- Five-epoch evaluation: all 24 validation frames and all 23 far-target frames.
- Condition intervention: every matched frame is paired with a deterministic
  cross-scene wrong Cube, and both arms decode identical Q0 and Q1 coordinates.
- Cache arrays opened by the pilot dataset:
  `target_xyz_confidence` and `target_rae_index` only.
- The pilot never opens CFAR arrays, test records, future Cube paths, or future
  targets. Training queries use only the current target occupancy cells.

The run manifest binds SHA256 values for the frozen manifest, scene split,
normalization, `dense_geometry.py`, `rald_wce_stage0.py`, the pilot source files,
and, for five-epoch modes, the formal-best checkpoint and metrics JSON.

## Query sampling

### Shared cell jitter

Positive and negative cells both receive independent
`Uniform[-0.5, 0.5)` jitter along range, azimuth, and elevation. Both classes
therefore cover a full cell. The earlier positive-only half-cell distribution
is forbidden because it permits a fractional-coordinate label shortcut.

Sampling is with replacement. This matches the audited RaLD class sampling and
does not create output points; exact-10k export remains capacity-one with a true
5 cm exclusion radius.

### Source-classwise and tiny modes

Each update samples exactly 10,000 coordinates:

| Class | Count |
| --- | ---: |
| Occupied | 625 |
| Empty global | 9,375 |

The occupancy objective is:

```text
0.1 * mean(BCE_positive) + 1.0 * mean(BCE_negative)
```

The tiny mode uses the same sampler and loss so its result diagnoses the current
architecture/optimization path rather than a separate objective.

### Range-class mode

Each update samples exactly 16,000 coordinates:

| Range | Positive | Global negative | Surface-shell negative |
| --- | ---: | ---: | ---: |
| 0-30 m | 700 | 2,500 | 2,500 |
| 30-60 m | 200 | 2,500 | 2,500 |
| 60-120 m | 100 | 2,500 | 2,500 |

Surface-shell negatives have Chebyshev distance 2-4 cells from a target cell.
Every negative source rejects the union of each occupied cell's 3x3x3
neighborhood. Each range must contain at least one positive target cell; a
missing class is a protocol failure and its quota is never reassigned.

For matched and wrong conditions, the loss first computes

```text
0.1 * mean(BCE_positive, range) + mean(BCE_negative, range)
```

within each of the three range classes and then averages the three results.

### Shared auxiliary objectives

Both occupancy losses retain the existing R-A1 same-query wrong-condition
margin, positive residual Smooth L1 term, and confidence Brier term. The pilot
therefore changes only the requested class loss and query sampler. FP32 model
parameters are trained under H200 BF16 autocast.

## Tiny memorization gate

- Maximum budget: 500 updates over the frozen eight-frame subset.
- Evaluation: the same exact-10k path on all eight train frames every 100
  updates.
- Per-evaluation pass:
  - mean Chamfer <= 1.0 m;
  - mean 2 m outlier fraction <= 10%;
  - mean completeness <= 0.75 m.
- Early stop: two consecutive evaluations pass all three checks.
- Terminal failure: update 500 does not complete two consecutive passes.

The terminal failure status is
`architecture_optimization_no_go_at_500_updates`. It does not by itself
distinguish insufficient capacity from optimization failure.

## Five-epoch screen and promotion

The two five-epoch modes run one update per frozen train frame:

```text
5 epochs * 76 frames = 380 updates
```

They evaluate at epoch 3 (update 228) and epoch 5 (update 380). Both evaluations
use the unchanged 500k Q0, 200k Q1, exact-10k range quotas, 5 cm minimum
distance, and wrong-condition intervention from R-A1.

The formal R-A1 metrics JSON contains far F1 but not far recall. Before pilot
training, the script therefore reloads the formal-best checkpoint and evaluates
it once with the same 24-frame pilot wrapper. This reproduces formal outlier,
median completeness, and far F1 within `1e-4`, then computes uncensored weighted
far recall@1m over the same 23 frames. The checkpoint and metrics hashes are
stored in `formal_reference.json`.

Epoch-3 continuation requires either:

- mean outlier fraction improves by at least 1 percentage point; or
- mean far recall@1m improves by at least 5 percentage points.

If neither condition passes, the run stops with
`stopped_epoch3_screen_no_go`.

Epoch-5 promotion requires all of:

- mean outlier fraction improves by at least 2 percentage points;
- mean far F1@1m improves by at least 25% relative;
- median completeness degrades by no more than 0.25 m.

Passing the pilot authorizes a separately frozen R-A2 formal experiment. It
does not pass the original R-A1 absolute geometry gate.

## Output and recovery contract

Every mode writes to a new independent directory. JSON files and PyTorch
checkpoints use temporary-file replacement. `run_manifest.json` contains a
canonical resume-contract hash over source, data, evaluator, mode, frozen
parameters, tiny frame identities, and formal-reference inputs.

`--resume` requires:

- an existing `run_manifest.json`;
- no completed `summary.json`;
- exact resume-contract identity;
- a matching atomic `last.pt`.

The checkpoint contains model, optimizer, Python/NumPy/Torch RNG state, update
count, completed evaluation points, loss history, and consecutive tiny passes.
Resume never changes a seed, frame order, quota, reference artifact, or gate.

## Verification and execution

This implementation task authorizes CPU tests only:

```bash
PYTHONPATH=code:code/scripts python -m pytest -q \
  code/tests/test_rald_wce_pilot.py
```

No GPU pilot is started by this task. A later authorized run must use
`wangning`, physical H200 GPU 0 or 2, and an `hym_*` Conda environment. Example:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=code:code/scripts

/home/wangning/miniforge3/envs/hym_radar/bin/python -u \
  code/scripts/train_rald_wce_pilot.py \
  --mode source_classwise \
  --data-root /home/wangning/Shared/l40s_wangning_radar/cube_dense_data/kradar_g0_cross_scene \
  --cache-root /home/wangning/Shared/l40s_wangning_radar/cube_dense_data/kradar_g0_cross_scene_cache_a7d06db \
  --manifest artifacts/g0/g0_audit_100_manifest.json \
  --scene-split artifacts/g0/g0_scene_split.json \
  --normalization /home/wangning/Shared/l40s_wangning_radar/cube_dense_runs/formal_0e5fe84_g1_recovery/g1_cube_normalization_train.json \
  --formal-best-checkpoint /absolute/path/to/formal_wce_best.pt \
  --formal-best-metrics /absolute/path/to/formal_wce_best_metrics.json \
  --output-dir /home/wangning/Shared/l40s_wangning_radar/cube_dense_runs/ra2_source_classwise_SOURCE \
  --source-commit SOURCE_COMMIT \
  --device cuda:0
```

Use `--mode tiny_memorization` without the two formal-best arguments. Use
`--resume` only with the same command and the same immutable source snapshot.
