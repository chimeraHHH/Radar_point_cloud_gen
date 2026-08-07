# Q1 RaLD-WCE Geometry-Quality Tiny Protocol

## Decision question

Formal R-A1 already contains a geometrically strong, deterministic 700,000
candidate pool, but binary occupancy confidence does not rank that pool well.
The validation-GT diagnostic improved mean Chamfer from 4.00895 m to 0.64102 m
by changing only the score used for exact export. The R-A2 binary
eight-frame/500-update memorization control also failed its frozen gate.

Q1 asks one narrow, falsifiable question:

> Can an independent continuous geometry-quality head learn to rank the
> unchanged formal R-A1 candidates on the frozen eight train frames?

This pilot is ranking-only. It does not authorize a new candidate generator,
residual, evaluator, Doppler output, temporal input, or validation/test claim.

## Frozen formal parent

The pilot requires:

- the formal R-A1 best checkpoint;
- its matching metrics JSON and run manifest;
- the existing `RaLDWCEField`, Q0/Q1 construction, residual, and geometry
  evaluator;
- Q0 quotas `166667/166667/166666`;
- Q1 anchor quotas `20000/4250/750`, with 8 samples per anchor;
- exactly 700,000 Q0+Q1 candidates;
- output quotas `8000/1700/300`;
- exact 10,000 output points and a true 5 cm capacity-one exclusion radius.

The formal field is loaded strictly, put in evaluation mode, assigned
`requires_grad=False`, and excluded from the optimizer and Q1 checkpoint.
Its state-dict digest is checked at every evaluation and at termination.

At update zero, the final layer of the new quality head is zero initialized.
The quality logit is:

```text
quality_logit = logit(detached_formal_occupancy_confidence)
              + learned_quality_delta
```

Therefore the update-zero exact export must select the same candidate rows as
formal occupancy confidence on every frozen tiny frame. Failure of this
control invalidates the run before training.

## Quality head

The independent head receives only:

1. the formal residual-refined normalized RAE coordinate;
2. the frozen formal condition latent from the current Full-RAED Cube;
3. detached formal occupancy confidence.

It applies a small Fourier coordinate embedding, projected condition
cross-attention, and an MLP quality correction. It has no API argument for
target geometry, test/future frames, Doppler targets, stochastic samples, or
best-of-k selection.

At inference, exactly one deterministic quality score is computed per formal
candidate. Only this score replaces occupancy confidence in
`exact_capacity_export`; coordinates, residual, Q0/Q1, range quotas, and the
5 cm rule are unchanged.

## Continuous training target

Ground truth is introduced only after the complete formal candidate field has
been built. For candidate `i`, let `j(i)` be its nearest current-frame target:

```text
d_i = ||candidate_xyz_i - target_xyz_j(i)||_2
w_i = clamp(target_confidence_j(i), 0) / max_j target_confidence_j
q_i = stopgrad(w_i * exp(-d_i / 1.0 m))
```

`target_xyz_confidence`, candidate coordinates, nearest indices, distances,
and quality targets are detached. Q1 cannot update or chase the formal
residual. The target is used only for training supervision and post-inference
geometry metrics; it is forbidden in candidate generation, scoring, and
exact export.

The nearest-neighbour calculation is chunked and never builds a
`700k x N_target` matrix.

## Training sample and loss

Every update selects 16,000 unique rows from one frozen 700k candidate field:

| Range | Rows |
| --- | ---: |
| 0-30 m | 12,800 |
| 30-60 m | 2,720 |
| 60-120 m | 480 |

These ratios match the frozen exact-output quotas. Within each range, half of
the rows are sampled from the top 10% continuous-quality pool and half
uniformly from the remaining candidates. Sampling is deterministic by frame,
update, and source seed. This is training-only supervision sampling; it does
not replace Q0/Q1 or exact export.

The loss averages soft-target BCE independently across the three ranges and
adds a within-range pairwise ranking term:

```text
L = mean_range soft_BCE(quality_logit, q)
  + 0.5 * mean_range softplus(-(s_high - s_low))
```

Pairs with target-quality separation below 0.05 are omitted. Only quality-head
parameters are optimized with AdamW; the formal field remains frozen.

## Data and provenance boundary

The run recomputes and requires the audited ordered cache byte digest:

```text
dd9d296cc10933fce12f4e050b4aa065f82752ab123171ec1a75e8cabbc06e4f
```

The run manifest also binds:

- per-cache SHA256 values in manifest order;
- per-Cube SHA256 values and an ordered digest for the eight tiny frames;
- manifest, scene split, normalization, geometry evaluator, and exact-export
  hashes;
- `info_arr.mat` and `arr_doppler.mat` resource hashes;
- formal checkpoint, metrics, and run-manifest hashes;
- source commit and hashes for the formal parent, quality implementation,
  evaluator, training script, and this protocol.

`arr_doppler.mat` is bound because it is part of the immutable K-Radar axis
resource set. Q1 does not read a Doppler target, predict Doppler, or evaluate a
Doppler metric.

The dataset opens only the current Cube plus
`target_xyz_confidence`/`target_rae_index`. The frozen eight training frames
are also the only tiny evaluation frames. Test and future records are
forbidden.

## Frozen gate

Budget: at most 500 updates.

Evaluation: exact-10k inference on all eight frozen train frames after updates
100, 200, 300, 400, and 500.

One evaluation passes only if all three hold:

- mean Chamfer <= 1.0 m;
- mean 2 m outlier fraction <= 10%;
- mean completeness <= 0.75 m.

Q1 tiny passes only after two consecutive evaluations pass all three checks.
The run then stops as `quality_ranking_tiny_passed_early`.

## Pre-registered failure decisions

- Update 500 without two consecutive gate passes:
  `quality_ranking_no_go_at_500_updates`.
- Any exact-10k or 5 cm capacity failure:
  `quality_ranking_no_go_exact_10000_capacity`.
- Ordered cache digest, Cube hash, source hash, formal checkpoint, run
  manifest, or resume-contract mismatch: hard provenance failure; no
  scientific result may be reported.
- Any update-zero selected-row mismatch: implementation invalid; do not train.
- Any formal-base state-dict change or gradient: implementation invalid; do
  not report the run.

Passing this tiny gate authorizes a separately frozen 76/24 Q1 pilot. It does
not pass G1, unlock Doppler/cycle/temporal work, or support a validation/test
claim.

## Verification and launch boundary

This implementation task authorizes CPU tests only:

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=code:code/scripts \
  /home/wangning/miniforge3/envs/hym_radar/bin/python -m pytest -q \
  code/tests/test_rald_wce_quality.py
```

No GPU training is launched by this task. A later launch must use `wangning`,
physical H200 GPU 0 or 2, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, and the
`hym_radar` environment. The script requires explicit paths to the formal
checkpoint, metrics, and run manifest and writes to a new output directory.
