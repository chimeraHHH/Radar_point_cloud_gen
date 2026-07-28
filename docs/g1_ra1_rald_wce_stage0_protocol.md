# G1 R-A1 RaLD-WCE Stage-0 Protocol

## Scope

R-A1 tests whether a condition-exclusive continuous occupancy field can replace
G1D's finite proposal support and hard proposal selection. Stage 0 consumes the
current Full-RAED Cube and exports exactly:

```text
10,000 x (x_m, y_m, z_m, confidence)
```

Doppler is deliberately absent. A Doppler head may be added only after this
geometry gate passes. Stage-0 artifacts must state
`doppler_head_implemented=false`; they are not evidence for the complete
XYZ+Doppler method.

## Official RaLD audit

The implementation boundary was checked against official RaLD commit
`ffec4b41241391734b1eda5c093de843c909eb8e`.

- `model/models_ae.py::KLAutoEncoder.decode` embeds only query coordinates and
  cross-attends latent tokens before predicting occupancy.
- `utils/utils.py::generate_query_points` creates a wide coordinate domain
  independently of the output points.
- `engine_generation.py` evaluates occupancy over the wide domain and can run a
  second query pass around positive predictions.
- `datasets/utils/query_helper.py::aug_query_helper` can sample with replacement
  and jitter when helper capacity is low.
- Official RaLD does not guarantee exact 10k, does not export confidence under
  this project's contract, and has no per-point Doppler head.

R-A1 adopts the coordinate-only decoder, wide query, and occupancy-dependent
refinement. It rejects the official helper fallback because this project
forbids copy, padding, or jitter-based duplicate filling.

## Anti-bypass architecture

```text
Full-RAED Cube
  -> global radar tokens
  -> 512 condition latents

normalized continuous RAE coordinate
  -> Fourier coordinate embedding
  -> cross-attention to condition latents
  -> occupancy logit + bounded local residual
```

The geometry decoder has no input for local Cube spectrum, energy, CFAR,
proposal source, proposal score, sparse radar XYZ, LiDAR target geometry, or
future frames. Cube information reaches geometry only through condition
latents.

## Training

Training uses bounded occupancy queries, not the 500k inference domain.

- Formal data: frozen 76 training frames; validation and test are excluded.
- Query count: 16,000 per frame.
- Occupied ratio: 6.25%; empty ratio: 93.75%, matching the direct unweighted BCE
  convention audited in RaLD.
- Positive queries: jittered within unique occupied RAE cells.
- Negative queries: unique empty-cell scrambled Sobol coordinates.
- Loss: one unweighted BCE over all queries, same-query wrong-condition margin,
  positive residual Smooth L1, and confidence Brier loss.
- Formal budget: 20 epochs, 1,520 updates, evaluation every five epochs.
- Smoke budget: one epoch over two train frames and two validation scenes.

## Inference

### Q0 wide occupancy

Formal Q0 is a fixed 500,000-coordinate scrambled Sobol domain:

| Range | Q0 quota |
| --- | ---: |
| 0-30 m | 166,667 |
| 30-60 m | 166,667 |
| 60-120 m | 166,666 |

The seed and Q0 coordinates are identical for every frame and every condition
intervention.

### Q1 occupancy refinement

Matched-condition Q0 occupancy ranks fixed per-range anchor quotas:

| Range | Q1 anchors |
| --- | ---: |
| 0-30 m | 20,000 |
| 30-60 m | 4,250 |
| 60-120 m | 750 |

Each anchor creates eight deterministic bounded local RAE queries, yielding
200,000 Q1 queries. Q1 coordinates are then frozen. The wrong-Cube arm decodes
the exact same Q0+Q1 coordinates; it cannot construct its own easier query set.

### Exact-10k export

Q0 and Q1 candidates are ranked by confidence with candidate ID as the stable
tie breaker. Selection applies a true Euclidean 5 cm exclusion radius and
frozen output quotas:

| Range | Output quota |
| --- | ---: |
| 0-30 m | 8,000 |
| 30-60 m | 1,700 |
| 60-120 m | 300 |

Each candidate has capacity one. Ground truth is not available to Q0, Q1,
ranking, range assignment, or export. If any quota cannot be filled, execution
writes `terminal_capacity_failure.json`, attempts no fallback, and exits with
status 2.

## Evaluation and gates

Formal evaluation is locked to the corrected
`code/eval/dense_geometry.py` SHA256:

```text
e1f970c1e827c7801c7c9018378603b89a7709b0b4bf261e740eab1575631e68
```

It must cover all 24 validation frames and all 23 frames containing a
60-120 m target. Test remains locked. Every frame is paired with a deterministic
cross-scene wrong Cube on identical queries.

Structural gates:

- matched and wrong arms both export exact 10k;
- observed minimum pair distance is at least 5 cm;
- no copy, padding, or duplicate jitter;
- allocated memory is below 55 GiB and reserved memory below 65 GiB.

Scientific gates:

- mean Chamfer <= 2.50 m;
- median completeness <= 2.4946 m;
- mean 2 m outlier fraction <= 25%;
- mean 60-120 m completeness <= 46.9407 m;
- wrong Cube worsens Chamfer by at least 1% on average;
- matched Cube wins on at least 75% of validation frames.

Smoke results are never promotion-eligible.

## H200 commands

All scientific tests and smoke execution run under user `wangning` on physical
H200 GPU 0 or 2. Physical GPU 1 is forbidden.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=code:code/scripts

/home/wangning/miniforge3/envs/hym_radar/bin/python -m pytest -q \
  code/tests/test_rald_wce_field.py \
  code/tests/test_preflight_rald_wce.py \
  code/tests/test_rald_wce_stage0.py \
  code/tests/test_train_rald_wce_stage0.py
```

The smoke command uses the immutable source snapshot and a new output path:

```bash
/home/wangning/miniforge3/envs/hym_radar/bin/python -u \
  code/scripts/train_rald_wce_stage0.py \
  --data-root /home/wangning/Shared/l40s_wangning_radar/cube_dense_data/kradar_g0_cross_scene \
  --cache-root /home/wangning/Shared/l40s_wangning_radar/cube_dense_data/kradar_g0_cross_scene_cache_a7d06db \
  --manifest artifacts/g0/g0_audit_100_manifest.json \
  --scene-split artifacts/g0/g0_scene_split.json \
  --normalization /home/wangning/Shared/l40s_wangning_radar/cube_dense_runs/formal_0e5fe84_g1_recovery/g1_cube_normalization_train.json \
  --output-dir /home/wangning/Shared/l40s_wangning_radar/cube_dense_runs/smoke_SOURCE_ra1_wce \
  --source-commit SOURCE_COMMIT \
  --device cuda:0 \
  --smoke
```

Removing `--smoke` starts the formal 20-epoch Stage-0 run. That action is not
authorized by this implementation task.
