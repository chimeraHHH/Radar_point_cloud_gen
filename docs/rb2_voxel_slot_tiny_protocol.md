# R-B2 Cube-Only Voxel-Slot One-Frame Protocol

## 1. Question and evidence boundary

This Stage-0 pilot asks one narrow question:

> Can a Cube-conditioned network memorize one frozen training frame while
> emitting exactly 10,000 points from a fixed `0.40 m` Cartesian lattice with
> four learned slots per selected voxel?

A pass establishes one-frame representational and optimization feasibility
only. It does not establish validation generalization, temporal consistency,
formal G1 geometry, or a comparison with RaLD-WCE. A failed candidate-support
gate rejects the Cube-only activation mechanism before optimization. A failed
500-update geometry gate rejects this specific representation/training recipe;
it is not evidence against every voxel or sparse-convolution model.

No GPU run is authorized by the implementation task that created this
protocol. Running the command below is a separate main-thread decision.

## 2. Frozen data and access contract

- Manifest SHA256:
  `645307a8bae351db51b55128043dae69bce5b928169d5fa25c1b9c55083de4e4`
- Scene-split SHA256:
  `61596bd50ce0bdab633c9ff0ec5ab5148c2c78a10dfc06a71d0d6f0a6c9427cc`
- Development split: `76 train + 24 validation`; test is forbidden.
- Pilot frame: lexicographically smallest `(sequence, radar_index)` in the
  frozen train partition. Target statistics do not select the frame.
- Inference inputs: current `cube_drae` and frozen range/azimuth/elevation
  axes only.
- Training/evaluation label: cache array `target_xyz_confidence` only.
- Forbidden: `target_rae_index`, CFAR, copied radar points, future Cube,
  validation/test input, target-conditioned candidate generation, and
  target-conditioned export.

The source tree, manifest, split, normalization, Cube, cache, and generated
candidate IDs are bound by SHA256 in `config.json`. Candidate count, per-range
count, and unique voxel-ID count are recomputed after generation and recorded as
explicit preflight checks.

## 3. Fixed representation

The model uses the same fine hypothesis that passed the GT-aided R-B2
structural oracle:

| Property | Frozen value |
|---|---:|
| Cartesian voxel size | `0.40 x 0.40 x 0.40 m` |
| Slots per selected voxel | `4` |
| Output point quotas | `8000 / 1700 / 300` |
| Output voxel quotas | `2000 / 425 / 75` |
| Candidate voxel quotas | `16000 / 3400 / 600` |
| Required point separation | `0.05 m` |

Candidate voxels are produced from the current Cube:

1. Reduce Doppler only for candidate ranking with
   `max_D log(1 + power)`.
2. Within each range stratum, rank current RAE cells by that Cube score.
3. Convert ranked RAE centers to immutable `0.40 m` Cartesian voxel IDs.
4. Remove duplicate IDs while preserving Cube rank.
5. If Cube peaks collide in one voxel, expand a fixed Manhattan-ordered
   Cartesian neighborhood up to radius four. Expansion is centered only on
   Cube-derived seeds and never sees the target.
6. Reject cells whose bounded slot box could leave its range stratum or the
   frozen K-Radar FOV.
7. Hard-fail if a candidate quota remains unfilled.

The network gathers the current 64-bin Doppler spectrum nearest each candidate
center, concatenates normalized Cartesian/RAE coordinates, and predicts:

- one voxel occupancy logit;
- four slot-confidence logits;
- four local XYZ residuals.

The four immutable slot anchors are the XY quadrants
`(-0.10,-0.10)`, `(-0.10,+0.10)`, `(+0.10,-0.10)`,
`(+0.10,+0.10) m`. Learned residuals are bounded by
`(+/-0.07, +/-0.07, +/-0.17) m`. Therefore every point remains at least
`0.03 m` inside its voxel, and the representation has a conservative global
pair-distance lower bound of `0.06 m`. This is part of the decoder
parameterization, not output jitter or post-hoc collision repair.

At export, candidates are ranked by voxel occupancy plus mean slot confidence.
The top `2000 / 425 / 75` whole voxels are selected and all four learned slots
are emitted. No point copy, padding, duplication, free center, or repair jitter
exists in the export function. Selected candidate positions and immutable voxel
IDs are checked for uniqueness before the report can mark duplication as false.

## 4. Supervision

GT is used only after the Cube-only candidate bank has been frozen for the
frame.

- A candidate voxel is positive when it contains at least one target point.
- The four slots correspond to target-local XY quadrants.
- If a quadrant contains multiple points, the highest-confidence point is the
  deterministic supervised representative.
- The target local offset is projected into the slot's representable box.
- Loss:
  `L = L_occ_focal + 0.5 L_slot_focal + 5.0 L_offset_smooth-L1`.

This pilot does not claim that one slot represents every return in a crowded
target voxel. Candidate support and slot multiplicity remain explicit
diagnostics.

## 5. Frozen gates and decisions

### 5.1 Candidate-support preflight

Training starts only if:

- target-occupied voxel recall in the Cube-only candidate bank is at least
  `20%`;
- target-confidence coverage in that bank is at least `30%`;
- every candidate ID is unique and belongs to the fixed lattice;
- all three candidate quotas are satisfied without GT, copy, or padding.

Failure decision: `no_go_cube_only_candidate_support`.

### 5.2 One-frame overfit

Train for at most `500` updates, evaluate every `50`, and require two
consecutive evaluations satisfying all checks:

- exactly `10,000` points;
- exact range quotas `8000 / 1700 / 300`;
- guaranteed minimum pair-distance lower bound at least `0.05 m`;
- no copy, padding, jitter, or duplicate candidate IDs;
- Cube-only candidate generation and export;
- Chamfer at most `1.00 m`;
- confidence-weighted mean completeness at most `0.75 m`;
- 2 m outlier fraction at most `10%`.

Pass decision: `go_one_frame_overfit_only`.

If the two-pass gate is not reached by update 500, the decision is
`no_go_representation_or_optimization`. A pass does not authorize a formal
training run; the main thread must inspect the candidate-support report,
geometry, hashes, and checkpoint before selecting the next route.

## 6. H200 command

Run only from a clean committed snapshot on an allowed H200:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python -u code/scripts/train_rb2_voxel_slot_tiny.py \
  --data-root /path/to/K-Radar \
  --cache-root /path/to/g0/cache \
  --manifest artifacts/g0/g0_audit_100_manifest.json \
  --scene-split artifacts/g0/g0_scene_split.json \
  --normalization-stats /path/to/cube_normalization.json \
  --output /path/to/rb2_voxel_slot_tiny_<source> \
  --source-commit "$(git rev-parse HEAD)" \
  --device cuda:0
```

The output directory is immutable. Existing non-empty output is a hard error.
The script itself verifies that the selected CUDA device reports `H200` and
supports BF16.

## 7. CPU checks

The model, target construction, loss, exact export, and hard-failure contracts
are CPU-testable:

```bash
PYTHONPATH=code python -m pytest -q \
  code/tests/test_rb2_voxel_slot_model.py

python -m py_compile \
  code/models/rb2_voxel_slot_model.py \
  code/losses/rb2_voxel_slot.py \
  code/scripts/train_rb2_voxel_slot_tiny.py \
  code/tests/test_rb2_voxel_slot_model.py
```
