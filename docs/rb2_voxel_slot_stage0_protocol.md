# R-B2 Cartesian Voxel-Slot Stage-0 Protocol

## 1. Scope

R-B2 tests a geometry representation, not a trained model. It asks whether a
fixed Cartesian sparse lattice with a bounded number of mutually exclusive
slots per active voxel has enough structural capacity to emit the frozen dense
point-cloud contract:

- exactly 10,000 XYZ points;
- range quotas `8,000 / 1,700 / 300` for `0-30 / 30-60 / 60-120 m`;
- global Euclidean separation of at least `0.05 m`;
- no copy, padding, output jitter, duplicated candidate IDs, or free centers.

The oracle uses target geometry to choose active support voxels, slot offsets,
candidate ranking, and final selection. It is therefore labelled
`unattainable_gt_aided_voxel_slot_heuristic`. It is **not** a strict
mathematical upper bound, a deployable inference result, or evidence that a
network can predict the selected structure.

No GPU or training is authorized by this protocol.

## 2. Frozen input and access boundary

The only data manifest is the frozen 100-frame G0 cohort:

```text
manifest SHA256:
645307a8bae351db51b55128043dae69bce5b928169d5fa25c1b9c55083de4e4
partitions: 76 train + 24 validation
test records: forbidden
```

For each selected frame, the program opens the source-bound NPZ and reads only
`target_xyz_confidence`. XYZ and target confidence enter the geometry metric.
The script has no Cube/data-root argument and never accesses Cube, CFAR,
Doppler, model checkpoints, or test data. Every cache file, target array,
source file, manifest, selected point set, and aggregate frame-source list is
bound by SHA256.

This target remains the frozen radar-observable LiDAR geometry target. The
oracle does not alter its provenance; it only avoids reading the Cube and CFAR
arrays during R-B2 evaluation.

## 3. K-Radar Cartesian support

The Cartesian lattice encloses the frozen physical support:

```text
range:    [0, 118.037109] m
azimuth:  [-53, 53] deg
elevation:[-18, 18] deg
```

Every target and candidate slot must also pass the spherical FOV test. Cell
identity is fixed by integer Cartesian indices. A slot may move only inside
its own cell and keeps a `0.0255 m` boundary margin, so slots in adjacent cells
cannot collapse through a shared boundary. Slots in one cell are greedily
accepted only when their mutual Euclidean distance is at least `0.05 m`.

Two representation hypotheses are frozen:

| Name | Voxel size | Slots per voxel | Template | Purpose |
|---|---:|---:|---:|---|
| `cartesian_0p40m_s4` | `0.40^3 m` | 4 | `4^3` bounded positions | Fine spatial support with modest local multiplicity |
| `cartesian_0p60m_s8` | `0.60^3 m` | 8 | `5^3` bounded positions | Coarser occupancy with greater multi-return capacity |

These configurations jointly audit the trade-off between occupied-voxel
count, slot saturation, and local multiplicity. They are not tuned after
seeing validation geometry.

## 4. GT-aided construction

For each frame and configuration:

1. Map target XYZ to unique fixed Cartesian voxel indices.
2. In every representable target-occupied voxel, form a bounded candidate
   library from clipped local target coordinates and a deterministic subcell
   template.
3. Use nearest-target distance to choose exactly `S` mutually exclusive slots.
4. If a frozen range quota lacks candidate capacity, activate empty fixed
   lattice cells nearest to target-supported seed cells. These quota-support
   cells still contain exactly `S` bounded slots. A range with no GT uses a
   deterministic interior seed and therefore retains the forced-output
   false-positive cost.
5. Rank all slots by nearest-target distance, break ties by immutable
   voxel-slot candidate ID, and apply global 5 cm selection.
6. Stop with a capacity failure if any quota cannot be filled. No fallback,
   copy, padding, duplication, or jitter is allowed.

The procedure is a feasible GT-aided heuristic. It does not solve the
combinatorial optimum over all possible active voxel sets, so reports must
retain `strict_upper_bound=false`.

## 5. Modes

| Mode | Frames | Purpose |
|---|---:|---|
| `preflight` | 2 | Deterministic first train and first validation frame; schema and mechanism check |
| `capacity` | 100 | All 76 train and 24 validation frames; structural capacity screen |
| `geometry` | 24 | Validation-only geometry ceiling under the GT-aided heuristic |

The geometry mode must identify exactly 23 validation frames with target
points in `60-120 m`. Far completeness is aggregated over all 23 such frames,
including a frame even if its selected output has no same-range prediction.
The remaining validation frame still obeys the forced 300-point far output
quota; it simply does not contribute a far-target completeness value.

## 6. Required report

Each frame and configuration records:

- target-occupied, activated target, unavailable target, quota-support, and
  total active voxel counts;
- occupied-voxel target multiplicity and slot-saturation fraction;
- total candidate count and candidate capacity by range;
- exact selected capacity by range and selected target/support slot counts;
- total and per-range 5 cm rejections and observed minimum pair distance;
- Chamfer, precision distance, confidence-weighted completeness, 2 m outlier
  rate, F-scores, and range-conditioned geometry;
- whether far-range GT is present;
- explicit GT-aided/non-model labels and all forbidden-operation booleans;
- source, data, target, candidate, and selection hashes.

## 7. Hard gates

A configuration passes a mode only when every requested frame satisfies:

1. exactly `10,000` selected points;
2. exact `8,000 / 1,700 / 300` range quotas;
3. observed global minimum pair distance at least `0.05 m`;
4. exactly `S` unique candidate slots per activated voxel;
5. every candidate remains inside its fixed cell and K-Radar FOV;
6. no copy, padding, jitter, duplicated candidate ID, or free center;
7. explicit `unattainable_gt_aided_heuristic=true`,
   `strict_upper_bound=false`, and `eligible_as_model_result=false`;
8. no test, Cube, or CFAR access.

Any capacity shortfall is a hard failure and the process exits non-zero after
atomically publishing the failure JSON. Passing R-B2 only authorizes a later
one-frame overfit implementation; it does not authorize a formal model run.

## 8. Commands

Run after committing the four R-B2 files so `--source-commit` can bind the
checked-out source:

```bash
python -u code/scripts/preflight_rb2_voxel_slot.py \
  --mode preflight \
  --manifest artifacts/g0/g0_audit_100_manifest.json \
  --cache-root /path/to/g0/cache \
  --source-commit "$(git rev-parse HEAD)" \
  --output /path/to/rb2_preflight.json

python -u code/scripts/preflight_rb2_voxel_slot.py \
  --mode capacity \
  --manifest artifacts/g0/g0_audit_100_manifest.json \
  --cache-root /path/to/g0/cache \
  --source-commit "$(git rev-parse HEAD)" \
  --output /path/to/rb2_capacity_100.json

python -u code/scripts/preflight_rb2_voxel_slot.py \
  --mode geometry \
  --manifest artifacts/g0/g0_audit_100_manifest.json \
  --cache-root /path/to/g0/cache \
  --source-commit "$(git rev-parse HEAD)" \
  --output /path/to/rb2_geometry_24val.json
```

The output path is exclusive and immutable. Existing JSON is never
overwritten.
