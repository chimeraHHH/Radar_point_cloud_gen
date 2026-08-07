# R-B2 Cube-only Candidate-support Diagnosis Protocol

Status: frozen before execution
Source under diagnosis: `8fe928069169624f2db40fad3d5daba86cb6740f`
Execution source: the clean committed snapshot containing this diagnostic
Execution boundary: read-only, no training, H200 only

## Question

The frozen one-frame R-B2 preflight on `seq01/radar00232` produced an exact,
unique 20,000-voxel candidate bank, but reached only `10.976%` target occupied
voxel recall and `31.691%` confidence-weighted coverage. This diagnosis asks one
bounded question:

> Can the current Cube-only score-plus-fixed-neighborhood activation family
> satisfy the frozen support gate by legally changing Doppler aggregation or
> increasing the candidate bank to at most 80,000 voxels?

This is not a model comparison. Ground truth never generates, ranks, expands,
filters, or exports a candidate.

## Frozen Inputs

- Current K-Radar DRAE Cube only.
- Frozen range, azimuth, and elevation axes.
- Frozen 100-frame manifest and leakage-audited 76/24 development split.
- `target_xyz_confidence` is loaded only after candidate construction to report
  support.
- One-frame mode is exactly train frame `seq01/radar00232`.
- Full mode is exactly all 76 train frames.

Validation frames, test data, CFAR, future Cubes, temporal context, LiDAR inputs,
and target-derived indices are forbidden candidate inputs.

## Frozen Sweep

All arms use:

- Cartesian voxel size: `0.40 m` from the existing R-B2 representation.
- Seed multiplier: `8`.
- Neighborhood radius: exactly `4` lattice cells.
- Range proportions: `80% / 17% / 3%`.
- Candidate bank sizes:
  - 20k: `(16000, 3400, 600)`.
  - 40k: `(32000, 6800, 1200)`.
  - 80k: `(64000, 13600, 2400)`.

The Doppler score modes are:

1. `max_d`: `max_D(log(1 + clamp(Cube, 0)))`. This is the frozen current arm.
2. `sum_d`: `sum_D(clamp(Cube, 0))`.
3. `log_sum_d`: `sum_D(log(1 + clamp(Cube, 0)))`.

The 3 x 3 cross-product is frozen before either audit. Neighborhood radius is
not swept: holding it at `4` isolates bank-size and Doppler-score effects.

## Structural Checks

Every arm on every frame must have:

- Exact requested candidate count.
- Exact per-range candidate quotas.
- Unique immutable Cartesian voxel IDs.
- Candidates derived only from the current Cube and axes.

The frozen current 20k `max_d` arm on `seq01/radar00232` must reproduce candidate
SHA-256
`247a3ea5bf3ce5175a40818aae1f8f47d72d8d647bddd39ed8210bfe660b0544`.
A mismatch hard-fails the diagnosis.

## Frozen Support Gate

For every selected train frame independently:

- Target occupied voxel recall must be at least `20%`.
- Confidence-weighted point coverage must be at least `30%`.

An arm passes only when both minima pass. The decision reports every score mode
that passes at the smallest candidate-bank size; it does not select one score
post hoc using target metrics.

- At least one arm passes: report the minimum legal bank and all eligible score
  modes at that bank.
- No arm passes: close the current Cube-score plus radius-4 activation family.

A failure does not close all Cube-only representations. A pass is only a
GT-aided candidate-support result and does not authorize training automatically.

## Execution Order

1. Run CPU tests in the H200 Conda environment with CUDA disabled.
2. Run one-frame preflight on physical H200 GPU 0 or 2.
3. Inspect exact/unique/hash checks, metrics, runtime, and memory.
4. Only after the one-frame result is reviewed may the 76-frame train audit run.

The full train audit must not be launched by this implementation task.

## Recorded Cost

For each arm and frame, the JSON records:

- Wall-clock candidate construction time.
- Baseline and peak CUDA allocated/reserved bytes.
- Incremental peak CUDA memory.
- Candidate tensor bytes.
- Process maximum RSS.
- Cube/cache file hashes and exact arrays read.

## Evidence Boundary

The output may support only a statement about candidate activation coverage on
the frozen training audit. It is not a learned model result, does not measure
point-cloud Chamfer/completeness/outliers, and provides no validation, test,
generalization, Doppler-quality, or temporal-consistency evidence.

The output records two distinct commits: `diagnosed_parent_source_commit`
identifies the frozen R-B2 implementation whose activation family is being
tested, while `source_commit` identifies the clean executable snapshot that
contains this sweep. Requiring both avoids the impossible contract in which a
new diagnostic would have to exist inside its older parent commit.
