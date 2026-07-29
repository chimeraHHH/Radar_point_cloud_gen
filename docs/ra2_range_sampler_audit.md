# R-A2 Range-Sampler Supervision Audit

## Decision boundary

This is a CPU-only, read-only eligibility audit. It does not instantiate
`RaLDWCEField`, start CUDA, edit a checkpoint, or change the R-A2 implementation.
It must run before interpreting `range_class_sampler` against
`source_classwise`.

The audit covers:

- positive support in all three frozen ranges for every train frame;
- cache provenance and XYZ/RAE consistency;
- global and surface-shell negative semantics;
- 3x3x3 ambiguity exclusion;
- per-range/per-class quotas and deterministic replay;
- range labels after full-cell jitter;
- manifest scene separation and test exclusion;
- cross-run input/source comparability;
- checkpoint/progress/RNG integrity when run directories are supplied.

## Static findings

### High: loss and exact-10k ranking optimize different objects

`sample_pilot_queries` and `rald_wce_pilot_loss` receive binary occupied/empty
cells but not `target_xyz_confidence`. The formal evaluator generates about
700,000 continuous Q0/Q1 candidates and selects exact 10k by confidence within
fixed range quotas, subject to the 5 cm capacity filter. Pointwise BCE and
global Brier calibration do not impose the within-range ordering needed by this
selection. Lower training loss therefore does not imply a better exact-10k
subset.

Minimum patch recommendation, not applied here: retain occupancy BCE but add a
frozen candidate-set ranking/calibration term using target distance or target
importance. This is the most direct response if long training still yields a
good candidate pool and poor selected geometry.

### High: actual cache/Cube inputs are absent from the resume contract

The current contract binds manifest, scene split, normalization, source files,
and baseline inputs. It does not bind `data_root`, `cache_root`, per-cache
digests, Cube files, or cache `source_commit`. A run can therefore resume from a
different cache root without violating the current contract.

The audit emits an ordered digest over all 100 cache SHA256 values. That digest
must be identical for every compared R-A2 mode. Minimum patch recommendation:
bind it, the cache source commit, data-root identity, and an ordered Cube
digest/index into `resume_contract`.

### High: formal reference is not integrity-bound after resume

On resume, `_formal_reference` trusts `formal_reference.json` after checking its
recorded baseline input hashes. The reference JSON's own digest is not in the
checkpoint contract. The audit can check internal metric/value consistency for
supplied run directories, but that cannot detect a coordinated edit.

Minimum patch recommendation: recompute or verify the reference against the
formal metrics on resume and bind the resulting JSON digest to the checkpoint.

### Conditional blocker: jitter can invalidate range quotas

The sampler assigns `range_class` from the integer cell center, then applies
independent full-cell jitter. A query next to 30 or 60 m can cross the physical
boundary while retaining the old class. Any observed crossing is a hard
failure because the claimed per-range quotas and normalized loss no longer
refer to the continuous coordinates seen by the model.

Minimum patch recommendation: sample/clamp range jitter inside the assigned
physical stratum, then assert the continuous query remains in that stratum.

### Medium: auxiliary terms are not range-normalized

The primary occupancy BCE averages each label class within each range. The Brier
term remains a global mean over 15,000 negatives and 1,000 positives, while the
residual term averages positives in the 700/200/100 sampling mix. Far-range
confidence and residuals are therefore not weighted like the primary BCE.
Report per-range auxiliary gradients before deciding whether to normalize them.

### Medium: query budgets differ

`source_classwise` decodes 10k training queries per update;
`range_class_sampler` decodes 16k. Both run 380 updates. The comparison is valid
as a bundled sampler intervention, but it is not an isolated loss ablation.
Match total query count if the paper claim needs to isolate normalization.

### Medium: resume is seeded, not guaranteed bitwise deterministic

Python, NumPy, CPU Torch, and CUDA RNG states are saved. Deterministic algorithm
settings, SDPA backend, physical GPU identity, driver, and
`CUDA_VISIBLE_DEVICES` are not frozen. A proper resume test must compare an
uninterrupted run and a split/resumed run at the same update.

## Frozen-data audit result on 2026-07-29

The CPU-only command below was executed against snapshot `a7f0335`, the frozen
76/24 manifest, and the cache root used by the existing G1 runs. CUDA remained
uninitialized. The current `range_class_sampler` is **not eligible to launch**.

Observed blockers:

- all 100 cache files lack `cache_schema_version`,
  `source_manifest_sha256`, and `source_commit`, although the current cache
  builder writes those fields;
- nine of 76 train frames cannot satisfy the frozen positive quotas:
  `seq46/radar00205`, `seq46/radar00404`, `seq47/radar00094`,
  `seq47/radar00514`, `seq52/radar00202`, `seq52/radar00401`,
  `seq57/radar00205`, `seq58/radar00205`, and `seq58/radar00404` have no
  60-120 m positive cell; the last frame also has no 30-60 m positive cell;
- on the 67 otherwise sampleable frames, all frozen source quotas and the
  index-space ambiguity/shell checks pass, but all 67 frames contain jittered
  queries assigned to the wrong physical range: 2,542 of 1,072,000 queries
  (0.237%);
- 1,001 of 654,552 train target points (0.153%) fall on the opposite side of a
  30/60 m boundary when range is computed from XYZ rather than the stored range
  bin center.

The physical shell diagnostic is not a hard label assertion, but it is large:
105,751 of 502,500 shell negatives (21.0%) lie within 1 m of a target point,
versus 14,311 of 502,500 global negatives (2.85%). The shell median nearest
target distance is 1.85 m and its fifth percentile is 0.49 m. Chebyshev
distance 2-4 in RAE index space is therefore not a uniform physical shell.

The ordered digest of the 100 cache file hashes is:

```text
dd9d296cc10933fce12f4e050b4aa065f82752ab123171ec1a75e8cabbc06e4f
```

This digest identifies the audited bytes but does not recover their missing
builder commit or manifest provenance.

### Minimum revision before a new range pilot

Do not reassign a missing positive quota to another range and do not silently
drop the nine frames. Freeze a new protocol with a per-frame range-availability
mask: absent positive strata contribute no fabricated positive term, and all
loss denominators and query budgets must be defined explicitly over observed
`(range, label)` strata. Apply the same eligible-frame/query budget to the
source control if the intended claim is an isolated sampler comparison.

For boundary cells, reject and redraw the range jitter until the continuous
query remains in its assigned physical stratum. For shell negatives, replace
the fixed RAE Chebyshev shell with a metric or ray-aware exclusion whose
distance distribution is frozen and reported. Finally, either rebuild the
caches with provenance metadata or bind the ordered cache digest to every
compared run and recover provenance from an independently archived cache
report.

## Hard-failure conditions

The script exits with status 2 after writing its JSON report if any condition
below fails:

1. Frozen manifest, scene split, normalization, evaluator, loss, or trainer hash
   differs.
2. Manifest counts are not 76 train/24 validation, identities repeat, a frame
   belongs to the wrong split, development scenes overlap, or a test scene
   appears.
3. Any cache is missing, unreadable, stale, mixed-source, non-finite,
   out-of-range, or inconsistent between `target_xyz_confidence` and
   `target_rae_index`.
4. Any train frame lacks at least one unique occupied cell in 0-30, 30-60, or
   60-120 m.
5. Deterministic replay changes any sampled tensor.
6. Positive/global/shell quotas differ from 700/2500/2500 in any range.
7. A negative enters the occupied-cell 3x3x3 exclusion band.
8. A shell negative is not at Chebyshev index distance 2-4 from an occupied
   cell.
9. A jittered continuous query crosses its assigned 30 or 60 m range boundary.
10. Supplied pilot manifests differ in source commit, frozen input hashes, or
    source-file hashes.
11. A supplied run directory has inconsistent checkpoint/progress state,
    resume-contract hash, RNG payload, or formal-reference values.

Metric proximity of a nominal negative to target geometry is reported but is
not a hard failure: a nearby coordinate can still represent free space. Large
fractions within 1 m indicate that the index-space shell is physically
anisotropic and should be inspected before calling it a clean hard-negative
set.

## H200 read-only command

Use the H200 Conda environment but hide all GPUs. The command reads the 100
cache files and replays the sampler on CPU:

```bash
cd /home/wangning/Workspace/radar_cube_dense/snapshots/SOURCE_COMMIT

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=code:code/scripts

/home/wangning/miniforge3/envs/hym_radar/bin/python -u \
  code/scripts/audit_ra2_range_sampler.py \
  --data-root /home/wangning/Shared/l40s_wangning_radar/cube_dense_data/kradar_g0_cross_scene \
  --cache-root /home/wangning/Shared/l40s_wangning_radar/cube_dense_data/kradar_g0_cross_scene_cache_a7d06db \
  --manifest artifacts/g0/g0_audit_100_manifest.json \
  --scene-split artifacts/g0/g0_scene_split.json \
  --normalization /home/wangning/Shared/l40s_wangning_radar/cube_dense_runs/formal_0e5fe84_g1_recovery/g1_cube_normalization_train.json \
  --pilot-run-manifest /absolute/source_classwise/run_manifest.json \
  --pilot-run-manifest /absolute/range_class_sampler/run_manifest.json \
  --pilot-run-dir /absolute/source_classwise \
  --pilot-run-dir /absolute/range_class_sampler \
  --output /home/wangning/Shared/l40s_wangning_radar/cube_dense_runs/ra2_range_sampler_audit.json
```

For pre-training eligibility, omit all `--pilot-run-manifest` and
`--pilot-run-dir` arguments. The required pre-training result is
`audit_passed=true`, no hard failures, 76 replayed train frames, exact quotas,
zero ambiguity/shell violations, zero range-class crossings, and one uniform
cache source commit. Preserve the ordered cache digest for every later mode.

CPU-only tests:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=code:code/scripts \
  /home/wangning/miniforge3/envs/hym_radar/bin/python -m pytest -q \
  code/tests/test_ra2_range_sampler_audit.py
```
