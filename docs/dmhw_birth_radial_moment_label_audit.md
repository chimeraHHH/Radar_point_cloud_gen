# D-MHW G-RM Label-Only Audit Protocol

## Scope

This protocol audits whether K-Radar can legally supply a geometry/track-derived
radial first-moment target for the 3,000 D-MHW birth points. It does not train a
model, build a dense-target cache, read a future Radar Cube, or authorize a
future Doppler-distribution claim.

The audit covers every legal direct three-horizon anchor:

| Partition | Required anchors |
|---|---:|
| train | 740 |
| validation | 160 |

The three frozen horizons are `0.5`, `1.5`, and `2.5 s`.

## Legal inputs

The builder may read only:

- the development temporal manifest;
- future OS2-64 LiDAR XYZ as label geometry;
- target and immediately preceding K-Radar box labels;
- target and preceding ego poses already stored in the manifest;
- radar/LiDAR calibration;
- static radar axes and the train-frozen P5 sign audit.

It must report exactly zero future-Cube paths and bytes and exactly zero test
records. Future LiDAR and annotations are training labels, so the allowed
method statement remains `radar-only inference with LiDAR/annotation-derived
training supervision`.

## Geometry denominator

The denominator is every finite future LiDAR point transformed into the target
radar frame and lying inside the static range, azimuth, and elevation axes. No
future Cube, CFAR response, target cache, or GT-guided candidate selector may
filter this denominator.

Invalid points remain in the denominator. In particular, background points
are invalid because this dataset slice does not independently verify that they
are static. Ego pose must not be used to manufacture background labels.

## Conservative validity rules

A point is valid only when all conditions hold:

1. It lies in exactly one target box and at least `0.1 m` inside every face.
2. The target track ID occurs exactly once.
3. The same track ID occurs exactly once in the immediately preceding label.
4. The class is unchanged.
5. Its rigidly mapped source point lies in exactly one source box, matches the
   same track, and is at least `0.1 m` inside every face.
6. Target and source poses reproduce the manifest radar transform to `1e-8`.

Duplicate IDs, missing tracks, class changes, target/source overlaps,
target/source boundary points, degenerate rays, and unverified background are
invalid.

## Radial target

For valid target point `y_t`, its local target-box coordinate is mapped into
the source box. The source point is rotated into the target radar axes with the
pose-derived adjacent-frame rotation, without removing ego translation. The
sensor-relative radial rate is

```text
range_rate = dot(y_t - R_target_from_source y_source, unit(y_t)) / dt
```

The scalar target follows the P5 train-frozen sign:

```text
positive_ego -> radial_target = -range_rate
negative_ego -> radial_target = +range_rate
```

It is wrapped onto the frozen 64-bin Doppler period. This supplies only a
wrapped scalar first moment, not spectral width, multimodality, sidelobes,
material response, or calibrated uncertainty.

Every valid ledger row must contain source and target frames, `dt`, track and
class, pose hash, source/target label hashes, LiDAR hash, calibration hash,
unwrapped and wrapped targets, and provenance
`tracked_rigid_dynamic|tracked_rigid_static_like`.

## Frozen gates

All gates are mandatory:

- exact `740/160` train/validation anchors;
- all anchor-horizon target usages audited;
- no frame errors;
- future-Cube path count and bytes read are both zero;
- test records read are zero;
- every valid label has complete provenance;
- overall valid coverage is at least `75%`;
- valid coverage at every horizon is at least `60%`;
- dynamic tracked labels span at least `10` independent windows;
- P5 agreement MAE is at most `0.5 m/s`.

P5 agreement is computed per tracked box: first take the median circular
point-to-box target difference within the box, then take the mean across boxes.
The sign audit is allowed only as `sign_only_calibration` when its validation
physics-prior gate remains failed.

Passing enables only:

> Birth points may use a 64-bin circular parameterization supervised through a
> geometry/track-derived radial first moment.

It does not enable a true or calibrated future Doppler-distribution claim.
Failure is a G-RM `no-go`; gates must not be changed after observing validation.

## H200 command

Run only on physical H200 GPU 0 or 2 with `hym_radar`; never expose physical
GPU 1.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=code

/home/wangning/miniforge3/envs/hym_radar/bin/python -m pytest -q \
  code/tests/test_dmhw_birth_radial_moment.py

/home/wangning/miniforge3/envs/hym_radar/bin/python -u \
  code/scripts/audit_dmhw_birth_radial_moment.py \
  --manifest /home/wangning/Shared/l40s_wangning_radar/cube_dense_runs/formal_ae535ae_parallel_stage0/g4_temporal_manifest_radar_frame.json \
  --data-root /home/wangning/Shared/l40s_wangning_radar/cube_dense_data/kradar_temporal_w48 \
  --static-doppler-audit /home/wangning/Shared/l40s_wangning_radar/cube_dense_runs/formal_0c86c1ad/static_doppler_snr/audit_q0p0.json \
  --output /home/wangning/Shared/l40s_wangning_radar/cube_dense_runs/g_rm_LABEL_SOURCE_COMMIT/g_rm_label_audit.json \
  --ledger-output /home/wangning/Shared/l40s_wangning_radar/cube_dense_runs/g_rm_LABEL_SOURCE_COMMIT/g_rm_valid_labels.jsonl.gz \
  --source-commit LABEL_SOURCE_COMMIT
```

No 500-update attribute training is authorized by this protocol.
