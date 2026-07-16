# Cube-to-Dense Data Protocol

## Source and provenance

- Dataset: official K-Radar full DRAE archive.
- Schema reference: K-Radar commit `f90c28993d1757ce0236632b4d912c92a14976b2`.
- Heavy data and caches live under `/storage/data/metaiot_data/wangning_radar` on `WHUServer-L40S`.
- Local execution is limited to source editing, version control, and artifact transport. Numerical audits run on CUDA on the GPU server.

The official File Station listing currently exposes 53 of the documented 58
sequence ZIPs. Sequences `15,16,17,19,20` are absent from `/KRadar` rather than
failed downloads, so they are explicitly excluded instead of assigned guessed
metadata. The availability snapshot is stored in
`artifacts/g0/g0_archive_availability.json`.

## Scene-isolated split

The union of official labelled frames is repartitioned at whole-sequence
granularity. A deterministic mixed-integer program minimizes frame and
road/time/weather distribution error under a hard frame-ratio tolerance and
requires every attribute represented by at least three sequences to occur in
all three partitions.

| Partition | Sequences | Frames | Achieved ratio |
|---|---:|---:|---:|
| Train | 37 | 22,419 | 69.961% |
| Validation | 8 | 4,836 | 15.091% |
| Test | 8 | 4,790 | 14.948% |

The split passes frame conservation, attribute coverage, ratio tolerance and
zero sequence-overlap checks. Adjacent frames cannot cross partitions. The
complete immutable mapping is `artifacts/g0/g0_scene_split.json`.

The 100-frame G0 cohort samples 76 train and 24 validation frames over all 45
non-test sequences. Every sequence contributes at least two evenly spaced
interior frames; test remains untouched. Official KITTI-format odometry rows
are attached one-to-one to label-defined OS2 timestamps. The cohort manifest is
`artifacts/g0/g0_audit_100_manifest.json`.

## Canonical tensor

The MATLAB variable is stored as `arrDREA` with shape `(64,256,37,107)` and dtype `float64`. The loader transposes it into canonical `(D,R,A,E)=(64,256,107,37)`, applies the official angular-axis reversal, makes the result contiguous, and casts to `float32` on CUDA.

| Axis | Bins | Minimum | Maximum | Median step |
|---|---:|---:|---:|---:|
| Doppler | 64 | -1.932591 m/s | 1.872198 m/s | 0.060393 m/s |
| Range | 256 | 0.000000 m | 118.037109 m | 0.462891 m |
| Azimuth | 107 | -53 deg | 53 deg | 1 deg |
| Elevation | 37 | -18 deg | 18 deg | 1 deg |

Raw power spans several orders of magnitude. Models must consume a documented robust/log normalization, while reports retain raw-power statistics for auditability.

## Synchronization and deskew

The label header maps radar, OS2-64, camera, and OS1-128 indices. For the eight-frame audit, the label timestamp equals the selected OS2 timestamp exactly. OS1 differs by 23.984 ms on average and is auxiliary unless explicitly motion-compensated.

OS2 PCD fields are:

```text
x y z intensity t reflectivity ring ambient range
```

The unsigned `t` field is a nanosecond offset from scan start and spans approximately 100 ms. The primary target therefore:

1. treats the OS2 timestamp as scan start;
2. converts each point time to `timestamp + t * 1e-9`;
3. interpolates odometry translation and trajectory-derived heading;
4. transforms each point into the radar frame at the Cube timestamp;
5. applies the official LiDAR-to-radar translation `[-2.54, 0.30, 0.70] m`.

The start-time hypothesis improves local radar threshold margin by 0.061 over no deskew on average. Center and end hypotheses are retained as audit controls.

## CFAR and round trip

The audit uses a 3D CA-CFAR detector on Doppler-max power with a 9-cell training kernel, 3-cell guard kernel, 3-cell local-maximum kernel, and false-alarm rate `1e-3`. Each retained spatial peak carries its exact Doppler argmax.

Round-trip validation converts every `XYZ+Doppler` point back to nearest physical DRAE bins and re-queries power. The eight-frame audit obtains 100% exact bin recovery and zero relative lookup error.

## Radar-observable LiDAR target

The geometry target is not all LiDAR points. For each deskewed OS2 point:

1. reject points outside radar range/azimuth/elevation support;
2. retain the first surface within 1 m for each angular bin;
3. query the 3x3x3 local radar peak and CA-CFAR noise estimate;
4. compute `margin = log((peak+1)/(alpha*noise+1))`;
5. compute `confidence = sigmoid(margin/0.5)`.

`confidence >= 0.5` defines a radar-observable positive. Across eight frames, 26.14% +/- 2.02% of first-surface points are positive. The correct angular convention exceeds a mirrored-azimuth null by 0.334 margin.

## Doppler convention

The official Doppler axis is used without reordering. Its circular alias period is approximately 3.865181 m/s. High-power peaks in the initial audit are predominantly zero-centered and do not support blindly subtracting an ego-radial term. The project therefore preserves the full 64-bin spectrum, uses distributional supervision, and treats compensation/alias unwrapping as an explicit later ablation.

## Minimal cache

Each audited frame stores a compressed NPZ with:

- `cfar_xyzd_power_snr`: CFAR geometry, Doppler, power, and local SNR;
- `cfar_drae_index`: exact Cube lookup indices;
- `target_xyz_confidence`: deskewed first-surface LiDAR and observability confidence;
- `target_rae_index`: target spatial lookup indices;
- `ego_velocity_xyz_mps`, `ego_speed_mps`, and `ego_yaw_rate_radps`.

The full DRAE tensor remains the source of truth and is not duplicated in the cache.

## Evidence boundary

The eight-frame sequence-1 audit passes all numerical checks and the
scene-isolated split is frozen. The selected 100-frame cross-scene cohort is
currently being downloaded and CRC-verified; therefore G0 remains open until
that complete CUDA audit passes.
