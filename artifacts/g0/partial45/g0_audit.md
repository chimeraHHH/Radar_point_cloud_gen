# K-Radar G0 Audit

Frames completed: 45
Gate pass: **True**

## Gate Checks

| Check | Result |
|---|---|
| `required_frame_count` | PASS |
| `schema_consistent` | PASS |
| `os2_label_sync_le_1ms` | PASS |
| `odometry_support_le_60ms` | PASS |
| `lidar_point_time_present` | PASS |
| `selected_deskew_not_worse_than_none` | PASS |
| `cfar_exact_roundtrip` | PASS |
| `observable_target_nonempty` | PASS |
| `observable_target_stable` | PASS |
| `correct_azimuth_beats_mirror` | PASS |
| `doppler_hypothesis_beats_random` | PASS |

## Aggregate Evidence

- Sequences covered: 21
- Partition frame counts: {'train': 39, 'validation': 6}
- Ego-speed range: 0.337 to 34.690 m/s
- OS2/label maximum timestamp delta: 0.000000 ms
- OS1/label mean absolute delta: 24.742 ms
- Odometry nearest-sample maximum delta: 0.000 ms
- Minimum exact CFAR round-trip rate: 1.000000
- Observable surface fraction: 0.2118 +/- 0.0967
- Correct-minus-mirrored angular margin: 0.3635
- Selected deskew-minus-no-deskew margin: 0.0194

The primary geometry target uses OS2-64 because its timestamp is the label timestamp. OS1-128 is retained as an asynchronous auxiliary source.
