# Partial 45-frame G0 Audit

This is an interim diagnostic artifact, not the final G0 decision. It covers the
largest contiguous prefix available when the K-Radar download was interrupted:
45 frames from 21 isolated sequences (39 train and 6 validation). The frozen
G0 requirement remains 100 frames.

## Aggregate result

- Successful frames: 45/45; frame errors: 0.
- Gate checks: 11/11 passed.
- Ego speed range: 0.337-34.690 m/s.
- OS2/label maximum timestamp delta: 0 ms.
- Official odometry nearest-sample maximum delta: 0 ms.
- Minimum exact CFAR Cube-bin round trip: 100%.
- Radar-observable surface fraction: 0.212 +/- 0.097.
- Correct-minus-mirrored azimuth margin: +0.364.
- Start-reference deskew-minus-no-deskew margin: +0.019 mean, +0.018 median.

## Deskew diagnostic

The scan-start timestamp convention remains the best fixed convention across
the partial cohort:

| Reference | Mean margin vs no deskew | Median | Positive frames |
|---|---:|---:|---:|
| start | +0.019 | +0.018 | 57.8% |
| center | -0.003 | +0.003 | 53.3% |
| end | -0.106 | -0.117 | 15.6% |

The mean start-reference advantage is small and 19/45 individual frames are
negative, so the final 100-frame audit must re-evaluate this gate. The current
evidence rejects a global center/end shift but does not claim that deskew
improves every frame.

## Visual inspection

The three archived panels include the original sequence-1 smoke frame, a
high-speed frame, and the worst deskew-margin frame. They show consistent axes,
no azimuth mirroring, aligned LiDAR surfaces and observable targets, finite
Doppler fields, and no plotting overlap. The sparse target in the worst frame
is a confidence/observability case rather than an axis flip.

Authoritative machine-readable evidence is in `g0_audit.json`; the generated
human-readable report is `g0_audit.md`.
