# Execution Checklist

## P0 / G0

- [x] Verify GPU server, CUDA environment and storage placement.
- [x] Reject TruckScenes PCD-only data as a full-Cube source.
- [x] Locate the official K-Radar full DRAE archive and enumerate synchronized fields.
- [x] Establish selective HTTP Range access without downloading 235 GB sequence archives.
- [x] Download and CRC-verify the eight-frame G0 sample on the GPU server.
- [x] Verify DRAE shape, physical axes, dtype, units and Doppler convention.
- [x] Verify label-defined radar/LiDAR pairing, calibration and odometry timing.
- [x] Deskew OS2-64 with its per-point nanosecond timestamp and interpolated ego pose.
- [x] Verify CFAR point spatial/Doppler round trip into Cube bins.
- [x] Define and audit the radar-observable LiDAR confidence target.
- [x] Download and independently CRC-verify the frozen 100-frame cross-scene cohort.
- [ ] Complete the 100-frame visual/numeric CUDA audit across isolated scenes.
- [x] Freeze scene-level train/validation/test split and minimal cache schema.
- [ ] Decide G0 using direct evidence.

## P1-P6

- [ ] Execute the remaining gates in `PLAN.md` after G0 passes.
