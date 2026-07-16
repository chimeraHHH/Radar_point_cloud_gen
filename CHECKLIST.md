# Execution Checklist

## P0 / G0

- [x] Verify GPU server, CUDA environment and storage placement.
- [x] Reject TruckScenes PCD-only data as a full-Cube source.
- [x] Locate the official K-Radar full DRAE archive and enumerate synchronized fields.
- [x] Establish selective HTTP Range access without downloading 235 GB sequence archives.
- [ ] Download and CRC-verify the eight-frame G0 sample on the GPU server.
- [ ] Verify DRAE shape, physical axes, dtype, units and Doppler convention.
- [ ] Verify label-defined radar/LiDAR pairing, calibration and odometry timing.
- [ ] Verify CFAR point spatial/Doppler round trip into Cube bins.
- [ ] Define and audit the radar-observable LiDAR confidence target.
- [ ] Freeze scene-level train/validation/test split and minimal cache schema.
- [ ] Decide G0 using direct evidence.

## P1-P6

- [ ] Execute the remaining gates in `PLAN.md` after G0 passes.
