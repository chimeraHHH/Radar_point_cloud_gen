# Intake State Audit

Date: 2026-07-16

## Trusted facts

- GPU host: `WHUServer-L40S`, eight NVIDIA L40S GPUs, CUDA-enabled `hym_radar` environment.
- Large data and run outputs must use `/storage/data/metaiot_data/wangning_radar`; root and SSD mounts are nearly full.
- Existing TruckScenes data contains radar PCD sweeps, not full RAED tensors.
- `/storage/ssd/metaiot_guest/lxl_data/kradar` contains converted LiDAR/camera/pose assets but no full Doppler tensor.
- The official K-Radar NAS exposes sequence archives through a read-only Synology File Station account.
- Sequence 1 archive size is 234,832,394,392 bytes. Its `radar_tesseract` directory contains 629 `arrDREA` MAT files of about 293 MB each, plus LiDAR, labels, calibration and timing metadata.
- Official K-Radar code commit used for schema interpretation: `f90c28993d1757ce0236632b4d912c92a14976b2`.

## Rejected anchors

- The empty `/home/metaiot_guest/data/k-radar` directory is not a dataset.
- Doppler-collapsed `radar_zyx_cube` is insufficient for the Full-RAED thesis.
- Prior PCD-only temporal generation results remain useful for P4 baselines but cannot prove G0-G3.

## Current anchor

P0/G0 starts from official K-Radar sequence 1 full `radar_tesseract` data, synchronized labels and LiDAR, official physical-axis resources and sequence-1 odometry. The first audit sample uses eight label-defined pairs (`00033_00001` through `00040_00008`).

## Direct G0 evidence

- All eight selected DRAE tensors and paired OS1/OS2 point clouds passed archive CRC verification.
- On disk, `arrDREA` is `float64` with shape `(64, 256, 37, 107)`; the canonical loader returns contiguous `(D,R,A,E)=(64,256,107,37)` and casts to `float32` on CUDA for computation.
- The label timestamp equals the selected OS2 timestamp exactly in all eight frames. OS1 is asynchronous by 23.984 ms on average; odometry's nearest sample is within 28.072 ms and is interpolated.
- OS2 is a rotating scan with per-point nanosecond offsets spanning about 100 ms. At the observed 13.8 m/s ego speed, ignoring this field permits about 1.38 m of scan distortion. Start-referenced GPU deskew improves the radar threshold margin by 0.061 on average.
- CA-CFAR points return to their exact DRAE bins in 100% of cases with zero power lookup error.
- The observable-target protocol retains 26.14% +/- 2.02% of first-surface LiDAR points at confidence >= 0.5. Correct azimuth beats a mirrored null by 0.334 log-threshold margin.
- High-power Doppler is empirically zero-centered. The audit therefore preserves the full spectrum and does not apply an unsupported ego-radial correction.
- The official File Station listing currently publishes 53/58 sequence archives; sequences 15, 16, 17, 19 and 20 are absent and are explicitly excluded rather than assigned inferred metadata.
- The frozen 53-sequence split contains 22,419 train, 4,836 validation and 4,790 test frames (69.961%/15.091%/14.948%) with zero sequence overlap and required attribute coverage in every partition.
- The 100-frame cross-scene audit manifest covers all 45 train/validation sequences with 76 train and 24 validation frames; test is untouched and all selected timestamps have one-to-one official odometry support.

Evidence: `artifacts/g0/g0_audit_8frame.json`, `artifacts/g0/g0_audit_8frame.md`, `artifacts/g0/g0_audit_seq01_00033.png`, `artifacts/g0/g0_archive_availability.json`, `artifacts/g0/g0_scene_split.json`, and `artifacts/g0/g0_audit_100_manifest.json`.

## Open risks

- The public archive has no per-radar timestamp text file; synchronization is encoded by official label mappings. Timing uncertainty must be reported rather than invented.
- Calibration files provide frame offset and planar translation; the official pipeline supplies the vertical offset from configuration. The selected vertical convention must be sensitivity-tested.
- Full sequence archives are hundreds of GB, so all acquisition must remain selective and resumable.
- The eight-frame result is a sequence-1 feasibility audit, not final cross-scene evidence. G0 remains open until the selected 100-frame CUDA audit completes and passes.
