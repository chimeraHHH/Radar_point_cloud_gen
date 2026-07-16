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

## Open risks

- The public archive has no per-radar timestamp text file; synchronization is encoded by official label mappings. Timing uncertainty must be reported rather than invented.
- Calibration files provide frame offset and planar translation; the official pipeline supplies the vertical offset from configuration. The selected vertical convention must be sensitivity-tested.
- Full sequence archives are hundreds of GB, so all acquisition must remain selective and resumable.
