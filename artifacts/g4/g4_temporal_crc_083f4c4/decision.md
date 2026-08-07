# G4 temporal data CRC decision

Status: `g4_temporal_crc_passed`

## Frozen evidence

- H200 host: `WHUServer-H200`, user `wangning`
- verifier run label/source checkout: `083f4c4`
- verifier source blob:
  `code/scripts/verify_kradar_g0_download.py@85ae907a1a411789b543475b6b5a21838c959211`
- temporal manifest source commit:
  `a7d06db1abcc69c20dfed381f0c2909b1a89f026`
- temporal manifest SHA-256:
  `110391d79922226ccd4145e7dfec47ed9c39c0e74e1009fce588a057b0fbe8d2`
- archived verification SHA-256:
  `0ed646c841ccfd88863cae6333c7b065016425c96d96eb97c53b941f01470fda`
- archived log SHA-256:
  `9c6bcd5281a0007b424baf2b9799de8a4b308c53febca4661d06df3b810c0df4`
- server output:
  `/home/wangning/Shared/l40s_wangning_radar/cube_dense_runs/g4_temporal_crc_083f4c4`

## Result

The required full-summary verification completed with exit status zero:

```text
passed = true
expected_frame_count = 2160
cube_count = 2160
lidar_count = 2160
label_count = 2160
verified_file_count = 6660
sequence_count = 45
member_set_matches = 45/45
invalid_files = []
errors = []
pending_sequences = []
active_sequences = []
```

Each sequence contained exactly 148 expected manifest members, with no missing,
unexpected, or duplicate member. The manifest partitions the 2,160 frames into
1,776 train and 384 validation frames and estimates 600.773 GiB of Cube plus
LiDAR payload. The verifier used retry-round-2 download summaries and accepted
all completed sequence sets.

Runtime was 1:56:21 wall time, 497.77 seconds user time, 330.73 seconds system
time, and 159,940 KiB maximum resident memory. The check is I/O and CPU bound;
it is a data-integrity gate rather than a model or GPU-performance experiment.

## Decision boundary

This result unlocks only the use of the frozen G4 temporal files after the
single-frame parent family is selected. It proves byte-level manifest agreement,
required file counts, and complete sequence membership. It does not prove label
correctness, calibration quality, temporal-model quality, Doppler generation,
generalization, or test performance. Validation files were checksum-verified as
data assets; no validation metric, model fitting, or route selection was run.
