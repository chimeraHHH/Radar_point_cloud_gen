# G0 Formal 100-Frame Failure Decision

Date: 2026-07-18

## Outcome

The first formal cross-scene G0 audit completed all 100 frames on an NVIDIA
H200 NVL with zero frame errors, but `gate_pass=false`. Nine of eleven checks
passed. The failed checks were:

- `selected_deskew_not_worse_than_none`: start-referenced deskew minus no
  deskew mean margin was `-0.0245556` over all frames and `-0.0258711` over the
  76 training frames.
- `observable_target_nonempty`: the implementation required every frame to
  exceed a `0.005` observable fraction even though the check and protocol only
  require a nonempty target. Sequences 51 and 52 had fractions `0.0026333` and
  `0.0023721`, respectively, but still retained positive target points.

The audit covered 45 train/validation sequences with 76 train and 24 validation
frames. Exact CFAR round trip was 1.0, correct-minus-mirrored azimuth margin was
0.3981, and mean observable fraction was 0.2087 +/- 0.1083.

## Decision

1. Freeze `lidar_time_reference=none` from training-only evidence. Training
   mean margins were `-0.8165` (none), `-0.8424` (start), `-0.8233` (center),
   and `-0.9614` (end). Validation did not select the reference.
2. Make `observable_target_nonempty` test the literal condition
   `observable_count > 0` for every frame. Continue reporting observable
   fraction and retain the existing cross-frame stability gate.
3. Rerun the complete 100-frame audit and rebuild dense-target caches under a
   new source commit. G1 remains blocked until the repaired aggregate passes.
4. Report the unresolved OS2 scan-origin convention as a limitation. Do not
   claim that deskew improves K-Radar alignment from the sequence-1 pilot.

## Provenance

- First formal source commit: `253f26c9bd03e47dcb8b1b15a5eaf9d1859a92b3`
- Audit manifest SHA-256:
  `645307a8bae351db51b55128043dae69bce5b928169d5fa25c1b9c55083de4e4`
- Scene split SHA-256:
  `61596bd50ce0bdab633c9ff0ec5ab5148c2c78a10dfc06a71d0d6f0a6c9427cc`
- Device: NVIDIA H200 NVL
- Torch: 2.12.1+cu130
