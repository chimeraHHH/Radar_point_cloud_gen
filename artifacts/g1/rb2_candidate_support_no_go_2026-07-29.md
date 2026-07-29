# R-B2 Cube-Only Candidate Support No-Go

Date: 2026-07-29

## Frozen Preflight

- Protocol: `rb2_cube_only_voxel_slot_tiny_v1`
- Source: `8fe928069169624f2db40fad3d5daba86cb6740f`
- Frame: frozen lexicographically first train frame, `seq01/radar00232`
- Device: physical H200 GPU2
- Inference inputs: current Cube and frozen RAE axes only
- Target use: support reporting after candidate generation only
- Training started: false

The Cube-only candidate builder emitted the exact frozen bank:

| Range | Candidate voxels |
|---|---:|
| 0-30 m | 16,000 |
| 30-60 m | 3,400 |
| 60-120 m | 600 |
| Total | 20,000 |

All 20,000 immutable voxel IDs were unique and every structural count check
passed.

## Support Gate

| Check | Result | Gate | Decision |
|---|---:|---:|---|
| Target-occupied voxel recall | 10.9763% (`416/3790`) | >=20% | Failed |
| Confidence-weighted target coverage | 31.6908% | >=30% | Passed |

The hard decision is `no_go_cube_only_candidate_support`. The model,
optimizer, and one-frame geometry gate were never instantiated or evaluated.

## Failure Scope

This result closes only the current `20k` candidate activation rule based on
max-over-D Cube score plus a radius-four fixed Cartesian neighborhood. It does
not close the fixed `0.40 m x 4-slot` representation: the independent GT-aided
structural oracle already showed that the representation has sufficient
capacity when relevant voxels are activated.

One bounded, preregistered support sweep is authorized before closing this
activation family:

- fixed Cube-only score definitions;
- fixed `20k/40k/80k` candidate banks;
- no target-conditioned candidate generation or selection;
- one-frame preflight followed by a 76-train-frame audit only for candidates
  that pass the frozen one-frame gate.

Training remains prohibited until a single frozen Cube-only configuration
reaches both `>=20%` occupied-voxel recall and `>=30%` confidence coverage.

## Integrity

- `config.json` SHA-256:
  `2b5d16700975c7f0852b60eaece9a1a1e3191cc1509ef1b04c1c16ba7206f908`
- `decision.json` SHA-256:
  `0602bc2155323684dab4700d4c7b7f2f37257bf442c5fdfac69872e139bcde3e`
- Cube SHA-256:
  `b21355705b672ec348f5c802fbed8b288077f58ad31d8a2675874f26cd4e4b2e`
- Cache SHA-256:
  `76380164d2460a5de33cb06e2779a98aae0f9260fe6b77ef92b2332f12ba8fae`
- Candidate-ID SHA-256:
  `247a3ea5bf3ce5175a40818aae1f8f47d72d8d647bddd39ed8210bfe660b0544`
