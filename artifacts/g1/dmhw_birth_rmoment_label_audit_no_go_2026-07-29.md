# D-MHW G-RM label-only audit: no-go

Date: 2026-07-29

Source commit: `e161be7aa3579d8bd4e8aaaea49ddf08f61aabce`

H200 snapshot:
`/home/wangning/Workspace/radar_cube_dense/snapshots/e161be7`

Run directory:
`/home/wangning/Shared/l40s_wangning_radar/cube_dense_runs/g_rm_e161be7_label_only`

## Verification

- Physical GPU visibility: H200 GPU 0 only; GPU 1 was never exposed.
- Conda environment: `hym_radar`.
- Targeted H200 integration tests: `31 passed in 1.49s`.
- Real audit: 740 train and 160 validation anchors, 2,700 horizon
  occurrences, and 1,800 unique target frames.
- Frame errors: 0.
- Future-Cube paths/bytes read: 0/0.
- Test records read: 0.
- Pose-composition maximum residual: 0.

## Frozen-gate result

| Gate | Result | Decision |
|---|---:|---|
| Overall valid coverage >=75% | 4.487% | fail |
| 0.5 s coverage >=60% | 4.731% | fail |
| 1.5 s coverage >=60% | 4.301% | fail |
| 2.5 s coverage >=60% | 4.434% | fail |
| Dynamic tracked windows >=10 | 40 | pass |
| P5 box-center range-rate MAE <=0.5 m/s | 0.1422 m/s | pass |
| Every valid label has provenance | 1,086,733/1,086,733 | pass |

Train coverage was 4.788%; validation coverage was 3.124%. The dominant
anchor-weighted invalid reason was unverified background
(`33,906,568` points). This is expected under the frozen conservative rule:
background geometry cannot receive a static ego-derived label without an
independent static/dynamic verification source.

The coordinate, sign, pose, track, and provenance contracts are internally
consistent. The failure is label support: tracked rigid boxes cover too little
of the future geometry to supervise the full birth branch.

## Artifacts

- Full machine-readable report:
  `artifacts/g1/dmhw_birth_rmoment_label_audit_e161be7.json`
- Full report SHA256:
  `cc65539938300d213e6b655bd44b6e5cf8e81a36a3ec7568dca2335633984d93`
- Remote valid-label ledger:
  `g_rm_e161be7_label_only/g_rm_valid_labels.jsonl.gz`
- Ledger SHA256:
  `bc2a0639a6b89bf2b97bce5b012fa8e71f5178f20abdd4ed1ec38563a16fdea8`
- Ledger size: 72 MB compressed.

## Decision

G-RM is `no-go` for the full D-MHW birth branch. The valid tracked subset may
remain a diagnostic target, but it cannot authorize birth radial-moment
training or a birth Doppler claim. Do not run the 500-update attribute stage
and do not relax the frozen coverage denominator after seeing this result.

The safe D-MHW attribute fallback remains A-NC: birth points emit XYZ and
confidence with `doppler_valid=false`; any 64-bin Doppler claim is restricted
to persistent points.
