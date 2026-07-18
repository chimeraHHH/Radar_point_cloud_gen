# Static Doppler failure recovery (2026-07-18)

## Parent failure

The original 100-frame static-background audit completed with no frame errors
but failed its frozen validation check. Train selected `positive_ego` with a
`0.096906 m/s` margin, while validation circular error was `1.020193 m/s`,
worse than the unchanged circular-random median baseline of `0.966296 m/s`.

Original report SHA-256:
`08159dea592d7de72828677982e313c4def9a1831cde4bf2c918ec36c1e12360`.

## Bounded recovery

The recovery kept the manifest, box margin, range cutoff, three wrapped
hypotheses, random baseline, train/validation split, and `0.05 m/s` train margin
fixed. Only the per-frame background SNR quantile varied over the preregistered
grid. Selection minimized train error among candidates meeting the train
margin; validation did not participate.

| SNR quantile | Train hypothesis | Train error | Train margin | Validation error | Validation beats random |
|---:|---|---:|---:|---:|---|
| 0.00 | positive_ego | 1.020373 | 0.096906 | 1.020193 | no |
| 0.50 | positive_ego | 1.022370 | 0.034516 | 1.043263 | no |
| 0.75 | positive_ego | 1.023505 | 0.003184 | 1.023335 | no |
| 0.90 | positive_ego | 1.043236 | 0.043846 | 1.089280 | no |

The train-only rule selected quantile `0.0`, which then failed validation. SNR
filtering therefore does not recover a stable analytic static convention on
this cohort.

## Route decision

- E5 physics-mixture is omitted from the formal G2 run and cannot support a
  physics-prior claim.
- E3 scalar and E4 distribution continue with the same failed audit recorded in
  provenance, because their comparison does not use the analytic static prior.
- G2 passes only if E4 satisfies the frozen distribution-versus-scalar gates.
- G3 proceeds from E4 only after that distribution gate passes.
- The full spectrum remains the source of truth; no sign, offset, SNR threshold,
  or validation criterion is tuned further.

Machine-readable selection:
`artifacts/g2/static_doppler_snr_selection_0c86c1ad.json`.

Authoritative server directory:
`/home/wangning/Shared/l40s_wangning_radar/cube_dense_runs/formal_0c86c1ad/static_doppler_snr`.
