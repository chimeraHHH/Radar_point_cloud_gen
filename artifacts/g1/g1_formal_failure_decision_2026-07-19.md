# G1 formal failure and bounded redesign decision (2026-07-19)

## Decision

The first formal G1 run fails. All six runs completed for 50 epochs on H200
with the three frozen seeds, and the scene-first paired comparison was produced.
Neither dense arm satisfies every frozen dense-versus-CFAR check, and Full-RAED
is significantly worse than RAE-Max on all preregistered Doppler-sensitive
geometry endpoints. G2/G3 therefore remain stopped.

Authoritative server comparison:

`/home/wangning/Shared/l40s_wangning_radar/cube_dense_runs/formal_a7d06db/g1_comparison_a7d06db1.json`

## Gate evidence

| Comparison | Endpoint | First | Second | Improvement or change | 95% CI / gate |
|---|---|---:|---:|---:|---|
| E1 RAE-Max vs E0 CFAR | Chamfer (m) | 10.6982 | 3.0063 | -71.9% | improvement CI `[6.1080, 9.8614]` |
| E1 RAE-Max vs E0 CFAR | F-score@1m | 0.1642 | 0.6443 | +0.4800 | improvement CI `[0.4184, 0.5389]` |
| E1 RAE-Max vs E0 CFAR | outlier >2m | 0.7991 | 0.2651 | -66.8% | fails absolute `<=0.25` by 0.0151 |
| E2 Full-RAED vs E1 | Chamfer (m) | 3.0063 | 3.8575 | +28.3% worse | degradation CI `[+5.8%, +52.3%]` |
| E2 Full-RAED vs E1 | far completeness (m) | 5.4513 | 18.1320 | +232.6% worse | improvement CI `[-18.2513, -8.2308]` |
| E2 Full-RAED vs E1 | far F-score@1m | 0.1081 | 0.0257 | -76.2% | improvement CI `[-0.1251, -0.0402]` |

The parameter-parity check passes: RAE-Max has 125,769 parameters and the
original Full-RAED model has 126,273, a 0.4007% increase.

## Diagnosis

The dense decoder is useful but misses the preregistered absolute outlier gate
by 1.51 percentage points. This threshold is not changed. The stronger failure
is the Full-RAED representation:

- Full-RAED reaches lower final training loss than RAE-Max for every seed, so
  the failure is not lack of optimization progress.
- Full-RAED validation occupancy loss rises sharply late in training, while its
  selected epochs are earlier and its far-range geometry remains much worse.
- The original encoder replaces the deterministic Doppler maximum with an
  unconstrained learned `64 -> 8` pointwise projection. On only 76 training
  frames, that projection can fit spectral patterns while discarding the stable
  high-energy geometry path.

This supports a bounded representation/regularization diagnosis. It does not
support a claim that the full spectrum helps geometry.

## One allowed redesign

The only recovery changes the Full-RAED input projection to

```text
Full-RAED feature = Project(RAE-Max) + ZeroInitResidualProject(64-bin spectrum).
```

The RAE-Max path, spatial U-Net, loss, point count, normalization, split,
seeds, epochs, optimizer, model selection, metrics, and all gate thresholds
remain fixed. The residual branch is zero initialized, so paired E1 and E2 are
functionally identical at initialization; full spectral information can only
add a learned residual. The added parameter ratio remains below 1%.

Both E1 and redesigned E2 must be rerun under the same new source commit. The
original six runs remain archived and are not overwritten. Test data remains
untouched.

## Stop rule

If this one bounded redesign does not pass the unchanged G1 gate, the project
permanently removes the claim that full Doppler spectra improve dense geometry
on the current cohort and does not start the planned G2/G3 chain to mask the
failure. The paper must then be narrowed or the representation/data scale must
be redesigned as a new research branch.
