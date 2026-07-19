# G1 Bounded Recovery Final Decision

## Immutable evidence

- Training source: `0e5fe8430892d57996ed26fa18f233cfa5e0c79b`.
- Seeds: `20260716`, `20260717`, `20260718`.
- Evaluation: eight validation scenes, scene-first paired bootstrap with 10,000
  resamples.
- Server artifact:
  `/home/wangning/Shared/l40s_wangning_radar/cube_dense_runs/formal_0e5fe84_g1_recovery/g1_comparison_0e5fe843.json`.
- Archived artifact: `g1_bounded_recovery_comparison_0e5fe843.json`.
- SHA-256:
  `ef8eb17bf6e9f72a9ea45800ef30e1ff363e830b17cfa3760c214041472afd17`.
- Frozen safeguards: outlier fraction at most `25%`; Full-RAED Chamfer
  degradation upper bound at most `2%`; parameter increase at most `1%`.

## Result

| Comparison | Chamfer | Outlier fraction | Far completeness | Formal consequence |
|---|---:|---:|---:|---|
| CFAR | `10.6982 m` | `79.909%` | `5.7723 m` | Diagnostic baseline |
| RAE-Max | `2.9306 m` | `25.697%` | `6.3150 m` | Strong geometry gain, but failed the fixed `25%` outlier gate |
| Full-RAED | `3.1024 m` | `26.896%` | `7.3789 m` | Failed both geometry safeguards |

Full-RAED increased Chamfer by `5.86%` relative to RAE-Max; the scene-first
95% interval was `+0.78%` to `+14.69%`. Its far-completeness mean distance
increased by `17.31%`, and neither far endpoint had a confident improvement.
The parameter parity check passed (`+0.413%`), so capacity mismatch does not
explain the failed gate.

The authoritative decision is:

```text
dense_beats_cfar = false
rae_max_beats_cfar = false
full_raed_beats_cfar = false
full_raed_doppler_sensitive_gain = false
full_raed_chamfer_nondegradation = false
g1_passed = false
```

## Routing decision

1. Original G1 is closed as failed. The claim that early Full-RAED fusion
   improves dense geometry is rejected for this protocol.
2. The original `0e5fe84` G2/G3 chain is not unlocked and cannot be used as
   evidence for C2 or C3.
3. G1B remains an independent representation study with unchanged safeguards.
   It may create a separately named geometry parent only after Stage A and the
   three-seed Stage B both pass.
4. RaLD-anchor RH1/RH2 may use a successful G1B candidate only as
   `independent_g1b_parent`. This is a late-fusion method branch, not a recovery
   of original G1.
5. If G1B has no Stage A survivor or fails Stage B, the current geometry family
   closes and no downstream training is authorized.
