# Fixed per-frame range-quota contract decision

Date: 2026-08-07

## Terminal decision

`fixed_range_quota_contract_no_go`

The hard per-frame `8000/1700/300` allocation over `[0,30)`, `[30,60)`, and
`[60,120)` m is not reusable. It is mathematically incompatible with the
frozen `CD <=0.8 m` and `outlier <=5%` capacity gates on the train cohort,
independently of model, candidate field, angular error, 5 cm packing, and
completeness.

## Bound execution

- source: `dc63dfc404d647f31409177ccca6de80726eb8d7`;
- protocol: `fixed_per_frame_range_quota_feasibility_v1`;
- host: `WHUServer-H200`;
- train frames: `76/76`;
- validation cache access: false;
- test access: false;
- elapsed: `11.2799 s`;
- report SHA-256:
  `d5d9eeaae99a97bb8d358ce2de7c6c4892e221e82b1e64eaf2897aa038cb627c`;
- time-log SHA-256:
  `4a7f5ab5ce4a5f391ae65e74ab62695e448768f8dcd5e7dbac04a23183985eb4`.

All 76 target-cache bytes and target tensors are individually bound in
`feasibility.json`. The audit accessed no Cube, CFAR, candidate field,
prediction, validation target, or test data.

## Strict result

For target radial ranges `T` and output interval `I_k`, the audit computes

```text
gap_k = inf_{r in I_k, t in T} abs(r-t)
precision_lower_bound = sum_k quota_k * gap_k / 10000
outlier_lower_bound = sum_k quota_k * 1[gap_k > 2m] / 10000.
```

The reverse triangle inequality makes both bounds independent of angle and
representation. Completeness is set to its unattainable best value of zero, so
the precision bound is also an optimistic Chamfer lower bound.

Two train frames are strictly impossible:

| frame | target counts near/mid/far | max target range | CD lower bound | outlier lower bound |
|---|---:|---:|---:|---:|
| `47:94` | `3286/2/0` | `32.6339 m` | `0.8210 m` | `3%` |
| `58:404` | `285/0/0` | `27.8774 m` | `1.3245 m` | `20%` |

Across all 76 frames, 67 have zero forced outlier lower bound, eight have 3%,
and one has 20%. The maximum optimistic Chamfer lower bound is `1.3245 m`.

## Consequences

1. The Q-Local-F0 terminal remains valid under its frozen protocol and training
   remains prohibited, but its no-go cannot be generalized to the scorer or
   the 700k candidate field.
2. No successor exporter, Q-Local rerun, or ray-hazard oracle may inherit hard
   per-frame `8000/1700/300` quotas.
3. The narrowest next falsification is `Q-Local-F0R`: keep Fresh-WCE-20, all
   700k candidate coordinates, Q0/Q1, residual, base confidence, scorer, loss,
   exact 10k count, and true 5 cm spacing fixed; change only the exporter from
   hard per-range quotas to one global capacity-constrained selection.
4. F0R must first run a zero-training GT-nearest capacity oracle on the same 12
   train-only frames. It must preserve all per-frame `CD <=0.8 m` and
   `outlier <=5%` gates and may not delete the failed frames or replace them
   with aggregate gates.
5. A variable multi-return ray-hazard representation is authorized only if the
   F0R same-candidate oracle still fails. Fixed `K=4/6` measured peaks remain
   closed by R-B1.

The future deployable exporter may not read GT or target range mass. If F0R
capacity passes, its learned scorer must allocate globally from Cube evidence,
and far-range behavior must be protected by explicit per-target-stratum recall
and completeness anti-collapse gates rather than a contradictory hard quota.
