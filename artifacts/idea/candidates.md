# Cube-to-dense candidate frontier

> 2026-08-07 successor addendum: Q-Local-F0 reached
> `qlocal_f0_capacity_no_go` under its frozen hard-range exporter, and a
> separate 76-frame proof reached `fixed_range_quota_contract_no_go`. The live
> route is Q-Local-F0R: preserve the fresh WCE 700k field and local Full-RAED
> scorer/loss, remove only hard per-frame range quotas, and first run the
> source-bound global exact-10k capacity oracle in
> `docs/qlocal_f0r_global_export_capacity_protocol.md`. A variable multi-return
> renewal-hazard field is authorized only after F0R scientific no-go; sparse
> ray-range transport remains third priority. None has accessed validation or
> test.

> Frozen on 2026-07-28. Test access is false. The existing G1D v2 run continues
> unchanged and does not select settings for these candidates.

> Corrected-evaluator addendum, 2026-07-28: the original far-completeness gate
> and every archived far value below used a censored metric that omitted
> target-bearing frames without same-bin predictions. They remain historical
> preregistration records, not valid current evidence. New runs must use
> `dense_geometry.py` from source `1561ac3` or later and bind all relative gates
> to `g1d_epoch15_corrected_geometry_control_1561ac3.json`. The absolute far
> gate is suspended until all parent arms are re-evaluated; it is not relaxed.

## Unified Stage-0 contract

Every learned route uses the existing 76/24 scene-held-out split, 10,000 output
points, G1D geometry metrics, scene-first reporting, and H200 GPU 0 or 2. A
candidate is not promoted because of lower loss or a single favorable metric.
It must expose its named mechanism with a paired control.

The authoritative final geometry gate remains:

| Metric | Required |
|---|---:|
| median Chamfer | `<= 2.50 m` |
| mean outlier fraction at 2 m | `<= 25%` |
| median completeness | `<= 0.65 m` |
| mean far completeness | `<= 8.0 m` |
| mean duplicate fraction at 5 cm | `<= 10%` |
| output count | exactly `10,000` |
| named learned-condition shuffle degradation | `>= 1%` |

Stage-0 may terminate a route early. It cannot relax this final gate.

## Candidate table

| ID | Primary mechanism | Stage-0 | Anti-win condition | Promotion rule |
|---|---|---|---|---|
| G1F | Candidate support oracle plus differentiable balanced transport allocation | F0 oracle on the frozen 32k pool; only then a 10-epoch soft selector | GT is used only for an unattainable diagnostic upper bound; learned inference sees no GT; all logits need finite nonzero gradients | F0 must show that one fixed-count subset can satisfy the geometry support gates; F1 must improve completeness and duplicates without worsening outlier above 25% |
| G1G | Condition-exclusive global allocation plus bounded hierarchical local patches | 20-epoch, one-seed allocator; 2.5k centers x 4 children | No local Cube spectrum, energy, or coordinate-neighborhood feature before center allocation; local refinement has a frozen offset radius | condition shuffle `>=1%`, duplicate `<=15%` at Stage-0, and at least 30% completeness improvement over matched G1D epoch-15 control |
| G1T | Ego/Doppler-warped history as a measurement proposal prior | no-train current-only, ego-union, Doppler-union comparison | Identical 10k export, current-Cube rescore, no LiDAR/GT selection, duplicate-aware deduplication fixed before evaluation | Doppler-union must beat ego-union on completeness and far completeness without more than 2 percentage points outlier or duplicate degradation |
| G1H | Frozen G1B equal-count tail replacement | no-train low-support isolated-tail replacement | Remove and add exactly the same number of points; no confidence masking or output-count reduction | Full geometry gate must pass; otherwise audit-only |

## Ordering and resource policy

1. Run G1F-F0 first because a failed candidate-support oracle closes both G1F-F1
   and any selector-only repair on that pool.
2. Run G1T concurrently because it is no-train and tests a disjoint source of
   geometric support.
3. Run G1H as a low-cost conservative control.
4. Start G1G training after its H200 preflight and when an allowed H200 has
   sufficient memory; it does not wait for G1D scientific selection.

No route may use physical GPU 1. Existing unrelated jobs are not interrupted.

## Decision matrix

| Outcome | Decision |
|---|---|
| G1F-F0 fails support | abandon selector-only repair; prioritize G1G representation and G1T support |
| G1F-F0 passes, F1 fails | candidate support is adequate but the selected transport objective is insufficient; do not tune beyond the frozen Stage-0 budget |
| G1G shuffle fails | architecture still permits condition bypass; close the route regardless of geometry |
| G1T Doppler does not beat ego | history cannot justify a Doppler-specific temporal mechanism on this cohort |
| G1H passes all gates | retain as a strong non-generative control; still require G1G or a new learned family for a method claim |
| More than one route passes | select by the complete unified gate, then resource cost; do not combine mechanisms until each has an independent ablation |

## Observed Stage-0 results

### G1F-F0: failed, selector-only route closed

Source `ca60d76174b62dd7364285655b7ae25f92700330` evaluated all 24
validation frames with the explicitly unattainable GT coverage oracle:

| Metric | Oracle result | Gate |
|---|---:|---:|
| median Chamfer | `2.8863 m` | `<=2.50 m` |
| mean outlier fraction | `24.853%` | `<=25%` |
| median completeness | `1.6513 m` | `<=0.65 m` |
| mean far completeness | `8.6533 m` | `<=8.0 m` |
| mean duplicate fraction | `14.015%` | `<=10%` |

Only the outlier gate passed. The full 32k pool had mean far-range GT recall of
only `17.80%` at 2 m, so the failure is not attributable to the learned hard
top-k selector alone. G1F-F1 balanced-transport training is not authorized.
Artifact: `artifacts/g1/g1f_f0_ca60d76.json`, SHA-256
`edaf94fa57b324abc3fe853438ed9146671ee8e73494e62339dff47514911320`.

The G1F geometry failure remains sufficient to close selector-only repair on
the frozen 32k pool because Chamfer, completeness, and duplicates also failed.
Its archived far-completeness value is censored and must not be cited.

### G1G: failed, condition-exclusive hierarchy closed

Source `c2a0ccb38b27c17ff4eaa7acc151189fd8b06958` completed the frozen
20-epoch H200 Stage-0. Static and dynamic anti-bypass checks passed, but the
scientific promotion gate failed:

| Metric | Result | Stage-0 requirement |
|---|---:|---:|
| Condition-shuffle Chamfer degradation | `-0.1211%` | `>=1%` |
| Mean duplicate fraction at 5 cm | `70.8108%` | `<=15%` |
| Median completeness | `1.2086 m` | `<=2.4946 m` |
| Mean outlier fraction at 2 m | `84.6854%` | `<=25%` |
| Corrected far completeness | `6.5635 m`, 23/23 frames | no worse than `46.9407 m` |

The route achieved broad coverage but bought it through off-surface points and
child collapse. The wrong cross-scene Cube was marginally better on average,
so nonzero allocation gradients did not establish useful condition dependence.
G1G is closed without extending training or relaxing its gates. Artifact:
`artifacts/g1/g1g_formal_stage0_decision_c2a0ccb.json`, SHA-256
`430a238531f45d5cbbe213007075334bf1516360ba7dc717284e8caa15fa4dde`.

### G1T: corrected result does not authorize a learned follow-up

The corrected no-train comparison found less than `0.05%` relative geometry
change between current-only, ego-union, and Doppler-union arms. All three arms
had about `15.9 m` Chamfer and `87.6%` outliers. The old proposal-union
mechanism is therefore closed as a learned parent even though its internal
relative boolean passed. An independently defined temporal model may still be
tested, but it cannot cite G1T as evidence that the current proposal mechanism
is useful.

### G1D endpoint: failed, bounded repairs closed

The source-`4c6150cd` run completed all 150 epochs, but the selected checkpoint
remained epoch 15. Corrected endpoint metrics degraded versus that checkpoint:
median Chamfer `4.4609 -> 5.4221 m`, mean outlier
`17.8875% -> 24.8396%`, corrected far completeness
`46.9407 -> 48.5173 m`, and duplicate `26.9721% -> 30.7279%`.

The source-`22f4aa6` D1/D2/D3 diagnostic covered all 24 validation frames and
23 far-target frames:

- scaling residual offsets below `1.0` always worsened Chamfer and
  completeness;
- wrong global condition changed mean Chamfer by `+1.05%`, with a scene-first
  interval spanning zero;
- occupancy-only ranking improved mean Chamfer `12.03%` but worsened outliers
  by `3.00 pp`;
- even validation-GT ranking on the same 32k pool failed the absolute geometry
  gate and worsened outliers by `2.42 pp`.

G1D is closed together with offset clipping and same-pool learned ranking.
Raw artifact: `artifacts/g1/g1d_epoch150_d1_d2_d3_22f4aa6.json`, SHA-256
`bae5d699a4c3d7be14a446c79aa4293b37509d1467b7b00516b48a3e32066e8c`.
