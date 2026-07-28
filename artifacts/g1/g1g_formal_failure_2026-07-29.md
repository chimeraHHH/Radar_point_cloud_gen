# G1G formal Stage-0 decision

> Date: 2026-07-29
>
> Source: `c2a0ccb38b27c17ff4eaa7acc151189fd8b06958`
>
> Protocol: `g1g_condition_exclusive_hierarchy_stage0_v1`
>
> Test accessed: false

## Decision

G1G completed the frozen 20-epoch, one-seed H200 Stage-0 and failed promotion.
The route is closed without an epoch extension or post-hoc gate change.

The static and dynamic anti-bypass checks passed: the measured Cube had no
pre-allocation gradient, the condition Cube drove allocation, and every
allocation block received finite nonzero geometry gradients. This establishes
that the intended information path was executable. It does not establish that
the model learned a useful condition-dependent allocation.

## Frozen result

| Metric | Result | Stage-0 requirement | Passed |
|---|---:|---:|---|
| Condition-shuffle Chamfer degradation | `-0.1211%` | `>=1%` | No |
| Mean duplicate fraction at 5 cm | `70.8108%` | `<=15%` | No |
| Median completeness | `1.2086 m` | `<=2.4946 m` | Yes |
| Mean outlier fraction at 2 m | `84.6854%` | `<=25%` | No |
| Corrected far completeness | `6.5635 m`, 23/23 frames | no worse than `46.9407 m` | Yes |
| Mean unique center fraction at 5 cm | `95.5467%` | `>=80%` abandonment check | Yes |

The hierarchy reached broad spatial coverage, including the far range, but
bought that coverage by placing most generated points off the target surface
and by collapsing many children into duplicate locations. The negative
condition-shuffle value means the wrong cross-scene Cube was marginally better
on average; therefore nonzero gradients did not translate into reliable
condition use.

## Evidence

- Best epoch: `20`
- Best checkpoint SHA-256:
  `96afa3ebeb956575b513e215024272dda6e11aa75c18503655b542cc97038db9`
- Archived decision:
  `artifacts/g1/g1g_formal_stage0_decision_c2a0ccb.json`
- Archived decision SHA-256:
  `430a238531f45d5cbbe213007075334bf1516360ba7dc717284e8caa15fa4dde`
- Corrected evaluator SHA-256:
  `e1f970c1e827c7801c7c9018378603b89a7709b0b4bf261e740eab1575631e68`

## Consequence

G1G remains a negative structural control: condition-exclusive global
allocation plus bounded center-child decoding does not by itself solve
fixed-count Cube-to-dense geometry. No claim of successful Full-RAED
conditioning or usable dense geometry may cite this run. The next independent
routes must change the output representation or transport mechanism rather
than extend G1G training.
