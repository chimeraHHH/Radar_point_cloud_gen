# RaLD and temporal radar prior-art audit

> Evidence scope: primary papers and official code linked below.  
> Purpose: constrain novelty claims and identify reusable mechanisms.

## RaLD code-level boundary

The official RaLD source at commit
[`ffec4b4`](https://github.com/MetaIoT-WHU/RaLD/tree/ffec4b41241391734b1eda5c093de843c909eb8e)
implements radar-conditioned latent generation and an arbitrary-query occupancy
decoder. Although its preprocessing can derive a peak-Doppler product, the
released generation configuration uses the intensity channel and does not emit
per-point Doppler. It is also a single-frame, variable-cardinality XYZ
generator rather than an exact-count temporal radar-state model.

Reusable parts are the mixed set latents, repeated radar cross-attention,
continuous coordinate queries, occupancy/query decoder, and EDM training
skeleton. They do not solve this repository's exact-10k selection, true 5 cm
spacing, generated Doppler, confidence calibration, or rollout contracts.

## Closest task families

| Work | Existing capability | Boundary relative to this project |
|---|---|---|
| [RaLD](https://ojs.aaai.org/index.php/AAAI/article/view/38946) | Single-frame radar-spectrum-conditioned dense XYZ generation | No per-point Doppler output or temporal state generation in the released path |
| [SDDiff](https://www.ijcai.org/proceedings/2025/979) | Joint spatial-Doppler radar refinement | Invalidates broad "first spatial-Doppler generation" claims |
| [RadarMP](https://github.com/chengrui7/RadarMP) | Consecutive radar tesseracts, point generation, and scene flow | Invalidates "all prior enhancement is single-frame" |
| [DoppDrive](https://openaccess.thecvf.com/content/ICCV2025/html/Haitman_DoppDrive_Doppler-Driven_Temporal_Aggregation_for_Improved_Radar_Object_Detection_ICCV_2025_paper.html) | Doppler propagation and age-gated history aggregation | Strong temporal control; it aggregates history rather than generating a complete current dense state |
| [RaFlow](https://arxiv.org/abs/2107.09720) | Doppler-supervised radar scene flow | Invalidates broad Doppler-displacement novelty |
| [Radar-Diffusion](https://arxiv.org/abs/2403.08460) and adjacent 4D radar generation work | Radar data generation | Invalidates broad "first radar generation" wording without a narrower state/output definition |

The search found no exact match that simultaneously performs history-radar-
conditioned future dense point-set generation, emits per-point Doppler, models
birth/death, and demonstrates stable autoregressive rollout. This is a scoped
search result, not proof of universal absence.

## Safe positioning

The defensible target is:

> History-radar-conditioned generation of a current or future dense
> radar-observable `XYZ + circular Doppler distribution + confidence` state,
> with explicit birth/death, ego-aware position-Doppler bidirectional
> consistency, and multi-step stability.

Unsafe statements include "first Cube-to-dense," "first Doppler densification,"
"first joint position and Doppler," "all prior methods are single-frame," and
"no work predicts future radar points." The paper must compare against
single-frame Cube generation, Doppler scene flow, temporal aggregation, and
future-radar prediction as distinct adjacent families.
