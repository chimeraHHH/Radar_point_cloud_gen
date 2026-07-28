# G1D failure diagnosis and parallel route freeze

Date: 2026-07-28

Scope: raw Full-RAED Cube to exactly 10,000 `XYZ + circular Doppler
distribution + confidence` points. This document freezes the next parallel
experiments after the long G1D run showed no useful late-stage gain.

## 1. Evidence-based diagnosis

The epoch-15 to epoch-125 trajectory is not consistent with insufficient
training:

| Metric | Epoch 15 | Epoch 125 | Change |
|---|---:|---:|---:|
| median Chamfer | 4.482 m | 5.407 m | +20.6% |
| median completeness | 3.581 m | 3.503 m | -0.078 m |
| mean outlier | 17.90% | 24.68% | +6.78 pp |
| mean duplicate | 26.94% | 30.41% | +3.47 pp |
| mean absolute offset | 1.030 bins | 4.294 bins | 4.17x |
| condition-shuffle Chamfer | -0.017% | +0.032% | effectively zero |

The most defensible mechanism is:

1. The frozen current-frame proposal pool has weak far-range support.
2. Hard top-k selection does not receive geometry gradients.
3. The selected zero-offset centers become worse after epoch 15.
4. The decoder increasingly uses large offsets to recover completeness.
5. Weakly weighted repulsion and offset penalties do not prevent crowding.
6. Local spectrum, energy, and measurement proposals remain available from
   the correct frame when only the global condition is shuffled.

This explains the observed combination: nearly flat completeness, worse
Chamfer/outlier/duplicates, saturated offsets, and negligible causal use of
the global Full-RAED condition.

The source-level audit also found an objective mismatch. Training minimizes a
squared distance excess while evaluation gates the fraction of points beyond
2 m. A model can lower a few extreme distances while moving more points just
past the 2 m threshold.

## 2. Immediate causal experiments

These inference-only experiments are higher information gain than another
long G1D training run.

### D1: offset dose response

- Freeze checkpoint, top-k indices, Cube, and evaluator.
- Evaluate joint coarse/final offset scale
  `alpha in {0, 0.25, 0.5, 0.75, 1.0}` on all 24 validation frames.
- Require exact 10k outputs and all 23 far-target frames.
- Authorize a bounded-offset repair only if a smaller alpha improves
  scene-first Chamfer with paired confidence interval below zero, reduces
  outlier and duplicate by at least 3 pp each, and worsens completeness by no
  more than 0.10 m.

### D2: measured-path versus condition-path intervention

Run the frozen 2x2 intervention:

| measured/local Cube | global condition Cube |
|---|---|
| current | current |
| current | wrong scene |
| wrong scene | current |
| wrong scene | wrong scene |

Use one frozen cross-scene derangement and identical queries. If changing the
condition changes Chamfer by less than 1% while changing the measured path
changes it by more than 10%, the bypass diagnosis is confirmed and G1D
training remains closed.

### D3: ranking-source intervention

On the same 32k pool compare occupancy+confidence, occupancy-only,
confidence-only, integrated energy, and a GT-distance diagnostic oracle.
Only an oracle improvement of at least 10% in Chamfer or completeness, with
outlier and duplicate no worse than +2 pp, can authorize a learned ranking
repair.

Combined planned budget for D1-D3: below 0.8 H200 GPU-hour.

## 3. Route A: RaLD-WCE

RaLD-WCE is the primary RaLD-derived route. It addresses both missing support
and the local-condition bypass.

```text
Full-RAED Cube
  -> 336 global radar tokens
  -> static + dynamic latent set
  -> repeated latent self-attention / radar cross-attention
  -> coordinate-only continuous occupancy decoder
  -> Q0 wide initial queries
  -> occupancy-dependent Q1 refinement
  -> true 5 cm Euclidean exclusion + frozen range quotas
  -> exactly 10,000 XYZ
  -> post-selection Doppler distribution and confidence heads
```

Frozen anti-bypass rules:

- Geometry queries may use only coordinate features and condition latents.
- Local spectrum, energy, CFAR score, and candidate source are forbidden
  before geometric selection.
- Wrong-condition evaluation must reuse identical query coordinates.
- The Doppler head is post-selection and cannot alter geometry in Stage 0.
- Confidence is calibrated occupancy probability, not an unconstrained score.

RaLD components that transfer are its hybrid set latent, repeated radar
cross-attention, coordinate-only implicit decoder, wide random/radar-guided
queries, and occupancy-dependent second query pass. The official released path
is intensity-only, variable-cardinality XYZ on short-range ColoRadar; its
range scale, class ratios, thresholding, and output contract cannot be copied
directly.

Primary sources:

- [RaLD paper](https://ojs.aaai.org/index.php/AAAI/article/download/38946/42908)
- [RaLD code at audited commit](https://github.com/MetaIoT-WHU/RaLD/tree/ffec4b41241391734b1eda5c093de843c909eb8e)
- [3DShape2VecSet](https://arxiv.org/abs/2301.11445)
- [ImplicitO](https://openaccess.thecvf.com/content/CVPR2023/papers/Agro_Implicit_Occupancy_Flow_Fields_for_Perception_and_Prediction_in_Self-Driving_CVPR_2023_paper.pdf)
- [UnO](https://openaccess.thecvf.com/content/CVPR2024/papers/Agro_UnO_Unsupervised_Occupancy_Fields_for_Perception_and_Forecasting_CVPR_2024_paper.pdf)
- [DIO](https://openaccess.thecvf.com/content/CVPR2025/papers/Diehl_DIO_Decomposable_Implicit_4D_Occupancy-Flow_World_Model_CVPR_2025_paper.pdf)

Stage-0 gates:

- exact 10k on all 24 frames;
- median Chamfer at most 2.50 m;
- median completeness at most 2.4946142 m;
- mean outlier at most 25%;
- mean duplicate at most 10%;
- corrected far completeness at most 46.9407304 m with sample count 23;
- fixed-query condition shuffle degrades Chamfer by at least 1%;
- matched condition beats wrong condition in at least 75% of frames.

Failure of the current R-A0 initial-query diagnostic cannot close the complete
RaLD-wide family because R-A0 has no learned occupancy-dependent second pass.

## 4. Route B: polar rectified flow

P-RF is independent of G1D, its 32k pool, selectors, anchors, and checkpoints.

- Source measure: exactly 10,000 fixed scrambled-Sobol particles.
- State: range empirical-CDF/logit plus normalized azimuth/elevation logits.
- Target: confidence-weighted, first-surface polar lifting with global 5 cm
  exclusion and no replacement.
- Pairing: deterministic per-frame hierarchical optimal transport; cross-scene
  minibatch OT is forbidden.
- Network: Full-RAED condition latents and chunked time-conditioned particle
  velocity; no 10k-by-10k self-attention.
- Inference: four-step Euler ODE, no top-k, NMS, copied fill, or best-of-k.
- Endpoint: 64-bin circular Doppler logits and confidence.

Primary sources:

- [Flow Matching](https://openreview.net/forum?id=PqvMRDCJT9t)
- [Rectified Flow](https://openreview.net/forum?id=XVjTT1nw5z)
- [Conditional Flow Matching code](https://github.com/atong01/conditional-flow-matching)
- [C2OT](https://openaccess.thecvf.com/content/ICCV2025/html/Cheng_The_Curse_of_Conditions_Analyzing_and_Improving_Optimal_Transport_for_ICCV_2025_paper.html)
- [PUFM](https://arxiv.org/abs/2501.15286)
- [RangeLDM](https://arxiv.org/abs/2403.10094)
- [RadarOcc](https://arxiv.org/abs/2405.14014)
- [SDDiff](https://arxiv.org/abs/2506.16936)

Stage-0 gates:

- exact finite 10k points on all 24 frames;
- mean duplicate at most 5%, worst frame at most 10%;
- median Chamfer no worse than corrected G1D control, 4.4609 m;
- median completeness at most 2.4946 m;
- corrected far completeness at most 37.5526 m on 23/23 frames;
- range-mass total variation at most 0.15;
- mean outlier at most 20%;
- correct Cube improves screen Chamfer by at least 1%;
- Doppler circular-W1 improves over the train marginal by at least 10%;
- confidence ECE at most 0.10.

## 5. Route C: T-WC

The existing temporal model is not yet a forced temporal generator. Its
history gate can be zero, its training path uses ego-only warp, and rollout
reads the real Cube at every step.

T-WC splits output into persistent and birth points:

- Persistent points must carry a valid history source identity.
- Radial displacement from circular Doppler and ego transform is analytic and
  mandatory.
- The network predicts tangential flow and only a bounded radial correction.
- Persistent generation requires current-Cube/history correlation and has no
  current-only bypass.
- Birth points cover newly observed regions.

Wrong conditions include cross-scene history, time reversal, ego-transform
permutation, Doppler zeroing, sign reversal, and circular bin rolls. Online
enhancement with a real current Cube and forecast without future Cubes are
separate protocols.

Primary sources:

- [DoppDrive](https://openaccess.thecvf.com/content/ICCV2025/html/Haitman_DoppDrive_Doppler-Driven_Temporal_Aggregation_for_Improved_Radar_Object_Detection_ICCV_2025_paper.html)
- [RaFlow](https://arxiv.org/abs/2203.01137)
- [RadarMP](https://arxiv.org/abs/2511.12117)
- [CMFlow](https://arxiv.org/abs/2303.00462)
- [Scheduled Sampling](https://proceedings.neurips.cc/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html)
- [Scheduled Sampling critique](https://arxiv.org/abs/1511.05101)

Minimum mechanism gates:

- 100% of persistent outputs have valid history IDs;
- wrong history degrades Chamfer or forward-Cube error by at least 3%;
- at least two Doppler interventions worsen dynamic inverse-radial error by
  at least 15%;
- matched current-frame geometry is within 2% of the best baseline;
- completeness or far completeness improves by at least 10%;
- 5-step and 10-step forecast Chamfer and radial error improve at least 10%
  over ego-only/Doppler-only baselines;
- 25-step failures remain in the denominator.

## 6. Corrected G1T result

The corrected no-train temporal proposal evaluation covered 352 frames and all
312 frames containing far targets. Internal relative checks passed, but the
effect size is negligible:

| Scene-first metric | T0 current | T1 ego | T2 Doppler |
|---|---:|---:|---:|
| Chamfer | 15.9242 m | 15.9159 m | 15.9199 m |
| completeness | 1.07038 m | 1.07045 m | 1.07015 m |
| far completeness | 2.74590 m | 2.74732 m | 2.74639 m |
| outlier | 87.572% | 87.580% | 87.591% |

T2 versus T1 improves completeness by only 0.028% and far completeness by
0.034%, while worsening Chamfer by 0.025% and outlier by 0.011 pp. This is a
support-mechanism diagnostic, not a usable point generator and not sufficient
to authorize learned temporal training.

The result artifact has `completed=false` because the source script encoded
three successful forbidden-access assertions as false and then applied
`all(checks.values())`. The metrics remain usable; the completion-check
semantics are repaired separately without changing the observed promotion
logic.

## 7. Current execution state

- G1D remains frozen and is running to its preregistered 150-epoch endpoint.
- G1G maximum-target H200 preflight passed on 18,836 targets with 8.30 GB peak
  reserved memory; its 20-epoch formal run is active on H200 GPU 0.
- Repaired G1R and R-A0 code passed 36 targeted and 311 full H200 tests.
- G1R single-frame and R-A0 maximum-frame preflights remain queued.
- Independent implementation skeletons for RaLD-WCE, P-RF, and T-WC are being
  built in disjoint new files and require mainline review plus H200 tests
  before any scientific run.

## 8. Novelty boundary

The project must not claim the first use of radar diffusion, radar flow,
multi-frame radar aggregation, raw radar-to-point generation, or joint spatial
and Doppler modeling.

The defensible target is the conjunction:

> Full-RAED Cube-conditioned, fixed-cardinality dense radar state generation
> with exactly 10,000 XYZ points, per-point circular Doppler distribution,
> confidence, explicit far-range support, causal condition-use tests, and a
> temporal physical cycle.

The literature search did not find this exact conjunction, but that statement
is scoped to the primary papers and official repositories reviewed here; it is
not a universal priority claim.
