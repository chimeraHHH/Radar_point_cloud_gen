# Cube-to-dense literature and mechanism survey

> Frozen on 2026-07-28 before the next Stage-0 results are observed.

## Search question

The relevant question is no longer whether multi-frame radar enhancement exists.
It does. The useful question is which published mechanisms can address the
specific failure observed in this repository:

1. poor allocation of a fixed 10,000-point budget;
2. excessive duplicates or outliers despite acceptable partial geometry;
3. negligible dependence on the global Full-RAED condition;
4. no generated per-point Doppler distribution or confidence;
5. no validated geometry support on which to build temporal consistency.

The search covered direct radar-to-point reconstruction, Doppler-aware temporal
radar, point completion, sparse set allocation, implicit/range representations,
and LiDAR sequence generation. Absence claims below are limited to the cited
papers and repositories inspected by 2026-07-28.

## Direct radar reconstruction and generation

| Work | Input and output | Mechanism worth borrowing | Boundary for this project |
|---|---|---|---|
| [RaLD](https://github.com/MetaIoT-WHU/RaLD) | Single radar spectrum to 10k XYZ | `512 x 32` target posterior, mixed occupied/empty implicit queries, radar-conditioned latent EDM | Official main path is intensity-only and coordinate-only; it does not generate pointwise Doppler or temporal state |
| [DenserRadar](https://github.com/hanzy21/DenserRadar) | Full single-frame radar tensor to dense occupancy/XYZ | Doppler bins retained as input channels; dense spatial supervision | No pointwise Doppler output or temporal generation |
| [SDDiff](https://www.ijcai.org/proceedings/2025/979) | Single-frame RAED-derived representation to dense point state | Iterative refinement and Doppler-consistency objective | Uses a peak-Doppler reduction rather than a calibrated circular Doppler distribution |
| [Radar-Mamba](https://palm.seu.edu.cn/zhangml/files/MM%2725.pdf) | Current plus two previous radar frames/features to enhanced dense radar points | Explicit spatiotemporal feature fusion while retaining Doppler features | Directly invalidates the broad claim that radar enhancement is uniformly single-frame |
| [RadarMP](https://github.com/chengrui7/RadarMP) | Two consecutive K-Radar tesseracts to radar points plus pointwise scene flow | Joint point generation and motion prediction; Doppler-guided temporal consistency | Closest task neighbor, but its output target is point geometry plus scene flow rather than a calibrated pointwise Doppler distribution/confidence state |

## Temporal radar and motion mechanisms

| Work | Mechanism | What can be reused | What remains different |
|---|---|---|---|
| [DoppDrive](https://yuvalhg.github.io/DoppDrive/) | Doppler-driven radial shift and adaptive history aggregation | No-train temporal proposal prior and a strong aggregation baseline | Reuses measured historical points; it does not generate a new dense current-frame state |
| [RaFlow](https://github.com/Toytiny/RaFlow) | Radar scene flow with radial-displacement consistency | Motion/velocity consistency diagnostics | Scene-flow estimation, not Cube-conditioned dense generation |
| [CMFlow](https://github.com/Toytiny/CMFlow) | Cross-modal supervision for radar scene flow | Dynamic/static separation and motion supervision | Requires a different training target and does not solve fixed-count allocation |
| [DoppDrive paper](https://arxiv.org/abs/2508.12330) | Doppler-corrected temporal radar aggregation | Ego-only versus Doppler-warp controls | Geometry gains can come from aggregation alone, so any generative temporal claim must beat this control |

## Point allocation and completion mechanisms

| Family | Source | Mechanism relevant to the failure |
|---|---|---|
| Hierarchical point completion | [AdaPoinTr](https://arxiv.org/abs/2301.04545), [SeedFormer](https://arxiv.org/abs/2207.10315), [SnowflakeNet](https://arxiv.org/abs/2108.04444) | Allocate coarse semantic seeds first, then expand each seed into a bounded local patch; this separates global point-budget allocation from local coordinate refinement |
| Balanced set transport | Entropic optimal transport and differentiable Sinkhorn matching | Gives all candidate scores a training signal and can explicitly constrain range/density mass, unlike hard top-k followed by unrelated random-query occupancy BCE |
| Ray/range factorization | [RangeLDM](https://github.com/WoodwindHu/RangeLDM), [LiDPM](https://github.com/astra-vision/LiDPM) | Converts unconstrained 3D point allocation into ray/range occupancy plus local decoding, which naturally exposes long-range coverage |
| Sparse implicit fields | [OctFusion](https://github.com/octree-nn/octfusion) | Coarse spatial allocation followed by local implicit refinement | Useful only if radar observability and fixed-count export remain explicit |

## Failure-to-mechanism map

| Repository evidence | Most plausible cause | Supported intervention |
|---|---|---|
| G1B reaches `2.0251 m` Chamfer but has `28.885%` outliers | tail precision failure, not global support failure | equal-count tail replacement control |
| G1D lowers outliers but completeness remains around `3.6 m` and duplicates approach `30%` | hard point-budget allocation and local offset crowding | candidate oracle plus balanced transport selector |
| G1D condition shuffle remains near zero | direct local Cube query state bypasses the global condition | condition-exclusive global allocator |
| G1D offsets grow while validation geometry worsens | local refinement is compensating for wrong allocation | bounded hierarchical patch refinement |
| Temporal Doppler warp previously beats ego-only at long horizons | history contains useful geometric support | temporal proposal prior, evaluated separately from generation |

## Correct novelty boundary

The following statement is **not allowed**:

> Existing radar enhancement is single-frame, and no work uses previous radar
> frames to improve the current point cloud.

Radar-Mamba, RadarMP, DREAM-PCD-style reconstruction, and DoppDrive invalidate
that broad statement.

The scoped target claim is:

> We study radar-observable state generation from Full-RAED measurements:
> fixed-count dense geometry, a calibrated circular Doppler distribution, and
> confidence are generated jointly and constrained by Cube reprojection and
> displacement-Doppler consistency. History is an optional proposal prior, while
> the current Cube remains the source of current-frame occupancy and velocity
> evidence.

Within the reviewed scope, no inspected method combines all of:

- history-aware Full-RAED conditioning;
- generation of a new fixed-count dense 3D radar set;
- a per-point circular Doppler distribution and confidence;
- Cube-to-point-to-Cube closure;
- a displacement-Doppler temporal constraint.

This is a scoped literature conclusion, not a universal absence proof.

## Resulting experiment frontier

The literature supports four distinct falsification routes:

1. **G1F:** test whether the 32k measurement-grounded candidate pool has enough
   support, then learn balanced allocation with differentiable transport.
2. **G1G:** force Full-RAED global condition to allocate hierarchical patch
   centers before any local Cube lookup.
3. **G1T:** measure the support added by ego- and Doppler-warped history before
   training a temporal generator.
4. **G1H:** test whether G1B's failure is confined to a replaceable outlier tail.

G1F is first because it distinguishes representation failure from selector
failure. G1G and G1T are independent mechanism tests. G1H is a conservative
control and cannot become the primary contribution without passing every frozen
geometry and condition gate.
