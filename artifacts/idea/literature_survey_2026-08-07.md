# Cube-to-dense literature and mechanism rescan

> Frozen on 2026-08-07 before Q-Local-F0 implementation or execution. The
> search used primary papers and official repositories. Test access is false.

## Decision question

The fresh replay reproduces the actionable R-A1 failure without reproducing the
deleted checkpoint identity. On the same 700,000 candidate coordinates, the
deployed confidence gives mean Chamfer/outlier `4.0261 m/31.7854%`, while a
non-deployable nearest-target score gives `0.6375 m/2.2771%`. Candidate support
is therefore strong on average; radar-only geometric-risk prediction is not.

The rescan asks which mechanism can test that missing link without relabeling a
closed occupancy, arbitrary-query, fixed-neighborhood, hierarchy, or latent
diffusion route.

## Direct radar reconstruction

| Work | Relevant mechanism | Boundary in this repository |
|---|---|---|
| [DenserRadar](https://arxiv.org/abs/2405.05131) and its [official repository](https://github.com/hanzy21/DenserRadar) | Single Full-Doppler tensor, dense spatial occupancy supervision | Closest to the already closed early Full-RAED occupancy family; no generated pointwise Doppler distribution or temporal state |
| [RaLD](https://arxiv.org/abs/2511.07067) and [official source](https://github.com/MetaIoT-WHU/RaLD/tree/ffec4b41241391734b1eda5c093de843c909eb8e) | Mixed set latents, coordinate-only implicit queries, radar-conditioned latent diffusion, wide inference queries | Its central mechanisms were separately tested by G1C/G1D/G1E/R-A1. More RaLD depth or queries is not a new route; coordinate-to-global-condition attention remains a useful control |
| [SDDiff](https://www.ijcai.org/proceedings/2025/979) and [official repository](https://github.com/StellarEsti/SDDiff) | Directional spatial-Doppler refinement | Strong novelty collision for broad spatial-Doppler claims; released code is not yet an implementation to import |
| [RaUF](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_RaUF_Learning_the_Spatial_Uncertainty_Field_of_Radar_CVPR_2026_paper.pdf) and [project repository](https://github.com/MetaIoT-WHU/RaUF) | Polar anisotropic uncertainty, spatial-Doppler interaction, likelihood-based dense reconstruction | Strongest direct collision with generic uncertainty/confidence novelty. Q-Local is a bounded geometry-parent experiment, not the paper's final novelty claim |
| [SD4R](https://arxiv.org/abs/2602.20653) and [official repository](https://github.com/lancelot0805/SD4R) | Foreground virtual points from sparse radar points for detection | Different input and foreground-only downstream objective; useful baseline context, not a Cube-to-full-scene state generator |

## Temporal and Doppler-aware radar

| Work | Relevant mechanism | Required treatment |
|---|---|---|
| [DoppDrive](https://openaccess.thecvf.com/content/ICCV2025/html/Haitman_DoppDrive_Doppler-Driven_Temporal_Aggregation_for_Improved_Radar_Object_Detection_ICCV_2025_paper.html) | Doppler radial propagation and age-gated history aggregation | Mandatory temporal aggregation control; it does not generate a new current dense state |
| [RadarMP](https://ojs.aaai.org/index.php/AAAI/article/view/37323) and [official source](https://github.com/chengrui7/RadarMP) | Consecutive tesseracts, Doppler spectrum embedding, point generation and scene flow | Invalidates a broad temporal-absence claim; its DopplerMLP is a later engineering reference |
| [Radar-Mamba](https://palm.seu.edu.cn/zhangml/files/MM%2725.pdf) | Current plus historical radar features for point enhancement | Another temporal enhancement baseline, not evidence for a single-frame parent |

## Ranking, distribution, and set allocation

| Family | Primary source | Transferable mechanism | Limit |
|---|---|---|---|
| Quality-aligned ranking | [Rank-DETR](https://proceedings.neurips.cc/paper_files/paper/2023/hash/34074479ee2186a9f236b8fd03635372-Abstract-Conference.html), [official source](https://github.com/LeapLabTHU/Rank-DETR) | Align the deployment ranking score with localization quality rather than class occupancy | Detection boxes are not fixed-count radar candidates |
| Distributional localization | [D-FINE](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6cf58a87e3097e7d1f9be3e8693a93de-Abstract-Conference.html), [official source](https://github.com/Peterande/D-FINE) | Predict an ordered distance distribution and rank by expected risk | The transfer is the output parameterization, not the detector architecture |
| Differentiable top-k | [SOFT top-k](https://arxiv.org/abs/2002.06504) | Entropic-OT relaxation of cardinality selection | Cannot repair an incorrect score, range quota, or 5 cm conflict by itself |
| Sparse/partial transport | [OT-M](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Optimal_Transport_Minimization_Crowd_Localization_on_Density_Maps_for_Semi-Supervised_CVPR_2023_paper.pdf), [official source](https://github.com/Elin24/OT-M) | Couple density mass and point allocation rather than score points independently | A dense `700k x target` transport matrix is prohibited; this is a fallback only |
| Ray-native generation | [Neural LiDAR Fields](https://arxiv.org/abs/2305.01643), [RangeLDM](https://arxiv.org/abs/2403.10094) | Ordered returns, ray drop, and range-native generation | A fixed `K=4/6` measured-peak construction already failed R-B1; any fallback must predict variable returns rather than reuse that rule |

## Mechanism decision

### Primary: Q-Local-F0

Freeze the newly named fresh replay field and its 700k coordinates. Predict a
six-bin nearest-surface distance distribution using:

- refined normalized RAE coordinate;
- detached fresh-parent confidence;
- fresh-parent global Cube latents;
- local normalized 64-bin Doppler spectrum;
- local integrated log energy, spectrum entropy, and RAE energy gradients.

The deployment score is negative expected clipped distance. The unchanged
capacity exporter enforces exact 10k, `8000/1700/300`, and 5 cm separation.
Ground truth only builds training risk labels and evaluation metrics.

This is not another binary occupancy head: its target, output distribution,
listwise loss, local Full-RAED evidence, and same-coordinate condition
intervention all differ. It is also not presented as the final novelty claim.

### Outside-family fallback: variable multi-return ray hazard

If Q-Local-F0 fails, the next representation family is a survival-normalized
range hazard on each azimuth/elevation ray with a variable number of learned
returns. A zero-training 76-frame capacity oracle must pass before any model is
implemented. Fixed measured-peak `K=4/6`, larger WCE pools, and another
Cartesian neighborhood sweep remain closed.

### Deferred fallback: sparse ray-range transport

Sparse partial transport is considered only after a ray-range hard-rounding
oracle stays within `0.15 m` Chamfer and `2 pp` outlier of the unattainable
pointwise oracle, under `2 s/frame` and `60 GiB`. It is not run in parallel with
Q-Local-F0.

## Novelty boundary after the rescan

The project must not claim first use of Doppler for densification, first joint
spatial-Doppler radar modeling, first radar confidence/uncertainty, or absence
of temporal radar enhancement.

The still-defensible target is the complete generated radar-observable state:

```text
Full-RAED (+ optional bounded history)
  -> exact-count XYZ
  + pointwise circular Doppler distribution
  + calibrated confidence
  -> differentiable point-to-RAED closure
  -> displacement-Doppler temporal consistency
```

Q-Local-F0 can provide a geometry parent. It cannot by itself support that
final claim.
