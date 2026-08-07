# Literature positioning audit: Cube-conditioned dense physical radar state

Date: 2026-08-07

## Audit question

The defensible research question is not whether radar point clouds have ever
been densified, temporally aggregated, or generated with Doppler. All three
already exist. The narrower question is whether a model can map historical and
current full RAED radar tensors to a dense current/future point state with:

- exact-count, spatially reliable XYZ geometry;
- a per-point circular radial-velocity distribution and calibrated confidence;
- point-to-RAED measurement closure; and
- displacement-to-Doppler forward/backward temporal consistency.

The current repository has not established that full claim. STDA-F0 is only a
zero-training geometry-capacity gate and cannot by itself authorize a Doppler,
cycle, temporal, or deployment claim.

## Verified nearest work

| Work | Input and temporal scope | Output | Doppler role | Consequence for this project |
|---|---|---|---|---|
| [RaLD, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38946) | One radar spectrum; latent diffusion with frustum LiDAR autoencoding and radar guidance | Dense `N x 3` LiDAR-like points | The published output is geometric XYZ, not a per-point Doppler state | Use its frustum VAE, order-invariant set latents, radar-token hierarchy, mixed static/dynamic priors, and query decoder as the strongest geometry reference. Do not describe the field as empty or the current implementation as source-equivalent without a matched audit. |
| [RadarHD](https://arxiv.org/abs/2206.09273) | Current radar image plus 40 past frames (2 s history) stacked as channels | Current LiDAR-like point cloud/image with improved persistence | Not a generated point-state attribute | Directly invalidates a blanket claim that prior enhancement is only single-frame point-cloud super-resolution. It is a temporal current-frame reconstructor, although it does not generate the proposed `XYZ + q(v_r)` physical state. |
| [DenserRadar](https://arxiv.org/abs/2405.05131) | Radar representation supervised by stitched dense LiDAR occupancy | Dense 3D radar detections | Radar sensing capability is used, but output emphasis is dense geometry | Strong dense-PCE baseline; compare task and target construction explicitly. |
| [PillarGen](https://arxiv.org/abs/2403.01663) | Radar point cloud transformed with pillar-based generation | Densified synthetic points | Not the central output state | Covers learned point-cloud densification and downstream detection utility. |
| [SDDiff, IJCAI 2025](https://www.ijcai.org/proceedings/2025/979) | Radar spatial-Doppler representation | Dense point-cloud extraction and ego-velocity estimation | Joint spatial-Doppler diffusion and iterative Doppler refinement | Invalidates claims that joint spatial/Doppler modeling or reciprocal geometry/velocity supervision is new in isolation. |
| [DoppDrive, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Haitman_DoppDrive_Doppler-Driven_Temporal_Aggregation_for_Improved_Radar_Object_Detection_ICCV_2025_paper.html) | Multiple sparse radar point frames | Doppler-warped temporal aggregate for detection | Radial displacement and per-point retention duration | The project must beat an aggregation baseline and show that current-Cube-conditioned generation creates or refreshes state rather than only reusing history. |
| [RadarMP](https://arxiv.org/abs/2511.12117) | Two consecutive low-level 4D radar tensors | Consistent radar points plus pointwise 3D scene flow | Doppler- and echo-guided self-supervision | This is the closest temporal and physical-consistency competitor. A claim based only on adjacent tensors, consistent detection, or Doppler/position coupling is not defensible. |
| [4D-RaDiff](https://arxiv.org/abs/2512.14235) | Box- or LiDAR-conditioned latent point generation | Synthetic radar point clouds for augmentation | Full radar point attributes include Doppler and RCS in the model's generated state | Invalidates a "first generated Doppler radar point cloud" claim. Its goal is radar simulation/augmentation rather than current Cube-to-dense reconstruction, which remains a task distinction rather than proof of novelty. |
| [RadarGen](https://arxiv.org/abs/2512.17897) | Multi-view images and BEV-aligned visual cues | Radar point clouds reconstructed from generated BEV maps | Generated maps carry Doppler and RCS | Further invalidates output-attribute novelty; the defensible difference is sensing input, measurement closure, and temporal state estimation. |
| [RaUF, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_RaUF_Learning_the_Spatial_Uncertainty_Field_of_Radar_CVPR_2026_paper.pdf) | Radar measurements with LiDAR and odometry supervision | Radar point extraction/uncertainty and velocity estimates | Velocity is explicitly supervised | Invalidates a first velocity-regression or first radar uncertainty-field claim. The project needs calibrated per-point distributions plus closed-loop consistency, not a scalar channel alone. |
| [Radar4D-VLM](https://arxiv.org/abs/2608.04130) | Ten consecutive 4D-radar point sweeps | Object/scene/kinematic reasoning heads | Explicit radial-velocity tokens | Very recent evidence that temporal radar-only state reasoning is active. It does not directly forecast a dense radar point state, but it narrows broad temporal novelty claims. |

## Prior-to-next-frame search result

The 2026-08-07 search did not surface a directly matched method whose primary
task is autoregressive generation of the next dense radar `XYZ + v_r` point
state from a previous generated radar point state while also conditioning on
the current full RAED Cube. This is a scoped search result, not proof of
absence and not a publishable "first" claim.

The nearest categories are materially different but must remain in the review:

1. temporal aggregation of measured sparse points, especially DoppDrive;
2. adjacent-tensor point extraction and scene flow, especially RadarMP;
3. temporal radar detection/reasoning without dense point-state forecasting;
4. camera/LiDAR/box-conditioned radar simulation, including RadarGen and
   4D-RaDiff; and
5. LiDAR point-cloud forecasting, which supplies forecasting baselines but not
   radar measurement closure.

The final pre-submission review must repeat this search and follow the citation
graphs of RadarMP, RaLD, DoppDrive, SDDiff, RaUF, and 4D-RaDiff.

## Claims that must not be used

- "Prior radar enhancement is single-frame only."
- "This is the first multi-frame radar enhancement method."
- "This is the first method to generate Doppler radar points."
- "This is the first joint spatial-Doppler radar model."
- "Doppler can be integrated directly as an unconstrained 3D displacement."
- "STDA is a new optimal-transport algorithm or a deployable selector."

## Defensible combined contribution boundary

Subject to future evidence, the strongest defensible contribution is a
measurement-closed physical radar state generator:

```text
historical full RAED tensors + current full RAED tensor
    -> RaLD-structured geometric state prior
    -> dense current/future {XYZ, circular q(v_r), confidence}
    -> point-to-RAED closure
    -> radial projection of scene displacement <-> Doppler consistency
```

The temporal constraint must use the radial projection of 3D displacement,
not treat scalar Doppler as a complete 3D velocity:

```text
((p_{t+1} - T_{t->t+1} p_t) / delta_t) dot r_hat
    <-> expected radial velocity under q_t(v_r)
```

Occlusion, birth/death, tangential ambiguity, ego motion, and Doppler aliasing
must be gated or represented probabilistically.

## RaLD modules to borrow and test

1. Frustum-aware LiDAR autoencoding and arbitrary occupancy queries.
2. Compact order-invariant set latents instead of a dense 3D diffusion grid.
3. Radar-spectrum token hierarchy and cross-attention at every denoising block.
4. Mixed random/prior queries, interpreted here as static support and dynamic
   refresh candidates.
5. Latent diffusion only after a deterministic geometry parent passes; it is
   not a repair for a failed candidate domain or selector.
6. Source-faithful matched baselines with the official variable-threshold
   decoder, alongside the project's exact-count output contract.

## Parallel method frontier after STDA-F0

Priority is conditional on the unique formal STDA result.

| Priority | Route | Hypothesis | Mandatory controls | Stop condition |
|---|---|---|---|---|
| 1 | STDA structured assignment | The frozen RaLD-WCE field contains enough target-independent support and needs structured allocation rather than pointwise ranking | packed-pointwise and round-robin controls; all-76 resource/provenance transaction | Any frozen capacity, cardinality, recipe, or utility no-go closes exactly the corresponding mechanism. |
| 2 | Identity-centered RaLD anchor residual | Preserve a trusted anchor exactly at initialization, refine only a gated dynamic/uncertain subset, and refresh Doppler from the current Cube | copy/anchor identity, residual-off, wrong-Cube, dynamic/static mask, no-harm hinge | Geometry or confidence worsens outside the preregistered trust region. |
| 3 | Bidirectional position-Doppler state model | A circular radial-velocity distribution plus forward/backward displacement consistency improves physical metrics without static collapse | scalar head, distribution head, no-cycle, one-way cycle, ego-only and RadarMP-like flow control | PCE/spectrum gain is not accompanied by geometry non-degradation and dynamic variance retention. |
| 4 | Explicit temporal latent memory | Maintaining a persistent latent state separates object persistence from measurement birth/death better than raw point aggregation | single-frame, DoppDrive, ego-only, detached history, shuffled history, scheduled sampling | Current-frame geometry or local Doppler degrades beyond the frozen tolerance. |
| 5 | Uncertain-subset residual diffusion | Diffusion is useful only for ambiguous births and multimodal Doppler, while deterministic anchors preserve geometry | deterministic route, full-set diffusion, uncertainty-shuffled subset | No calibrated uncertainty benefit or excessive geometry tax. |
| 6 | Dynamic-subset refinement | Most bridge geometry tax comes from unnecessarily rewriting static support | all-point refinement, static-only, dynamic-only, zero-refinement | Dynamic gains do not offset whole-scene geometry loss. |
| 7 | Alternate representation reset | If exact-count point allocation repeatedly fails, predict a radar-observable field/state first and sample points only for evaluation | RaLD variable-threshold output, voxel/field baseline, exact-count adapter | Field quality does not survive point extraction or remains non-deployable. |

## Evidence discipline

- Literature facts are parsed claims supported by the linked primary paper or
  official proceedings page.
- The absence of a directly matched autoregressive method is a hypothesis from
  the current search, not a novelty certificate.
- Repository results remain computed claims only when bound to source, split,
  environment, logs, and immutable artifacts.
- A STDA pass authorizes only a separately frozen learned allocation gate. It
  does not unlock Doppler, cycle, temporal training, test access, or a paper
  claim by itself.
