# G1D to RaLD Source Map

## Audited upstream

- Repository: official RaLD release
- Commit: `ffec4b41241391734b1eda5c093de843c909eb8e`
- License: Apache-2.0
- Audit date: 2026-07-28

G1D is a deterministic K-Radar query-field geometry gate. It borrows RaLD's
information flow but is not the RaLD VAE or EDM. The later G3L-D gate remains
responsible for testing latent diffusion.

## Adopted mechanisms

| RaLD mechanism | Upstream source | G1D realization | Verification |
|---|---|---|---|
| Occupied and empty arbitrary queries | `datasets/aligned_coloradar/Coloradar_dataset.py:237-294`; `configs/generation/ge_indoor_cfg_aniso_mix_view_cone_unfreeze_enc_ints_only.yml:45-46` | Exactly 625 occupied and 9,375 reliable-empty RAE queries per frame | Per-frame count records, occupancy recall/FPR, sampler tests |
| Static and input-dependent dynamic latents | `model/models_ae.py:322-387` | `Qd` cross-attends radar-proposal tokens; `Proj(Qs + Qd)` | Mixed-latent gradient gate |
| Post-mix input cross-attention and FFN | `model/models_ae.py:392-396` | Residual proposal cross-attention followed by FFN | Module gradient gate |
| Arbitrary spatial query decoder | `model/models_ae.py:408-424` | RAE query embeddings cross-attend the latent set and emit occupancy, confidence, and bounded offset | Query decoder gradient and arbitrary-query tests |
| Radar condition in every latent block | `model/models_radar_generation.py:133-169`; `model/models_radar_generation.py:215-229` | Each of 24 blocks applies self-attention, Full-RAED cross-attention, and FFN | All 24 condition blocks must have nonzero finite gradients |
| Coarse query then local refinement | `engine_generation.py:249-310` | 32,000 radar-seeded coarse queries, fixed top 2,500, four local queries each | Exact per-frame 32k/2.5k/10k count gate |
| Full latent-diffusion schedule reserved for later | `model/models_radar_generation.py:235-295`; `model/models_radar_generation.py:314-449` | Not used in G1D; frozen for G3L-D: EDM noise law and 18-step Heun sampler | G1D provenance labels the model deterministic |

## K-Radar-specific replacements

| Upstream choice | G1D replacement | Reason |
|---|---|---|
| Intensity-only radar condition (`model/models_radar_generation.py:377-390`) | Complete normalized 64-bin Full-RAED Cube | Preserve Doppler evidence instead of collapsing it before geometry generation |
| ColoRadar 15.8 m view cone | K-Radar train-only axes and 120 m range | Avoid transferring incompatible spatial scales |
| Random 500k inference grid and optional CFAR helper | Deterministic Full-RAED energy NMS seeds | Prevent helper leakage and bound inference cost |
| Thresholded variable-count output | Fixed 10,000-point coarse-to-refine output | Make geometry, duplicate, confidence, and downstream comparisons controlled |
| Unqualified random empty voxels | Range-stratified empty cells outside a 3x3x3 occupied dilation | Avoid ambiguous negatives and preserve far-range supervision |
| Intensity-only point query state | RAE coordinate, all 64 local bins, absolute integrated log energy, and normalized range | Retain local Doppler shape and absolute radar evidence |

## Claim boundary

Passing G1D supports only this claim: a RaLD-structured arbitrary query field can
improve deterministic Cube-to-dense geometry under the frozen K-Radar gate. It
does not establish latent diffusion, generated Doppler, temporal consistency, or
downstream utility. Those require G3L-D, G2D, G4L-D, and P5 respectively.
