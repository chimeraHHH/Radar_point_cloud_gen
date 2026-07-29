# Parallel literature and geometry-route decision

Date: 2026-07-29 CST

## Decision

The project remains a two-stage method:

```text
Stage I
current Full-RAED Cube
  -> radar-only single-frame geometry parent
  -> dense XYZ + geometry confidence

Stage II
history + current Cube + Doppler
  -> gated Doppler warp and current-frame refresh
  -> dense XYZ + circular Doppler distribution + confidence
  -> point-to-Cube physical closure
```

Stage I is an enabling component, not the main novelty. LiDAR, semantics, and
other privileged labels may be used during training, but inference must remain
current-Cube only. Stage II retains the paper contribution: generating and
refreshing a physically consistent point state rather than aggregating observed
points.

The completed negative results rule out specific mechanisms, not the full
research question:

- G1/G1B: early Full-RAED occupancy fusion did not beat the matched RAE-Max
  parent gate.
- G1D: the finite query pool, residual scaling, and same-pool ranking repairs
  failed.
- G1G: free center-child set generation collapsed and sprayed points.
- P-RF: the target adapter lacked exact-10k capacity in 10 of 57 sparse frames.
- R-A1/WCE: wide queries learned Cube dependence and coverage, but failed
  precision and outlier gates.
- G-RM: future birth-point radial supervision covered only 4.487% of geometry.

## Literature boundary

Primary sources support four distinct lessons:

1. [RaLD](https://ojs.aaai.org/index.php/AAAI/article/view/38946) demonstrates
   radar-spectrum-conditioned dense XYZ generation through a frustum LiDAR
   autoencoder and latent diffusion. Its public method does not close this
   project's per-point Doppler, confidence, temporal refresh, or point-to-Cube
   cycle requirements.
2. [RadarOcc](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b81d83165e3145a2e7d33bb5e33ea913-Abstract-Conference.html)
   shows that direct 4D radar tensors benefit from Doppler descriptors,
   range-aware processing, spherical encoding, and spherical-to-Cartesian
   aggregation. Its output is semantic occupancy, not a dense physical point
   state.
3. [TULIP](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_TULIP_Transformer_for_Upsampling_of_LiDAR_Point_Clouds_CVPR_2024_paper.html)
   supports range/ray-aware decoding and explicitly reports floating-point and
   long-range failure modes that unstructured interpolation can create.
4. [RadarDistill](https://openaccess.thecvf.com/content/CVPR2024/html/Bang_RadarDistill_Boosting_Radar-based_Object_Detection_Performance_via_Knowledge_Distillation_from_CVPR_2024_paper.html)
   establishes the legality of a LiDAR teacher used only during training, while
   retaining radar-only inference. It is detection evidence, not direct
   Cube-to-point generation evidence.

[GRT](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_Towards_Foundational_Models_for_Single-Chip_Radar_ICCV_2025_paper.html)
further shows that raw 4D radar representation quality and data scale matter:
its one-million-sample study predicts occupancy and semantics from raw radar.
The current 76-frame training cohort is therefore a material limitation, and
masked pretraining is only an initialization ablation after a decoder is
selected.

No reviewed primary source was found that jointly provides:

```text
single-frame Full-RAED Cube
  -> dense XYZ
  + pointwise circular Doppler distribution
  + confidence
  + temporal refresh
  + point-to-Cube closure
```

This is a scoped novelty boundary, not a universal absence claim.

## Parallel falsification ladder

The next work is ordered by evidence cost rather than by model size.

### D0: frozen-checkpoint diagnostics

Run without training:

- R-A1 candidate support and validation-GT diagnostic ranking;
- residual-on versus residual-off;
- positive-logit capacity by range;
- frozen output quota versus train-distribution quota;
- RAE-Max 2.5k/5k/7.5k/10k density-quality Pareto.

These tests separate support, ranking, residual, quota, and exact-cardinality
failure. Validation GT may appear only in explicitly unattainable diagnostics.

### R-B1: ray-wise multi-return geometry

Represent each azimuth-elevation ray by at most `K` ordered continuous range
returns with existence, sub-bin offset, and confidence. This changes the
geometry parameterization from global occupancy top-k to a sensor-aligned
multi-echo process.

Before training:

- audit `K=4` and `K=6` on all 100 development frames;
- require exact-10k and 5 cm capacity without copy, padding, or output jitter;
- evaluate the GT-aided structural diagnostic on all 24 validation frames and
  all 23 far-target frames.

### R-B2: spherical encoder to Cartesian voxel slots

Borrow RadarOcc's spherical processing boundary, but decode to a fixed
Cartesian lattice with bounded, mutually exclusive point slots. This removes
the free-center collapse mode of G1G and changes the output basis from RAE
occupancy.

Before training:

- audit at least two voxel-size/slot-count configurations;
- require exact-10k range quotas and 5 cm capacity;
- fail hard when any frame cannot meet the contract.

### R-B3: LiDAR geometry manifold and Cube-to-latent distillation

If R-B1 and R-B2 structural oracles fail, train a range/frustum-aware LiDAR
autoencoder first, then predict its geometry latent from the current Full-RAED
Cube. Distillation is restricted to radar-observable or LiDAR-active regions;
unknown space cannot be forced to match the LiDAR teacher.

The LiDAR autoencoder must independently pass the complete geometry gate before
Cube adaptation. This route is not a restart of the archived RaLD matched AE:
it requires variable-cardinality masking, observability-aware targets, and a
separate representation oracle.

## WCE-only bounded pilots

R-A1 itself is closed. A small R-A2 mechanism study is permitted only because
it tests documented supervision mismatches:

- tiny-set memorization on eight train frames, maximum 500 updates;
- RaLD classwise `0.1 positive / 1.0 empty` BCE;
- range-class sampling with equal full-cell jitter and an ambiguity band;
- confidence-soft or truncated UDF supervision only if the earlier pilot
  establishes memorization and a metric-aligned improvement.

Failure to memorize closes the WCE architecture family. Passing memorization
does not reopen R-A1; a pilot must still improve outlier and far precision under
the unchanged validation protocol before a new Stage-0 is authorized.

## Innovation boundary

The geometry parent, distillation, or pretraining route is not claimed as the
paper's central novelty. The defensible contribution remains:

> Under radar-only inference, generate dense geometry, confidence, and a
> pointwise circular Doppler state from the current Full-RAED Cube; use
> Doppler-warped history only as a prior, refresh geometry and velocity from the
> current Cube, and enforce position-velocity consistency through a
> point-to-Cube reverse constraint.

Until birth-point supervision is solved, Doppler claims are restricted to
persistent points and birth points must retain `doppler_valid=false`.
