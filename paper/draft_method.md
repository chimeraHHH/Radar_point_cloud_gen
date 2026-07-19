# Method Blueprint: Full-RAED Cube to Dense Radar Points

> Authority: current Cube-to-dense implementation and frozen G0-G4 protocols
> Updated: 2026-07-19 17:03 CST
> Evidence rule: this file defines the method, not experimental success

## 1. Problem Formulation

For time `t`, the canonical radar measurement is a non-negative RAED Cube

```text
C_t in R_+^(D x R x A x E),  (D,R,A,E) = (64,256,107,37).
```

The model reconstructs a fixed-size radar-observable point set

```text
P_t = {(p_i, q_i, v_hat_i, c_i)}_(i=1)^N,  N = 10,000,
```

where `p_i=(x_i,y_i,z_i)`, `q_i` is a 64-bin circular Doppler distribution, `v_hat_i` is its circular scalar summary, and `c_i` is visibility confidence. The target is not the complete LiDAR surface; it is the subset supported by radar field of view, Cube energy, first-surface visibility, and range constraints.

## 2. Cube Normalization and Matched Encoders

Power values are normalized with train-only statistics:

```text
C_tilde = clip((log10(C_t + 1) - mu_train) / s_train, -4, 4).
```

The geometry experiment uses matched input encoders:

- **RAE-Max:** one channel obtained by maximizing over Doppler.
- **RAE-Moments:** peak power, circular Doppler mean, and spread.
- **Full-RAED:** all 64 Doppler bins as input channels.

Each representation is projected to `b=8` channels with a `1x1x1` convolution. The same residual 3D U-Net then operates over the RAE grid:

```text
F0 = Res3D_b(Proj(C_tilde))
F1 = Res3D_2b(Conv3D_s2(F0))
Z  = Res3D_4b(Conv3D_s2(F1))
U1 = Res3D_2b([Up(Z), F1])
F  = Res3D_b([Up(U1), F0]).
```

Each residual block contains two `3x3x3` convolutions, GroupNorm, SiLU, and an optional `1x1x1` skip projection. Upsampling is trilinear. A `1x1x1` occupancy head produces logits on the `256x107x37` RAE grid.

## 3. Radar-Observable Occupancy and Dense Decoding

Radar-observable LiDAR points are projected into the RAE grid. If multiple targets share a cell, their maximum visibility confidence defines the soft target `Y_rae in [0,1]`.

The occupancy objective is

```text
L_occ = L_soft-focal + 0.25 L_soft-dice.
```

At inference, global top-k decoding selects 10,000 distinct RAE cells. Every cell is decoded once, preventing duplicated points from creating artificial density. The occupancy probability also provides point confidence.

### 3.1 RaLD-Inspired Anchor Latent Refinement

The selected occupancy grid remains responsible for long-range anchor allocation. We
borrow RaLD's mixed set-latent mechanism after this parent rather than replacing
the complete geometry decoder. For normalized anchor coordinates `u_i` and
parent features `f_i`, Fourier coordinate embeddings and projected features form
anchor tokens:

```text
a_i = Fourier(u_i) + W_f f_i.
```

The complete RAED Cube is encoded by the RaLD radar hierarchy into 336 spatial
condition tokens `R`; unlike upstream RaLD, all 64 Doppler bins participate.
Static and input-dependent latent queries cross-attend to both the complete
anchor set and `R`, followed by latent self-attention:

```text
z_dyn = z_dyn^0 + CrossAttn(z_dyn^0, {a_i}) + CrossAttn(z_dyn^0, R),
Z = Transformer(W_z(z_static + z_dyn)).
```

Each anchor then queries `Z` to obtain a globally contextualized point feature.
A zero-initialized physical head first predicts a bounded RAE offset. The Cube
is queried again at the final continuous position, and the same RaLD query
feature predicts confidence plus a residual 64-bin Doppler distribution:

```text
u_i = u_anchor_i + 0.5 tanh(Delta u_i),
q_i = Softmax(log(q_cube(u_i) + eps) + Delta l_i).
```

Thus the initial hybrid exactly preserves anchor positions, parent confidence,
and measured Doppler while learning globally radar-conditioned point-set
corrections. The independent RaLD point VAE was rejected by the K-Radar
long-range Chamfer gate, so this anchor hybrid is evaluated as a separately
gated candidate and cannot be treated as an established contribution before
RH1/RH2 pass. The current independent route additionally requires a passing
G1B geometry parent; the failed original G1 is never relabeled.

This separation also isolates early and late spectral fusion. The original G1
showed that Full-RAED early fusion degraded Chamfer relative to RAE-Max and is
closed. The independent G1B route instead selects a physically compressed
geometry allocator, freezes it, and introduces all 64 Doppler bins through
RaLD radar-token cross-attention. No G1 result is relabeled by this branch.

### 3.2 RaLD-Faithful Physical Latent Diffusion Candidate

The anchor refiner above is deterministic and does not instantiate RaLD's
central latent-diffusion mechanism. G3L therefore encodes an unordered target
point-state set into a `512 x 32` Gaussian posterior. Each target token contains
normalized RAE, its complete 64-bin Doppler distribution, circular moments, and
confidence. A 24-layer latent Transformer decodes only the frozen parent
anchors, which act as radar-guided query initialization rather than a dense
free-space occupancy scan.

After this physical VAE passes, its encoder and decoder are frozen and a
24-layer EDM Transformer models the latent distribution conditioned on the
current Full-RAED tokens. Formal inference uses the RaLD 18-step Heun sampler.
The decoded query features still use the final-position Cube spectrum and the
same geometry, Doppler, confidence, and cycle heads. This branch is separately
gated; until G3L passes, the method is described as RaLD-inspired anchor
refinement rather than latent diffusion.

## 4. Point-Conditioned Doppler Prediction

RaLD query features and the Cube spectrum at each final continuous RAE location
are passed to matched physical heads.

### 4.1 Scalar Head

The scalar baseline starts from the circular mean of the local Cube spectrum,
predicts a bounded circular residual, and wraps the result to the sensor alias
interval. For distributional metrics, it is converted to a fixed one-bin-width
wrapped Gaussian.

### 4.2 Distribution Head

The distribution head predicts 64 residual logits over the measured spectrum:

```text
q_i(d) = softmax(log(q_cube(u_i,d) + eps) + W_d f_i),
v_hat_i = CircMean(q_i, v_D).
```

Training uses the normalized `log1p(power)` spectrum queried from the current
Cube at radar-observable target locations matched to each generated point:

```text
L_dop = - sum_i sum_d w_i q_i*(d) log q_i(d),
```

where `w_i` is the frozen geometry-parent confidence, preventing learned
confidence from hiding Doppler errors. Circular mean, scalar error, and
Wasserstein distance explicitly respect the Doppler alias period. G2R trains
matched scalar and distribution RaLD arms with cycle disabled and compares the
learned distribution against the unmodified Cube spectrum at the same final
point position.

### 4.3 Rejected Analytic Static Mixture

The implementation supports a candidate mixture

```text
q_i(v) = s_i q_static(v | p_i, v_ego) + (1-s_i) q_dynamic(v | C_t, p_i).
```

However, the frozen static-background audit selected `positive_ego` on train with a `0.096906 m/s` margin, while validation error `1.020193 m/s` was worse than the circular-random reference `0.966296 m/s`. A bounded SNR-quantile recovery also failed. Therefore this head is excluded from formal G2 and cannot support an analytic physics-prior claim. It remains documented only as a rejected branch.

## 5. Continuous RAE Point Parameterization

For each selected cell, the cycle model predicts a bounded sub-bin offset:

```text
Delta u_i = 0.5 tanh(h_offset(f_i)),
u_i = (r_i, a_i, e_i) + Delta u_i.
```

Each component lies in `[-0.5,0.5]` bins. Interpolated physical range, azimuth, and elevation are converted to Cartesian coordinates:

```text
p_i = [rho_i cos(eps_i) cos(alpha_i),
       rho_i cos(eps_i) sin(alpha_i),
       rho_i sin(eps_i)].
```

A confidence-weighted symmetric Chamfer term is applied with weight `0.1`.
All G3R arms start from the exact same passing cycle-free G2R checkpoint and
train the same offset head, so cycle arms receive neither an unmatched
continuous-coordinate advantage nor cycle-contaminated pretraining.

## 6. Differentiable Point-to-Cube Rendering

The renderer assigns every point to the eight neighboring RAE cells with normalized trilinear weights and deposits its 64-bin Doppler probability mass:

```text
C_hat(d,r,a,e) = sum_i c_i w_i(r,a,e | u_i) q_i(d).
```

Boundary weights are renormalized, so valid point mass is conserved. Gradients propagate to sub-bin offsets, Doppler logits, and confidence.

The complete cycle objective is

```text
L_cycle = L_local-spectrum-KL
        + 0.25 L_Doppler-marginal
        + 0.25 L_spatial-energy
        + L_confidence-floor.
```

- `L_local-spectrum-KL` compares spectra only at rendered, energetic Cube cells.
- `L_Doppler-marginal` compares frame-level Doppler mass.
- `L_spatial-energy` compares normalized log energy against the strongest 10,000 target cells.
- `L_confidence-floor` penalizes mean confidence below `0.1`.

Success cannot be inferred from the floor alone. G3R separately requires
relative confidence and covered-cell retention of at least 90%, no significant
ECE degradation, and an acceptable sub-bin offset saturation fraction.

The matched G3R arms are:

```text
C0: no cycle
C1: local spectrum KL
C2: C1 + Doppler marginal
C3: C2 + spatial energy.
```

## 7. Current-Observation Temporal Refinement

The temporal model always receives the current Cube. A previous predicted point set is only an auxiliary prior.

The previous implementation fused history into a separate `CubeCycleNet` and
depended on an analytic static-Doppler convention. That route is closed because
it cannot load the selected RaLD checkpoint and the static convention failed
validation. G4R injects the historical prior at three RaLD-structured
deterministic locations. If G3L passes, G4L additionally conditions the EDM on
the ego-aligned historical physical point state.

The preregistered main prior first uses only the unambiguous ego transform:

```text
p_prior_i = T_(t<-t-1) p_i.
```

The complete historical Doppler distribution remains an input feature. A raw
Doppler displacement candidate is evaluated only as a sensitivity baseline:

```text
p_prior,dopp_i = T_(t<-t-1)(p_i + v_hat_i Delta_t r_hat_i).
```

This avoids reviving the rejected analytic static prior. Historical point
features contain normalized XYZ, confidence, circular Doppler sine/cosine, and
spectral entropy.

Three matched G4R fusion families are defined:

1. **TR4 token fusion:** rasterize prior confidence and Doppler moments; a
   zero-gated hierarchy injects them into the 336 current Full-RAED tokens.
2. **TR5 latent fusion:** prior points become set tokens; only RaLD dynamic
   mixed latents cross-attend to them before the latent Transformer.
3. **TR6 query refinement:** each decoded anchor query receives its nearest
   warped prior feature and relative position before predicting final geometry
   and Doppler.

All arms retain the current Cube, final-position spectrum query, and G3R cycle.
Temporal matching is measured in the ego-aligned frame without a static-PCE or
analytic dynamic-fraction gate.

Temporal training uses 20 epochs: the zero-gated RaLD temporal adapter is
isolated for five epochs, joint fine-tuning starts at epoch six, and scheduled
sampling increases linearly from 0 to 0.4. Recurrent predictions are detached.
Teacher history comes from the frozen selected G3R parent, never from LiDAR.

## 8. Training Objectives and Staging

Single-frame geometry:

```text
L_G1B = L_occ.
```

Doppler generation:

```text
L_G2R = L_geo + L_dop + 0.1 L_conf + 0.01 L_offset.
```

Cycle training:

```text
L_G3R = L_G2R + 0.1 L_cycle_variant.
```

RaLD physical latent diffusion:

```text
L_G3L-VAE = L_G3R + beta_KL L_KL,
L_G3L-EDM = E_sigma[w(sigma) ||D(z + sigma epsilon, sigma, R) - z||_2^2].
```

Temporal training:

```text
L_G4R = L_G3R + 0.1 L_temporal-match.
```

Modules are released only after their parent gate closes. All main comparisons use three fixed seeds and scene-first paired bootstrap. The untouched test split is released only after the temporal family is frozen.

## 9. Evidence Boundaries

- G0 passed and validates the data/geometry pipeline, not model quality.
- The original G1 failed; independent G1B geometry selection is pending.
- Doppler distribution gain remains pending G2R after RH passes.
- Cube-cycle value remains pending G3R.
- RaLD-faithful `512 x 32` physical VAE/EDM value remains pending G3L.
- Temporal value remains pending verified data and G4R evaluation.
- The analytic static mixture failed and is not a contribution.
- TruckScenes Doppler-warp and scheduled-sampling results motivate the temporal design but are not K-Radar Cube-to-dense evidence.
- The RaLD-anchor hybrid passed component-scale RH0 only; its training gain
  remains pending G1B, RH1, and RH2. G2R/G3R code exists but has no eligible
  result yet.
