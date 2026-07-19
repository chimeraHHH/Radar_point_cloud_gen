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

The occupancy grid remains responsible for long-range anchor allocation. We
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
A zero-initialized physical head predicts a bounded RAE offset, confidence, and
a residual 64-bin Doppler distribution over the local measured Cube spectrum:

```text
q_i = Softmax(log(q_cube_i + eps) + Delta l_i).
```

Thus the initial hybrid exactly preserves anchor positions, parent confidence,
and measured Doppler while learning globally radar-conditioned point-set
corrections. The independent RaLD point VAE was rejected by the K-Radar
long-range Chamfer gate, so this anchor hybrid is evaluated as a separately
gated candidate and cannot be treated as an established contribution before
RH1/RH2 pass.

## 4. Point-Conditioned Doppler Prediction

Features are gathered at selected RAE locations and passed through a Linear-SiLU projection.

### 4.1 Scalar Head

The scalar baseline predicts one Doppler value and wraps it to the sensor alias interval. For distributional metrics, it is converted to a fixed-width wrapped Gaussian.

### 4.2 Distribution Head

The distribution head predicts 64 logits:

```text
q_i(d) = softmax(W_d f_i),
v_hat_i = CircMean(q_i, v_D).
```

Training uses the normalized `log1p(power)` spectrum queried from the current Cube at the target location:

```text
L_dop = - sum_i sum_d w_i q_i*(d) log q_i(d),
```

where `w_i` is target visibility confidence. Circular mean, scalar error, and Wasserstein distance explicitly respect the Doppler alias period.

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

A confidence-weighted symmetric Chamfer term is applied with weight `0.1`. All G3 arms train the same offset head, so cycle arms do not receive an unmatched continuous-coordinate advantage.

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

Success cannot be inferred from the floor alone. G3 separately requires relative confidence and covered-cell retention of at least 90%, no significant ECE degradation, and an acceptable sub-bin offset saturation fraction.

The matched G3 arms are:

```text
C0: no cycle
C1: local spectrum KL
C2: C1 + Doppler marginal
C3: C2 + spatial energy.
```

## 7. Current-Observation Temporal Refinement

The temporal model always receives the current Cube. A previous predicted point set is only an auxiliary prior.

For previous point `i`, residual Doppler and the dynamic gate are

```text
v_res_i = wrap(v_hat_i - v_static_i)
g_i = 1[abs(v_res_i) > 1.0 m/s].
```

The point is first advected radially and then transformed into the current frame:

```text
p_adv_i = p_i + g_i v_res_i Delta_t r_hat_i
p_prior_i = T_(t<-t-1) p_adv_i.
```

Static or low-residual points receive only ego motion. Prior point features contain normalized XYZ, confidence, circular Doppler sine/cosine, spectral entropy, and the dynamic gate.

Three matched fusion families are implemented:

1. **Concat:** rasterize energy, circular moments, entropy, and gate into five RAE channels, project, concatenate, and fuse with the current Cube feature.
2. **Local cross-attention:** each current query attends to its eight nearest prior tokens with relative XYZ encoding.
3. **Draft refinement:** the nearest warped point is a draft; a learned 3D gate interpolates between prior and learned offsets.

The cross-frame radial residual is

```text
e_rad_i = abs((||p_j,t|| - ||p_i,t-1^ego||)
              - 0.5(v_res_i,t-1 + v_res_j,t) Delta_t).
```

Confidence and a Gaussian function of match distance weight this loss. Temporal training uses 20 epochs: the temporal head is isolated for five epochs, joint fine-tuning starts at epoch six, and scheduled sampling increases linearly from 0 to 0.4. Recurrent predictions are detached. Teacher history comes from the frozen single-frame parent, never from LiDAR ground truth.

## 8. Training Objectives and Staging

Single-frame geometry:

```text
L_G1 = L_occ.
```

Doppler generation:

```text
L_G2 = L_occ + L_dop + 0.1 L_geo.
```

Cycle training:

```text
L_G3 = L_occ + L_dop + 0.1 L_geo + L_cycle_variant.
```

Temporal training:

```text
L_G4 = L_occ + L_dop + 0.1 L_geo
     + I_C3 L_cycle + 0.1 L_radial.
```

Modules are released only after their parent gate closes. All main comparisons use three fixed seeds and scene-first paired bootstrap. The untouched test split is released only after the temporal family is frozen.

## 9. Evidence Boundaries

- G0 passed and validates the data/geometry pipeline, not model quality.
- Full-RAED geometry gain remains pending G1 comparison.
- Doppler distribution gain remains pending G2.
- Cube-cycle value remains pending G3.
- Temporal value remains pending verified G4 data and G4 evaluation.
- The analytic static mixture failed and is not a contribution.
- TruckScenes Doppler-warp and scheduled-sampling results motivate the temporal design but are not K-Radar Cube-to-dense evidence.
- The RaLD-anchor hybrid passed component-scale RH0 only; its training gain remains pending RH1/RH2.
