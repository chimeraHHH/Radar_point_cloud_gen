# G2 Cube-to-Point Doppler Protocol

## Scope

G2 tests whether preserving the Doppler spectrum at each generated location is
more useful and physically faithful than compressing the same Cube evidence to
one regressed scalar. G2 starts only if G1 passes. It uses the frozen K-Radar
scene split, train-only normalization, and the matched Full-RAED geometry
checkpoint from the corresponding G1 seed.

The input Cube already contains Doppler measurements. Therefore, recovering a
Doppler value from the Cube is not itself claimed as novel. The scientific
question is whether a generated dense point representation should retain and
calibrate the local Doppler distribution, and whether an analytic ego-motion
prior can resolve static/dynamic ambiguity without destroying dynamic modes.

## Compared Systems

| ID | Geometry | Doppler output | Purpose |
|---|---|---|---|
| E2 | Full-RAED G1 model | none | frozen geometry reference |
| Q0 | E2 positions | direct local Cube spectrum | nonlearned sensor-query reference |
| E3 | E2 initialization | scalar regression head | single-value baseline |
| E4 | E2 initialization | calibrated 64-bin distribution head | distribution representation test |
| E5 | E2 initialization | static-prior/dynamic-residual mixture | physics mechanism test |

Q0 must remain in every Doppler table. E4 or E5 is not credited for merely
copying the observed Cube spectrum; gains must appear in calibration,
ambiguity handling, geometry, physical consistency, or a downstream endpoint.

## Position-Conditioned Spectrum Target

For every unique radar-observable LiDAR RAE cell, query the complete 64-bin
input spectrum at that location. For continuous generated positions, use
trilinear RAE interpolation before normalizing over Doppler. Convert power to a
probability target with `log1p(power)`, nonnegative clipping, and fixed additive
smoothing of `1e-4` before normalization.

The scalar target is the circular spectral mean. Scalar errors use circular
distance over the measured Doppler period. For distribution metrics, convert
E3 to a fixed one-bin-width wrapped Gaussian; its width is not tuned on the
validation set.

All systems output 10,000 geometry points. E3-E5 attach either one radial
velocity or one 64-bin distribution to exactly those points. Empty or
out-of-FOV queries are reported and cannot be silently removed.

## Static/Dynamic Physics Mixture

Before E5, calibrate the dataset convention using only train-partition CFAR
points outside every annotated 3D box. Compare the three wrapped hypotheses
`-dot(v_ego, r_hat)`, `+dot(v_ego, r_hat)`, and zero-centered compensation.
Freeze the train winner only if its frame-median circular error beats the
runner-up by at least `0.05 m/s`, then evaluate that frozen choice on validation.
No validation frame participates in selecting the sign or compensation mode.

For a point with radial unit vector `r_hat` and platform velocity `v_ego`, the
uncompensated static radial velocity is

```text
v_static = -dot(v_ego, r_hat).
```

K-Radar aliasing is handled with circular Doppler distance on the measured
axis. The analytic distribution is centered at the frozen calibrated
hypothesis, which may be zero if the Cube is already ego compensated. E5
predicts a static probability and a dynamic residual distribution:

```text
p(v_r) = p_static * p_analytic(v_r | v_ego, r_hat)
       + (1 - p_static) * p_dynamic(v_r | Cube, position).
```

The analytic component is a wrapped Gaussian with one-bin fixed standard
deviation. A detached soft static/dynamic supervision target is derived from
the distance between the observed local spectrum and the calibrated static
center: at most
`1.0 m/s` is static, at least `2.0 m/s` is dynamic, and the interval is linearly
interpolated. The target is a training label, not a test-time oracle.

E5 receives ego motion explicitly. Under an ego-radial convention,
counterfactual interventions must alter the analytic component; under a
zero-centered compensated convention, the correct response is invariance near
zero. The learned dynamic component must remain responsive to Cube evidence and
must not be overwritten by the static prior.

## Training Protocol

- Seeds: `20260716`, `20260717`, `20260718`.
- Initialize each E3-E5 run from the same-seed E2 best checkpoint.
- First 5 epochs: train Doppler and mixture heads only.
- Next 25 epochs: jointly fine-tune with head learning rate `3e-4` and geometry
  backbone learning rate `3e-5`.
- Optimizer: AdamW, weight decay `1e-4`; cosine schedule over 30 epochs.
- Numeric mode: BF16 autocast with FP32 parameters.
- Common model selection: lowest validation local-spectrum NLL after converting
  E3 to its fixed wrapped-Gaussian distribution.
- Main and gate statistics use all three seeds and scene-first paired bootstrap.

Losses are the existing occupancy loss plus the mode-specific Doppler loss. E5
adds static-gate binary cross entropy with weight `0.25`. No loss weight or
static/dynamic threshold is tuned after viewing G2 validation outcomes.

## Metrics

### Doppler fidelity

- local spectrum NLL and KL;
- circular W1 and mode-bin accuracy;
- circular scalar MAE for the reported point velocity;
- CD-Doppler against radar-observable target points with queried spectra.

### Physics and calibration

- static and dynamic circular velocity error separately;
- PCE at `0.25`, `0.5`, and `1.0 m/s` on the static subset;
- distribution confidence ECE and NLL;
- predicted versus target dynamic fraction;
- convention-aware ego-speed counterfactual response at multipliers `0`,
  `0.5`, `1`, `1.5`, and `2`, including slope and monotonicity or invariance.

### Geometry safeguard

- the complete G1 geometry suite, evaluated on the same validation frames;
- E5 versus E4 relative Chamfer change with a paired confidence interval.

## G2 Decision Rule

G2 passes only when all conditions hold:

1. E4 improves over E3 with a 95% confidence interval excluding zero on local
   spectrum NLL and on at least one of circular W1 or CD-Doppler.
2. E5 improves over E4 with a 95% confidence interval excluding zero on static
   PCE and on at least one of spectrum NLL, ECE, or CD-Doppler.
3. The upper 95% confidence bound of E5's relative geometry Chamfer degradation
   versus E4 is at most 2%.
4. E5's predicted dynamic fraction is between 0.5 and 1.5 times the target
   dynamic fraction and at least 5%; otherwise the physics mixture is declared
   collapsed.
5. Counterfactual ego-speed response matches the frozen sensor convention. For
   an ego-radial convention it is monotonic with the calibrated sign; for a
   zero-centered compensated convention the absolute fitted slope is at most
   `0.1`. Saturation outside the training range is reported but is not by
   itself a failure.

If E4 does not beat E3, inspect circular-label construction and spectrum-query
calibration before changing the model. If E5 fails while E4 passes, proceed to
the Cube-cycle study with E4 as the single-frame anchor and report the analytic
mixture as a negative result rather than weakening this gate.

## Required Artifacts

- Q0 and E3-E5 per-frame metric JSON for all three seeds;
- configuration, provenance, best/last checkpoints, and exact E2 parent hash;
- paired scene-first bootstrap report and G2 decision note;
- reliability curves and static/dynamic distribution panels;
- counterfactual dose-response table and figure;
- fixed-frame and worst-five qualitative comparisons;
- geometry nondegradation report using the frozen G1 evaluator.
