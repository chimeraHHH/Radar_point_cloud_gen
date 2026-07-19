# G3L: RaLD-Faithful Physical Latent Diffusion Protocol

## Purpose

RH/G2R/G3R borrow RaLD's radar hierarchy, mixed static/dynamic queries, latent
self-attention, and implicit query decoding, but their latent state is a
deterministic `512 x model_dim` feature array. They do not reproduce RaLD's
central generative mechanism: a `512 x 32` Gaussian point-set latent modeled by
a radar-conditioned EDM Transformer. G3L tests that missing mechanism without
discarding the independently gated long-range geometry parent.

The implementation source is the official RaLD repository at commit
`ffec4b41241391734b1eda5c093de843c909eb8e`. G3L adapts its representation and
EDM schedule, not its intensity-only input, short-range coordinates, CFAR query
helper, or geometry-only output.

## Architecture

1. The selected G3R parent supplies exactly 10k confidence-ranked RAE anchors
   and spatial features. These are the only decoder queries; G3L never scans a
   dense free-space grid.
2. A target point-state token contains normalized RAE, the complete 64-bin
   circular Doppler distribution, its sine/cosine moments, and confidence.
3. RaLD static and input-dependent dynamic queries cross-attend to the unordered
   target point-state set and parameterize a `512 x 32` Gaussian posterior.
4. A 24-layer latent Transformer decodes the posterior only at the parent
   anchors. The resulting query feature enters the existing final-position
   Cube query and physical head to produce bounded RAE offset, 64-bin Doppler
   distribution, and confidence.
5. After the physical VAE is frozen, a 24-layer EDM Transformer learns the
   `512 x 32` latent distribution conditioned on Full-RAED radar tokens. Formal
   sampling uses the RaLD schedule: 18 Heun steps, `sigma_min=0.002`,
   `sigma_max=80`, and `rho=7`.

This preserves the project's distinguishing physics: all Doppler bins condition
generation, geometry and Doppler refer to the same final continuous position,
and the decoded point state is evaluated through the point-to-RAED cycle.

## Gates

### G3L-0: component verification

- target point permutation leaves posterior statistics unchanged;
- anchor permutation permutes decoded query features identically;
- default dimensions are `512 x 32`, 24 decoder layers, 24 EDM layers, and 18
  sampling steps;
- EDM gradients reach the Full-RAED encoder after the zero-output first update;
- fixed sampling seeds are bitwise deterministic;
- no occupancy head, full-space mesh, CFAR helper, or best-of-k selection exists.

### G3L-1: physical VAE

Initialize from each selected G3R `full` parent, freeze that parent, and train
the posterior encoder, anchor decoder, and physical head. The target Doppler
distribution is queried from the measured Cube at each target point. Use the
same geometry, Doppler, confidence, offset, and full-cycle losses as G3R plus a
KL term with a frozen warmup schedule.

The posterior-mean reconstruction must retain G3R Chamfer within 2%, local
spectrum KL and circular W1 within 5%, and at least 90% confidence and covered
cells. Posterior variance must remain finite and nonzero; a constant posterior
or decoder that ignores Doppler/confidence fails.

### G3L-2: Full-RAED-conditioned EDM

Freeze the passing G3L-1 encoder/decoder and train the official-scale
radar-conditioned EDM on cached train-only posterior means. Use three frozen
seeds and 100 epochs, matching RaLD's published epoch count while recording the
exact update count. Evaluation uses one frame-derived sampling seed fixed
before metrics; best-of-k is prohibited.

The sampled model must retain deterministic G3R Chamfer within 5%, local
spectrum KL and circular W1 within 5%, and at least 90% confidence and coverage.
Shuffling the Full-RAED condition across scenes must confidently worsen either
Chamfer or local spectrum KL, proving the sampler did not learn an unconditional
latent prior. Four-seed diversity is descriptive only and cannot select output.

## Temporal Consequence

If G3L passes, the RaLD-faithful temporal mainline is `G4L`: ego-aligned
historical point-state tokens condition the EDM together with the current Cube.
The current deterministic token/latent/query G4R family remains a matched
control and engineering fallback. If G3L fails, the paper may claim structural
RaLD borrowing in RH/G2R/G3R but must not call the main generator latent
diffusion.

## Evidence Boundary

G3L is not a reopening of the failed standalone matched RaLD occupancy AE. That
model failed long-range allocation because it queried the whole field and
ranked a fixed top-10k output. G3L uses an independently passing geometry parent
as radar-guided query initialization and asks whether RaLD's latent diffusion
improves physical state generation on that fixed support.
