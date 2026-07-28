# G1E RaLD latent-EDM protocol

## Decision and scope

G1E is frozen before the G1D Stage-A result is known. It answers a different
question from G1D:

- G1D tests a deterministic Cube-conditioned query field with direct local
  Cube evidence in the decoder.
- G1E tests RaLD's central two-stage mechanism: a LiDAR occupancy VAE defines a
  `512 x 32` target latent space, and a radar-conditioned EDM generates that
  latent before a coordinate-only implicit decoder reconstructs geometry.

The upstream reference is the official RaLD repository at commit
`ffec4b41241391734b1eda5c093de843c909eb8e`. G1E is not a relabeling or repair
of G1D, and G1D metrics do not select G1E hyperparameters.

## Source-faithful mechanisms

G1E must preserve the following RaLD mechanisms:

1. `10,000` uniformly sampled target RAE points.
2. `512` static and `512` input-dependent dynamic queries mixed before the
   posterior cross-attention.
3. A Gaussian posterior with shape `512 x 32`.
4. A 24-layer latent self-attention decoder.
5. Exactly `625` occupied and `9,375` empty training queries. Both classes use
   within-cell `[-0.5, 0.5]` jitter.
6. Coordinate Fourier embedding as the only input to the occupancy query
   decoder.
7. Training loss
   `0.1 BCE_occupied + 1.0 BCE_empty + 1e-3 KL`.
8. EDM clean-latent prediction with `P_mean=-1.2`, `P_std=1.2`,
   `sigma_data=1`, and an 18-step Heun sampler over
   `sigma_max=80`, `sigma_min=0.002`, `rho=7`.
9. A 24-layer denoiser in which every block contains latent self-attention,
   radar cross-attention, feed-forward processing, and noise conditioning.

The all-query unweighted BCE used by upstream evaluation is not a training
loss.

## K-Radar adaptation

The official short-range random query grid is not transferred unchanged to the
120 m K-Radar domain. G1E uses a deterministic, non-learned Full-RAED proposal
support:

1. Reuse the source-bound G1D proposal cache to obtain `32,000` Cube-energy NMS
   coarse RAE queries.
2. Decode occupancy for all coarse queries using only coordinate embedding and
   the generated or target latent.
3. Select the top `2,500` coarse queries by occupancy logit.
4. Apply the frozen four-way sub-cell refinement pattern to produce exactly
   `10,000` output coordinates.

Cube values, local spectra, integrated energy, range features, learned G1D
offsets, and anchor features must not enter the implicit decoder. The Cube may
affect geometry only through the deterministic proposal support and the EDM
condition. CFAR helpers remain prohibited.

The radar encoder keeps the complete normalized 64-bin RAED input and produces
`16 x 7 x 3 = 336` condition tokens. This is the single intentional condition
change from RaLD's released intensity-only configuration.

## D0: retrospective query-support diagnosis

Before any new training, evaluate the existing source-bound one-frame
`R1-fidelity` and `R1-KRadar` checkpoints with the G1E proposal support. Keep
their posterior, decoder weights, target frame, and output count unchanged.

D0 passes only if at least one checkpoint satisfies all of:

- proposal-support Chamfer `<= 5.0 m`;
- outlier fraction at 2 m `<= 25%`;
- Chamfer improves by at least `30%` relative to that checkpoint's archived
  full-grid top-10k result;
- exactly `32,000 / 2,500 / 10,000` coarse, selected, and final queries;
- no test frame, CFAR helper, learned G1D checkpoint, or local Cube decoder
  feature is accessed.

If D0 fails, query allocation is not sufficient to rescue the existing RaLD
occupancy representation. G1E closes without VAE or EDM retraining.

### D0 outcome

D0 ran on H200 source `da6f8a5f` after `216` regression tests passed. Both
archived runs preserved their source-bound best checkpoint and train frame:

| Archived VAE | Full-grid Chamfer | Proposal-support Chamfer | Relative change | Outlier@2m |
|---|---:|---:|---:|---:|
| R1-fidelity | 10.9985 m | 10.7680 m | -2.10% | 7.32% |
| R1-KRadar | 9.9612 m | 11.4948 m | +15.40% | 7.41% |

Both arms preserved the exact `1,000 / 32,000 / 2,500 / 10,000`
seed/coarse/selected/final counts and used only normalized coordinates plus the
target latent in the decoder. Neither reached `5.0 m` or the required 30%
improvement. D0 therefore failed and E1/E2 are not authorized. The result is
archived at `artifacts/g1/g1e_d0_da6f8a5f.json` with SHA-256
`f59d1afbf015e2004e32575826b437afa1ffad535b2bc318170ac037f8c147e5`.

## E1: target-latent VAE

D0 is the only authorization for E1. Train one Stage-A seed (`20260719`) for
150 epochs on the frozen train split. Validation uses the deterministic
posterior mean and the G1E proposal support; stochastic best-of-k selection is
forbidden.

E1 must pass all frozen G1D geometry gates:

- median Chamfer `<= 2.50 m`;
- median completeness `<= 0.65 m`;
- mean outlier fraction at 2 m `<= 25%`;
- far-range completeness `<= 8.0 m`;
- duplicate fraction at 5 cm `<= 10%`;
- occupied-query recall `>= 80%`;
- empty-query false-positive rate `<= 20%`.

Cross-scene target-latent shuffling must worsen occupancy BCE or Chamfer by at
least `1%`. All posterior, latent decoder, and coordinate-query modules must
receive finite nonzero gradients. E1 failure is a target occupancy
representation failure and does not test radar conditioning.

## E2: Full-RAED latent EDM

Only a passing E1 checkpoint may create the frozen train-latent cache. E2
trains one Stage-A model seed for 100 epochs and selects the final epoch rather
than validation-best sampling noise.

Formal evaluation uses three fixed sampling seeds per frame and reports their
predeclared aggregate, never best-of-k. E2 must:

- pass the E1 geometry gates;
- make cross-scene radar-condition shuffling worsen denoising loss or Chamfer
  by at least `1%`;
- deliver finite nonzero gradients to the Full-RAED encoder and all 24
  radar-cross-attention blocks;
- preserve the `512 x 32` latent shape and official EDM schedule;
- keep the test split untouched.

Stage B trains model seeds `20260720` and `20260721` only after Stage A passes.
The three-seed comparison must show that, relative to the frozen G1D
zero-offset proposal control, Chamfer and outlier do not both regress and at
least one scene-first bootstrap upper confidence bound is below zero.

## Required ablations and successors

After E2 passes:

- compare the released RaLD intensity-only condition against Full-RAED under
  matched capacity and optimization;
- G2E adds scalar and circular-distribution Doppler heads to the same
  latent-query feature, followed by the Cube-point cycle gate;
- G4E adds ego-warped historical physical-state tokens to the EDM condition,
  while retaining the current Cube;
- P5 remains locked until the G1E/G2E/G4E family is frozen.

Geometry passing alone supports only RaLD-style Cube-to-dense latent
generation. It does not support generated Doppler, temporal consistency, or
downstream claims.
