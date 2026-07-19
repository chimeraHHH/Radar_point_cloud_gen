# RaLD-inspired physical point generation protocol

## Scope

The matched RaLD baseline no-go applies to that baseline's K-Radar adaptation
and frozen one-frame gate. It does not reject RaLD's latent point-generation
architecture. This protocol promotes the reusable parts of RaLD into a separate
main-method candidate without retroactively changing G1 or its recovery.

## Borrowed from RaLD

1. An order-invariant mixed-query point VAE with `512 x 32` latent tokens.
2. An implicit occupancy decoder evaluated at arbitrary normalized RAE queries.
3. A spatial radar encoder flattened into range, azimuth, and elevation tokens.
4. A radar-conditioned latent Transformer with adaptive normalization.
5. EDM preconditioning and deterministic 18-step second-order sampling.

The Apache-2.0 upstream reference is fixed at
`ffec4b41241391734b1eda5c093de843c909eb8e`.

## Required changes for this work

RaLD's intensity-only RAE condition is replaced by a learned projection of all
64 Doppler bins before the spatial radar encoder. The implicit decoder exposes
per-query features. A zero-initialized physical head then predicts:

- bounded continuous RAE offsets;
- a 64-bin circular Doppler distribution initialized to the local measured Cube
  spectrum;
- independent point confidence.

The generated points are subsequently rendered through the existing
point-to-RAED soft splatter for local-spectrum, Doppler-marginal, and spatial
cycle losses. CFAR query helpers remain prohibited in the main method.

## Geometry adaptation

The prior matched baseline sampled target surfaces and positive queries with
radar-observability confidence. This suppressed long-range geometry. The
RaLD-inspired method instead freezes uniform target-surface and occupied-cell
sampling, matching the upstream binary-occupancy semantics more closely.
Confidence remains an output target and an evaluation weight; it is not an
occupancy label.

## Execution gates

### R0: component integrity

- all 64 input Doppler channels receive gradient through the spectral projection;
- full native K-Radar Cube produces exactly `16 x 7 x 3 = 336` radar tokens;
- physical-head initialization preserves the queried Cube spectrum, zero offset,
  and neutral confidence;
- full-scale forward/backward fits one H200 and records peak memory.

### R1: point-AE feasibility

Run one frame for 100 epochs with the original RaLD AE overfit thresholds. The
only protocol differences from AE-B1 are uniform surface and positive-query
sampling. This is a new method branch, not a second matched-baseline repair.

The initial R1 run failed only the Chamfer check (`8.8966 m`) while passing
outlier, F-score, confidence, and optimization checks. Inspection found a
remaining upstream-fidelity mismatch: official RaLD samples positive queries
uniformly inside occupied voxels, whereas R1 supervised exact continuous LiDAR
locations and decoded at grid centers. One named `R1-fidelity` run is therefore
allowed with occupied-voxel jitter and the official 625/9,375 positive/negative
query ratio. All other R1 settings and thresholds remain frozen. If this run
fails, R2 remains locked.

`R1-fidelity` also failed only Chamfer (`10.9985 m`) and exposed the expected
short-range-domain mismatch: the upstream `0.1 / 1.0` positive/negative loss
weights produce high-precision but low-coverage occupancy on K-Radar. One final
`R1-KRadar` adaptation is allowed before abandoning the independent point VAE:
retain uniform surface sampling, occupied-voxel jitter, the 625/9,375 query
ratio, and all model/optimizer settings, but set positive and negative occupancy
weights to `1.0 / 1.0`. The original R1 gate remains unchanged.

If `R1-KRadar` fails, the independent point-VAE path closes and R2 does not run.
The already verified Full-RAED radar-token encoder, latent Transformer/EDM, and
physical query head may still be fused with the deterministic frustum-occupancy
parent under a separately named hybrid protocol.

### R2: conditional latent feasibility

Only after R1 passes, cache the frozen latent and overfit the full-RAED
conditioned EDM on the same frame. Require decreasing latent RMSE and decoded
geometry that remains within the R1 gate.

### R3: development comparison

Only after R2 passes, train one seed on all 76 train frames and compare against
the frozen frustum-occupancy parent on validation. Release three seeds only when
geometry does not regress beyond the G1 tolerance and the latent generator adds
either sample quality or physical-head capacity not available to the parent.

## Evidence boundary

R0-R3 do not alter the current `0e5fe84` G1 decision. If that G1 fails, the
RaLD-inspired model is a separately named redesign and requires an explicit
promotion decision before any G2/G3 successor is launched.
