# Cube-to-dense current board packet

## Incumbent and contract

The current frozen run is G1D v2, source `4c6150cd`: a deterministic
Full-RAED-conditioned query field with 512 working latents, 24 conditioned
Transformer blocks, 32k coarse queries, and 10k refined output points. Direct
local Cube spectrum, standardized energy, and range enter every query-decoder
call.

The authoritative geometry thresholds are recorded in
`code/scripts/compare_rald_query_field.py`. Test access remains false.

## Durable results

| Route | Strongest result | Decision |
|---|---|---|
| G1 recovery | Full-RAED did not satisfy the frozen original gate | Closed |
| G1B spectrum | `full_raed_rank2`: Chamfer `2.0251 m`, outlier `28.885%`, far completeness `7.5757 m` | Failed outlier gate; no Stage B |
| G1C | No scientific training result | Superseded before metrics |
| G1D invalid runs | Loss, energy scale, label-phase, and shuffle-pair defects | Engineering-only; excluded |
| G1E-D0 | R1-fidelity `10.9985 -> 10.7680 m`; R1-KRadar `9.9612 -> 11.4948 m` under proposal support | Failed; independent occupancy VAE/EDM closed |

G1E-D0 shows that replacing full-grid top-10k with deterministic radar proposal
support is not sufficient to rescue the archived RaLD latent-only occupancy
decoder. This removes query allocation as the sole explanation for the old
point-VAE failure.

## Active intermediate evidence

G1D v2 is still training and has no scientific result. Through epoch 65:

- best selection occurred at epoch 15;
- epoch-15 Chamfer was `4.4823 m`, completeness `3.5811 m`, and outlier
  `17.90%`;
- occupied recall increased from `24.1%` at epoch 15 to about `50.4%` at epoch
  65;
- epoch-65 Chamfer was `5.2683 m`, completeness `3.5822 m`, outlier about
  `23.45%`, and duplicate fraction about `29.16%`;
- cross-scene condition-shuffle Chamfer degradation remained below `0.2%`,
  versus the frozen `1%` gate.

This is monitoring evidence only. The run continues unchanged to 150 epochs.

## Current contradiction

Two partially successful behaviors do not combine:

1. Compact Full-RAED occupancy models can approach the Chamfer and far-range
   gates, but retain too many spatial outliers.
2. The large G1D query field lowers outliers and increases occupied recall, but
   has poor completeness, excessive duplicates, and almost no measurable
   dependence on its global radar condition.

The likely architectural conflict is that direct local Cube evidence is strong
enough to dominate query scoring, while the 24-layer global condition path has
no exclusive information or bottleneck forcing it to matter.

## Stale routes not to reopen

- original G1 threshold relaxation;
- G1B variants with only a new spectral rank or scalar summary;
- G1C deterministic refiner under a new name;
- another point-VAE run that differs only in positive/negative weights;
- RaLD full-grid or proposal-support occupancy decoding without new target
  representation evidence;
- confidence-based masking that improves outlier by discarding coverage;
- best-of-k latent diffusion;
- G2/G4/P5 launched from a failed geometry family.

## Open candidate families

The parallel literature and code audit must compare at least:

1. **Measurement/objective:** range- and density-balanced transport or set
   matching that directly controls coverage, outliers, and duplicates.
2. **Mechanism:** condition-exclusive global allocation followed by local
   refinement, with the local Cube bypass removed or gated.
3. **Representation:** factorized occupancy/ray or range-view generation that
   better matches long-range radar geometry than full 3D point queries.
4. **Temporal-first:** ego/Doppler-warped history as a proposal prior, while the
   current Cube refreshes occupancy and Doppler.
5. **Conservative baseline:** retain the compact G1B geometry family and attack
   only its outlier tail with a source-independent, non-collapsing mechanism.

Only candidates with a distinct mechanism, official prior-art support, and a
one-to-two-hour falsification run may enter the next experiment frontier.

