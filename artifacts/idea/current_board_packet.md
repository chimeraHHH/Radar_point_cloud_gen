# Cube-to-dense current board packet

> Refreshed 2026-08-07 at repository commit `083f4c4`. Test access is false.

## Incumbent

There is no passing single-frame geometry incumbent. The strongest current
mechanism evidence is the fresh R-A1 replay candidate pool, not its deployed
output:

- current-confidence exact-10k: Chamfer `4.0261 m`, outlier `31.7854%`;
- validation-GT-nearest exact-10k on identical candidates:
  Chamfer `0.6375 m`, outlier `2.2771%`, far completeness `0.7046 m`;
- candidate coordinates, range quotas, exporter, and 5 cm spacing are fixed.

This isolates geometric-utility prediction and structured selection as an open
mechanism. It does not authorize reuse of the deleted R-A1 identity.

## Final route decisions

| Route | Decisive result | Boundary |
|---|---|---|
| RAE-Max | Chamfer `2.9306 m`, outlier `25.697%` | Useful compact baseline; geometry gate failed |
| Full-RAED early fusion | `+5.86%` Chamfer vs RAE-Max | Early spectrum fusion rejected |
| G1D direct query field | Endpoint degraded; condition effect weak; duplicates high | Deterministic arbitrary-query family closed |
| G1G hierarchy | Completeness/far support improved, but outlier `84.685%`, duplicates `70.811%`, shuffle `-0.121%` | Current condition-exclusive patch allocator closed |
| G1E/RaLD latent occupancy | Source-faithful proposal-support gate failed | Current occupancy VAE/EDM route closed |
| G1F fixed 32k selector oracle | GT-aided subset still failed Chamfer/completeness/duplicates | Selector-only repair on that pool closed |
| R-A1/R-A2 | Wide candidates exist, but binary occupancy/confidence does not select them | Binary score recipe closed |
| Q1-R | Replay failed four original-endpoint equivalence tolerances | Quality hypothesis untested; old-parent route closed |
| R-B2 | 80k `max_d` recall passed on 76/76, confidence coverage failed on 3 frames | Current score-plus-fixed-neighborhood activation closed |
| G1T | Ego/Doppler union changed geometry by less than `0.05%` and remained very poor | Current no-train history proposal route closed |

## Important contradiction

The project has moved past the question of whether a sufficiently wide spatial
support can exist. On two independently trained R-A1 pools, an unattainable
geometric score selects an excellent exact-10k subset, while binary occupancy
confidence fails badly.

The unresolved question is whether radar-only evidence can predict that utility
without:

1. inheriting a deleted checkpoint identity;
2. reading target geometry at inference;
3. collapsing coverage, range quotas, or point diversity;
4. bypassing the global Cube condition through local energy alone.

## Closed assumptions

- early Full-RAED channel fusion is not sufficient;
- larger deterministic query fields are not sufficient;
- binary occupancy probability is not geometric quality;
- reducing the 10k output count is not an allowed or sufficient repair;
- fixed-neighborhood voxel activation is not robust across all train frames;
- the current ego/Doppler union proposal does not supply useful geometry;
- diffusion, Doppler, cycle, or temporal modules cannot compensate for a failed
  single-frame geometry parent.

## Selected live mechanism

**Q-Local-F0** freezes the fresh replay as `Fresh-WCE-20`, keeps its 700k
candidate coordinates and exact exporter, and predicts a six-bin geometric-risk
distribution from global Cube latents plus local 64-bin Doppler evidence. It is
train-only, uses eight fit sequences plus four unseen training sequences, and
has a 500-update/7200-GPU-second hard stop.

The same-coordinate wrong-Cube control swaps only global and local Cube
evidence; candidates, parent confidence, target, and exporter remain fixed.
This directly tests radar-conditioned scoring instead of another support field.

If Q-Local-F0 fails, the outside-family fallback is a variable multi-return
ray-hazard representation. Sparse ray-range transport is deferred behind its
own hard-rounding oracle. Adaptive fixed-neighborhood support is closed with
R-B2 and is no longer listed as live.

## Stale routes not to reopen

- threshold relaxation or cardinality reduction;
- G1B spectral-rank variants;
- another G1C/G1D deterministic query refiner;
- G1G with only a different patch count or radius;
- R-A2 binary occupancy with a different class sampler;
- Q1-R under the original-parent claim;
- R-B2 with only a larger fixed bank or neighborhood;
- G1T with only a longer history;
- any downstream Doppler/temporal run before a new parent passes.

## Independent infrastructure progress

The 45-sequence, 2,160-frame G4 download is complete. A fresh exact-member,
size, and CRC audit is running from clean H200 source `083f4c4`; temporal
training remains locked regardless of the data-audit outcome.
