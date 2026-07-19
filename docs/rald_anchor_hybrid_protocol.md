# RaLD-anchor hybrid protocol

## Motivation

Three one-frame studies showed that the independent RaLD point VAE can produce
high-precision K-Radar points but cannot allocate a fixed top-10k implicit
occupancy budget across the 0-120 m field of view. Its latent EDM is therefore
not trained. The reusable RaLD mechanism is moved after the geometry parent
instead of replacing it.

## Architecture

1. A formally selected frozen frustum-occupancy parent produces top-10k RAE
   anchor cells and per-anchor spatial features. If G1 passes, this is the
   Full-RAED arm. If G1 fails but RAE-Max independently passes the frozen CFAR
   geometry gate, RAE-Max may be used only in the separately named G1R/RH
   late-fusion recovery. If neither original arm passes, RH waits for an
   independently passing G1B Stage B candidate and records the route as
   `independent_g1b_parent`.
2. Each anchor combines Fourier RAE coordinates with its parent feature.
3. A RaLD Full-RAED hierarchy projects all 64 Doppler bins into 336 spatial
   radar-condition tokens.
4. RaLD static and dynamic mixed queries cross-attend to both the complete
   anchor set and the radar-condition tokens, producing 512 order-invariant
   latent tokens.
5. A latent self-attention stack models global point-set context.
6. Anchor queries cross-attend back to the latent set.
7. The zero-initialized physical query head predicts bounded RAE offsets,
   residual circular Doppler distributions over the local measured Cube
   spectrum, and confidence.

This hybrid borrows RaLD's radar-token hierarchy, mixed set-latent bottleneck,
and implicit query decoder while retaining the current model's verified
long-range geometry allocation.

## Gates

### RH0: component verification

- anchor-order permutation leaves latent tokens unchanged;
- permuting anchors permutes per-anchor outputs identically;
- 10k anchors and 512 latent tokens fit one H200;
- initialization preserves geometry anchors and measured Doppler spectra.

Status: passed at source `a1c862a` on physical H200 GPU 2. The native Cube
produced 336 Full-RAED tokens with gradients through all 64 Doppler bins. The
10k-anchor hybrid produced `512 x 512` latent tokens and 10k offset outputs with
28,730,436 parameters and 1.35 GB peak allocated CUDA memory. Evidence:
`artifacts/baselines/rald/anchor_hybrid_rh0_a1c862a.json`.

RH0 originally verified the radar-token and anchor-latent branches separately.
Source `a7c36d4` connects the 336 Full-RAED tokens directly to the dynamic mixed
latents and passes 101 repository tests on H200. Native two-step integration is
tracked as RH0.5 and remains pending an available H200.

### RH1: one-frame physical refinement

Use a frozen G1 geometry parent. Train the radar-token encoder and hybrid
refiner, never the parent, on one train frame. Require nonzero physical-head
gradient on step one and nonzero set-latent and radar-token gradients on step
two, bounded offset saturation, no confidence collapse, and geometry no worse
than the parent while Doppler NLL improves over direct Cube query.

Parent selection is automatic and immutable after the formal G1 comparison.
Using RAE-Max after a Full-RAED early-fusion failure does not change the G1
decision: it tests the distinct hypothesis that RaLD late fusion can preserve a
strong geometry allocator while injecting complete spectral context.

The bounded G1 recovery ultimately rejected both original parent routes. RH now
requires an independently passing G1B Stage B candidate. Its exact parent runs,
training source commit, and summary hash are part of RH provenance; appearance
or modification of a G1B file cannot alter a run on an original G1 route.

### RH2: development ablation

Run only after the current G1 family is formally decided. Compare the selected
parent against `+RaLD-anchor-hybrid` using the same development frames, seeds,
and scene-first statistics. This is a newly named method branch and cannot be
reported as the original G1 recovery.

When original G1 fails, RH2 does not wait for or unlock the original G2/G3
summary. It records that dependency as unavailable by protocol and remains an
independent late-fusion comparison.

## Evidence boundary

The hybrid does not reopen the failed independent RaLD AE or authorize its
latent-cache/EDM chain. It also does not unlock the original G2/G3 chain after a
failed G1. G1R/RH has its own RH1/RH2 gates and must be reported as a separate
late-fusion recovery branch.
