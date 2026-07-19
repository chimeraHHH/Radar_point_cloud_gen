# RaLD-anchor hybrid protocol

## Motivation

Three one-frame studies showed that the independent RaLD point VAE can produce
high-precision K-Radar points but cannot allocate a fixed top-10k implicit
occupancy budget across the 0-120 m field of view. Its latent EDM is therefore
not trained. The reusable RaLD mechanism is moved after the geometry parent
instead of replacing it.

## Architecture

1. The frozen frustum-occupancy parent produces top-10k RAE anchor cells and
   per-anchor spatial features.
2. Each anchor combines Fourier RAE coordinates with its parent feature.
3. RaLD static and dynamic mixed queries cross-attend to all anchors, producing
   512 order-invariant latent tokens.
4. A latent self-attention stack models global point-set context.
5. Anchor queries cross-attend back to the latent set.
6. The zero-initialized physical query head predicts bounded RAE offsets,
   residual circular Doppler distributions over the local measured Cube
   spectrum, and confidence.

This hybrid borrows RaLD's set-latent bottleneck and implicit query decoder while
retaining the current model's verified long-range geometry allocation.

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

### RH1: one-frame physical refinement

Use a frozen G1 geometry parent. Train only the hybrid refiner on one train
frame. Require nonzero gradients after the zero-initialized first step, bounded
offset saturation, no confidence collapse, and geometry no worse than the
parent while Doppler NLL improves over direct Cube query.

### RH2: development ablation

Run only after the current G1 family is formally decided. Compare the selected
parent against `+RaLD-anchor-hybrid` using the same development frames, seeds,
and scene-first statistics. This is a newly named method branch and cannot be
reported as the original G1 recovery.

## Evidence boundary

The hybrid does not reopen the failed independent RaLD AE or authorize its
latent-cache/EDM chain. It also does not unlock G2/G3 before the frozen parent
gate closes.
