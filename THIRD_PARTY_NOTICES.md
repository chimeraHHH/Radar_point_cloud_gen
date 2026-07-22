# Third-Party Notices

## RaLD

The RaLD-inspired and G3L modules adapt the architecture and sampling protocol
described by:

- Ruijie Zhang, Bixin Zeng, Shengpeng Wang, Fuhui Zhou, and Wei Wang,
  "RaLD: Generating High-Resolution 3D Radar Point Clouds with Latent
  Diffusion."
- Official source: <https://github.com/MetaIoT-WHU/RaLD>
- Audited source commit: `ffec4b41241391734b1eda5c093de843c909eb8e`
- Upstream license: Apache License 2.0

This repository changes the upstream problem formulation and implementation:
it uses K-Radar's native grid, the complete 64-bin RAED Cube, independently
gated long-range anchor queries, per-point circular Doppler distributions,
confidence, final-position Cube lookup, and point-to-RAED cycle constraints.
