# Matched RaLD baseline decision (2026-07-19)

## Decision

Stop the matched RaLD **baseline** training chain after the preregistered B1
repair. Do not train its RAE-Sum latent EDM or include that checkpoint as a
quantitative main baseline for the K-Radar protocol.

This decision does not affect the Cube-to-dense Module A gate. RaLD remains a
direct architecture source for implicit point occupancy and latent diffusion.
The separate Full-RAED physical generator is governed by
[`rald_inspired_mainline_protocol.md`](rald_inspired_mainline_protocol.md) and
must not be conflated with the failed matched baseline.

## Evidence

The official RaLD checkpoint is not a fair main baseline because it is trained
for ColoRadar, consumes an intensity-only RAE condition, and does not emit
point-level Doppler or confidence. A from-scratch matched K-Radar implementation
was therefore built and verified at the official model scale.

The first one-frame autoencoder run used target confidence as the positive
occupancy label and failed three geometry checks. The B1 repair changed only
that label to binary occupancy, matching the official RaLD formulation. B1 then
passed four of five original gate checks:

| Check | B1 result | Gate | Status |
|---|---:|---:|---|
| Chamfer distance | 9.1444 m | <= 5.0 m | fail |
| Outlier fraction at 2 m | 0.0920 | <= 0.5 | pass |
| F-score at 1 m | 0.3220 | >= 0.2 | pass |
| Mean top-10k confidence | 0.3822 | >= 0.05 | pass |
| Train-loss reduction | 70.24% | >= 30% | pass |

The final epoch makes the residual failure interpretable: near-range
completeness was `1.1579 m`, while 30-60 m completeness was `13.8297 m` and the
30-60 m F-score at 1 m was zero. The model learned high-precision occupied
predictions but allocated its fixed top-10k budget to the confidence-dominant
near range instead of covering the full target distribution.

## Evidence boundary

This is an implementation-specific no-go under the matched protocol, not a
claim that RaLD is generally ineffective. A range-balanced sampling redesign
could address the diagnosed failure, but selecting it from this one-frame result
would be a second adaptive repair and is outside the frozen baseline budget.

The main paper may cite RaLD as evidence that prior Cube-to-point generation is
single-frame and geometry-only. It must not report the failed matched run as a
competitive headline number without the no-go context.
