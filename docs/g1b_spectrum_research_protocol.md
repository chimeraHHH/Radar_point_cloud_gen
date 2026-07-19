# G1B Physics-Compressed Spectrum Research Protocol

## Status And Boundary

G1B is an independent research branch that may start only after the bounded G1
recovery is formally decided. It does not reopen G1, weaken its thresholds, or
permit the `0e5fe84` G2/G3 queue to run after a failed G1. Any successful G1B
representation creates a new parent family and requires a separately named
G2B/G3B evidence chain.

## Question

The original Full-RAED projection applies an unconstrained `64 -> 8` mapping at
every RAE cell. G1B tests whether geometry benefits from a smaller, physically
structured summary of the circular Doppler spectrum instead of the complete
per-cell spectrum.

## Candidates

All candidates retain the matched RAE-Max spatial backbone, occupancy target,
decoder, point count, optimizer, and evaluation code.

| ID | Representation | Added spectral capacity |
|---|---|---|
| B0 | RAE-Max | none |
| B1 | max + linear mean/std | existing three-channel diagnostic |
| B2 | max + real/imag circular harmonics at orders 1 and 2 | five input channels |
| B3 | RAE-Max + zero-initialized rank-2 residual of the 64-bin spectrum | `64 -> 2 -> 8` |

B2 preserves circular phase and distinguishes spectra that share coarse linear
moments. B3 is exactly function-matched to B0 at initialization and limits the
learned full-spectrum correction to rank 2. No candidate may exceed a 1%
parameter increase over B0.

## Two-Stage Gate

### Stage A: One-Seed No-Go Screen

- Seed: `20260716`.
- Epochs: 15.
- Data: the frozen G0 train/validation cohort; test remains sealed.
- Arms: B0-B3 from scratch under one new source commit.
- Report all G1 geometry endpoints and spectral-branch gradient/RMS diagnostics.

A candidate survives only if its validation median Chamfer is no worse than B0
by 2%, its mean outlier fraction is at most 25%, and it improves at least one of
overall Chamfer, 60-120 m completeness, or 60-120 m F-score. The screen is a
research decision, not G1 evidence.

### Stage B: Independent Three-Seed Decision

Freeze the single surviving representation by the following order: lowest
median Chamfer, then lowest far completeness, then lower parameter count. Rerun
B0 and the frozen candidate for 50 epochs at seeds `20260716`, `20260717`, and
`20260718`. Use the original scene-first paired bootstrap and unchanged G1
geometry safeguards.

G1B passes only if the candidate's overall Chamfer degradation upper 95% bound
is at most 2%, the outlier fraction is at most 25%, and one preregistered
Doppler-sensitive geometry endpoint improves with a 95% interval excluding
zero. A failed Stage B closes the physics-compressed spectrum branch.

## Required Evidence

- exact source commit and frozen G0 artifact hashes;
- one-seed B0-B3 screen with no missing arms;
- parameter count, first-step gradient, and residual/trunk RMS diagnostics;
- three-seed paired report for the single frozen survivor;
- explicit statement that G1B is not the original G1 recovery;
- no G2B/G3B launch before the independent G1B decision.
