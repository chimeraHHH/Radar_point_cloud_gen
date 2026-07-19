# Paper Outline: Full-RAED Cube to Physically Consistent Dense Radar Points

> Status: evidence-driven outline, updated 2026-07-19 16:20 CST
> Main task: current-frame 4D Radar Cube -> 10,000 radar-observable `XYZ + Doppler distribution + confidence` points
> Optional extension: warped historical prediction as a prior corrected by the current Cube

## One-Sentence Thesis

Dense radar reconstruction should preserve the full Doppler spectrum and jointly infer point geometry, circular Doppler uncertainty, and visibility, with generated points required to explain the measured Cube through a differentiable point-to-Cube cycle.

## Evidence Status

| Claim | Gate | Status | Paper treatment |
|---|---|---|---|
| K-Radar Cube, LiDAR target, axes, timing, and observability are reliable | G0 | Passed: 100/100 frames, 11/11 checks | Methods / data protocol |
| Full-RAED improves dense geometry over matched RAE-Max | G1 | First formal run failed; bounded recovery has two Full-RAED seeds complete and the third running | Do not state as result |
| A per-point Doppler distribution is better than a scalar head | G2 | Pending G1 | Candidate contribution |
| Differentiable point-to-Cube cycle adds independent value without confidence collapse | G3 | Pending G2 | Central candidate contribution |
| Current-Cube temporal refresh improves stability and accuracy | G4 | Manifest passed; 12/45 development sequences complete and download active | Optional main text or appendix |
| RaLD is a protocol-matched quantitative baseline | Baseline gate | Official checkpoint is domain/output mismatched; matched AE failed the frozen one-frame Chamfer gate after one repair | Related work and no-go record only |
| Analytic static-Doppler mixture is valid on this cohort | Static audit | Failed validation and bounded recovery | Negative result; E5 removed |
| Frozen test, downstream velocity, slices, and efficiency | P5 | Test remains untouched | Final evidence only |

## Proposed Paper Structure

### Abstract

Write last. It may contain only claims that passed their frozen gate. Required elements:

1. Problem gap: prior dense radar reconstruction focuses mainly on geometry and commonly compresses or omits the Doppler axis.
2. Method: full-RAED encoder, dense occupancy decoding, point-conditioned circular Doppler distributions, confidence, and differentiable Cube cycle.
3. Optional temporal extension: historical points are a physical prior, not a replacement for the current Cube.
4. Results: fill only after G1-G3 and P5 are frozen.
5. Limitation: analytic static prior was not stable on the current K-Radar cohort.

### 1. Introduction

Paragraph 1: 4D radar provides spatial and Doppler measurements, yet downstream stacks usually consume sparse CFAR points or geometry-only reconstructions.

Paragraph 2: geometry-only densification is under-constrained. A dense point should carry a locally supported velocity distribution and confidence, and the resulting set should remain explainable by the measured Cube.

Paragraph 3: formulate full-RAED-conditioned dense radar reconstruction. Explain the two directions: Cube-to-point inference and differentiable point-to-Cube verification.

Paragraph 4: optional temporal prior. A gated Doppler warp propagates a previous prediction, while the current Cube remains the source of new evidence and corrects stale history.

Contributions must be selected from the claim ledger after gates close. Do not include the rejected analytic static mixture.

### 2. Related Work

1. Radar point-cloud densification and radar-to-LiDAR generation.
2. Radar representations using RAE, RAED, tensors, spectra, and learned Cube encoders.
3. Doppler-aware detection, motion estimation, scene flow, and temporal aggregation.
4. Differentiable rendering and cycle consistency for set-to-grid and grid-to-set reconstruction.

Required boundary: DoppDrive-style aggregation reuses historical points; this work generates a new current-frame set conditioned on the current Cube.

RaLD is an architecture and task-positioning reference, not a headline numeric
baseline in the current protocol. Its official checkpoint is trained on
ColoRadar with an intensity-only condition and geometry-only output. The
from-scratch K-Radar matched AE did not pass its frozen one-frame geometry gate,
so its latent diffusion stage was not trained.

### 3. Problem Formulation and Data Protocol

1. Canonical Cube `C_t in R^(64 x 256 x 107 x 37)` and train-only normalization.
2. Output set of 10,000 points with `XYZ`, 64-bin circular Doppler distribution, scalar summary, and confidence.
3. Radar-observable LiDAR target, RAE occupancy, scene-isolated train/validation/test split.
4. G0 axis, timing, synchronization, CFAR round-trip, and observability checks.
5. Explicit evidence boundary: no-deskew was selected from train-only evidence; the static-Doppler convention did not validate.

### 4. Method

#### 4.1 Matched Cube Encoder and Dense Occupancy

- RAE-Max and Full-RAED differ only at the input projection.
- Shared lightweight 3D residual U-Net over the RAE grid.
- Soft occupancy supervision and unique top-10k RAE-cell decoding.

#### 4.2 Point-Conditioned Circular Doppler Distribution

- Query spatial features at selected cells.
- Predict scalar or 64-bin distribution under a matched geometry parent.
- Circular mean, wrapped errors, circular W1, and confidence-aware supervision.

#### 4.3 Continuous Point Parameterization

- Predict bounded `[-0.5, 0.5]` offsets in range, azimuth, and elevation bins.
- Convert continuous RAE coordinates to Cartesian XYZ.
- Keep the offset head structurally matched in all cycle ablations.

#### 4.4 Differentiable Point-to-Cube Cycle

- Trilinear spatial splatting and per-point Doppler mass.
- Local spectrum KL, global Doppler marginal KL, and sparse spatial-energy loss.
- Confidence floor plus explicit anti-collapse checks on confidence, coverage, ECE, and offset saturation.

#### 4.5 Current-Observation Temporal Refinement

- Gated residual-Doppler warp followed by ego transform.
- Concat, local cross-attention, and draft-refinement variants.
- Current Cube always remains present.
- Scheduled sampling reaches 0.4; recurrent predictions are detached.

### 5. Experiments

#### 5.1 Setup

- K-Radar sequence-isolated split: 37 train, 8 validation, 8 untouched test sequences.
- G0/G1 development cohort: 100 frames from 45 train/validation sequences.
- G4 cohort: 45 windows x 48 frames = 2,160 frames, test excluded.
- Three seeds: 20260716, 20260717, 20260718.
- Scene-first paired bootstrap; no frame-level pseudo-replication.

#### 5.2 Full Doppler Spectrum for Geometry

- E0 official CFAR, E1 RAE-Max, E2 Full-RAED.
- Geometry, completeness, distance slices, duplicates, outliers, and qualitative failures.
- This section exists in the main paper only if G1 passes.

#### 5.3 Per-Point Doppler Representation

- Q0 direct spectrum query, E3 scalar, E4 distribution.
- NLL/KL, circular W1, mode accuracy, scalar circular MAE, CD-Doppler, PCE, and ECE.
- E5 appears only as a preregistered failed branch in appendix/limitations.

#### 5.4 Cube-Point Cycle

- C0 no cycle; C1 local spectrum; C2 plus marginal; C3 plus spatial energy.
- Jointly report geometry, spectrum, confidence, coverage, ECE, and saturation.
- Include fixed-frame panels and worst-five failures.

#### 5.5 Temporal Extension

- Single-frame T0, aggregation T3, concat T4, cross-attention T5, draft refinement T6.
- Matched radial error, flicker, Doppler refresh, geometry, and 1/5/10/25-step rollout.
- Move to appendix if G4 does not improve both temporal consistency and current-frame quality.

#### 5.6 Frozen Test and Downstream Value

- Test is released only after G4 family freeze.
- Object radial-velocity estimation, distance/weather/class slices, latency, memory, and failure taxonomy.
- Test results are descriptive and cannot select a model.

### 6. Limitations

1. The static-Doppler analytic convention did not generalize from train to validation; E5 is omitted.
2. Radar-observable LiDAR targets are proxies, not ground-truth radar returns.
3. The current development cohort is small; all claims require frozen test confirmation.
4. Doppler is radial and aliased; tangential motion remains under-observed.
5. Temporal results depend on completion and verification of a 600.8 GiB official-data cohort.

## Main Figures

1. **Figure 1:** Full-RAED Cube -> dense point set -> differentiable Cube cycle; temporal prior shown as optional side branch.
2. **Figure 2:** selected RAE locations, 64-bin Doppler distributions, continuous offsets, and soft splatting.
3. **Figure 3:** geometry/Doppler/confidence qualitative comparison and failure cases.
4. **Figure 4:** cycle ablation with spectrum, coverage, confidence, and calibration diagnostics.
5. **Figure 5:** temporal rollout only if G4 passes; otherwise supplementary.

## Main Tables

1. **Table 1:** CFAR, matched RAE-Max, and Full-RAED geometry results with scene-first uncertainty.
2. **Table 2:** Q0, scalar, distribution, and cycle ablations with geometry plus Doppler metrics.
3. **Table 3:** temporal comparison, conditional on G4 pass.
4. **Table 4:** frozen-test downstream velocity, slices, latency, and memory.

## Stop and Branch Rules

- G1 recovery fails: close the current Full-RAED parent and stop its G2/G3 queue. The separately preregistered G1B spectrum screen may run, but it cannot silently replace the parent or release later gates without a new explicit decision.
- G2 fails: Doppler distribution is not a contribution; investigate supervision before G3.
- G3 fails: do not claim a strong bidirectional method; reposition as a dense reconstruction baseline or redesign the renderer.
- G4 fails: temporal branch moves to appendix and does not weaken the single-frame paper.
- P5 contradicts validation: report the gap; do not reopen model selection on test.

## Archived Prior Work Boundary

TruckScenes FlowRadar/DopplerConsist results support the design motivation for gated Doppler warp and scheduled sampling. They are not evidence that the current K-Radar Cube-to-dense model passes G1-G4 and are excluded from current main-result tables.
