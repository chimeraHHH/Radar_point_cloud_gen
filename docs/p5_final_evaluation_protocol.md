# P5 Frozen Test, Generalization, and Downstream Protocol

## Evidence Boundary

P5 is reporting-only. The single-frame parent arm is frozen by G3 and the
temporal fusion family is frozen by the G4 validation decision before any test
Cube is downloaded or evaluated. Test results cannot change a model,
checkpoint, threshold, loss, fusion family, or paper gate.

If G3 fails, its matched C0 fallback remains the declared single-frame parent.
If G4 fails, temporal results are reported as appendix evidence and cannot
replace the single-frame main result. Neither fallback retroactively passes a
failed validation gate.

## Frozen Test Cohort

- Source split: the existing sequence-isolated K-Radar split.
- Partition: all eight untouched test sequences.
- Sampling: one centered contiguous 48-frame window per sequence.
- Size: 384 frames and 376 adjacent-frame pairs.
- Data: full RAED Cube, synchronized OS2 LiDAR, official labels and odometry.
- Construction: the same manifest, CRC, deskew, CFAR and radar-observable
  target code used by G4. Test receives a separate immutable manifest, download
  verification report, dense cache report and parent-prediction cache.

The selection rule uses no model output, object count, weather, speed or target
quality. No test sequence may occur in a train or validation artifact.

## Frozen Methods

- T0: the same-seed G3-selected C0/C3 single-frame parent.
- T3: the frozen four-history DoppDrive-style aggregation baseline.
- T*: the G4-selected learned temporal family, evaluated with strict recurrent
  state for all 48 frames even when G4 failed.

All three seeds `20260716`, `20260717`, and `20260718` are evaluated. The first
frame of every window is the same frozen parent prediction for T0, T3 and T*.

## Primary Metrics

Report the existing current-frame geometry, Doppler spectrum, static PCE,
confidence, covered-cell and temporal consistency metrics. Report strict
rollout at steps 1, 5, 10 and 25. Inference latency uses three warm-up passes,
CUDA events, peak allocated memory and exact 10,000-point throughput.

Test comparisons use paired seed-by-scene bootstrap with 10,000 resamples and
seed `20260718`. Confidence intervals are descriptive; no test-derived
pass/fail rule or model reselection is allowed.

## Downstream Velocity Task

The required downstream task is object-centric radial velocity estimation.
Official K-Radar track IDs match boxes between adjacent frames. The target is
the finite-difference range rate of the matched box center. For every method,
generated points inside the current box vote with confidence in circular
Doppler space using the frozen Cube alias period. Report:

- box radial-speed MAE and RMSE under circular error;
- fractions within 0.5 and 1.0 m/s;
- box support rate at 1, 5 and 10 generated points;
- circular-resultant strength and unsupported-box count.

The same zero-margin boxes also select radar-observable LiDAR targets. Report
object-level Chamfer, completeness and 1 m F-score so official class, distance
and speed slices cover geometry as well as radial velocity.

The box inclusion margin is zero and cannot be tuned on test. This task tests
whether a dense `XYZ + Doppler` cloud supports a physical downstream estimate;
it is not presented as a replacement for a trained 3D detector.

## Generalization Slices

Report every primary and downstream endpoint by:

- sequence description tags: road type, time of day and weather;
- range: 0-30, 30-60 and 60-120 m;
- absolute box radial speed: 0-0.5, 0.5-2 and at least 2 m/s;
- official object class;
- static and dynamic generated-point subsets.

Every slice records seed, scene, frame and object counts. Empty or undercovered
slices remain explicit and are never silently dropped from a claimed average.

## Failure Taxonomy

For each seed and method, rank frames without manual intervention into frozen
categories: geometry outlier, Doppler mismatch, static-PCE failure, confidence
collapse, coverage collapse and long-rollout drift. Save the worst five frames
per category and the same fixed eight windows for qualitative rendering.

## Required Artifacts

- immutable test manifest and CRC verification;
- complete dense-target and three parent-prediction caches;
- T0/T3 and T* per-frame predictions for all seeds;
- scene-first bootstrap JSON;
- object radial-velocity JSON with all box-level observations;
- weather, range, speed and class tables;
- efficiency table and failure taxonomy;
- fixed-window figures, worst-five figures and video-ready frame sequences;
- checkpoint hashes, evaluator commit, environment and command provenance.
