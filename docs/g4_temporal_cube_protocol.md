# G4 Current-Cube Temporal Prior Protocol

## Scope

G4 tests whether a warped historical prediction is useful as a prior for the
current-frame Cube-to-dense task. It does not redefine the task as historical
point aggregation: every learned temporal arm must consume the current full
RAED Cube and must output a newly inferred `XYZ + Doppler distribution +
confidence` point cloud.

G4 starts after a single-frame parent has been frozen. The parent is same-seed
C3 when G3 passes; if G3 fails, the matched same-seed C0 continuous-point arm
is used and the Cube-cycle claim remains failed. C0 keeps the exact offset and
point-query architecture while removing all cycle losses, so temporal arms
remain structurally comparable. This fallback cannot retroactively pass G3.

## Frozen Temporal Cohort

- Source: official K-Radar labels, OS2 timestamps, and official odometry.
- Split: the existing sequence-isolated G0 split; official within-sequence
  train/test lists are not used as learning partitions.
- Development partitions: 37 train sequences and 8 validation sequences.
- Selection: one centered, contiguous 48-frame window from every sequence.
- Size: 1,776 train frames and 384 validation frames in 45 windows; each window
  spans approximately 4.7 seconds.
- Test: all 8 test sequences remain untouched until P5 final evaluation.
- Manifest: `build_kradar_temporal_manifest.py` records timestamps, poses, and
  the exact `previous -> current` transform and rejects time discontinuities,
  duplicate sensor triplets, invalid rotations, or sequence leakage.

The centered-window rule uses no model output, LiDAR quality metric, object
label, speed threshold, or validation result. The formal manifest is immutable
after download begins.

## Historical Prior

For a generated previous-frame point `p`, transform it into the current sensor
frame using the manifest pose. The ego-only control applies only this rigid
transform. The Doppler prior first applies a gated radial displacement in the
previous sensor frame:

```text
v_res = wrap(v_r - v_static)
g = 1(|v_res| > 1.0 m/s)
p_adv = p + g * v_res * dt * r_hat
p_prior = T_current_from_previous * p_adv
```

`v_static` follows the frozen convention selected by the 100-frame static
Doppler audit. For zero-centered compensated Cubes it is zero. Doppler values
are treated as circular quantities with the calibrated 64-bin period.

Historical inputs are always model predictions. LiDAR targets, boxes, future
frames, and target-frame CFAR points are never used to construct the prior.

## Matched Arms

All learned arms use the same parent, point count, current Cube, optimizer,
training frames, losses, and random seeds.

| ID | Method | Purpose |
|---|---|---|
| T0 | Frozen single-frame parent | No-history reference |
| T1 | Ego-only recurrent copy | Geometry-only historical lower bound |
| T2 | Gated Doppler recurrent copy | Isolate the warp mechanism |
| T3 | DoppDrive-style aggregation | Strong non-generative temporal baseline |
| T4 | Cube feature + rasterized-prior concat | Simple learned fusion |
| T5 | Current-Cube query to prior-point cross-attention | Token fusion |
| T6 | Doppler-warp draft refinement | Current Cube corrects a physical draft |

T3 aggregates the current T0 prediction and up to four warped historical T0
predictions, then applies confidence-aware RAE voxel suppression and returns
exactly 10,000 points. It may reuse history but cannot hallucinate new evidence.
This controls for the core behavior of Doppler-driven temporal aggregation.
The frozen suppression score is `confidence * exp(-age / 4)`, where age is the
integer history depth in frames. One highest-scoring point is retained per
integer RAE voxel; if fewer than 10,000 unique voxels remain, suppressed points
are restored in score order. Exact ties prefer the newer, then lower-indexed,
candidate. These rules are fixed before any temporal result is observed.

T4 rasterizes prior position, confidence, circular Doppler mean, and Doppler
entropy into the current RAE grid before concatenation. T5 uses bounded local
cross-attention rather than global attention. T6 initializes point coordinates
and Doppler distributions from the nearest physical draft and predicts bounded
residuals from the current Cube features.

## Selection and Training

1. Run a five-epoch seed-`20260716` fusion preflight for T4-T6 with frozen
   single-frame weights.
2. Select one fusion family using the preregistered validation score
   `temporal_radial_error + 0.25 * current_chamfer + 0.25 * local_spectrum_KL`.
3. Run the selected family at seeds `20260716`, `20260717`, and `20260718` for
   20 epochs: five temporal-head-only epochs followed by 15 joint epochs.
4. Use AdamW with temporal-head learning rate `3e-4`, inherited-backbone
   learning rate `3e-5`, weight decay `1e-4`, BF16 autocast, and cosine decay.
5. From epoch 6, linearly increase scheduled-sampling probability from zero to
   `0.4`. Recurrent predictions are detached when reused as history.

Teacher mode still uses the frozen parent prediction at `t-1`, never ground
truth points. Scheduled sampling replaces that parent prediction with the
temporal model's own previous output and therefore addresses only rollout
distribution shift, without privileged supervision.

## Metrics

Report every metric per frame and aggregate with scene-first paired bootstrap.

- Current accuracy: Chamfer, F-score at 0.5/1/2 m, completeness, Cube local KL,
  Doppler W1, static PCE, confidence ECE, and covered-cell count.
- Temporal consistency: matched-point
  `|delta_range - mean(v_r) * delta_t|`, ego-compensated occupancy flicker, and
  Doppler refresh error against the current Cube.
- Rollout: the same metrics at 1, 5, 10, and 25 recurrent steps, corresponding
  to approximately 0.1, 0.5, 1.0, and 2.5 seconds.
- Efficiency: parameters added, peak GPU memory, latency, and point throughput.

Dynamic/static and distance-stratified results are mandatory. Smoothness alone
is not evidence of success because a static or low-confidence cloud can appear
temporally stable.

## G4 Decision Rule

The selected temporal model passes G4 only when all conditions hold across the
three formal seeds:

1. versus T0, the 95% paired interval excludes zero in the favorable direction
   for both temporal radial error and occupancy flicker;
2. versus T3, geometry Chamfer or a frozen downstream metric improves with a
   95% paired interval excluding zero;
3. current-frame Chamfer degradation versus T0 has an upper 95% relative bound
   of at most 2%, and Cube local KL/static PCE do not degrade by more than 5%;
4. at 25 rollout steps, mean confidence and covered-cell count each retain at
   least 90% of T0 and no metric is missing for any validation sequence;
5. all model/data/checkpoint provenance matches and test sequences remain
   untouched.

Conditions 3 and 4 use conservative paired-bootstrap bounds: non-degradation
thresholds are applied to the upper 95% relative-change bound, while the 90%
retention thresholds are applied to the lower 95% ratio bound. The bootstrap
resamples seeds and validation sequences; frames within a sequence are averaged
before inference and are not treated as independent observations.

T1/T2 mechanism results and the T4-T6 preflight cannot replace the selected
three-seed comparison. If G4 fails, temporal fusion is moved to the appendix;
the single-frame paper proceeds according to G2/G3 without weakening this gate.

## Required Artifacts

- immutable temporal manifest and CRC-verified download manifests;
- CUDA temporal-prior verifier covering coordinate round trips, ego/Doppler
  warps, raster conservation, descending axes, and nonzero gradients;
- T4-T6 preflight report and frozen selection decision;
- same-seed T0-T3 and selected temporal outputs on every validation frame;
- scheduled-sampling logs and teacher/recurrent exposure counts;
- one-, five-, ten-, and 25-step rollout reports;
- scene-first paired bootstrap JSON and G4 decision Markdown;
- fixed-window and worst-five visualizations;
- latency, memory, and throughput report.
