# D-MHW birth-Doppler supervision without a future Cube

Date: 2026-07-29

Status: final literature and repository decision; no long training was run.

Scope: resolve the supervision contract for the 3,000 birth points in D-MHW
while preserving causal Full-RAED inputs, radar-only inference, and an exact
10,000-point output.

## 1. Decision

The reviewed literature and code do not provide a legal way to obtain a
**true future 64-bin radar Doppler spectrum** for a newly born point without a
future radar measurement or a calibrated radar simulator.

Future geometry can identify position and kinematic displacement. Its radial
projection provides a scalar radial velocity, or equivalently the first
circular moment after wrapping onto the project Doppler axis. It does not
identify spectral width, multimodality, sidelobes, material response, or
measurement noise. Those are radar-measurement properties.

The frozen decision is therefore:

1. **Recommend G-RM: geometry/track-derived radial-moment supervision.**
   Preserve the 64-bin circular output parameterization, but supervise and
   claim only its wrapped radial first moment for birth points. Do not claim a
   measured or calibrated future Doppler distribution.
2. **Reject C-MC as the primary solution:** a current-Cube measurement cycle
   is useful as a causal regularizer, but genuine future births are not
   observable in the current Cube. It cannot establish true future
   birth-Doppler accuracy.
3. **Retain A-NC as the safe fallback:** abstain from birth Doppler, output
   birth XYZ plus confidence, and restrict the 64-bin Doppler claim to
   persistent points.

These routes are mutually exclusive in Stage 0. They must not be fused after
observing validation results.

## 2. Repository audit

### 2.1 What is already causal

- `code/models/dmhw_direct_world.py:429-443` accepts only history/current
  Cubes and explicitly rejects `future_cube_drae`.
- `code/dmhw/contracts.py:49-181` builds three-horizon examples from strictly
  historical/current inputs and records `future_cube_exposed=false`.
- `code/dmhw/contracts.py:208-315` audits future LiDAR geometry as supervision
  without loading a future Cube.
- `docs/dmhw_direct_multi_horizon_stage0.md:9-22` freezes direct
  `0.5/1.5/2.5 s` prediction and no autoregressive rollout.

Using future LiDAR geometry, future labels, or ego poses as **training
targets** does not violate the no-future-Cube input contract. It does mean the
method is not radar-only in training. The defensible statement is
`radar-only inference with LiDAR/annotation-derived training supervision`.

### 2.2 The unresolved head

- `code/models/dmhw_direct_world.py:559-592` decodes birth XYZ, 64 Doppler
  logits, and confidence from causal latents.
- `code/models/dmhw_direct_world.py:614-640` exposes the birth distribution in
  the final exact-count output.
- `code/losses/dmhw_objective.py:103-122` states that real K-Radar targets do
  not provide point-aligned birth Doppler.
- `code/losses/dmhw_objective.py:164-191` rejects point-aligned
  Doppler/confidence supervision unless it is explicitly synthetic.
- `code/losses/dmhw_objective.py:192-215` therefore reduces real supervision
  to geometry plus persistent inverse physics.

The H200 preflight proves that all 64 birth-head rows can receive gradients.
It does not prove that those gradients correspond to a legal real-data
target.

### 2.3 Claim lock

- `paper/claim_evidence_ledger.md:C10` keeps joint fixed-count geometry,
  circular Doppler, and confidence locked until a geometry parent and physical
  closure pass.
- The D-MHW row in the same ledger records that real birth-Doppler
  supervision remains unresolved without a future Cube.
- `docs/dmhw_direct_multi_horizon_stage0.md:119-152` makes this a formal data
  authorization gate.

This report resolves a legal **radial-moment** target. It does not unlock the
stronger true-distribution claim.

## 3. Evidence from official papers and repositories

| Work | What is supervised or used | Relevance | Boundary for D-MHW |
|---|---|---|---|
| [DoppDrive paper](https://openaccess.thecvf.com/content/ICCV2025/html/Haitman_DoppDrive_Doppler-Driven_Temporal_Aggregation_for_Improved_Radar_Object_Detection_ICCV_2025_paper.html), [official repository](https://github.com/yuvalHG/DoppDrive) | Measured pointwise scalar Doppler shifts historical radar points radially | Supports mandatory radial motion and Doppler interventions | It aggregates measured points; it does not supervise forecast birth attributes or output a distribution |
| [RaFlow paper](https://arxiv.org/abs/2203.01137), [official code at `c01897f`](https://github.com/Toytiny/RaFlow/tree/c01897faeb5d1e766ba899a677b38b1282b142e4) | Soft Chamfer to the next radar point cloud, spatial smoothness, and source-point radial displacement from measured RRV | Direct evidence that a scalar radial projection is a useful scene-flow constraint | It predicts source-point flow from two radar point clouds; no births and no 64-bin output distribution |
| [CMFlow paper](https://openaccess.thecvf.com/content/CVPR2023/html/Ding_Hidden_Gems_4D_Radar_Scene_Flow_Learning_Using_Cross-Modal_Supervision_CVPR_2023_paper.html), [official code at `16a095a`](https://github.com/Toytiny/CMFlow/tree/16a095a250453b4fe1363e4ec9dddd20d0d64e4f) | RaFlow losses plus LiDAR/camera/odometry and track-derived pseudo scene flow | Strongest precedent for using tracked boxes and cross-modal geometry to create a flow target | It supervises 3D flow for existing source radar points, not a future birth spectrum |
| [RadarSFEMOS paper](https://miasgroup.tongji.edu.cn/_upload/tpl/06/23/1571/template1571/pdf/ral2025_liu.pdf), [official code at `79e7433`](https://github.com/nubot-nudt/RadarSFEMOS/tree/79e7433dca85dd974818c87c37460e9b2eb8510b) | Pseudo scene flow plus scalar radial-displacement loss from measured RRV | Independently confirms the flow-dot-ray scalar constraint | It consumes two radar point clouds and does not predict a Doppler distribution |
| [RadarMP paper](https://arxiv.org/abs/2511.12117), [official code at `bd03fd7`](https://github.com/chengrui7/RadarMP/tree/bd03fd7f5e1da87f620f3b1775fa34e476d0a950) | Two consecutive tesseracts; energy-flow consistency uses both source and target tensors; Doppler channels yield scalar max/expectation cues | Closest Cube-level motion method and useful renderer/loss reference | Its target-frame tesseract is a future Cube for forecasting and is forbidden by the D-MHW contract |
| [D2-World paper](https://arxiv.org/abs/2411.17027), [official code at `f7c3061`](https://github.com/zhanghm1995/D2-World/tree/f7c3061845b2f0c0ccc5ba7789cd78e68394dc34) | Non-autoregressive multi-future occupancy/point forecasting with future point/ray losses and decoupled flow | Supports D-MHW direct multi-horizon structure and future geometry labels | It has no radar spectral attribute target |
| [Point Cloud Forecasting as a Proxy for 4D Occupancy Forecasting](https://github.com/tarashakhurana/4d-occ-forecasting), [Occ4cast code at `f37d8f4`](https://github.com/ai4ce/Occ4cast/tree/f37d8f47aca976c4f920d4acd0f98caa93a24f73) | Past LiDAR inputs and future LiDAR/occupancy supervision | Shows that causal forecasting routinely uses future geometry as a label | Geometry and occupancy do not determine a future radar Doppler spectrum |
| [ImplicitO](https://openaccess.thecvf.com/content/CVPR2023/html/Agro_Implicit_Occupancy_Flow_Fields_for_Perception_and_Prediction_in_Self-Driving_CVPR_2023_paper.html), [UnO](https://openaccess.thecvf.com/content/CVPR2024/papers/Agro_UnO_Unsupervised_Occupancy_Fields_for_Perception_and_Forecasting_CVPR_2024_paper.pdf), [DIO](https://openaccess.thecvf.com/content/CVPR2025/html/Diehl_DIO_Decomposable_Implicit_4D_Occupancy-Flow_World_Model_CVPR_2025_paper.html) | Continuous occupancy/flow learned from LiDAR observations and future ray/depth supervision | Supports latent occupancy-flow and renderer-based geometry supervision | None supplies radar Doppler-distribution labels |

### 3.1 Source-level checks that determine the decision

- RaFlow `losses/loss.py:78-103` constrains
  `dot(flow, ray) = measured_RRV * dt`. This is scalar supervision.
- CMFlow `losses/radar_loss.py:100-151` keeps the same scalar radial loss.
  Its preprocessing uses tracked objects and cross-modal geometry to generate
  pseudo 3D flow, which is the closest official precedent for G-RM.
- RadarSFEMOS `losses/radar_loss.py:100-155` again combines a next-cloud
  geometry loss with scalar radial displacement.
- RadarMP `Losses/RadarMPLoss.py:42-47,88-97,163-186,217-227` reads both
  `x1` and `x2`; the target tesseract is structurally required by its
  energy/flow losses. Copying that loss would violate no-future-Cube.
- D2-World and the 4D occupancy repositories supervise future geometry,
  occupancy, depth, or rays. They support causal target construction, not a
  future radar spectrum.

## 4. Recommended route: G-RM

G-RM means **Geometry/track-derived Radial Moment**.

### 4.1 Target construction

For each future LiDAR target point \(y_j^h\) at horizon \(h\):

1. Transform geometry into the target radar frame with the audited ego pose.
2. Parse the K-Radar box and track ID at the target frame and its immediately
   preceding labeled frame.
3. For a point inside one unique, consistently classed tracked box, derive a
   rigid object displacement from the two boxes.
4. For a confidently static background point, derive displacement from ego
   motion.
5. Mark unmatched tracks, duplicate IDs, class changes, overlapping boxes,
   box-boundary points, and unverified dynamic background as invalid.
6. Compute the scalar radial target

   \[
   v_{r,j}^{\star}
   =
   \frac{1}{\Delta t}
   \left\langle
   \Delta y_j,\,
   \frac{y_j^h}{\lVert y_j^h\rVert_2}
   \right\rangle .
   \]

7. Wrap \(v_{r,j}^{\star}\) onto the frozen 64-bin circular period.

This uses future geometry, labels, and poses only as training supervision. It
must never load a future tesseract or query a future radar spectrum.

The repository already contains compatible track parsing and adjacent-frame
range-rate logic in `code/scripts/eval_p5_object_velocity.py:93-136` and
`:550-614`. That implementation is evidence for coordinate and track handling,
not a ready-made birth target adapter.

### 4.2 Prediction-to-target association

Birth points have no identity. Associate predicted birth XYZ to future target
geometry with a frozen, stop-gradient, one-to-one local assignment:

- range-stratified Sinkhorn or mutual nearest neighbor;
- maximum match radius frozen before training;
- no GT-guided inference;
- no best-of-k;
- unmatched predictions receive no radial target and must be accounted for in
  confidence calibration.

The assignment must be identical for the scalar and circular controls.

### 4.3 Circular objective

Let \(p_{ik}\) be the predicted 64-bin probability, and let
\(\theta_k=2\pi(v_k-v_{\min})/P\). Supervise the first circular moment:

\[
\mu_i =
\sum_k p_{ik}
\begin{bmatrix}
\cos\theta_k\\
\sin\theta_k
\end{bmatrix},
\qquad
L_{\mathrm{moment}}
=
1-
\frac{\mu_i}{\max(\lVert\mu_i\rVert,\epsilon)}
\cdot
\begin{bmatrix}
\cos\theta_i^\star\\
\sin\theta_i^\star
\end{bmatrix}.
\]

The resultant strength \(\lVert\mu_i\rVert\) is reported but is not called
calibrated uncertainty without a distributional target.

A narrow von Mises or Gaussian soft label may be used only as a numerical
implementation of scalar uncertainty whose width is fixed from train-only
label noise. It remains a pseudo-distribution and must not be described as a
measured radar spectrum.

### 4.4 Minimal code interfaces required

No implementation is authorized by this report, but the minimum future change
set is:

```text
BirthRadialMomentTarget
  xyz_m:                (B,H,M,3)
  radial_velocity_mps:  (B,H,M)
  valid_mask:           (B,H,M)
  provenance:           static_ego | tracked_rigid | invalid
  source_frame_ids
  label_file_hashes
  future_cube_accessed: false

associate_birth_targets(
  predicted_birth_xyz_m,
  target_xyz_m,
  target_valid_mask
) -> target_index, matched_mask, distance_m

birth_circular_moment_loss(
  birth_doppler_probability,
  matched_radial_velocity_mps,
  matched_mask,
  doppler_axis,
  circular_period
)
```

`dmhw_stage0_loss` must receive a separate real-data
`birth_radial_moment_target`; it must not reuse the existing
`aligned_synthetic_attributes` escape hatch.

The output metadata must record:

```text
birth_doppler_supervision = geometry_track_radial_moment
birth_distribution_ground_truth = false
birth_distribution_claim_enabled = false
birth_radial_moment_claim_enabled = <gate result>
```

### 4.5 Stage-0, controls, and budget

Stage 0 is an attribute-only falsification experiment, not a long D-MHW run.

First run a label-only audit over all 740 train and 160 validation anchors.
Only after it passes, freeze geometry and train the attribute head for at most
500 updates on one permitted H200.

Controls:

| Arm | Output | Purpose |
|---|---|---|
| G0 | Train-only marginal 64-bin distribution | Detect dataset-prior success |
| G1 | Scalar radial-velocity head with wrapped Huber loss | Determine whether 64 bins add value over scalar regression |
| G2 | Proposed 64-bin circular first-moment loss | Candidate |
| G3 | G2 with scene-shuffled track/ego targets | Detect label or association shortcuts |
| G4 | Persistent-only inverse-physics head | Ensure birth gains are not inherited from persistent points |

No control may read a future Cube. All arms use identical geometry,
association, causal inputs, update count, and seed.

### 4.6 Hard gates

**Legality and provenance**

- Future-Cube path count and bytes read are exactly zero.
- Test records read are exactly zero.
- Every valid scalar has a source frame, target frame, \(\Delta t\), ego-pose
  hash, label hash, and provenance class.
- Valid label coverage is at least 75% overall and at least 60% at each
  horizon; invalid points remain in the denominator.
- The dynamic tracked subset spans at least 10 independent temporal windows.
- Point/box radial targets agree with the existing train-only P5
  box-center range-rate convention within `0.5 m/s` median absolute
  difference; otherwise sign or frame conventions are unresolved.

**Scientific effect**

- G2 birth radial circular MAE improves at least 10% over G0 overall and on
  the tracked-dynamic subset.
- G2 is no worse than scalar G1 by more than `0.05 m/s`; otherwise the 64-bin
  parameterization has no demonstrated benefit.
- Correct targets outperform shuffled G3 by at least 10% in radial MAE.
- Doppler sign flip and a non-zero circular bin roll each worsen dynamic
  radial loss by at least 15%.
- Birth confidence AUROC for matched versus unmatched geometry exceeds G0 by
  at least 5 percentage points.
- Frozen geometry Chamfer, completeness, outlier, and duplicate metrics change
  by at most numerical tolerance. In any later joint run, Chamfer may worsen
  by no more than 2%.

**Decision**

- Passing all gates enables only:
  “Birth points use a 64-bin circular parameterization supervised through a
  geometry-derived radial first moment.”
- It does not enable:
  “Birth points have a true or calibrated future Doppler distribution.”
- Failure closes G-RM; do not extend updates or replace the gate after seeing
  validation results.

## 5. Alternative rejected as primary: C-MC

C-MC means **Causal current-Cube Measurement Cycle**.

### 5.1 Proposed mechanism

Inverse-transport predicted future points to the current radar frame, render a
current-time RAE-D spectrum, and compare it with held-out cells of the current
Cube. The encoder must not see the held-out cells used as the cycle target.

Minimum interface:

```text
render_future_state_to_current_cube(
  future_xyz,
  future_doppler_probability,
  confidence,
  inverse_motion,
  current_visibility
) -> rendered_spectrum, observable_mask

masked_current_cube_cycle_loss(
  rendered_spectrum,
  held_out_current_cube,
  observable_mask
)
```

### 5.2 Why it is not the recommended solution

- A genuine birth caused by future entry, disocclusion, or a new reflection
  has no current measurement to reconstruct.
- The current Cube is already an input, so identity copying is a severe
  shortcut even with masked cells.
- The persistent branch can satisfy an aggregate renderer while the birth
  distribution remains arbitrary.
- RadarMP's useful energy-flow losses require a second tesseract. Using the
  target tesseract would violate the D-MHW contract.

Thus C-MC is an unsupervised regularizer for causal consistency, not real
future birth-Doppler supervision.

### 5.3 Controls and hard rejection gates

Controls:

- C0: no birth cycle;
- C1: detached local-spectrum copy at inverse-projected coordinates;
- C2: geometry/energy renderer without Doppler bins;
- C3: full circular renderer cycle;
- C4: current-Cube scene shuffle and circular bin roll.

Hard gates:

- zero future-Cube and zero test access;
- at least 20% of current target cells are hidden from the encoder;
- C3 held-out current-spectrum circular W1 improves at least 10% over C0 and
  C1;
- scene shuffle and bin roll each worsen cycle loss by at least 15%;
- C3 improves an external geometry-derived radial-moment diagnostic by at
  least 10% over C0;
- true-birth/no-current-support results are reported separately;
- if C1 is within 1% of C3, if birth confidence collapses, or if only the
  persistent branch explains the renderer, reject C-MC.

Even if these gates pass, the allowed claim is only
`current-measurement-cycle-consistent latent circular distribution`. A true
future distribution claim remains disabled.

## 6. Safe fallback: A-NC

A-NC means **Attribute abstention / no claim**.

### 6.1 Mechanism

- Persistent 7,000 points retain their measured and inverse-physics-constrained
  64-bin circular Doppler distribution.
- Birth 3,000 points output XYZ and confidence.
- For tensor compatibility, a birth 64-bin vector may be emitted only with
  `doppler_valid=false`; it is excluded from every Doppler loss and metric.
- Prefer an explicit `doppler_valid_mask` over silently returning an untrained
  softmax.

### 6.2 Minimal interface

```text
output["doppler_valid_mask"] = output["persistent_mask"]
output["birth_attribute_status"] = "abstain"

evaluate_doppler(..., doppler_valid_mask=<required>)
```

The evaluator must fail closed if a whole-cloud Doppler metric is requested
without the validity mask.

### 6.3 Controls and gates

- A0: frozen uniform birth vector plus invalid mask;
- A1: untrained learned birth head plus invalid mask;
- A2: remove birth head entirely in an interface-only branch.

Gates:

- exact 10,000 XYZ and confidence outputs remain unchanged;
- valid Doppler count is exactly the persistent count;
- no birth Doppler gradient enters geometry, confidence, or the causal
  encoder;
- A0/A1/A2 geometry metrics are identical within numerical tolerance;
- all reports separate `doppler_valid_count` and `birth_abstain_count`;
- full-cloud Doppler evaluation without a mask raises an error.

This route fully preserves no-future-Cube and radar-only inference. It does
not preserve the claim that all 10,000 output points carry a meaningful
64-bin Doppler distribution.

## 7. Claim compatibility matrix

| Route | No future Cube | Radar-only inference | Radar-only training | 64-bin tensor interface | Legal birth radial claim | True/calibrated birth 64-bin claim |
|---|---|---|---|---|---|---|
| **G-RM recommended** | Yes | Yes | No, uses LiDAR/track/pose labels | Yes | Yes, only after gates | **No** |
| **C-MC rejected primary** | Yes | Yes | Yes | Yes | Only self-consistency; external validation required | **No** |
| **A-NC fallback** | Yes | Yes | Yes for birth attributes | Compatibility vector may remain masked | No birth radial claim | **No** |
| RadarMP-style target-Cube loss | **No** | Not applicable to frozen contract | Yes, radar tensors | Yes in principle | Could supervise measurement moments | Could use a real spectrum, but violates the core constraint |

The phrase `64-bin circular distribution` must be separated into:

1. **representation claim:** the head emits a normalized 64-bin circular
   vector; and
2. **accuracy/calibration claim:** the vector matches the future radar
   measurement distribution.

G-RM supports the first and tests its first moment. It does not support the
second.

## 8. Final execution order

1. Keep real D-MHW training locked behind the existing eligible-geometry-parent
   and dense-target-cache gates.
2. Implement only the G-RM label audit first. Do not train before provenance,
   coverage, sign, and dynamic-window gates pass.
3. If the audit passes, run the 500-update head-only G0-G4 Stage 0 on H200.
4. Promote G-RM only as radial-moment supervision.
5. If G-RM fails, adopt A-NC. Do not rescue the result with C-MC.
6. Use C-MC later only as an independently named regularizer experiment after
   the main supervision decision is closed.

## 9. Evidence boundary

This report is based on repository inspection and official papers/code. It
does not contain a new empirical D-MHW result.

The strongest defensible conclusion is:

> Future LiDAR geometry and tracked object motion can legally supervise the
> wrapped radial first moment of a birth-point Doppler head without exposing a
> future Cube. They cannot provide the true future 64-bin radar spectrum.

Any paper statement stronger than this requires either:

- a future radar Cube, which violates the frozen D-MHW constraint; or
- a calibrated radar simulator/teacher, which would provide synthetic or
  teacher-derived rather than real distribution supervision and must be
  claimed accordingly.
