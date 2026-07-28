# Failure-driven Stage-0 frontier for Cube-to-dense geometry

> Date: 2026-07-28
>
> Scope: independent diagnosis and experiment design for G1D/G1F/G1G/G1T.
>
> Constraint: no new scientific result is claimed here. G1D epoch-100 values are
> monitoring evidence from a read-only H200 snapshot on 2026-07-29, not a
> completed gate. Test remains untouched.

## 1. Decision summary

The current failures are best explained by a coupled but ordered chain:

1. **The current G1D proposal mechanism is range-biased and lacks far-range
   support.** The strongest evidence is G1F-F0: even a GT-aware oracle cannot
   recover the full geometry gate from the frozen 32k proposal pool, and the
   complete pool recalls only `17.80%` of 60--120 m target mass within 2 m.
2. **G1D's global Full-RAED condition is bypassed by local query evidence.**
   Every proposal, training query, coarse query, and final query receives the
   measured local 64-bin spectrum, absolute energy, range, and coordinates.
   Shuffling only the 336 global condition tokens leaves this stronger path
   intact. Nonzero gradients prove reachability, not causal use.
3. **The training objective does not directly supervise the inference
   allocation problem.** Random LiDAR-derived occupied/empty queries supervise
   occupancy BCE, while inference selects hard top-k points from a different
   32k radar-template pool. Only selected points receive geometry gradients.
4. **Duplicate growth is a consequence of allocation error plus a many-to-one
   set objective, not merely a weak repulsion coefficient.** Fixed template
   expansion and boundary clipping already yield `14.015%` duplicates under the
   G1F oracle; learned offsets then increase the monitoring value to about
   `29.90%`.
5. **Far-range failure is reinforced by supervision imbalance.** Target
   confidence is derived from local CFAR margin, positive queries are sampled
   in proportion to that confidence, and weighted completeness gives weak
   distant surfaces less influence. Meanwhile negatives are approximately
   balanced across range thirds and proposal ranking is global.

Therefore the next frontier should not be another selector on the same 32k
pool, a larger Transformer, or a repulsion-weight sweep. The most informative
three routes are:

- a K-Radar range-aware RaLD-style query field that changes proposal support;
- the already implemented condition-exclusive G1G hierarchy;
- the no-train G1T history-support diagnostic.

Explicit inference-allocation supervision and condition contrast remain
important Stage-0 routes, but they should be launched only if the first three
experiments leave the corresponding root cause unresolved.

## 2. Evidence packet

### 2.1 Trusted observations

| Evidence | Observation | What it rules out |
|---|---:|---|
| G1B `full_raed_rank2` | Chamfer `2.0251 m`, outlier `28.885%`, far completeness `7.5757 m` | Full-grid occupancy has useful broad support; its dominant failure is precision/tail control, not total geometric blindness |
| G1D epoch 15 monitoring | Chamfer `4.4823 m`, completeness `3.5811 m`, outlier `17.90%`, far completeness `8.1239 m` | Measurement-seeded queries can lower outliers, but do not cover the target |
| G1D epoch 100 monitoring | Chamfer `5.5644 m`, completeness `3.5872 m`, outlier `24.242%`, far completeness `8.7814 m`, duplicates `29.898%` | Longer optimization is not repairing completeness; offset/occupancy growth is buying recall with crowding and precision loss |
| G1D epoch 100 condition intervention | shuffled-condition Chamfer change `0.0163%`; occupancy-BCE change `0.0024%` | The global condition is not causally material under the current decoder, despite nonzero gradients |
| G1F-F0 GT oracle | Chamfer `2.8863 m`, completeness `1.6513 m`, far completeness `8.6533 m`, outlier `24.853%`, duplicates `14.015%` | Hard top-k is not the sole failure; a selector-only repair on the same pool is closed |
| G1F complete-pool support | 60--120 m weighted GT recall at 2 m `17.80%`, median `0%` over eligible frames | The frozen proposal representation is missing far support before learned selection |
| G1G preflight | 41.68M parameters, 336 tokens, 2,500 centers, 4 children, 10k output, no measured-Cube allocation gradient, all condition gradients present | The anti-bypass architecture is executable; preflight does not establish geometry quality |

The formal evidence boundaries remain those in
[`paper/claim_evidence_ledger.md`](../../../paper/claim_evidence_ledger.md).
G1D epoch-100 numbers above are monitoring-only and cannot update that ledger.

### 2.2 Code facts that determine causality

1. The radar-observable target confidence is a sigmoid of local CFAR margin and
   is zeroed outside the first angular surface
   ([`code/cube_dense/observability.py`](../../../code/cube_dense/observability.py),
   `observable_lidar_target`). The dense occupancy cache stores the maximum
   confidence per RAE cell
   ([`code/cube_dense/dataset.py`](../../../code/cube_dense/dataset.py),
   `KRadarCubeDataset.__getitem__`).
2. G1D proposals rank the sum of `log1p` power over all 64 Doppler bins, then
   apply a fixed `(5,5,3)` index-space NMS
   ([`code/models/rald_query_field.py`](../../../code/models/rald_query_field.py),
   `integrated_log_energy` and `stable_radar_proposals`). This is not
   range-calibrated. A fixed angular kernel suppresses a larger physical area
   at long range.
3. Every G1D query token receives coordinates, the local normalized 64-bin
   spectrum, standardized absolute energy, and normalized range
   (`RaLDQueryField.query_tokens`). These features are computed from the
   unshuffled measured Cube.
4. Only `radar_encoder(condition_cube_drae)` is changed by the condition-shuffle
   intervention. Proposal indices, proposal tokens, local query spectra, energy,
   and coordinates remain from the correct frame
   (`RaLDQueryField.forward`). The current near-zero shuffle response is thus an
   expected shortcut signature, not evidence that Full-RAED is uninformative.
5. Inference is a detached 1,000-seed proposal pool, fixed 32-way expansion,
   hard top-2,500 selection, and four-way local expansion. Training occupancy
   queries are instead 625 confidence-sampled positives and 9,375
   range-stratified negatives. The two query distributions are not coupled by a
   coarse-allocation loss.
6. The geometry objective is nearest-neighbor Chamfer plus an outlier hinge.
   It does not penalize multiple predictions assigned to the same target.
   Repulsion activates only below `0.10 m` with weight `0.02`; the reported
   duplicate threshold is `0.05 m`. Wrongly allocated children can therefore
   move toward the same easy surface while reducing Chamfer.

### 2.3 What was borrowed from literature, and what was not

- [RaLD](https://github.com/MetaIoT-WHU/RaLD) motivates occupied/empty arbitrary
  queries, set latents, radar-conditioned latent blocks, and implicit occupancy
  decoding. The proposed range-aware route below is **RaLD-style plus a
  K-Radar-specific range proposal prior**; range calibration is not attributed
  to upstream RaLD.
- [AdaPoinTr](https://arxiv.org/abs/2301.04545),
  [SeedFormer](https://arxiv.org/abs/2207.10315), and
  [SnowflakeNet](https://arxiv.org/abs/2108.04444) motivate separating global
  seed allocation from bounded local patch generation. G1G is the minimal
  repository-native test of that mechanism.
- [RangeLDM](https://github.com/WoodwindHu/RangeLDM) and
  [LiDPM](https://github.com/astra-vision/LiDPM) support treating range/ray
  allocation as an explicit representation issue rather than an unconstrained
  Cartesian point-selection problem.
- [DoppDrive](https://openaccess.thecvf.com/content/ICCV2025/html/Haitman_DoppDrive_Doppler-Driven_Temporal_Aggregation_for_Improved_Radar_Object_Detection_ICCV_2025_paper.html)
  motivates ego-only versus Doppler-warped history controls.
  [RadarMP](https://arxiv.org/abs/2511.12117) shows that adjacent Cube
  observations and Doppler-motion consistency can jointly improve point
  generation. These works justify G1T as a support test, not as a novelty claim.

## 3. Most likely causal chain

### 3.1 Primary chain: range attenuation to missing proposal support

```text
range-dependent radar attenuation and weak/occluded returns
    -> lower target CFAR margin and lower integrated Cube energy
    -> confidence-weighted positives under-sample weak far surfaces
    -> global energy ranking allocates most seeds near range
    -> fixed angular NMS removes physically larger far neighborhoods
    -> 32 fixed local templates cannot create absent far modes
    -> hard top-k selects from an already incomplete support set
    -> far completeness and total completeness plateau
```

Confidence: **high**. G1F's complete-pool recall is the decisive evidence. The
oracle's selected far quota is also small because it follows target confidence
mass, so the full-pool `17.80%` recall is more diagnostic than the selected
oracle metric alone.

### 3.2 Secondary chain: local evidence to condition bypass

```text
correct-frame local spectrum + energy + range + coordinates at every query
    -> proposal tokens and decoder can solve local occupancy without global tokens
    -> residual cross-attention can shrink toward an identity/no-op path
    -> every condition block still receives gradients
    -> cross-scene condition shuffle changes neither occupancy nor geometry
```

Confidence: **high for bypass, medium for geometry causality**. The bypass is
real, but forcing dependence may not improve geometry if the global encoder
does not resolve missing far support. A condition-contrast experiment must
therefore require correct-condition geometry improvement, not merely a larger
shuffle gap.

### 3.3 Duplicate chain

```text
energy-clustered seeds + fixed template expansion + boundary clipping
    -> repeated or near-repeated candidate coordinates before learning
    -> hard selection has no capacity/coverage constraint
    -> four children per selected center share nearly identical context
    -> Chamfer rewards moving several children to the same nearby target
    -> large bounded offsets compensate for wrong center allocation
    -> recall and duplicates rise together while completeness remains flat
```

Confidence: **high**. The G1F oracle has unique candidate indices but still
`14.015%` coordinate duplicates, demonstrating a structural component before
learned offsets. G1D later reaches about `29.90%`.

### 3.4 Why G1B and G1D fail differently

G1B predicts the full occupancy grid and therefore retains broad range support,
but its balanced focal/dice objective and global top-10k export do not directly
control point precision. It reaches good Chamfer/far completeness with an
outlier tail. G1D starts from high-energy measured locations and therefore
improves precision/outliers, but loses weak surfaces and allocates several
children around the same evidence. The missing method is not a larger version
of either branch; it must combine **broad range support** with **capacity-aware
local precision**.

## 4. Shared Stage-0 contract

All learned experiments below must use:

- the frozen 76/24 train/validation split and no test access;
- fixed seed `20260716`;
- exactly 10,000 exported points;
- evaluation at epochs 5, 10, 15, and 20 when the route reaches those epochs;
- no hyperparameter sweep, best-of-k sampling, confidence masking, or output
  count reduction;
- frame-first metrics plus scene-first aggregation;
- the same geometry evaluator, 2 m outlier definition, 0.05 m duplicate
  definition, and 60--120 m far slice;
- a matched control from the same initialization and optimization budget;
- no fusion between routes during Stage-0.

Common screening limits, reused from the current G1G contract:

| Metric | Stage-0 limit |
|---|---:|
| median completeness | `<= 2.5068 m` (30% better than G1D epoch 15) |
| mean outlier fraction at 2 m | `<= 25%` |
| mean duplicate fraction at 0.05 m | `<= 15%` |
| mean far completeness | `<= 8.1239 m` |
| exact output count | `10,000` on every frame |

These are screening limits, not replacements for the final G1 gate
(`2.50/25%/0.65/8.0/10%` for Chamfer/outlier/completeness/far/duplicates).

## 5. Stage-0 experiment matrix

| ID | Route | Isolated factor | Budget | Primary diagnostic |
|---|---|---|---:|---|
| R0 | Range-aware RaLD-style query field | proposal support and physical range allocation | oracle + 10 epochs | far proposal recall and completeness |
| O0 | Explicit coarse occupancy/energy supervision | train/inference allocation mismatch | two 10-epoch arms | support of selected centers |
| C0 | Cross-scene condition contrast | shortcut versus useful global condition | 5 epochs | causal shuffle gap with correct-frame gain |
| H0 | G1G hierarchy/dynamic centers | global allocation topology and child crowding | 20 epochs | condition dependence plus duplicates |
| T0 | G1T temporal proposal | missing observation versus model allocation | 0 epochs | ego and Doppler history support |

### 5.1 R0: range-aware RaLD-style query field

**Unique hypothesis.** G1D's main geometry failure is caused by global
integrated-energy ranking and fixed index-space NMS, which under-allocate weak
mid/far measurements. The RaLD-style implicit decoder is usable if its query
support is made range-aware before hard selection.

**Minimal code change.**

1. Keep `RaLDQueryField`, its 336 tokens, 512 latents, decoder, losses, and
   32k-to-10k cardinalities unchanged.
2. Add a train-only range profile for integrated log energy:
   `z(r,a,e)=(E-median_r(E))/(MAD_r(E)+epsilon)`. Concatenate or average this
   calibrated score with the existing absolute score; do not remove absolute
   energy from the decoder.
3. Replace fixed angular NMS width with a range-dependent width corresponding to
   one preregistered physical lateral radius. The range-axis width remains
   fixed in meters. No learned proposal module is introduced.
4. Freeze proposal and selected-center range quotas at `75%/20%/5%` for
   `0--30/30--60/60--120 m`. This is a single Stage-0 setting, not a sweep.
5. Reuse the current G1F oracle before any training. Only an oracle pass
   authorizes a 10-epoch fine-tune from the G1D epoch-15 checkpoint.

**Controls.**

- `R0-control`: the original G1D energy ranking, `(5,5,3)` NMS, and proposal
  cache, evaluated from the same checkpoint.
- `R0-z-only`: calibrated energy with original NMS and no range quota. This is a
  no-train attribution control; it is not separately tuned.
- Existing G1F-F0 remains the fixed-pool lower support reference.

**Budget.**

- No-train oracle over all 24 validation frames.
- If authorized: 10 epochs, 760 optimizer updates, evaluations at epochs 5 and
  10. Hard stop at epoch 10.

**Preregistered promotion gate.**

The new no-train 32k pool must first satisfy all:

- 60--120 m GT recall at 2 m `>=30%` (baseline `17.80%`);
- G1F-style oracle median completeness `<=1.20 m`;
- oracle far completeness `<=8.0 m`;
- oracle outlier `<=25%`;
- oracle duplicate fraction `<=10%`.

The learned arm then must satisfy the common Stage-0 limits and improve far
completeness by at least `10%` relative to its matched 10-epoch vanilla control.
If the oracle fails, eliminate R0 training. If the oracle passes but training
fails, proposal support is adequate and the remaining cause is allocation
supervision or topology.

**Root-cause discrimination.**

- Oracle and learned arm pass: range-biased support is the primary cause.
- Oracle passes, learned arm fails: train/inference allocation mismatch is
  primary; advance O0, not another proposal variant.
- Oracle fails despite range calibration: current-frame Cube peaks do not
  contain enough support; prioritize T0 or a dense/ray representation.
- Far improves but outliers exceed 25%: range support was missing, but
  observability/precision calibration is the next bottleneck.

### 5.2 O0: explicit coarse occupancy and Cube-energy supervision

**Unique hypothesis.** G1D has usable movable queries, but random arbitrary-query
BCE does not train the actual 32k inference scores. Directly supervising every
coarse inference query will fix selection without changing proposal support.

**Minimal code change.**

1. Expose the 32k coarse logits and refined coarse coordinates already computed
   in `RaLDQueryField.forward`.
2. Query the cached radar-observable occupancy at those coordinates after a
   fixed one-cell dilation and apply range-balanced focal BCE to all 32k logits.
3. Add a within-range pairwise ranking term between coarse score and current
   Cube integrated-energy percentile. It is an auxiliary radar-evidence term,
   not a replacement for the LiDAR-derived occupancy target.
4. Preserve hard top-2,500 export; form all auxiliary losses before top-k so
   every coarse score receives a gradient.

**Controls.**

- `O0-control`: resume the same G1D epoch-15 checkpoint for 10 epochs with the
  original loss.
- `O0-occ`: add only direct coarse occupancy supervision.
- `O0-occ-energy`: add occupancy plus energy ranking. Both arms use identical
  initialization, learning rate, and data order.

**Budget.**

- 10 epochs per learned arm, evaluated at epochs 5 and 10.
- The original matched continuation supplies the control; total new route
  budget is 20 learned epochs.

**Preregistered promotion gate.**

- finite nonzero gradient on all 32k coarse logits before hard top-k;
- selected-center 2 m target-support recall improves by `>=10` percentage
  points over `O0-control`;
- common Stage-0 geometry limits all pass;
- `O0-occ-energy` must improve far completeness by `>=5%` over `O0-occ`
  without more than `2` percentage points of outlier or duplicate degradation.

Eliminate the energy term if its incremental gate fails. Eliminate the entire
route if direct occupancy supervision improves coarse AUROC but not
selected-center support or geometry.

**Root-cause discrimination.**

- `O0-occ` passes while R0 does not: offsets can reach targets; the main failure
  was allocation-loss mismatch rather than initial support.
- Only the energy arm passes: cached occupancy is too sparse/misaligned and
  measured Cube support is the useful supervision.
- Coarse classification improves but geometry does not: hard set allocation or
  child topology, not score learning, is causal; advance H0.
- Neither arm changes support: the representation lacks candidate modes; O0 is
  closed.

### 5.3 C0: cross-scene condition contrast / task-level mutual information

**Unique hypothesis.** The global Full-RAED encoder contains useful
scene-specific geometry, but local query evidence creates a shortcut. A
task-level contrastive intervention can force correct condition information
into the output without changing proposal support.

**Minimal code change.**

1. For each training frame, reuse the existing deterministic cross-scene
   derangement and run a correct-condition and wrong-condition forward while
   holding measured Cube, proposals, local spectra, and occupancy queries fixed.
2. Add a stop-gradient margin:

   ```text
   L_cond = relu(m_occ + L_occ(correct) - stopgrad(L_occ(wrong)))
          + relu(m_geo + L_soft_geo(correct) - stopgrad(L_soft_geo(wrong)))
   ```

   Only the correct branch receives gradient from this term, preventing a fake
   win obtained solely by making the wrong branch arbitrarily bad.
3. Do not add another encoder, decoder, proposal source, or local-feature
   dropout in this Stage-0.

**Controls.**

- `C0-control`: two forwards with the contrastive loss weight fixed to zero.
- `C0-label-shuffle`: use an independently permuted wrong condition only for
  the diagnostic report, not model selection.

**Budget.**

- Fine-tune from the same G1D epoch-15 checkpoint for 5 epochs.
- One seed, one frozen margin, no sweep. The two-forward cost is bounded by the
  five-epoch cap.

**Preregistered promotion gate.**

- mean cross-scene condition-shuffle Chamfer degradation `>=3%`;
- mean shuffled occupancy-BCE degradation `>=3%`;
- correct-condition median completeness improves by `>=10%` over `C0-control`;
- correct-condition Chamfer, outlier, duplicate, and far metrics do not regress
  by more than `2%` relative or `2` percentage points as applicable;
- no same-sequence negative pair.

Eliminate C0 if it creates a shuffle gap without improving correct-condition
geometry. Such a result proves controllability of the dependency metric, not
utility of global condition information.

**Root-cause discrimination.**

- Dependence and geometry both improve: condition bypass is causal.
- Dependence rises but geometry is flat: bypass is real but secondary to
  support/allocation.
- Dependence cannot rise: the 336-token global representation or its fusion is
  too weak; a larger MI weight is not authorized.
- Geometry improves without a shuffle gap: the gain is ordinary fine-tuning,
  not condition use.

### 5.4 H0: G1G condition-exclusive hierarchy / dynamic center allocation

**Unique hypothesis.** The fixed 10k budget must be allocated globally before
local Cube evidence is exposed. Bounded four-child patches then prevent local
offset compensation and reduce duplicate crowding.

**Existing minimal implementation.**

- 336 global Full-RAED tokens;
- 2,500 global center queries;
- no local Cube spectrum, energy, neighborhood, or proposal score before center
  allocation;
- four children per center, each bounded to half a Cube cell;
- 10k output, center coverage/existence/repulsion, and bounded child diversity.

The H200 preflight already verified the information-path contract. The current
model uses a fixed interior `25 x 20 x 5` normalized RAE template grid plus a
bounded global residual. This is an intentional Stage-0 prior, but it means a
failure with a correct shuffle response can still identify a **range-mass
prior** problem rather than falsify hierarchy in general.

**Controls.**

- zero-local-refinement;
- child-collapse;
- cross-scene condition shuffle;
- center uniqueness and range-mass reports;
- G1D epoch-15 metrics as the fixed external control. No G1D weights are loaded.

**Budget.**

- One seed, 20 epochs, 1,520 optimizer updates.
- Evaluations at epochs 5, 10, 15, and 20.
- No architecture or loss-weight repair after seeing Stage-0.

**Preregistered promotion gate.**

Use the already encoded G1G gate without modification:

- condition-shuffle Chamfer degradation `>=1%`;
- duplicate fraction `<=15%`;
- median completeness `<=2.5068 m`;
- outlier fraction `<=25%`;
- far completeness `<=8.1239 m`;
- at least `80%` unique centers at 0.05 m.

**Root-cause discrimination.**

- All gates pass: global allocation topology is the strongest geometry parent.
- Shuffle passes and duplicates improve, but completeness/far fail: hierarchy
  fixes crowding, while center range allocation remains wrong.
- Geometry improves but shuffle fails: local child refinement still dominates;
  H0 cannot support a condition-dependent method claim.
- Shuffle passes but outlier is high: the fixed global template prior allocates
  unsupported centers; compare its range mass to G1B before changing child
  losses.
- Centers are unique but children collapse: patch diversity/local spectrum
  decoder is the cause, not global allocation.

### 5.5 T0: G1T temporal proposal support

**Unique hypothesis.** Some weak or occluded current-frame surfaces are absent
from the current Cube proposal pool but are present in recent radar history.
Ego alignment should restore static support; Doppler displacement should add
dynamic support beyond ego-only aggregation.

**Minimal code change.**

No model training or new architecture. Reuse the implemented:

- `T0-current`: current proposals only;
- `T1-ego`: current plus four ego-warped history frames;
- `T2-doppler`: the same union with measured Doppler radial displacement;
- identical current-Cube rescoring, `(5,5,3)` duplicate suppression, and exact
  10k export.

**Controls.**

- `T1-ego` is the required non-Doppler temporal control.
- Every arm has the same history count, current-Cube rescore, NMS, and output
  count.
- Source age and static/dynamic, near/far slices are mandatory.

**Budget.**

- Zero training epochs.
- Evaluate all 384 validation windows after the source-bound cache finishes.
- No LiDAR/GT enters selection.

**Preregistered promotion gate.**

- `T2-doppler` improves completeness and far completeness over `T1-ego`;
- outlier and duplicate fractions increase by no more than `2` percentage
  points versus `T1-ego`;
- `T2-doppler` Chamfer is better than `T0-current`;
- transforms, timestamp sign, history count, and current-Cube rescoring all
  verify.

**Root-cause discrimination.**

- `T1` and `T2` both beat current, but `T2` does not beat `T1`: current-frame
  support is missing, but the useful history is mainly static/ego-aligned.
- `T2` uniquely passes: dynamic historical support is real and a
  Doppler-specific temporal proposal is justified.
- History increases candidates but current-Cube rescoring removes them: weak
  current evidence, not warp quality, is the limiting mechanism.
- No temporal arm helps: missing target support is not recoverable from the
  four-frame radar history under the current observability definition.

## 6. Cross-route decision matrix

| R0 | O0 | C0 | H0 | T0 | Most defensible diagnosis |
|---|---|---|---|---|---|
| pass | not needed/fail | any | any | any | range-biased current proposal support is primary |
| oracle pass, learned fail | pass | any | any | any | train/inference allocation supervision is primary |
| fail | fail | pass with geometry | fail | fail | global condition shortcut is primary, but support remains a risk |
| fail | fail | dependence-only | pass | any | allocation topology, not condition information content, is primary |
| fail | fail | fail | shuffle pass, far fail | pass | current-frame observability/support is primary; history supplies missing modes |
| fail | fail | fail | fail | fail | current target/representation is not recoverable by bounded model repair; open a new dense ray/range representation program |

Two apparent wins are explicitly rejected:

1. a larger condition-shuffle gap without correct-condition geometry gain;
2. lower duplicates obtained by confidence masking, lower output count, or
   removing far points.

## 7. H200 launch recommendation

Only three routes should enter the immediate H200 queue.

### Priority 1: G1T temporal support on GPU0

GPU0 is currently building the 384-frame temporal validation cache. Let that
source-bound build finish, then run G1T immediately on GPU0. It is no-train,
tests a root cause disjoint from G1D/G1G, and should release GPU0 quickly.

### Priority 2: R0 range-aware proposal support on GPU0

After G1T, run only the R0 no-train oracle on GPU0. Launch its 10-epoch learned
arm only if the oracle reaches the preregistered support gates. This prevents
spending another training run on an inadequate candidate pool.

### Priority 3: G1G hierarchy on GPU2

GPU2 remains reserved for the unchanged G1D v2 run through epoch 150. Do not
preempt it. Once the frozen endpoint and hashes are archived, run the completed
G1G smoke test and one-seed 20-epoch Stage-0 on GPU2. G1G's measured preflight
peak was about `8.63 GB`, but it should still run alone for clean timing and
failure attribution.

```text
GPU0: temporal cache -> G1T (0 epoch) -> R0 oracle -> R0 10 epochs if authorized
GPU2: G1D v2 to epoch 150 -> archive -> G1G smoke -> G1G 20 epochs
GPU1: never use
```

Do not launch O0 and C0 concurrently with these first three routes. Authorize
O0 only if R0's new oracle passes but its learned selector fails. Authorize C0
only if a route has adequate support/geometry yet still fails the condition
shuffle gate. This ordering preserves causal interpretability while still
testing three distinct mechanisms in parallel across the two allowed H200s.

## 8. Final research judgment

The strongest current conclusion is not “the network needs more training.”
G1D's optimization trajectory, G1F's oracle failure, and the code's query paths
jointly indicate a representation/allocation mismatch:

> Current G1D is precise where strong measured peaks already exist, but its
> range-biased proposal pool does not cover the full radar-observable target;
> local query evidence then bypasses the global condition, and the set loss
> spends repeated children on easy surfaces instead of creating missing modes.

The next successful geometry parent must demonstrate three properties
independently before Doppler generation or Cube-cycle claims are unlocked:

1. far-range support exists before hard selection;
2. global Full-RAED condition causally changes allocation;
3. fixed-count coverage is improved without duplicate or outlier inflation.

R0, H0, and T0 are the smallest current experiments that distinguish these
three requirements without reopening rejected branches or rewriting the
project.
