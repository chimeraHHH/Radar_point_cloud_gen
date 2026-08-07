# VRH-F0 variable-return renewal-hazard capacity protocol

Status: frozen before implementation and execution

Independent read-only audit returned `FREEZE` for candidate SHA-256
`137cecfa30673d4155c3571efbde3b691733ba9e7d284671b380a0a787e85a9b`.

## Decision question and evidence boundary

Can one fixed radar-frustum mark field, decoded as an ordered variable-return
process and exported through a one-next-event-per-ray frontier, contain an exact
10,000-point, true-5-cm set that passes every frozen train-only geometry and
anti-collapse gate?

VRH-F0 is a zero-training, GT-aided, non-deployable representation/export
capacity falsification. It is not a learned model, radar observability result,
Doppler result, Cube-cycle result, temporal result, validation result, or test
result. Ground truth is used only by `fit_gt_oracle_marks` and the post-export
metric evaluator; the support builder, renewal decoder, and both exporters do
not receive the raw target tensor, target IDs, or audit sidecar. A pass validates
only the complete frozen recipe. It is a
deterministic best-found diagnostic, not a mathematical upper bound on the
variable-renewal family.

The formal report must state the target-use boundary literally:

```text
target_used_for_mark_hazard_fit = true
target_used_for_mark_offsets = true
target_used_for_mark_priority = true
target_used_for_seed_reservation = false
decoder_target_input = false
exporter_target_input = false
metric_evaluator_target_input = true
```

The fitted offsets can exactly represent a target coordinate when it lies in a
cell. That is legal only inside this explicitly non-deployable capacity oracle.
The report must count exact target-coordinate coincidences. It may not present
them as generated points or model performance.

## Relation to closed routes

Three predecessor boundaries are terminal:

1. R-B1 used direct GT-supported peaks, capped every native ray at fixed
   `K=4/6`, and inherited the later-disproved `8000/1700/300` quotas. Its two
   audited frames produced only `6798/4567` peaks for `K=4` and `7939/5104`
   for `K=6`.
2. The 76-frame radial proof established that the hard per-frame range quotas
   are incompatible with the geometry contract on two train frames. VRH-F0
   records range counts but never uses a range quota.
3. Q-Local-F0R removed those quotas and passed Chamfer, outlier, exact-count,
   and spacing on 12/12 frames, but retained all target-bearing range strata on
   only 2/12. Independent point scores followed by one global top-set export
   are therefore closed on the frozen Fresh-WCE field.

VRH-F0 is a paired combination test, not a single-mechanism causal ablation. Its
decision arm combines one `2R x 2A x 2E` polar lattice, bounded target-fitted
marks, renewal MAP decoding, and sequential frontier exposure. A mandatory
same-mark all-events control removes only frontier blocking. A pass cannot be
attributed to renewal unless the frozen renewal-activity certificate is also
true. Even then, the certificate proves that renewal changed the effective set,
not that it alone caused a geometry improvement.

## Primary-source mechanism boundary

- [RaLD](https://ojs.aaai.org/index.php/AAAI/article/view/38946) contributes
  scene-level frustum latents, order-invariant mixed set latents, and continuous
  coordinate queries. A learned successor must use those condition and query
  interfaces rather than copy RaLD's released thresholded intensity export.
- [DenserRadar](https://arxiv.org/abs/2405.05131) motivates the one frozen
  `2R x 2A x 2E` Full-Doppler frustum resolution. No resolution sweep is legal.
- [Neural LiDAR Fields](https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_Neural_LiDAR_Fields_for_Novel_View_Synthesis_ICCV_2023_paper.pdf)
  motivates resetting transmittance after a return. NFL itself predicts at
  most two returns and is not reused as a radar model.
- [PLiNK](https://arxiv.org/abs/2411.01725) motivates a range-wise probability
  density/CDF with multiple peaks. VRH adds explicit sequential events and STOP.
- [Neural Hawkes](https://papers.neurips.cc/paper_files/paper/2017/hash/6463c88460bd63bbe256e495c63aa40b-Abstract.html)
  motivates conditioning a next event on previous events. VRH is a bounded
  range process, not a claim that radar returns obey a Hawkes excitation law.

These sources motivate an architecture hypothesis; they do not supply evidence
for the frozen gate.

## Frozen cohort and source binding

The formal cohort and order are identical to Q-Local-F0/F0R:

```text
fit:
  1:232, 9:440, 21:402, 27:403,
  29:399, 35:403, 47:514, 58:404
unseen_train:
  5:429, 26:403, 42:408, 50:405
```

All 12 identities belong to the frozen 76-frame train split. Validation, test,
future Cube, and future target access are false. The run binds:

- manifest SHA-256
  `645307a8bae351db51b55128043dae69bce5b928169d5fa25c1b9c55083de4e4`;
- scene-split SHA-256
  `61596bd50ce0bdab633c9ff0ec5ab5148c2c78a10dfc06a71d0d6f0a6c9427cc`;
- the exact target-cache paths, bytes, tensor hashes, frame order, and group
  labels recorded by archived F0R report SHA-256
  `505644667ba3f73b58ee138608307e82632fe92bc3336f538c74767ab9c148ae`;
- corrected dense-geometry evaluator SHA-256
  `e1f970c1e827c7801c7c9018378603b89a7709b0b4bf261e740eab1575631e68`;
- K-Radar range, azimuth, and elevation axis source bytes;
- the protocol-freeze Git blob, clean implementation commit, all implementation
  and evaluator sources, physical H200 index/UUID/PCI identity, and atomic
  terminal output.

The capacity oracle does not read Radar Cube or CFAR arrays. Cube evidence is
reserved for a separately frozen target-free learnability gate.

## Target-independent `2R x 2A x 2E` support

Let each strictly increasing native center axis be `x[0:n]`. For azimuth and
elevation, construct native Voronoi edges as

```text
edge[0] = x[0] - 0.5 * (x[1] - x[0])
edge[i] = 0.5 * (x[i-1] + x[i]), 0 < i < n
edge[n] = x[n-1] + 0.5 * (x[n-1] - x[n-2]).
```

For range, use the same interior midpoints, clamp only the unphysical lower
half-cell to `range_edge[0]=0.0 m`, and retain the natural upper half-step:

```text
range_edge[256] = range_center[255]
                  + 0.5 * (range_center[255]-range_center[254]).
```

For the frozen K-Radar axis this upper edge must equal
`118.2685546875 m`. No support is extrapolated to the 120-m metric boundary.
Split every resulting native interval at its midpoint, in float64, yielding:

```text
R2 = 512
A2 = 214
E2 = 74
subrays = A2 * E2 = 15,836
ray-range cells = R2 * A2 * E2 = 8,108,032
```

The stable cell ID is

```text
cell_id = ((a2 * E2) + e2) * R2 + r2.
```

Angular support is the non-wrapping radar FOV defined by the extrapolated axis
edges; its width must be less than `2*pi`. Input assignment uses half-open cells
`[lower, upper)`, except that a value bitwise equal to the final angular edge is
assigned to the last cell. Azimuth is canonicalized to `[-pi, pi)` before that
non-wrapping assignment. Range support is the half-open interval
`[0,118.2685546875)` m. A target outside angular support or this natural range
support stays in global geometry,
is recorded as out-of-support, and causes scientific failure if it has positive
effective confidence. It is not silently clipped out of evaluation.

Event parameters use the strict float64 interior of a cell:

```text
lower_interior = nextafter(lower, upper)
upper_interior = nextafter(upper, lower)
```

The fitted `(r,a,e)` is clamped to that closed interior. A cell without three
representable interiors is implementation-invalid. This rule prevents two
neighboring cell IDs from producing the same shared-boundary coordinate.

The complete centers, bounds, legal-range intersections, stable-ID stream, and
schema are built and SHA-256 hashed before the target loader is callable. Cells
cannot be created, deleted, or reindexed after target access. Each cell can emit
at most one event. Thus a ray has an explicit finite structural upper bound of
512 cell events, but there is no fixed per-ray `K`, learned return slot count, or
return-count quota.

## Frozen GT-aided mark fitter

`fit_gt_oracle_marks(target)` is the only function allowed to receive the target
tensor. It cannot receive a requested count, range class, output mask, selected
cell list, or exporter state. It writes two physically separate products.

The decoder-visible `ModelMarks` contains one model-shaped record per cell:

```text
(base_hazard, delta_r, delta_a, delta_e,
 opaque_distance_priority_key, confidence)
```

The audit-only `FitSidecar` contains:

```text
(nearest_target_id, fitted_distance_m, exact_target_coordinate_match)
```

`FitSidecar` is never passed to, imported by, or opened by the decoder or either
exporter. Its file path is absent from their configuration. Decoder/exporter
source modules may depend only on the support and `ModelMarks` schemas.

Target rows are canonicalized by lexicographically sorting float64
`(r,a,e,x,y,z,confidence)`. Exact duplicate rows are indistinguishable; their
canonical IDs are the contiguous post-sort ranks, and selecting any duplicate
must produce the same mark bytes. Input-row permutation must not change the
mark-field hash or either output-set hash.

For every cell in ascending stable-ID order:

1. Query the nearest canonical target to the fixed cell center in Euclidean
   XYZ. Compare squared distances in float64. If the two nearest distances differ
   by at most `1e-12 m^2`, inspect every target within that tolerance and choose
   the smallest canonical target ID.
2. Convert that target to canonical spherical coordinates, clamp each coordinate
   to the cell interior, and convert the fitted coordinate back to float64 XYZ.
3. Set audit-only `fitted_distance_m` to the Euclidean distance from fitted XYZ
   to the chosen target XYZ. Set the decoder-visible opaque priority to
   `round_ties_to_even(1e9*fitted_distance_m)` as a signed 64-bit integer.
4. Set `confidence=max(target_confidence,0)`; non-finite target values are
   implementation-invalid.
5. With the single frozen constants `tau=0.20 m` and `epsilon=2^-24`, set

   ```text
   base_hazard = clip(exp(-fitted_distance_m / tau),
                      epsilon, 1-epsilon).
   ```

Here `tau=0.20 m` is a one-time subcell-scale design constant, not a fitted or
physically unique value; `epsilon=2^-24` is a finite probability floor. Neither
may be swept. This nearest-target bounded-offset field is the sole allowed
target-conditioned
lattice lifting. It is intentionally explicit and is not claimed to be learned.
No seed set, seed count, support weight, FPS traversal, hard range allocation,
target-class allocation, alternative fitter, threshold sweep, or solver sweep is
allowed. The report records nearest-target lifting fanout per canonical target,
including minimum/median/maximum, and the exact-coordinate coincidence count.

## Variable-return renewal decoder

The decoder API accepts only the immutable support and `ModelMarks` blob. It
cannot import the fitter or sidecar module. For ray `q`, initialize

```text
c_0 = 0
H_0 = (depth=0, previous_cell_id=NONE, previous_radius=NONE).
```

At state `m`, a bin `b` is legal only for `c_m <= b < R2`. Its effective hazard
is `base_hazard(q,b)`, except that it is exactly zero when a previous event exists
and its fitted radius is less than `previous_radius + 0.05 m`. This is the one
frozen return-conditioned refractory rule. It is not a fitted binary mask.

For every legal bin and explicit STOP, compute in float64 log space:

```text
log P(B_m=b | H_m) = log h(q,b | H_m)
                     + sum_{j=c_m}^{b-1} log1p(-h(q,j | H_m))

log P(STOP | H_m) = sum_{j=c_m}^{R2-1} log1p(-h(q,j | H_m)).
```

An effective zero hazard has event log-probability `-inf` and contributes zero to
the survival sum. Non-finite event choices are removed before quantization and
cannot win MAP; STOP remains finite because fitted hazards are at most
`1-epsilon`. At `c_m=R2`, STOP has probability one. The last legal bin and STOP
therefore remain separate choices. An empty ray is one whose initial MAP choice
is STOP.

For finite values define `Q_log(x)=round_ties_to_even(1e12*x)`. The MAP choice
maximizes `Q_log(log P)`; a tied event precedes STOP, and tied events use smaller
`r2`. `Q_log` is never called on `+/-inf` or NaN. No infinite logits or post-hoc
0/1 oracle mask is used.

If STOP wins, the ray closes. If event `b` wins, append exactly that event to the
ray's canonical stream and advance:

```text
c_{m+1} = b + 1
H_{m+1} = (depth=m+1, previous_cell_id=cell_id(q,b),
           previous_radius=fitted_radius(q,b)).
```

The survival interval is therefore reset after every decoded event. Before
either exporter runs, `decode_canonical_streams(support, model_marks)` fully
unrolls every ray to STOP once, in ascending subray order, and freezes one
canonical per-ray stream. The event sequence is variable length, ordered in
range, and capped only by finite support. Neither acceptance nor global export
order can change this stream.

Canonical hashes use a sorted-key, UTF-8, LF-terminated schema header followed
by C-contiguous little-endian raw arrays in a fixed column order. `ModelMarks`
columns are `(base_hazard:f8, delta_r:f8, delta_a:f8, delta_e:f8,
opaque_distance_priority_key:i8, confidence:f8)`. Canonical-stream columns are
`(ray_id:u2, stream_index:u2, cell_id:u4, depth:u2, q_log_probability:i8,
opaque_distance_priority_key:i8, confidence:f8, r:f8, a:f8, e:f8)`. The sidecar has a separate
hash and is not part of either exporter input.

## Sequential frontier exporter

The exporter accepts only the frozen canonical streams; it cannot receive
`ModelMarks`, `FitSidecar`, target rows, target classes, preselected IDs, a seed
mask, or a desired per-range count. Integer tie keys are:

```text
Q_conf(w) = round_ties_to_even(1e9*w)
heap_key = (-Q_log(event_log_probability),
            opaque_distance_priority_key,
            -Q_conf(confidence),
            stable_cell_id).
```

The complete frozen algorithm is:

```text
accepted = []
next_index[q] = 0 for every q in ascending subray ID
frontier = empty min-heap

for q in ascending subray ID:
    if canonical_stream[q] is nonempty:
        push(frontier, canonical_stream[q][0].heap_key,
             canonical_stream[q][0])

while len(accepted) < 10000 and frontier is not empty:
    event = pop(frontier)
    if no accepted XYZ has squared float64 distance < 0.05^2:
        append event to accepted
    consume event
    next_index[event.q] += 1
    if next_index[event.q] < len(canonical_stream[event.q]):
        next_event = canonical_stream[event.q][next_index[event.q]]
        push(frontier, next_event.heap_key, next_event)

return accepted
```

The online spacing test uses a cubic spatial hash of side `0.05 m`, floor-based
integer cell coordinates, and all 27 neighboring hash cells. Distance exactly
`0.05 m` is accepted; distance below it is rejected. A rejected event is logged
and consumed, never replaced by a duplicate. The run stops only at exactly
10,000 accepted events or an exhausted frontier. It does not revisit a cell or
restart a ray. Failure to reach 10,000 is a scientific capacity failure after
all remaining frames are still evaluated. Its exposure-trace hash is computed
separately from the canonical-stream hash over
`(pop_index:u8, ray_id:u2, stream_index:u2, cell_id:u4, accepted:u1)` records.

## Mandatory same-mark flat-exposure control

For every frame, the control receives the exact same already-frozen canonical
streams. It exposes every event from those streams to one heap at once and
applies the identical 5-cm greedy acceptance until 10,000 or exhaustion. It
changes only the one-next-event frontier constraint; it does not decode again,
refit marks, alter scores, or read the sidecar.

Both arms bind the same canonical-stream hash and have distinct exposure-trace
and selected-set hashes. The renewal-activity certificate is true only when:

1. both arms bind the one pre-export canonical-stream hash;
2. at least one of 12 accepted-cell-ID hashes differs between decision and
   control; and
3. the decision arm accepts at least one event of decoded depth two or greater.

Every scientific gate below is also evaluated on the control. Renewal utility is
certified only when the decision arm passes every gate on all 12 frames and the
control fails at least one identically defined per-frame structural, geometry,
stratum, first-return, or later-return gate that the decision arm passes. If both
arms pass all gates, or activity is false, the terminal is
`vrh_f0_lattice_only`, because the frontier was not required by this test.

## Frozen evaluators and scientific gates

Global geometry must call `geometry_report` and `nearest_distance` from the
source-bound `code/eval/dense_geometry.py` used by F0R. Chamfer, 2-m outlier,
confidence-weighted completeness, and range masks retain exactly those source
semantics. Per-stratum completeness and recall@1m reproduce
`target_stratum_retention` from the archived F0R preflight: nonnegative target
confidence, target-to-prediction nearest XYZ distance, confidence-weighted mean,
and confidence-weighted fraction at distance `<=1.0 m`. A target-bearing stratum
with zero effective confidence is implementation-invalid, as in F0R.

Every one of the 12 decision-arm frames must pass:

- exact 10,000 selected events, unique stable cell IDs, and unique finite XYZ;
- every fitted `(r,a,e)` inside its source-cell interior;
- decoder/export replay reproducing the event trace and selected IDs exactly;
- independent KD-tree minimum pair distance `>=0.05-1e-12 m`;
- zero positive-confidence target weight outside frozen support;
- no direct target input to decoder/exporter, padding, random jitter, duplicate
  fallback, range quota, frame deletion, best-of-k, threshold sweep, or rerun
  with another lattice/fitter/export algorithm;
- Chamfer `<=0.8 m`;
- 2-m outlier fraction `<=5%`;
- every target-bearing `[0,30)`, `[30,60)`, or `[60,120)` m stratum no worse
  than its archived fixed-quota reference:

  ```text
  VRH completeness <= archived fixed completeness + 1e-6 m
  VRH recall@1m >= archived fixed recall@1m - 1e-6.
  ```

The fixed reference is a metric boundary only; its retired output quotas are not
reused.

## Renewal-specific first/later-return gate

Map every in-support target to the same `(a2,e2)` subray. Within each subray,
sort by `(range, canonical_target_id)` and form groups greedily: start a group at
the first remaining point and append subsequent points only while
`max_range-min_range < 0.05 m`; otherwise start a new group. The nearest group is
`first`; all remaining groups are `later`.

For a group, use `max(confidence,0)` as point weight. Its representative range is
the confidence-weighted mean; if the group weight is zero, use the unweighted
mean but retain zero effective weight. The group weight is the sum of its point
weights. Assign the group to a range stratum by representative range. A nonempty
evaluated class with zero total effective weight is implementation-invalid.

Match all groups and accepted events once per subray and arm using an
order-preserving dynamic program. Events are sorted by `(fitted_radius,
stable_cell_id)` and cannot be reused. A first group may match only an accepted
decoded-depth-1 event. A later group may match only an accepted event of decoded
depth at least two when the same DP path has already matched an earlier target
group to a lower-range accepted event on that ray. Thus a neighboring ray, an
unrelated unmatched predecessor, a first event, or an already matched event
cannot cover a later target group.

For each eligible pair, radial cost is `abs(group_range-event_range)`. Skipping
an event costs zero. Leaving a target group unmatched assigns it `120.0 m`.
The DP minimizes integer weighted cost using
`round_ties_to_even(1e9*group_weight) *
round_ties_to_even(1e6*assigned_distance_m)`; ties prefer more matched groups,
then the lexicographically smaller stable-cell-ID sequence. The report stores
the assignment and independently verifies monotonicity and non-reuse.

For every nonempty `range stratum x {first,later}` class on every frame:

```text
completeness = sum(group_weight * assigned_distance) / sum(group_weight)
recall@1m = sum(group_weight * [assigned_distance <= 1m]) / sum(group_weight)

completeness <= 1.0 m
recall@1m >= 0.80.
```

Aggregate geometry cannot replace a failed frame, range stratum, first-return
class, or later-return class.

## Implementation-validity tests

Before the one formal run, a clean H200 snapshot must pass targeted and full
regressions covering:

- exact `512/214/74`, 8,108,032 cells, natural
  `118.2685546875 m` upper range edge, bounds, half-open assignment, angular
  final-edge assignment, non-wrapping FOV, stable-ID order, and support digest;
- target loader unreachable before the support digest is committed;
- target-row permutation and exact-duplicate invariance;
- nearest-target tolerance/tie rule, strict-interior clamp, range clipping,
  out-of-support accounting, and lifting-fanout reporting;
- finite hazard formula and the frozen `tau/epsilon` values;
- `c_0`, last-bin-vs-STOP, empty ray, `0/1/2/7/11` returns, hit-reset-next-hit,
  refractory 4.999-cm suppression, 5.000-cm allowance, and more than six returns;
- separate `ModelMarks`/`FitSidecar` serialization and source-dependency checks;
  decoder/exporter signatures, imports, configurations, and opened files must
  contain no target rows/IDs, sidecar, selected mask, seed reservation, fixed
  return cap, range quota, or hidden fill fallback;
- non-finite event removal before `Q_log`, finite STOP, and NaN hard failure;
- exact pseudocode order, heap integer keys, accept/reject consumption, frontier
  exhaustion, and replay/hash equality;
- one complete canonical stream decoded before export, the same stream hash for
  both arms, distinct exposure-trace hashes, and a synthetic case in which
  frontier blocking changes the selected-cell hash;
- a decision-pass/control-fail synthetic utility case and a both-pass synthetic
  case that must route to `vrh_f0_lattice_only`;
- online and independent spacing: 4.999 cm rejected, 5.000 cm accepted;
- span grouping for `0/4/8 cm`, same-ray monotone one-to-one matching, no event
  reuse, rejection of an unrelated unmatched predecessor, first-only failure on
  a later group, and 120-m unmatched cost;
- exact-count hard failure without copy, padding, jitter, duplicate repair, or
  alternative selection;
- F0R-style near-collapse where aggregate Chamfer passes but a target stratum
  fails;
- exact reuse of F0R confidence-weighted geometry and retention semantics;
- source/data/frame-order/GPU tamper, validation/test path, partial-output,
  atomic-write, scientific-exception, OOM, and timeout tests.

The formal run is implementation-invalid if any source, input, support, cohort,
API, numeric-boundary, hash, GPU, transaction, or evaluator-binding check fails.
It must be repaired and rerun under the identical frozen protocol and cannot
route scientifically.

## Runtime, terminals, and routing

The single formal run executes as `wangning` on the H200 server, using physical
H200 GPU 0 or 2 for any CUDA work; physical GPU 1/RTX 5000 is forbidden. The
Conda environment name begins with `hym_`. The gate has a two-hour wall budget
and a 60-GiB peak CUDA-memory ceiling. CPU work on the H200 host is permitted and
reported. The atomic output includes source/protocol blobs, all input/support
hashes, target-use flags, schema headers, separate model-mark/sidecar hashes,
canonical-stream hash, compact per-arm exposure traces, selected hashes,
decision/control geometry, range and first/later metrics, exact-match/fanout
diagnostics, physical GPU provenance, elapsed time, peak host memory, and peak
CUDA memory. Full 8.1-million-cell marks remain canonical binary arrays rather
than expanded JSON.

Scientific failure does not permit early exit: all 12 frames and both arms must
complete and be archived. Terminal states are:

- `vrh_f0_capacity_passed`: implementation valid, all 12 decision frames pass
  every scientific gate, renewal activity is certified, and the flat-exposure
  control fails at least one corresponding gate passed by the decision arm.
  This authorizes only a separately frozen target-free one-frame
  mark-field/renewal learnability and condition-use gate.
- `vrh_f0_lattice_only`: all decision-arm scientific gates pass, but renewal
  activity is false or the flat-exposure control also passes every gate. The
  result may support only a fixed-lattice/mark-field capacity statement and does
  not authorize a renewal claim or learned renewal route.
- `vrh_f0_capacity_no_go`: implementation valid and both arms finish all 12,
  but at least one decision-arm scientific gate fails. This closes only the
  exact `2x lattice + frozen nearest-target mark fit + MAP renewal + frontier`
  recipe. It does not prove that variable-return renewal representations are
  impossible.
- `vrh_f0_implementation_invalid`: a binding, API, numeric, transaction,
  evaluator, or correctness check fails. Repair and rerun the same gate.
- `vrh_f0_resource_invalid`: OOM, wall timeout, memory-limit violation, host
  interruption, or an incomplete 12-frame/two-arm run. It carries no scientific
  capacity conclusion.

After a valid no-go or lattice-only terminal, the project stop-rule routes to a
separately frozen sparse ray-range partial-transport hard-rounding oracle. That
route is an engineering decision, not a family-wide impossibility theorem. No
status may be inferred from an aggregate mean, partial cohort, uncommitted
source, manually inspected intermediate file, or the non-decision control alone.
