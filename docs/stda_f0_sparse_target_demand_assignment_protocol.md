# STDA-F0 parity-packed sparse target-demand assignment protocol

> Status: **FINAL IMMUTABLE FREEZE CANDIDATE; ACTIVATION IS EXTERNAL AND SHA-BOUND**  
> Date: 2026-08-08  
> Scope: zero-training, train-only, non-deployable capacity/mechanism gate  
> Predecessor: `vrh_f0_capacity_no_go` at source `380f3ea`

## 1. Decision question and naming boundary

Can a target-demand-coupled, graph-constrained bipartite assignment select
exactly 10,000 distinct points from one target-independent largest-color parity
support derived from the unchanged 700k Fresh-WCE field, while satisfying
strict 5 cm spacing, absolute geometry, target-bearing range retention, and
representation-neutral range-by-return retention?

`STDA` means **sparse target-demand assignment**. The F0 mechanism is a sparse
rectangular linear assignment on a parity-packed candidate support. It is not
called ray-range transport, optimal transport, hard rounding, or a unique
solver. Its `K=256` graph is not claimed equivalent to a dense assignment.
Partial-transport notation is useful background, not an algorithmic novelty
claim. The assignment primitive can only be an internal capacity mechanism for
the project's Full-RAED exact-count state generator.

This gate does not train a model, produce a deployable selector, unlock
Doppler/cycle/temporal work, or access validation/test/future data.

These protocol bytes never change merely to mark a freeze. Section 15 defines
the external, SHA-bound activation rule. A `FREEZE` audit of the exact bytes plus
its committed freeze record activates them; a `NO-FREEZE` verdict requires new
bytes and another complete audit.

## 2. Frozen predecessors, source, and cohort

The formal run must bind:

- the Fresh-WCE-20 checkpoint and normalization hashes used by Q-Local-F0R;
- Q-Local-F0R and VRH-F0 source/report/protocol hashes;
- the target-free predecessor candidate-hash manifest at
  `artifacts/idea/stda_f0_candidate_hash_manifest.json`, SHA-256
  `1d650272a021c36ee2d92219ebec3b1d91508aefdb5242c0e6fcd323d6bed2bf`;
- the corrected evaluator at `code/eval/dense_geometry.py`, SHA-256
  `e1f970c1e827c7801c7c9018378603b89a7709b0b4bf261e740eab1575631e68`;
- the structural evaluator at `code/eval/vrh_f0_metrics.py`, SHA-256
  `3450f83aa7c4add4198f008995b4928e759391a5d74e8a72b0eedf6da2185a42`;
- all 76 frames of the accepted training split in the archived canonical order;
- every raw-Cube, target-cache, checkpoint, source, and environment hash;
- one clean source commit and one allowed physical H200 identity.

The earlier 12-frame F0R/VRH cohort is design history, not an unseen cohort.
No frame is called unseen. All 76 training frames are formal; validation, test,
future frames, and alternative cohorts are forbidden. All 76 frames complete
after the formal transaction starts. A scientific failure does not stop later
frames. Only implementation-invalid or resource-invalid conditions may abort.

The target-free audit previously performed on train frame `1:232` is recorded
only as pre-freeze engineering history. It cannot select constants or count as
formal evidence. After this protocol freezes, all smoke tests are synthetic;
the target-conditioned 76-frame execution is launched once.

## 3. Explicit target-use semantics

Every report and callable boundary must expose these exact booleans:

```text
support_target_input=false
demand_target_conditioned=true
graph_target_conditioned=true
cost_target_conditioned=true
control_score_target_conditioned=true
assignment_target_conditioned=true
solver_raw_target=false
deployment_claim=false
```

The graph and integer edge costs remain GT-conditioned even though the solver
does not receive raw target rows. API isolation is not deployability. A pass
proves only candidate/support/allocation capacity and authorizes a separate
Cube-conditioned learnability gate.

## 4. Two-process target isolation

The formal transaction has two non-overlapping phases.

### 4.1 Support phase

The support-only program is exactly the single CUDA-visible support child
described in Section 13. It processes all 76 frames in that one OS process,
creates no descendant processes, and exits before any target loader starts.
The minimal launcher installs `sys.addaudithook` before importing NumPy,
PyTorch, any project/data module, or opening any non-bootstrap path; interpreter
bootstrap reads are separately allowlisted. Candidate reconstruction, support
construction, and support serialization all occur in this audited process. It
must:

1. load each Cube directly through the frozen `load_tesseract` path;
2. never call `PilotCubeDataset.__getitem__`, because that path loads GT;
3. reconstruct the exact 700k Fresh-WCE Q0/Q1, refined XYZ, and base confidence;
4. use a new candidate-only verifier with no target or nearest-target argument;
5. verify the candidate hashes for the 12 predecessor frames against the
   frozen target-free expected-hash manifest and record fresh content hashes on
   the other 64 frames;
6. build and serialize the target-independent support in Section 5;
7. commit and rehash every support before the process exits.

The support process accepts no target/cache path in its signature, argv,
environment, config, import-side global, or IPC. A Python audit hook records
every file-open event. Read access is restricted to the pinned interpreter/
environment roots, clean source snapshot, checkpoint, normalization, manifest,
scene split, axis resources, the literal-SHA candidate-hash manifest, and the
current raw-Cube path; write access is restricted to its staging subtree.
Cache-root, LiDAR, label, current-frame target, CFAR, and future-frame paths are
explicit denials even if they share a parent data root.
The full canonicalized open ledger and allow/deny decision are hashed. Static
signature/import checks and an adversarial fake-target-path test are mandatory.
Any current-frame GT-bearing or target-derived sidecar access is
implementation-invalid. This prohibition does not relabel the explicitly
allowed, hash-bound trained Fresh-WCE checkpoint as target-free; the checkpoint
is historical model input, whereas `support_target_input=false` concerns the
current frame and its GT/cache artifacts.

### 4.2 Oracle phase

Only after the complete support manifest has been fsynced, hashed, re-read, and
the support process has exited may a new oracle process load targets. The
oracle receives immutable support bytes and cannot reconstruct, alter, recolor,
or replace candidates. Target/hash checks omitted from the support phase occur
here and cannot modify the committed support.

## 5. Target-independent parity-packed 5 cm support

Deserialize every final candidate XYZ as float32 and let `h=0.05 m`:

```text
g_i = floor(x_i / h) in Z^3
color_i = (g_ix mod 2, g_iy mod 2, g_iz mod 2) in {0,1}^3
```

The implementation does not evaluate `x/h` approximately. It promotes each
float32 value exactly with `as_integer_ratio()` and computes `g=floor(20*x)` by
integer floor division, identically to the independent spacing verifier.

All XYZ and base confidence values must be finite. Within each occupied cell,
retain the numerically highest frozen base confidence, then the smallest stable
candidate ID. For each of the eight parity colors, collect retained cells.
Select the color with largest cardinality, breaking ties by binary color ID
`4*x+2*y+z`. This decision is committed before target access. Best-of-color
geometry selection, shifted grids, adaptive cells, padding, copies, duplicates,
jitter, and a fallback color are forbidden.

The parity construction is spacing-safe in exact arithmetic, but the proof is
not the operational gate. After each support and exported point set is
serialized as little-endian float32 XYZ and deserialized, the independent
verifier performs complete spatial-hash enumeration:

1. promote each float32 coordinate exactly to a Python float and obtain its
   dyadic `(numerator, denominator)` with `as_integer_ratio()`;
2. compute the exact hash cell `floor(20*x)` by integer floor division;
3. visit points in stable row order and enumerate all prior points in the 27
   cells with offsets in `{-1,0,1}^3`;
4. compare the exact rational squared distance with `1/400` by integer cross
   multiplication, with no float tolerance.

This enumeration is complete: a pair with Euclidean distance below `1/20`
must differ by less than `1/20` on every coordinate, so its exact hash-cell
indices differ by at most one on every axis. Reject iff exact squared distance
is below `1/400`; equality passes. The verifier reads serialized bytes and
cannot import the support builder's spacing code.

The same verifier must cover negative cells, cell boundaries, ULP-adjacent
coordinates, `4.9999/5.0000 cm`, and float32 serialization round trips. If the
largest color contains fewer than 10,000 points, the frame has a packed-support
scientific failure reported as `stda_f0_packed_support_capacity_no_go`; no graph
is constructed and no repair is allowed.

## 6. Permutation-invariant target demand

The sole target-reading fitter consumes `target_xyz_confidence` after support
commitment.

1. Read XYZ/confidence as little-endian float32. Non-finite values or negative
   confidence are implementation-invalid.
2. Canonicalize every signed-zero XYZ/confidence bit pattern to positive zero;
   drop and count zero-confidence rows.
3. Group rows by bit-identical canonical float32 XYZ bytes. Within a group,
   sort positive confidence values by their uint32 bit pattern, promote each
   exactly to Python float, and set `weight=math.fsum(values)`. Require finite
   positive weight.
4. Convert each aggregate XYZ to continuous float64 RAE with the frozen axes.
   Sort atoms by `(range, azimuth, elevation, x, y, z, xyz_byte_key)`. Original
   row IDs are forbidden from canonical IDs, sorting, or hashes.
5. Let `weights` follow atom order, `total=math.fsum(weights)`, and
   `cdf[i]=math.fsum(weights[:i+1])/total`. Require finite positive `total`, a
   finite nondecreasing CDF, and bit-exact `cdf[-1]==1.0`; no endpoint overwrite
   is permitted.
6. For `k=0..9999`, set float64 `u=(2*k+1)/20000`, then
   `j=int(np.searchsorted(np.asarray(cdf,dtype='<f8'),u,side='right'))`.
   `j==atom_count` or any out-of-range result is implementation-invalid. Slot
   `k` inherits atom `j`.

This creates exactly 10,000 stable demand slots. Repeated atom IDs are allowed;
missing or extra slot IDs are forbidden. Random target-row permutations, zeros,
and signed zeros must preserve hashes. Duplicate split/merge equivalence means
only this: for the same canonical XYZ, the sorted positive float32 confidence
multisets have bit-identical float64 `math.fsum` results. Such mass-preserving
splits/merges must preserve atom/demand hashes; arbitrary float32 splits are not
claimed equivalent. The demand is target-mass discretization, not a range quota.

The fitter also emits a separately labelled immutable pointwise-control
sidecar: the nearest canonical target distance for every packed support point.
It is GT-conditioned and cannot enter the decision solver.

Demand/graph/control fitting uses only the positive-confidence canonical atoms.
Metric evaluation separately reloads the original immutable float32
`target_xyz_confidence` bytes and passes all rows and original confidence values
to the frozen evaluator, exactly as the predecessors did. Zero-confidence rows
therefore cannot receive demand, but their metric treatment is not silently
changed by STDA.

## 7. Frozen graph and integer cost

Rows are 10,000 demand slots and columns are every point in the committed
support. The support is canonically ordered by ascending original stable
candidate ID; `support_rank` is its zero-based row index in that order. Build
`cKDTree(support_xyz_float64, leafsize=16, compact_nodes=True,
balanced_tree=True, copy_data=True)`. For each slot, construct exactly `K=256`
distinct Euclidean XYZ nearest neighbors. RAE differences are report metadata.

Boundary handling is part of the graph definition. Call
`query(slot_xyz,k=256,p=2,eps=0,workers=1)`, take the provisional 256th distance,
then call `query_ball_point(slot_xyz,r=nextafter(d_256,+inf),p=2,eps=0,
workers=1,return_sorted=True)`. For each candidate, compute float64
`d2=(dx*dx+dy*dy)+dz*dz` in that exact operation order and
`distance_m=math.sqrt(d2)`. Sort by `(squared_distance,
stable_candidate_id)` with no approximate-tie tolerance and retain the first
256. Fewer than 256 returned candidates is implementation-invalid. Tests must
include more than 256 exactly equidistant candidates, ULP-separated distances,
and input/support-order permutations.

For each retained edge define:

```text
distance_um = int(np.rint(np.float64(distance_m * 1e6)))  # ties-to-even
edge_cost = distance_um * (support_cardinality + 1) + support_rank + 1
```

Costs are positive int64 values and every edge cost must be below `2^53`.
For every row, compute its maximum edge cost and require the Python-integer sum
of all 10,000 row maxima to be below `2^63`; this bounds every full-assignment
objective. Recompute the selected objective with Python integers after solving.
This is the frozen composite graph cost, not a floating tie perturbation and
not a proof of a mathematically unique optimum.

Within each CSR row, edges are serialized by ascending support column. The
canonical arrays are little-endian `indptr:<i8`, `indices:<i4`, and
`data:<i8`; shape is `(10000,support_cardinality)`, indices are unique and
strictly increasing per row, and `indptr[-1]==2,560,000`. Slot/atom IDs are
`<i8`, slot/atom XYZ are `<f8`, support IDs/cells are `<i8`, support XYZ/base
confidence are `<f4`, and color is `<u1`. The full CSR, edge-distance metadata,
support/demand hashes, and graph hash are serialized before solving.

`K=256` defines a graph-constrained assignment. No omitted-edge optimality,
reduced-cost, shielding, dense-OT equivalence, adaptive K, radius expansion,
or target-stratum edge pruning is claimed or allowed.

## 8. Cardinality certificate and unmatched mass

Run both SciPy maximum bipartite matching and a separately implemented,
deterministic Hopcroft-Karp verifier on the same graph. Require equal maximum
cardinality and validate all matched edges independently. Record:

```text
source_capacity = support_cardinality
requested_target_mass = 10000
transported_mass = maximum_matching_cardinality
unmatched_target_mass = 10000 - transported_mass
unused_source_mass = support_cardinality - transported_mass
```

If cardinality is below 10,000, emit the alternating reachable slot/support
sets from the Hopcroft-Karp terminal search. The independent verifier reloads
the complete CSR, forms `S` from the reported reachable slots, reconstructs
`N(S)` as the union of every stored neighbor of every slot in `S`, and requires
bit-identical sorted `N(S)` plus the strict Hall inequality `|N(S)|<|S|`.
It also requires `10000-cardinality >= |S|-|N(S)|` and records the deficit,
raw-set hashes, target atom/range/ray/multiplicity summaries, and both matching
hashes. A failed Hall verification is implementation-invalid. A valid deficit
is `stda_f0_graph_cardinality_no_go`, not a geometry or solver failure. Only
full cardinality permits a decision export.

## 9. Three matched arms

### 9.1 Decision: sparse full assignment

Run `scipy.sparse.csgraph.min_weight_full_bipartite_matching` on the frozen CSR.
Every slot must be used once, every support point at most once, and the export
must contain exactly 10,000 points. Run the solver twice in-process and once in
a clean subprocess; require identical assignment, selected-ID, export, and
objective hashes under the pinned environment. This certifies fixed-version
reproducibility only, not uniqueness.

### 9.2 Control A: packed pointwise

On the same committed support and canonical positive target atoms, construct a
separate target `cKDTree` with the Section 7 constructor parameters. For each
support point, call `query(k=1,p=2,eps=0,workers=1)`, gather all targets through
`query_ball_point(r=nextafter(d_1,+inf),p=2,eps=0,workers=1,
return_sorted=True)`, and recompute float64 squared distances in the exact
Section 7 operation order. Choose nearest target by
`(squared_distance,canonical_atom_id)` and serialize nearest atom ID `<i8`,
squared distance `<f8`, and `math.sqrt(squared_distance)` `<f8`. Sort support
points by `(squared_distance,stable_candidate_id)` and select the first 10,000.
This control shares support and target with the decision arm; it does not share
demand slots or the decision KNN graph.

The archived F0R global pointwise arm used the full 700k field. It is reported
only as historical context and is not a matched gate or baseline for relative
pass/fail tolerances.

### 9.3 Control B: target-atom round-robin greedy

This arm shares support, demand slots, graph, and integer costs with the
decision arm but has no global reassignment. Group slots by canonical target
atom. Define
`frame_key=f"seq{sequence:02d}/radar{radar_index:05d}".encode("ascii")` and
`atom_byte_key=canonical_xyz_<f4_bytes || aggregate_weight_<f8_bytes`.
Only atoms receiving at least one demand slot are active. Their fixed order is
ascending
`(SHA256(b"stda_f0_greedy_atom_order_v1\0" + frame_key + b"\0" +
atom_byte_key), canonical_atom_id)`. Let `ordered` be this fixed list and
`atom_count=len(ordered)`. Starting with `r=0`, repeat rounds `r=0,1,...` while
any slot remains unconsumed. In each round, visit
`ordered[(r+i) % atom_count]` for `i=0,1,...,atom_count-1`; skip an atom only if
all its slots are consumed. For every visited nonexhausted atom, consume its
smallest remaining stable slot ID and choose the unused graph neighbor with
smallest `(integer_edge_cost,support_rank)`. No backtracking or expansion is
allowed. A failed slot is marked consumed without a candidate, then the same
forward visit order continues.

This order prevents contiguous near-to-far slot bias while remaining fixed and
replayable. If a slot has no unused neighbor, record greedy capacity failure
and continue reporting all possible diagnostics; do not repair the arm.

## 10. Independent replay and API boundaries

Required modules and signatures are:

- `stda_f0_support.py`: target-free support construction and commitment;
- `stda_f0_fit.py`: sole target loader, canonical demand/graph/control sidecars;
- `stda_f0_round.py`: maximum matching, decision and controls; it cannot import
  fitter, target-loader, or metric modules;
- `stda_f0_structure.py`: GT-visible, representation-neutral range-by-return
  grouping, monotone DP, and class metrics;
- `stda_f0_verify.py`: independent byte-level replay, exact spacing, Hall
  certificate verification, and an independent structural-DP implementation
  without importing the structure solver;
- `preflight_stda_f0_capacity.py`: two-phase orchestration and publication.

The solver receives immutable support plus graph bytes, never raw target rows,
target IDs, nearest-target vectors, or metric callbacks. The exporter receives
support plus assignment only. Metrics execute only after immutable export.

Replay verifies count, finite/unique bytes, edge membership, slot/candidate
capacities, control ordering, objective, strict spacing, serialization, and all
content hashes. Static signature/import/open-file audits and adversarial tests
are mandatory. Existing Fresh-WCE, Q-Local, VRH, and evaluator source files are
read-only predecessors and cannot be modified by STDA.

## 11. Frozen scientific gates

All geometry uses the archived float32 CUDA evaluator and identical chunking.
No SciPy float64 reimplementation may decide a geometry metric.

For every frame, every arm is evaluated with this shared scientific gate vector:

- Chamfer `<=0.8 m`;
- 2 m outlier fraction `<=5%`;
- exact 10,000, finite/unique XYZ, strict 5 cm, and valid structural domain;
- for every positive-confidence target-bearing `0--30`, `30--60`, and
  `60--120 m` stratum:
  completeness `<=1.0 m` and recall@1m `>=0.80`;
- for each nonempty positive-confidence representation-neutral Cartesian
  product class in `{0--30,30--60,60--120 m} x {first,later}`:
  completeness `<=1.0 m` and recall@1m `>=0.80`.

The fixed vector keys are:

```text
exact_10000
finite_xyz
unique_xyz_bytes
strict_spacing_5cm
structural_domain_valid
chamfer_le_0p8
outlier_le_0p05
range_0_30_completeness_le_1m
range_0_30_recall_1m_ge_0p8
range_30_60_completeness_le_1m
range_30_60_recall_1m_ge_0p8
range_60_120_completeness_le_1m
range_60_120_recall_1m_ge_0p8
range_0_30_first_completeness_le_1m
range_0_30_first_recall_1m_ge_0p8
range_0_30_later_completeness_le_1m
range_0_30_later_recall_1m_ge_0p8
range_30_60_first_completeness_le_1m
range_30_60_first_recall_1m_ge_0p8
range_30_60_later_completeness_le_1m
range_30_60_later_recall_1m_ge_0p8
range_60_120_first_completeness_le_1m
range_60_120_first_recall_1m_ge_0p8
range_60_120_later_completeness_le_1m
range_60_120_later_recall_1m_ge_0p8
```

Every key stores `{applicable, passed}`. The seven global keys are always
applicable. A range or range-by-return key is applicable iff its target class
has positive total confidence; otherwise `applicable=false` and it is neutral
for all arms. Target-dependent applicability is byte-identical across arms. An
arm passes the vector iff every applicable key passes. Arm-specific graph-edge,
solver, ordering, serialization, and replay checks are implementation checks:
any failure makes the entire run implementation-invalid and cannot count as a
scientific control failure. A valid greedy exhaustion produces fewer than
10,000 points and therefore fails shared key `exact_10000`. "Corresponding
gate" means the identical applicable shared-vector key passes for the decision
and fails for that control.

The new `stda_f0_structure.py` evaluator is executable independently of VRH's
`ExportResult`:

1. canonicalize each unique final float32 XYZ by `(r,a,e,x,y,z,xyz_bytes)` and
   assign that rank as `event_id`; assignment/slot/support IDs are not inputs;
2. use the frozen VRH angular edges to assign a subray. Every output must have
   range in `[0,118.2685546875]` and valid angles; otherwise its arm fails the
   shared structural support key. Multiple events in one VRH cell remain
   distinct;
3. in each subray sort events by `(range,event_id)` and infer ordinal depth
   `1..n`; equal-range events remain ordered by event ID;
4. canonicalize all positive target atoms with range `[0,120)`, require valid
   angular support, sort by `(range,canonical_target_id)`, and form consecutive
   groups exactly as `vrh_f0_metrics.py`: a group starts at the first remaining
   target and absorbs following targets while `next_range-first_range<0.05 m`;
5. set group representative range to its confidence-weighted mean, class to
   its representative range stratum and `first` for group index zero in a
   subray or `later` otherwise;
6. run the same ordered DP state `(group_index,event_index,previously_matched)`.
   Unmatched distance is `120 m`; a first group may match only inferred depth 1;
   a later group may match only depth at least 2 after an earlier group in that
   subray was matched. Minimize Python-integer
   `rint(1e9*group_weight)*rint(1e6*distance)`, then maximize matched count,
   then lexicographically minimize the complete canonical-group assignment
   tuple. In `(ray_id,group_index)` order, each tuple entry is the matched
   `event_id` or the unmatched sentinel `2^63-1`; the sentinel is greater than
   every valid event ID;
7. independently re-solve every subray DP without importing the structure
   evaluator. Sum subray integer costs and matched counts, concatenate all
   assignments in canonical `(ray_id,group_index)` order, and compute the global
   key `(integer_cost,-matched_count,complete_assignment_tuple)`. Require both
   that key and the complete mapping hash to equal the evaluator output. Also
   replay every matched and unmatched entry for eligibility, order, nonreuse,
   distance, integer objective, and input/output hash;
8. for each nonempty range-by-return class, compute confidence-weighted mean
   assigned distance and confidence-weighted fraction with distance `<=1 m`.

Tests freeze multiple events in one cell, equal-range events, first/later
eligibility, skipped events/groups, every one of the six classes, output/target
angular and range boundaries, target row permutations, export row permutations,
and independent replay corruption. A target outside `[0,120)` or the angular
support is a shared structural failure, not silently omitted.

All gates are absolute and evaluated separately on all 76 frames. No mean,
median, historical relative tolerance, or control success rescues a decision
failure. All arms receive the same evaluator and reporting.

## 12. Mutually exclusive terminal statuses

After completing all scientifically valid frames, apply this precedence:

1. `stda_f0_implementation_invalid`: any source/input/API/hash/replay/numerical/
   provenance/atomicity violation.
2. `stda_f0_resource_invalid`: a valid implementation exceeds Section 13.
3. `stda_f0_packed_support_capacity_no_go`: any frame's largest parity support
   contains fewer than 10,000 points.
4. `stda_f0_graph_cardinality_no_go`: every parity support has at least 10,000
   points, but any decision graph has maximum
   cardinality below 10,000 while packed support has at least 10,000.
5. `stda_f0_assignment_recipe_no_go`: every graph has full cardinality, but the
   decision arm fails any scientific gate on any frame.
6. `stda_f0_packed_support_only`: decision passes every shared-vector key on all
   76 frames and packed pointwise does too. Assignment/demand utility is not shown.
7. `stda_f0_demand_allocation_only`: decision passes every shared-vector key on
   all 76 frames, packed pointwise fails at least one corresponding key, and
   greedy passes every shared key. Global
   assignment utility is not shown; only the simpler demand allocation survives.
8. `stda_f0_assignment_utility_passed`: decision passes every shared-vector key
   on all 76 frames; packed pointwise and greedy each fail at least one identical
   key passed by the decision; every implementation/resource check passes.

Scientific conclusions are status-specific. A packed-support capacity no-go
closes only the exact Section 5 parity construction; because no graph is built,
it supports no conclusion about demand, graph, cost, or assignment. A graph-
cardinality no-go closes only full-cardinality feasibility of that parity
support plus the frozen `K=256` graph; it says nothing about another support or
graph. An assignment-recipe no-go closes only the frozen demand, graph, cost,
and full-assignment combination. `packed_support_only` and
`demand_allocation_only` retain only the named simpler mechanism and do not
establish global-assignment utility. None of these statuses proves that other
set allocation, ray demand, dense assignment, or learned selection is
impossible.

Only `stda_f0_assignment_utility_passed` authorizes a separately frozen,
target-free Cube-conditioned learnability gate. No F0 status directly supports
a paper novelty or unlocks Doppler/cycle/temporal work.

## 13. Resource and environment contract

All pre-formal smoke tests use fixed-seed synthetic arrays only. The formal run
uses user `wangning`, environment `hym_radar`, exactly one visible allowed H200
physical GPU (GPU0 or GPU2), and never GPU1/RTX5000. Bind and report:

- Python `3.10.20`, NumPy `2.2.6`, SciPy `1.15.3`, PyTorch
  `2.12.1+cu130`, CUDA/driver, GPU UUID/PCI ID/name;
- hostname, CPU model, logical core count, user, git source/protocol hashes;
- `PYTHONHASHSEED=0`, locale `C`, timezone `UTC`;
- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `NUMEXPR_NUM_THREADS=1` and all solver-relevant environment variables.

The orchestrator itself has no CUDA context and permits exactly one CUDA-visible
child process at a time. Candidate reconstruction and CUDA metric evaluation
run in separately launched CUDA-visible children. Every matching, replay,
structural, serialization, publication, monitoring, and other helper subprocess
sets `CUDA_VISIBLE_DEVICES=""`. Overlapping CUDA-visible children make the run
implementation-invalid.

Measure the orchestrator and every descendant process, including peak RSS,
per-process `VmSwap`, CUDA allocated/reserved, and NVML per-process memory. At
50 ms intervals, enumerate every live process-tree PID, join those PIDs to the
NVML process table for the single allowed physical GPU, and sum their resident
GPU memory for that sample. Record the peak of those process-tree sums, not only
the largest individual PID. Use
`time.perf_counter_ns()` for every named interval. Call `torch.cuda.synchronize()`
immediately before and after candidate reconstruction and CUDA metric intervals,
and before each CPU allocation-core boundary, so asynchronous GPU work cannot
spill into another interval. Frozen ceilings are:

- graph entries exactly 2,560,000 for every graph-valid frame;
- peak process-tree host RSS `<=60 GiB` and peak process-tree `VmSwap=0`;
- `max(torch peak allocated, torch peak reserved,
  peak summed process-tree NVML memory)
  <=60 GiB`;
- allocation core `<=30.0 s/frame`, defined as the sum of that frame's support-
  phase packing/serialization interval and oracle-phase demand, graph,
  cardinality, three decision-solver executions, both controls, all independent
  replays, and serialization intervals. Candidate-network reconstruction and
  CUDA metric evaluation are excluded;
- complete per-frame total `<=120 s`, where
  `frame_total_ns=support_frame_ns+oracle_frame_ns`. `support_frame_ns` starts
  immediately before opening that frame's Cube and ends after its support bytes
  and manifest entry are fsynced and rehashed. `oracle_frame_ns` starts
  immediately before opening that frame's target bytes and ends after all three
  arms, CUDA metrics, structural metrics, replays, and frame evidence are
  fsynced and rehashed;
- complete 76-frame formal work transaction `<=7,200 s`.
  `transaction_work_ns` starts immediately before spawning the support process
  and ends after both final bundle paths have been renamed, parent-fsynced, and
  independently reverified, immediately before terminal-status determination
  and `TRANSACTION.json` or `FAILURE.json` serialization. It includes all
  support/oracle work, monitoring, global manifests, shared-bundle verification,
  complete local-mirror copying and verification, both final bundle renames,
  and cross-host reverification. `record_publication_ns` starts immediately at
  that boundary and covers terminal-status determination, record serialization,
  atomic publication, and parent fsync. It is reported in the post-publication
  log, is not serialized into the record being timed, and is not an input to the
  7,200-second status decision.

The two process phases record frame-keyed nanosecond intervals; the orchestrator
joins and sums them without overlap. Candidate reconstruction, each of the
three solver calls, allocation components, metric evaluation, and publication
times are reported separately. A resource excess is invalid, never a scientific
no-go; optimize implementation and rerun the identical frozen protocol.

## 14. Content-addressed dual-copy evidence and one commit point

No final bundle path or transaction record may exist before launch. Create one
staging directory on the shared H200/L40S filesystem and one under
`/home/wangning/stda_evidence_mirror/` on the H200-local filesystem. Their
absolute paths and filesystem device IDs must differ. Every written file is
closed and `fsync`ed; every containing directory and staging parent is fsynced.
Every JSON object named in this section uses the exact bytes below; timestamps
are integer Unix nanoseconds.

```python
json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
           allow_nan=False).encode("ascii") + b"\n"
```

First build the shared staging bundle. `payload_manifest.json` is canonical JSON
listing relative path, size, and SHA-256 for every payload except itself and
`BUNDLE_COMPLETE.json`. After all payloads and the manifest are synced, write
canonical `BUNDLE_COMPLETE.json`; it contains schema, protocol/source hashes,
payload count, and the SHA-256 of the exact manifest bytes, but no scientific or
transaction status. Define the immutable bundle root as:

```text
SHA256(b"stda_f0_bundle_v2\0" +
       payload_manifest_bytes + b"\0" + BUNDLE_COMPLETE_json_bytes)
```

Neither metadata file claims to cover itself. Re-read every shared-staging byte
and verify the root. Copy the complete bundle, including every graph and
sidecar, into the H200-local staging directory. Check every copied file against
the manifest, fsync it, re-read the complete local staging bundle, and require
the identical root and byte counts. Independently reverify the shared staging
copy from L40S and the local staging copy from H200.

Only after both staging copies pass, atomically rename each to the same basename
`stda_f0_<source8>_<protocol8>_<bundle_root16>` in its respective final parent
and fsync both parents. Reverify both final paths, stop `transaction_work_ns`,
and determine Section 12 status. Only a status in 3--8 constructs canonical
`TRANSACTION.json`, binding schema, full protocol/source hashes, that exact
terminal status, measured resource values including `transaction_work_ns`, both
final absolute paths, mount sources, filesystem types/device IDs, payload and
complete-bundle byte counts, the identical bundle root, and all verification
timestamps. Define:

```text
transaction_root = SHA256(b"stda_f0_transaction_v1\0" + TRANSACTION_json_bytes)
```

Write and fsync the record to a shared transaction staging file, re-read its
hash, then atomically rename it to
`stda_f0_transaction_<source8>_<protocol8>_<transaction_root16>.json` and fsync
the transaction parent. This single rename is the global commit point. A
Section 12 status 3--8 exists only when that record and both bound final bundle
paths all verify; a bundle directory without the transaction record is an
orphan, not evidence and not a terminal scientific result.

A crash between the two bundle renames, or between bundle publication and the
transaction commit, may leave staging directories or one/two orphan final
bundles. Recovery must label and quarantine them without overwrite or reuse
before any rerun. Every Section 12 status 1 or 2 condition at every execution,
verification, resource, or publication stage emits a separate canonical,
content-addressed `FAILURE.json`, subject to status precedence: any
implementation-invalid condition takes precedence over any resource excess.
The mandatory schema contains the full protocol SHA-256, full source commit,
terminal status, failed stage and code, integer event time, complete measured-
resource object or explicit null fields, available artifact path/size/SHA-256
records, any available bundle/transaction roots and paths, and environment/
provenance hashes. It cannot publish or preserve a scientific status. Its exact
canonical bytes define
`failure_root=SHA256(b"stda_f0_failure_v1\0"+FAILURE_json_bytes)`. For each
available server evidence parent, shared and H200-local, write and fsync the
bytes to staging, re-read the full hash, atomically rename to
`stda_f0_failure_<source8>_<protocol8>_<failure_root16>.json`, and fsync the
parent. Require at least one verified server copy. If the failed stage makes one
filesystem unavailable, the schema names that unavailable destination.
A status 1 or 2 becomes evidentially committed only after the exact
`FAILURE.json` bytes and full `failure_root` are copied byte-for-byte into the
repository, independently rehashed, committed, and pushed to GitHub. The Git
artifact path and hash are then recorded in the decision/report; server paths
alone never establish a failure status.

Both committed complete copies are retained through paper finalization. For
statuses 3--8, Git/GitHub archives the compact report, decision, log, payload
manifest, `BUNDLE_COMPLETE.json`, protocol/source hashes, exact
`TRANSACTION.json`, and transaction root. Statuses 1--2 use the mandatory
Git/GitHub failure route above. Server paths alone are never the only evidence.

## 15. External SHA-bound freeze activation

Once submitted for audit, these bytes remain immutable. An independent
read-only audit must return `FREEZE` while naming the exact protocol SHA-256 and
approving:

- STDA naming and non-novelty boundary;
- all-76 cohort, one formal target-conditioned launch, and test lock;
- two-process target isolation and candidate-only reconstruction;
- complete exact float32 5 cm enumeration and boundary tests;
- permutation-invariant target canonicalization and CDF semantics;
- K-boundary ties, integer cost, dual cardinality certificate, and replay;
- total structural-DP mapping tie-break and complete mapping replay;
- byte- and arithmetic-exact matched controls, shared gate vector, unbiased
  greedy order, and six
  range-by-return absolute classes;
- mutually exclusive statuses and narrow no-go conclusions;
- serialized CUDA-child execution and summed process-tree memory accounting;
- dual-copy bundle construction plus the one-record global commit point.

The `FREEZE` audit artifact and a canonical external freeze record containing
the exact protocol SHA-256, audit-artifact path/SHA-256, verdict, and activation
time are serialized by the Section 14 JSON rule and committed together without
editing this file. That Git commit is the
freeze commit. Its resulting commit hash is recorded by the first implementation
source and every run report, not written back into this protocol or freeze
record. No scientific implementation commit may precede, or be the same commit
as, the freeze commit. A `NO-FREEZE` audit does not activate these bytes.
