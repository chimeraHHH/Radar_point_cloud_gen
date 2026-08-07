# STDA-F0 pre-freeze audit, round 1

> Date: 2026-08-08  
> Audited draft SHA-256:
> `a53ccf63c7ff0bd147a34b90a5b274d6df81b2a1358dfc834ef81ae537261c79`  
> Original file: untracked `docs/srpt_f0_sparse_ray_transport_protocol.md`  
> Verdict: **NO-FREEZE (4/4 independent read-only audits)**

## Independent roles

1. mathematical/method audit;
2. code/system/isolation audit;
3. primary-literature and novelty audit;
4. hostile protocol/reproducibility audit.

No auditor modified repository files. The original SRPT draft was never frozen,
implemented, or used for a scientific run.

## Shared blockers

- The proposed graph was Euclidean XYZ KNN with Euclidean cost. RAE appeared
  only as metadata, so `ray-range transport` was an inaccurate name.
- The solver directly produced an integer rectangular assignment; there was no
  soft plan or `hard rounding` stage.
- Sparse/partial assignment and support subset selection have direct prior art.
  The primitive cannot be claimed as the project's novelty.
- `>=0.05-1e-6 m` admitted sub-5-cm pairs and violated the strict output
  contract.
- Floating `support_rank*1e-15` did not ensure a stable lexicographic cost or a
  unique optimum. Solver-version reproducibility and independent replay were
  missing.
- Target canonicalization used original row IDs and was not invariant to target
  row permutations or duplicate splitting.
- `cKDTree.query(k=256)` did not define the 256th-neighbor tie boundary.
- Graph cardinality failure and full-cardinality geometry failure were merged
  into one overly broad no-go.
- The existing data path loaded target tensors before candidate support
  commitment, violating target isolation.
- Pointwise and greedy controls did not have fully matched inputs or an
  order-neutral greedy schedule.
- The 12-frame cohort reused the words fit/unseen after those frames had already
  informed design; a new formal cohort required all 76 training frames.
- Resource ceilings, subprocess memory, environment binding, and atomic
  content-addressed publication were incomplete or contradictory.

## Literature boundary

The audit added direct precedents that make algorithm-level OT novelty unsafe:

- [Partial Optimal Transport for Support Subset Selection](https://openreview.net/forum?id=75CcopPxIr);
- [HOT-POT](https://arxiv.org/abs/2601.12423), which already combines ray
  distance, partial transport, sparse point sets, and integer partial matching;
- [Chapel et al., NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/1e6e25d952a0d639b676ee20d0519ee2-Abstract.html);
- [Schmitzer sparse shielding](https://arxiv.org/abs/1510.05466);
- [Smooth and Sparse Optimal Transport](https://proceedings.mlr.press/v84/blondel18a.html);
- [Partial Transport for Point-Cloud Registration](https://arxiv.org/abs/2309.15787);
- SuperGlue, OT-M, SampleNet, Support Points, SparseMAP, LP-SparseMAP, and
  differentiable perturbed optimizers as adjacent assignment/selection work;
- NFL, PLiNK, RangeLDM, LiDAR4D, NeuRadar, and RadarSplat as adjacent
  ray/range/reconstruction work.

The safe paper novelty remains the complete Full-RAED state target: exact-count
XYZ plus generated circular Doppler and confidence under Cube reprojection and
temporal physical closure. STDA can only be an internal capacity/allocation
component.

## Revision decision

The route was renamed **STDA-F0: parity-packed sparse target-demand bipartite
assignment**. The revised protocol:

- uses all 76 train frames and forbids validation/test/future access;
- separates target-free support construction and target-conditioned oracle
  fitting into processes that do not overlap;
- adds an exact float32-dyadic 5 cm comparator;
- canonicalizes duplicate target mass without row IDs;
- freezes K-boundary ties and positive int64 composite costs;
- uses SciPy plus independent Hopcroft-Karp cardinality certificates;
- compares assignment with packed-pointwise and target-atom round-robin greedy;
- adds absolute range and representation-neutral first/later gates;
- separates graph-cardinality and assignment-recipe no-go conclusions;
- binds subprocess resources and content-addressed atomic evidence.

The revised protocol remains `NOT FROZEN` until round-2 independent re-audit.
