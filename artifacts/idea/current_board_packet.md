# Cube-to-dense current board packet

> Refreshed 2026-08-07 after source `380f3ea` VRH-F0 terminal. Test access
> is false.

## Incumbent

There is no passing single-frame geometry incumbent. The strongest current
mechanism evidence is the fresh R-A1 replay candidate pool, not its deployed
output:

- current-confidence exact-10k: Chamfer `4.0261 m`, outlier `31.7854%`;
- validation-GT-nearest exact-10k on identical candidates:
  Chamfer `0.6375 m`, outlier `2.2771%`, far completeness `0.7046 m`;
- candidate coordinates and 5 cm spacing are fixed; the old per-frame range
  quotas are now proven contradictory and retired.

This isolates geometric-utility prediction and structured selection as an open
mechanism. It does not authorize reuse of the deleted R-A1 identity.

## Final route decisions

| Route | Decisive result | Boundary |
|---|---|---|
| RAE-Max | Chamfer `2.9306 m`, outlier `25.697%` | Useful compact baseline; geometry gate failed |
| Full-RAED early fusion | `+5.86%` Chamfer vs RAE-Max | Early spectrum fusion rejected |
| G1D direct query field | Endpoint degraded; condition effect weak; duplicates high | Deterministic arbitrary-query family closed |
| G1G hierarchy | Completeness/far support improved, but outlier `84.685%`, duplicates `70.811%`, shuffle `-0.121%` | Current condition-exclusive patch allocator closed |
| G1E/RaLD latent occupancy | Source-faithful proposal-support gate failed | Current occupancy VAE/EDM route closed |
| G1F fixed 32k selector oracle | GT-aided subset still failed Chamfer/completeness/duplicates | Selector-only repair on that pool closed |
| R-A1/R-A2 | Wide candidates exist, but binary occupancy/confidence does not select them | Binary score recipe closed |
| Q1-R | Replay failed four original-endpoint equivalence tolerances | Quality hypothesis untested; old-parent route closed |
| Q-Local-F0 | 10/12 fixed-export oracle frames passed; `47:514` and `58:404` failed | Scorer untrained; terminal under the original exporter only |
| Fixed range quotas | 76-frame audit found strict CD lower-bound failures on `47:94` and `58:404` | Hard `8000/1700/300` per-frame allocation retired from all successors |
| Q-Local-F0R | CD/outlier/structure passed 12/12 after global export, but target-stratum retention passed only 2/12 | Pointwise global ranking on the frozen 700k field closed; scorer untrained |
| VRH-F0 | Renewal activity passed, but sequential/flat mean CD was `0.5494/0.1095 m`; strata passed 3/12 vs 4/12 and first/later passed 0/12 in both | Frozen renewal/frontier recipe closed; learned renewal untrained |
| R-B2 | 80k `max_d` recall passed on 76/76, confidence coverage failed on 3 frames | Current score-plus-fixed-neighborhood activation closed |
| G1T | Ego/Doppler union changed geometry by less than `0.05%` and remained very poor | Current no-train history proposal route closed |

## Important contradiction

The project has moved past the question of whether a sufficiently wide spatial
support can exist. On two independently trained R-A1 pools, an unattainable
geometric score selects an excellent exact-10k subset, while binary occupancy
confidence fails badly.

F0R sharpened the unresolved question. The candidate field can support low
aggregate error, but an independent per-point utility score followed by one
global exact-count export spends almost all capacity on dense near surfaces.
Ten of 12 frames lose target-bearing middle/far strata even under an
unattainable GT-nearest score. VRH-F0 then tested an ordered variable-return
process. It certified active later-return selection but did not improve the
shared fitted stream: sequential geometry was worse than flat exposure, three
frames failed exact count, and every frame failed the complete first/later
gate. The next question is whether sparse set-level transport can allocate
geometry without:

1. inheriting a deleted checkpoint identity;
2. reading target geometry at inference;
3. collapsing target-bearing range coverage or point diversity;
4. reverting to fixed per-ray return counts or hard range quotas;
5. bypassing the global Cube condition through local energy alone.

## Closed assumptions

- early Full-RAED channel fusion is not sufficient;
- larger deterministic query fields are not sufficient;
- binary occupancy probability is not geometric quality;
- reducing the 10k output count is not an allowed or sufficient repair;
- fixed-neighborhood voxel activation is not robust across all train frames;
- the current ego/Doppler union proposal does not supply useful geometry;
- diffusion, Doppler, cycle, or temporal modules cannot compensate for a failed
  single-frame geometry parent.

## Selected live mechanism

**Sparse ray-range partial transport** is now the only eligible geometry
capacity route, but its protocol is not yet frozen. It must couple point budget
and range coverage on a sparse ray-range graph without constructing a dense
`700k x target` matrix. The first gate is a zero-training hard-rounding oracle,
not a learned transport model.

The protocol must freeze source support, sparse edge construction, mass and
cardinality semantics, true 5 cm hard rounding, per-frame geometry and strata
gates, a target-free deployment API, and explicit runtime/memory limits before
implementation. It must compare its rounded result with the same unattainable
pointwise oracle and cannot use GT-derived range quotas or masks at deployment.
Only a complete train-only capacity and resource pass may authorize a learned
Cube-conditioned transport scorer.

Q-Local training, fixed-neighborhood support, fixed per-ray `K`, hard per-frame
range quotas, pointwise global export, and the frozen VRH-F0 recipe are closed.

## Stale routes not to reopen

- threshold relaxation or cardinality reduction;
- hard per-frame `8000/1700/300` range quotas;
- G1B spectral-rank variants;
- another G1C/G1D deterministic query refiner;
- G1G with only a different patch count or radius;
- R-A2 binary occupancy with a different class sampler;
- Q1-R under the original-parent claim;
- Q-Local pointwise scoring plus one unconstrained global export on the frozen
  Fresh-WCE field;
- R-B2 with only a larger fixed bank or neighborhood;
- G1T with only a longer history;
- any downstream Doppler/temporal run before a new parent passes.

## Independent infrastructure progress

The 45-sequence, 2,160-frame G4 download and full-summary CRC audit are complete.
Source label `083f4c4` verified all 6,660 files and 45/45 member sets with no
missing, unexpected, duplicate, invalid, pending, or failed entries. Evidence is
archived under `artifacts/g4/g4_temporal_crc_083f4c4/`. Temporal training remains
locked because no single-frame geometry parent has passed.
