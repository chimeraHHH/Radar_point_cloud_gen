# Cube-to-dense current board packet

> Refreshed 2026-08-07 after source `34579a2` Q-Local-F0R terminal. Test access
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
| R-B2 | 80k `max_d` recall passed on 76/76, confidence coverage failed on 3 frames | Current score-plus-fixed-neighborhood activation closed |
| G1T | Ego/Doppler union changed geometry by less than `0.05%` and remained very poor | Current no-train history proposal route closed |

## Important contradiction

The project has moved past the question of whether a sufficiently wide spatial
support can exist. On two independently trained R-A1 pools, an unattainable
geometric score selects an excellent exact-10k subset, while binary occupancy
confidence fails badly.

F0R now sharpens the unresolved question. The candidate field can support low
aggregate error, but an independent per-point utility score followed by one
global exact-count export spends almost all capacity on dense near surfaces.
Ten of 12 frames lose target-bearing middle/far strata even under an
unattainable GT-nearest score. The next question is whether an ordered
variable-return process can allocate geometry without:

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

**Variable multi-return renewal hazard** is now the only authorized geometry
capacity route. It replaces independent candidate ranking with an ordered
radial process on each azimuth/elevation ray: a return terminates the current
survival interval, then renews the process so the same ray can emit a
data-dependent number of later returns. This is a representation change, not a
new threshold for the closed R-B1 fixed-`K=4/6` peak extractor.

The first gate remains zero-training and train-only. Before implementation, a
separate protocol must freeze the ray lattice, legal target-aided capacity
construction, global exact-10k/true-5-cm export, per-frame geometry gates, and
target-stratum anti-collapse checks. A deployable field may use only current
Cube evidence; GT is restricted to the non-deployable capacity oracle and
metrics.

Sparse ray-range partial transport remains third priority behind its own
hard-rounding oracle. Q-Local training, fixed-neighborhood support, fixed
per-ray `K`, and hard per-frame range quotas are closed.

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

The 45-sequence, 2,160-frame G4 download is complete. A fresh exact-member,
size, and CRC audit is running from clean H200 source `083f4c4`; temporal
training remains locked regardless of the data-audit outcome.
