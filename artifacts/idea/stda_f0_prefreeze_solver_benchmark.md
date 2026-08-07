# STDA-F0 pre-freeze solver benchmark

> Date: 2026-08-07  
> Role: synthetic engineering feasibility only; not a scientific result  
> Host: `WHUServer-H200`, environment `/home/wangning/miniforge3/envs/hym_radar`

No Cube, candidate cache, target, validation, test, or future frame was read in
the benchmark. Fixed-seed synthetic arrays only tested whether the proposed
sparse graph size is categorically incompatible with the H200 host. The result
does not validate real support, demand, graph, geometry, or protocol gates.

## Environment

- SciPy `1.15.3`
- graph shape: `10,000 x 80,000`
- neighbors per slot: `256`
- CSR nonzeros: `2,560,000`

## Timings

| Component | Synthetic size | Wall time | Peak process RSS |
|---|---:|---:|---:|
| parity-cell support construction | 700,000 XYZ/confidence rows | 0.207 s | 101 MiB |
| cKDTree build | 80,000 candidates | 0.018 s | included below |
| cKDTree query | 10,000 queries x 256 neighbors | 0.025 s | 105 MiB |
| CSR graph construction | 2,560,000 weighted edges | 0.092 s | included below |
| maximum bipartite matching | same CSR | 0.002 s | included below |
| minimum-weight full bipartite matching | same CSR | 0.383 s | 121 MiB |

The weighted solver returned 10,000 rows matched to 10,000 unique columns. The
synthetic parity packing produced about 87k points in its largest color, but
that distribution-specific count is not evidence about Fresh-WCE.

## Additional design history

A later read-only target-free audit reconstructed train frame `1:232` on H200
GPU2 and reproduced its archived 700k candidate hash. It observed 693,924
occupied 5 cm cells, 87,415 points in the largest parity color, and a minimum
packed-support spacing of 0.05422 m. This used no target but is still a real
formal-cohort frame, so it is design history only and cannot select constants,
satisfy a gate, or serve as the formal smoke.

After protocol freeze, every smoke is synthetic and the target-conditioned
76-frame formal transaction is launched once.

## Boundary

The timings justify protocol audit only. They do not freeze a scientific
conclusion or relax any real-data resource gate. The formal implementation must
report candidate reconstruction, allocation core, all arms, replay, metrics,
host RSS/swap, and CUDA memory separately for every train frame.
