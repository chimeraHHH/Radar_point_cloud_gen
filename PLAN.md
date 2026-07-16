# Cube-to-Dense Execution Plan

The authoritative scientific plan is `docs/work_plan_cube_to_dense_topconf.md`.
This file tracks execution state and evidence locations.

| Phase | Gate | State | Required evidence |
|---|---|---|---|
| P0 | G0 | In progress | Full DRAE schema, synchronized LiDAR/odometry/calibration, CFAR round trip, observability audit |
| P1 | G1 | Pending | Reproducible RAE-Max and Full-RAED Cube-to-XYZ baselines |
| P2 | G2 | Pending | Doppler distribution head, static/dynamic physics split, counterfactual response |
| P3 | G3 | Pending | Differentiable point-to-Cube renderer and cycle ablation |
| P4 | G4 | Pending | Historical Doppler-warp prior with current-Cube refinement |
| P5-P6 | Final | Pending | Scale, generalization, downstream, efficiency, paper package and completion audit |

## Compute boundary

All data processing, training and evaluation run on `WHUServer-L40S`. The local
worktree is used only for source editing, version control and transport.
