# Cube-to-dense objective contract

## Real objective

Given the current K-Radar Full-RAED Cube and, only in the temporal stage, a
bounded history of radar observations plus ego motion, generate a fixed-count
dense radar point state:

```text
10,000 x (x, y, z, circular Doppler distribution, confidence)
```

The output must improve spatial utility while preserving measured motion
evidence. The final method must support current-frame geometry, generated
Doppler, cross-frame consistency, and untouched-test downstream evaluation.

## Current Stage-0 target

The immediate target is not the full paper claim. It is to find a single-frame
geometry mechanism that passes the frozen development contract and can serve as
the support for Doppler and temporal generation:

- median Chamfer `<= 2.50 m`;
- mean outlier fraction at 2 m `<= 25%`;
- median completeness `<= 0.65 m`;
- mean far-range completeness `<= 8.0 m`;
- mean duplicate fraction within 5 cm `<= 10%`;
- fixed output count `10,000`;
- a learned condition or branch must degrade by at least `1%` under its
  preregistered cross-scene shuffle control.

These thresholds are decision gates, not tunable objectives.

## Trusted proxies

- scene-held-out validation under the source-bound 76/24 train/validation
  split;
- per-frame geometry plus scene-first aggregation;
- explicit zero-offset, local-evidence, condition-shuffle, and representation
  controls;
- positive occupancy recall and empty-space false-positive rate;
- range slices and duplicate-rate diagnostics;
- complete transitive source, data, checkpoint, and artifact hashes.

## False-progress signals

The following do not justify promotion:

- lower training loss without validation geometry improvement;
- improved precision/outlier obtained by collapsing occupied recall;
- lower Chamfer obtained by duplicating points or losing far-range coverage;
- nonzero condition gradients when cross-scene condition shuffling has no
  measurable effect;
- attaching measured Cube Doppler after geometry and describing it as generated
  Doppler;
- best-of-k diffusion samples;
- changing query counts, validation frames, thresholds, or test access after
  observing a result;
- reopening G1/G1B/G1C/G1D/G1E under a new name without a mechanism-level
  change and a new frozen protocol.

## Hard constraints

- all scientific computation runs as `wangning` on physical H200 GPU 0 or 2;
- physical GPU 1 is never used;
- existing unrelated GPU work is not stopped or modified;
- Conda environments use the `hym_*` namespace;
- the current G1D source `4c6150cd` runs unchanged to its frozen endpoint;
- test remains locked until a complete single-frame, Doppler, and temporal
  family is frozen;
- no CFAR or LiDAR-derived inference helper may silently carry the main method.

## Search and experiment budget

The next idea pass must cover direct radar literature, RaLD and latent
generation, cross-domain sparse-to-dense mechanisms, and Doppler-aware temporal
models. It should promote at most three mutually distinct candidates.

Each candidate needs a source-bound Stage-0 that:

- runs in approximately one to two H200 hours;
- changes one primary mechanism family;
- retains the current split and metrics;
- has an explicit anti-win condition and abandonment rule;
- can fail without releasing Stage B or the test split.

