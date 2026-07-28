# Direct Multi-Horizon Doppler World: Stage-0 Protocol

## 1. Scope

This route is an independent mechanism preflight for long-horizon degradation.
It borrows the direct multi-future prediction abstraction from D²-World and
related world-model work, but it is not a source-faithful reproduction.

The route predicts all three future states in one forward pass:

```text
causal history Full-RAED Cubes + current Full-RAED Cube
                             + current 10k radar state
                             + ego transforms
  -> {0.5 s, 1.5 s, 2.5 s} x 10k (XYZ + 64-bin Doppler + confidence)
```

There is no autoregressive feedback between horizons. The model API rejects a
future Cube. Development manifests containing test records are also rejected.

This document authorizes only synthetic structure and H200 forward/backward
preflight. It does not authorize real training.

## 2. Evidence audit

- Corrected G1T produced approximately `15.9 m` Chamfer and `87.6%` outliers
  for all current/ego/Doppler arms. The Doppler-history effect was below
  `0.05%` and was judged non-material.
- G3R was skipped because its geometry parent failed. It provides no positive
  Cube-cycle result for this route.
- G4/G4R did not produce an eligible temporal model result. The old route is
  closed and the successor remained locked behind G3R.
- The earlier FlowRadar rollout is design motivation only. Its TruckScenes
  result cannot validate K-Radar Full-RAED generation.
- The existing forced T-WC preflight proves only a one-target-time synthetic
  mechanism. It does not test direct multi-horizon prediction.

The D-MHW route therefore starts from a clean synthetic contract and carries no
performance claim from G1T, G3, G4, or FlowRadar.

## 3. Frozen architecture

### 3.1 Causal condition

- Three strictly historical Full-RAED Cubes and one current Full-RAED Cube.
- Every Cube has project shape `64 x 256 x 107 x 37`.
- A memory-bounded encoder pools all four physical axes; all 64 Doppler
  channels must receive non-zero gradient.
- A chronological GRU encodes historical Cube latents. A separate current-Cube
  latent remains present in every horizon.
- The direct heads receive fixed horizon embeddings for `0.5/1.5/2.5 s`.

### 3.2 Persistent branch

- Exactly 7,000 current-state points are selected by valid confidence.
- Every selected point carries a unique non-negative source ID.
- The same IDs and point order are preserved across all three horizons.
- Geometry must pass through analytic radial Doppler integration and the
  supplied current-to-target ego transform.
- The learned head outputs only:
  - tangential velocity bounded by `8 m/s`;
  - radial calibration bounded by `0.75 m/s`.
- There is no learned absolute-coordinate head for persistent points.

### 3.3 Birth branch

- Exactly 3,000 learned birth queries are decoded per horizon.
- Birth queries receive only causal history/current Full-RAED latents and the
  direct horizon embedding.
- Birth points have source ID `-1`; they cannot copy or pad persistent points.
- The branch predicts XYZ, a 64-bin circular Doppler distribution, and
  confidence.

### 3.4 Output and loss

- Output is exactly `B x 3 x 10,000`.
- Geometry uses a symmetric set loss at each horizon.
- Persistent points use inverse displacement-Doppler consistency and a motion
  cycle check.
- Tangential and radial residuals receive an explicit magnitude prior.
- Point-aligned Doppler and confidence labels are allowed only in synthetic
  contract tests. The loss rejects those labels as real-data supervision.

## 4. Five mandatory interventions

Every intervention must change all three horizon outputs and retain a finite,
non-zero autograd path:

1. `wrong_history`: cross-batch history Cube and observed-state replacement.
2. `zero_history`: historical Full-RAED Cubes set to zero.
3. `doppler_sign_flip`: observed circular Doppler distribution reflected about
   zero velocity.
4. `time_disorder`: historical Cubes reversed while timestamps remain fixed.
5. `cube_mismatch`: current Full-RAED Cube replaced across scenes.

The preflight archives XYZ, Doppler, and confidence deltas per horizon plus the
graph-gradient norm for every intervention.

## 5. Real supervision audit

The upgraded G4 manifest contains 45 causal windows of 48 frames. Under a
pre-registered `0.05 s` maximum horizon error and three historical frames, the
server audit found:

| Partition | Legal direct three-horizon anchors |
|---|---:|
| train | 740 |
| validation | 160 |

All required historical/current raw Cubes and future LiDAR geometry files are
present. No test record is present or read. Future Cubes are not part of an
example input.

The prepared train dense-target cache is currently empty, while the existing
validation cache contains 384 frames. Raw data can form legal geometry labels,
but a source-bound train cache must be built and audited before formal
training.

K-Radar future geometry does not provide birth-point identity or a point-aligned
future Doppler distribution without reading the future Cube. Persistent
Doppler can be constrained through displacement, but real birth-Doppler
supervision remains unresolved. It must be resolved without weakening the
no-future-Cube contract before any attribute claim.

## 6. Stage-0 gates

### 6.1 Mechanism preflight gate

All checks are mandatory:

- H200 physical GPU 0 or 2; never GPU 1.
- Exact Full-RAED shape and exact `3 x 10,000` output.
- Persistent/birth split exactly `7,000/3,000`.
- Persistent IDs identical across horizons; birth IDs all `-1`.
- Non-zero analytic Doppler displacement.
- Tangential orthogonality error below `1e-5`.
- Learned motion remains within the frozen bounds.
- Finite loss and finite non-zero gradients.
- All 64 current and historical Cube channels receive gradient.
- All 64 rows of persistent and birth Doppler heads receive gradient.
- All five interventions are measurable at every horizon and differentiable.
- Peak allocated memory below `55 GB`; peak reserved memory below `65 GB`.

### 6.2 Data authorization gate

Formal training remains locked until all are true:

- an eligible fixed-count geometry parent passes the project gate;
- all 740 train and 160 validation anchors have source-bound dense geometry
  targets;
- birth-Doppler supervision is defined without future-Cube access;
- train/validation scene separation and zero test access are re-audited.

### 6.3 Future real Stage-0 promotion gate

These thresholds are frozen before any real training:

- `0.5 s` Chamfer no worse than the best eligible single-step parent by more
  than `3%`;
- `1.5 s` and `2.5 s` Chamfer each improve over the corrected Doppler-warp
  control by at least `5%`;
- `2.5 s` radial error improves over ego/Doppler-only controls by at least
  `10%`;
- wrong history, time disorder, and Cube mismatch each worsen the relevant
  loss by at least `3%`;
- Doppler sign flip worsens dynamic radial error by at least `15%`;
- duplicate and outlier fractions worsen by no more than `2 percentage
  points`;
- persistent and birth outputs are both used at every horizon.

Failure of any hard gate closes this direct route. Training duration must not be
extended to rescue a structural failure.

## 7. Evidence boundary

A passing preflight proves only that the direct three-horizon graph is
well-formed, causal by interface, differentiable, exact-count, and executable
at maximum cardinality on H200. It does not prove forecasting accuracy,
temporal consistency, real Doppler quality, or improvement over any baseline.
