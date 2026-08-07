# Q-Local-F0: local Full-RAED distributional geometric risk

Status: selected for protocol freeze; no result observed

## Hypothesis

The fresh WCE field already contains a strong exact-10k subset, but binary
occupancy confidence does not estimate candidate-to-surface risk. A scorer that
reads the local full Doppler spectrum and predicts a distance distribution can
learn that risk without changing candidate coordinates.

## Mechanism

For each frozen candidate `i`, predict six ordered distance probabilities:

```text
p_i = softmax(f(rae_i, base_logit_i, global_latent(C), local64(C, rae_i)))
s_i = -sum_b p_i[b] * distance_center[b]
```

Distance centers are frozen at
`[0.05, 0.175, 0.375, 0.75, 1.5, 3.0] m`; distances are clipped to `4 m`.
Training uses soft adjacent-bin distribution supervision plus within-range
listwise ranking. At inference the target is absent and `s_i` is passed to the
unchanged exact-capacity exporter.

## Closest closed routes

- R-A1/R-A2 predict binary occupancy confidence.
- Q1-R proposed a continuous scalar correction from coordinate, global latent,
  and base confidence, but never trained because its deleted-parent equivalence
  prerequisite failed.
- G1D changes candidate coordinates and reads local Cube evidence inside an
  arbitrary-query refiner.
- G1F tested a weak 32k pool, not the independently verified 700k pool.

Q-Local changes neither the field nor exporter. It changes the score target,
probability structure, and evidence path. It is a newly named fresh-parent
experiment and does not inherit Q1-R authorization.

## Frozen train-only cohort

Fit frames, one per sequence:

```text
1:232, 9:440, 21:402, 27:403,
29:399, 35:403, 47:514, 58:404
```

Unseen train-sequence frames:

```text
5:429, 26:403, 42:408, 50:405
```

No development-validation or test frame may be read. Wrong-Cube mappings are
fixed cycles within the fit and unseen groups. Candidate coordinates, parent
confidence, Q0/Q1, residual, exporter, and targets remain matched-frame data;
only global and local Cube evidence are swapped in the intervention.

## Falsification order

1. Rebuild all 12 candidate pools from raw frozen inputs and bind Cube, target,
   candidate, query, checkpoint, source, GPU UUID, and PCI bus hashes.
2. Require the GT-nearest exact exporter to satisfy `CD <=0.8 m` and
   `outlier <=5%` on every frame. This is a non-deployable capacity test.
3. Require initial-score export to reproduce the fresh-parent baseline exactly.
4. Train at most 500 updates on only the eight fit frames.
5. Evaluate immutable checkpoints at updates 100, 200, 300, 400, and 500 on
   both fit and four unseen train-sequence frames.

## Pass and stop rules

The run passes only when two consecutive evaluations satisfy every fit gate:

- mean Chamfer `<=1.0 m`;
- mean completeness `<=0.75 m`;
- mean outlier at 2 m `<=10%`;
- exact 10k, fixed range quotas, and true minimum spacing `>=5 cm`.

At the same selected checkpoint, the unseen-train group must satisfy:

- mean Chamfer improves at least 30% over fresh-parent confidence;
- mean outlier improves at least 5 percentage points;
- matched Cube beats wrong Cube on at least 3/4 frames;
- wrong Cube worsens mean Chamfer at least 5%;
- all structural export checks pass.

Hard stop is the earlier of 500 updates or 7200 GPU-seconds. No epoch extension,
threshold tuning, reduced cardinality, padding, jitter fill, best-of-k, target
inference, or development/test access is allowed.

## Unlock

Passing authorizes one frozen 76-train/24-development geometry run. It does not
pass G1, unlock Doppler/cycle/temporal training, or establish novelty.
