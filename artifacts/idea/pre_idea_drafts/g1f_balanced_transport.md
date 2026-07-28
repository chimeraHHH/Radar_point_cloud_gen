# G1F Stage-0: candidate ceiling and balanced transport

## Hypothesis

G1D's 32,000 measurement-grounded proposals contain adequate geometric support,
but hard top-k and separately sampled occupancy BCE allocate the 10,000-point
budget poorly. A balanced transport selector should improve coverage and
duplicates while retaining the low outlier behavior of measurement-grounded
queries.

## F0 diagnostic oracle

- Source: the frozen G1D v2 proposal cache before learned offsets.
- Pool: exactly 32,000 unique proposal coordinates per frame.
- Export: exactly 10,000 points.
- GT access: allowed only inside this diagnostic upper bound.
- Selection: range-stratified, capacity-constrained nearest support assignment;
  repeated selection of one candidate is forbidden.
- Report: the complete G1D geometry table, proposal-to-GT recall by range,
  candidate density, and the unattainable oracle label on every output artifact.

F0 passes only if the oracle export satisfies Chamfer, outlier, completeness,
far completeness, duplicate, and count gates. Condition shuffle is not
applicable to an oracle. Failure closes F1.

## F1 learned selector

- Freeze the proposal generator and offsets.
- Predict one mass/logit for every candidate using the global Full-RAED
  condition and proposal coordinates/range features.
- Use entropic balanced optimal transport between candidates and train-only GT
  points, with fixed range-bin mass constraints.
- Use a straight-through or deterministic top-10k export only after the soft
  transport loss is formed.
- Verify finite nonzero gradients for all candidate logits, including candidates
  outside the exported hard top-k.
- Train one seed for 10 epochs. No test access and no hyperparameter sweep.

## Controls

- the frozen G1D hard top-k candidate export;
- score shuffle within a frame;
- cross-scene Full-RAED condition shuffle;
- range-mass constraint disabled, with all other settings fixed.

## Abandonment

Stop after F0 if the candidate pool lacks support. Stop after F1 if the learned
selector exceeds 25% outliers, fails to improve both completeness and duplicate
fraction over its matched hard-top-k control, or has condition shuffle below
1%.
