# G1T Stage-0: temporal proposal support

## Hypothesis

The current Cube candidate support may miss weak or occluded surfaces that are
present in recent radar history. Ego-aligned history should add static support,
and Doppler-aware radial warping should further improve dynamic support.

## Arms

- `T0-current`: current-frame Cube proposals only.
- `T1-ego`: current proposals union ego-motion-warped history.
- `T2-doppler`: current proposals union ego-motion plus measured
  Doppler-displacement-warped history.

Every arm:

- uses exactly four history frames;
- is rescored by the current Cube;
- applies the same deterministic `(5, 5, 3)` RAE-cell duplicate suppression;
- exports exactly 10,000 points;
- uses no LiDAR, GT, or learned G1D score at inference.

The existing temporal warp implementation is reused only after its coordinate,
time-sign, and frame-transform tests pass.

The residual-Doppler split uses the train-selected `positive_ego` convention.
This is only a fixed sign mapping for the DoppDrive-style proposal control; the
failed validation audit still prohibits an analytic static-physics claim.

## Metrics

Report the complete G1D geometry table, current-Cube support, source-frame age,
static/dynamic and near/far slices (fixed boundary `60 m`), and the fraction of
exported points supplied by history. Scene-first uncertainty is mandatory
because the validation cohort is small.

## Promotion

`T2-doppler` must improve completeness and far completeness over `T1-ego`, while
increasing neither outlier nor duplicate fraction by more than two percentage
points. It must also outperform `T0-current` on Chamfer. Otherwise the
Doppler-specific temporal proposal route closes.

Passing G1T establishes proposal support only. It does not establish generated
Doppler, future prediction, or temporal generative consistency.
