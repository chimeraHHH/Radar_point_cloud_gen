# G1C: RaLD-Guided Query Geometry Protocol

## Decision boundary

G1C is a new, independently named branch created after G1B Stage A produced no
survivor. It does not reopen G1 or G1B, and it cannot use any failed occupancy
checkpoint as initialization. The authoritative G1B report is
`artifacts/g1/g1b_final/g1b_screen_3fa7ae88.json`.

The motivation is specific: the G1B `full_raed_rank2` candidate reached a
validation median Chamfer of `2.0251 m`, but its mean outlier fraction remained
`28.885%` against the frozen `25%` limit. Its errors were concentrated in
sequences 12, 43, and 51. Continuing the same occupancy family would be an
unregistered threshold repair. G1C instead changes the geometry abstraction
from dense-grid occupancy ranking to RaLD-style radar-guided point queries.

## Frozen architecture

1. Normalize the complete 64-bin Full-RAED Cube with train-only statistics.
2. Rank spatial cells by normalized integrated radar energy. Apply deterministic
   3D non-maximum suppression and retain exactly 1,000 base seeds.
3. Expand every seed into 10 deterministic fractional RAE query templates,
   producing exactly 10,000 initial queries. No CFAR helper, LiDAR query, failed
   occupancy logit, or test frame is allowed.
4. Embed each query from normalized RAE, its complete local 64-bin spectrum, and
   a learned template embedding.
5. Encode the Cube as 336 Full-RAED radar tokens. RaLD static and input-dependent
   dynamic queries form 512 mixed latents, followed by 24 latent Transformer
   layers and query cross-attention.
6. Predict bounded continuous RAE residuals, existence confidence, and the local
   64-bin Doppler distribution. G1C geometry training uses only XYZ/confidence;
   the distribution output is retained for the later G2C physical gate.

The fixed residual bounds are `8` range bins, `4` azimuth bins, and `2`
elevation bins. Final coordinates are clamped only to the valid Cube domain.

## Frozen losses

The one-frame geometry objective is

```text
L = L_chamfer
  + 0.25 L_outlier_hinge(2 m)
  + 0.10 L_existence
  + 0.02 L_offset
  + 0.02 L_within_seed_repulsion.
```

`L_outlier_hinge` is the mean squared excess over 2 m for generated-to-target
nearest-neighbour distances. `L_within_seed_repulsion` operates only among the
10 queries expanded from the same radar seed and penalizes pairwise distance
below `0.10 m`; it prevents passing through duplicate points.

## Stage A: one-seed no-go screen

- seed: `20260716`;
- 30 epochs, AdamW, cosine schedule;
- fixed development train/validation split and target cache;
- validation checkpoint selected by
  `Chamfer + 2 * max(outlier_fraction_2m - 0.25, 0)`;
- no architecture, loss, point-count, or threshold sweep.

Stage A passes only if all checks hold on the full validation cohort:

- median Chamfer `<= 2.50 m`;
- mean outlier fraction at 2 m `<= 0.25`;
- median completeness mean distance `<= 0.65 m`;
- mean 60-120 m completeness mean distance `<= 8.0 m`;
- duplicate fraction within `0.05 m` `<= 0.10`;
- mean confidence `>= 0.10`;
- all 64 Doppler bins and Full-RAED radar tokens receive nonzero gradients by
  the second optimizer step.

No bounded repair is permitted after a Stage A failure.

## Stage B: three-seed confirmation

Only a passing Stage A authorizes seeds `20260717` and `20260718` with the exact
same configuration. The three-seed scene-first bootstrap must satisfy:

- Chamfer upper 95% confidence bound `<= 2.50 m`;
- outlier upper 95% confidence bound `<= 0.25`;
- completeness upper 95% confidence bound `<= 0.65 m`;
- duplicate-fraction upper 95% confidence bound `<= 0.10`.

If G1C passes, its three checkpoints become the only eligible geometry family
for new successors named RH-C, G2C, G3C, G3L-C, and G4L-C. Existing G1B/RH/G2R/
G3R reports remain failed or skipped and are never rewritten.

If G1C fails, Cube-to-dense single-frame geometry is no-go under the current
data and compute protocol. The project then preserves the negative evidence,
does not release P5 test, and limits further work to paper/report consolidation
or a separately proposed future dataset/model program.
