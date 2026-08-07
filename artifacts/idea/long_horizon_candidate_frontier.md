# Long-horizon candidate frontier

> Revised 2026-08-07 after the VRH-F0 scientific no-go.

## Failure diagnosis

The archived TruckScenes result does not support "exposure bias is the only
problem." Scheduled sampling improves late-horizon CD and Doppler freshness,
but the bridge already pays a large geometry tax at its first generated step.
The full-set refiner is rewriting a strong `copy_dopp` scaffold rather than
making a bounded correction. The old rollout evaluator also used independent
random subsamples across arms, so small arm differences require a paired replay
before publication.

The K-Radar geometry line is blocked even earlier. VRH-F0 certified that
sequential renewal was active, yet it worsened mean CD from `0.1095 m` for flat
exposure to `0.5494 m`, retained all target-bearing strata on only 3/12 frames,
and failed every frame-level first/later gate. A learned renewal model is not
authorized from this recipe.

## Ordered candidates

| Priority | Candidate | Required first gate | Reason |
|---:|---|---|---|
| 1 | STDA-F0 sparse target-demand assignment | Frozen zero-training all-76 oracle; exact 10k/strict 5 cm; absolute geometry/range/first-later gates; matched pointwise and greedy controls | Internal graph-constrained capacity test after pointwise and renewal exporters failed; no OT or ray-range novelty claim |
| 2 | Copy-anchored Doppler refresh | Paired one-step evaluator; XYZ remains bitwise `copy_dopp`; current Cube may update only Doppler/confidence | Preserves the strongest known geometric scaffold while testing current-observation refresh |
| 3 | Trust-region RaLD refiner | Zero gate, bounded XYZ residual, no-harm hinge against paired `copy_dopp` | Tests generative correction without allowing full-set geometric destruction |
| 4 | Selective renewal mixer | Persistent copied points plus Cube-supported births; dynamic-only or confidence-gated replacement | Makes birth/death explicit and limits refinement to uncertain/dynamic subsets |
| 5 | Free-running teacher/student distillation | Student trained on its own rollout states; paired multi-horizon gate | Addresses exposure bias only after one-step no-harm is established |
| 6 | Scene-flow/occupancy intermediate state | One-step geometry and condition-use gate before rollout | Provides a structured motion state when direct point regeneration remains unstable |
| 7 | Latent recurrent state | Last resort after simpler bounded routes fail | Highest implementation and identifiability risk |

## Execution rules

1. Repair the rollout evaluator so every arm shares the initial sample, GT
   sample, frame cohort, and random seed.
2. Do not use scheduled sampling to excuse a one-step geometry regression.
3. Require a no-harm gate against `copy_dopp` before any multi-step claim.
4. Report geometry, range coverage, Doppler freshness, confidence calibration,
   and rollout stability together.
5. Do not start the temporal candidates until a K-Radar single-frame geometry
   parent passes its independent train/validation gate.
