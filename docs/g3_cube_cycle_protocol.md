# G3 Differentiable Cube-Cycle Protocol

## Scope

G3 tests the paper's central bidirectional claim: generated points should not
only inherit attributes from a 4D Radar Cube, but should also reconstruct the
measured Cube evidence when differentiably projected back. G3 starts only after
G2 is decided. If E5 passes, E5 is the parent; if the distribution result E4
passes but the analytic mixture E5 fails, E4 is retained as the declared
single-frame parent and E5 remains a negative result.

The renderer uses continuous RAE coordinates, 64-bin Doppler distributions,
and predicted point confidence. It performs trilinear spatial splatting and
preserves each point's Doppler probability mass. It does not attempt to recover
complex IQ phase.

## Compared Systems

| ID | Added point parameterization | Cube loss | Purpose |
|---|---|---|---|
| C0 | bounded continuous RAE offsets | none | architecture-matched control |
| C1 | same | local covered-cell spectrum KL | local mechanism ablation |
| C2 | same | local KL + global Doppler marginal KL | marginal mechanism ablation |
| C3 | same | local + marginal + sparse spatial energy | primary E6 method |

All four systems initialize from the same-seed G2 parent. Every generated point
has an offset bounded to `[-0.5, 0.5]` bin in range, azimuth, and elevation.
The occupancy decoder, Doppler head, optimizer, schedule, output count, and
evaluation code are otherwise matched. C3 is the only preregistered primary G3
comparison. C1/C2 cannot replace C3 after observing results.

## Renderer and Loss

For point `i`, continuous RAE coordinate `u_i`, confidence `c_i`, and Doppler
distribution `p_i(d)`, splat to the eight neighboring spatial cells:

```text
Cube_hat[d, r, a, e] = sum_i c_i * w_i(r,a,e | u_i) * p_i(d).
```

Trilinear weights are renormalized at Cube boundaries, so each in-FOV point
contributes exactly `c_i` total mass. Required unit tests verify energy
conservation and nonzero gradients to offsets, Doppler logits, and confidence.

The three loss terms are:

1. confidence-weighted local spectrum KL at rendered cells;
2. global Doppler marginal KL over the full Cube;
3. normalized spatial-energy loss over the union of rendered cells and the
   10,000 strongest target Cube cells.

Fixed weights are `1.0`, `0.25`, and `0.25`. A confidence-floor penalty of
weight `1.0` activates below mean confidence `0.1`, but this floor alone is not
accepted as evidence against confidence collapse.

## Training Protocol

- Seeds: `20260716`, `20260717`, `20260718`.
- Parent: same-seed best G2 checkpoint and exact parent SHA-256.
- Epochs: 20.
- Optimizer: AdamW, head/offset learning rate `3e-4`, inherited backbone
  learning rate `3e-5`, weight decay `1e-4`.
- Scheduler: cosine annealing over all 20 epochs.
- Numeric mode: BF16 autocast with FP32 parameters.
- Model selection: lowest common validation score
  `local_spectrum_KL + 0.25 * geometry_Chamfer`.
- Main statistics: all three seeds, all frozen validation scenes, scene-first
  paired bootstrap.

## Confidence-Escape Audit

For every frame, report mean confidence, confidence quantiles, ECE, rendered
energy, covered-cell count, and the correlation between confidence and input
Cube energy. C3 passes the anti-collapse check only if:

- its mean confidence is at least 90% of C0's paired mean;
- its covered-cell count is at least 90% of C0's paired count;
- its confidence ECE does not degrade by more than an absolute `0.02`.

These checks are evaluated with paired confidence intervals. A model that
reduces cycle loss by suppressing confidence or coverage fails G3 regardless
of other metrics.

## Robustness Matrix

Evaluate C0 and C3 without retraining under:

- additive log-power noise at fixed SNR levels `20`, `10`, and `5 dB`;
- Doppler-bin circular shifts of `-2`, `-1`, `+1`, and `+2` bins;
- azimuth/elevation calibration offsets of `0.25` and `0.5` bin;
- confidence temperature multipliers `0.5`, `1`, and `2`.

Report degradation curves for Cube KL, PCE, geometry Chamfer, and confidence
coverage. Robustness results diagnose the mechanism and do not replace the
clean-data gate.

## G3 Decision Rule

G3 passes only when C3 versus C0 satisfies all conditions:

1. confidence-interval-excluding-zero improvement in local spectrum KL;
2. confidence-interval-excluding-zero improvement in at least one second metric
   class: static PCE, geometry Chamfer/F-score, or a frozen downstream metric;
3. geometry Chamfer relative degradation upper 95% bound at most 2% if geometry
   is not the improving class;
4. all three confidence-escape checks pass;
5. no data/provenance mismatch and no failed seed.

If G3 fails, do not describe the project as a successful Cube-point closed
loop. Inspect renderer normalization, offset saturation, and confidence escape.
If one redesign still fails, retain E4/E5 as the result and reposition the
cycle as a negative or diagnostic experiment, as required by the top-level
stop rule.

## Required Artifacts

- C0-C3 configurations, parent hashes, best/last checkpoints, and logs;
- renderer unit-test report with conservation and gradient checks;
- per-frame clean and robustness metrics for all seeds;
- paired bootstrap and G3 decision JSON/Markdown;
- fixed-frame Cube/point/reprojected-Cube panels and worst-five failures;
- confidence reliability and coverage plots;
- ablation table separating local, marginal, and spatial-energy losses.
