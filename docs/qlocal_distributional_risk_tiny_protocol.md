# Q-Local-F0 Distributional Geometric-Risk Tiny Protocol

Status: frozen before implementation and execution

## Decision question

Can local Full-RAED evidence predict geometric risk well enough to select a
useful exact-10k subset from a newly named fresh replay field, while preserving
candidate coordinates and demonstrating an initial unseen-train-sequence and
wrong-Cube effect?

This is a geometry-parent falsification. It is not a Doppler-output, cycle,
temporal, development-validation, or test experiment.

## Fresh parent boundary

The only allowed parent bytes are:

```text
/home/wangning/Shared/l40s_wangning_radar/cube_dense_runs/
  formal_f2a9489_ra1_wce_replay_20260807/checkpoint_epoch020.pt
```

Frozen SHA-256:

```text
c0ebb4c1509c3d36b6834bd39977cb1cda2fcaff1747eabc97ebb41af0f06ce4
```

The source family is `f2a9489d40323d1ef45d85de958f4aea8126e1c8`, epoch
20, seed `20260716`. The run must call this `Fresh-WCE-20`. It must not claim
checkpoint identity or endpoint equivalence with deleted R-A1. The failed Q1-R
certificate is evidence of that boundary, not an authorizer.

Q0, occupancy-dependent Q1, residual coordinates, parent confidence, candidate
count, range quotas, exporter, and minimum spacing remain frozen:

```text
Q0: 166667 / 166667 / 166666
Q1 anchors: 20000 / 4250 / 750, 8 samples each
candidate count: 700000
output: 8000 / 1700 / 300 = 10000
minimum Euclidean spacing: 0.05 m
```

## Frozen cohort

The cohort is selected only from the existing 76-frame train partition.

Fit group, in identity order:

```text
(1,232), (9,440), (21,402), (27,403),
(29,399), (35,403), (47,514), (58,404)
```

Unseen train-sequence group, in identity order:

```text
(5,429), (26,403), (42,408), (50,405)
```

Wrong-Cube maps are frozen cycles:

```text
fit:    1->9->21->27->29->35->47->58->1
unseen: 5->26->42->50->5
```

The implementation must reject missing, duplicate, reordered, wrong-partition,
or wrong-pair records. It must bind each raw Cube and target-cache SHA-256 plus
the ordered cohort digest. Validation and test partitions are forbidden.

## Candidate capacity gate

Before model construction, all 12 frames are rebuilt from raw frozen Cube and
cache bytes. The target is opened only after the target-free 700k candidate
field and its hashes have been committed in memory.

For each frame, use negative nearest-target distance only as an explicitly
non-deployable score and run the exact exporter. Every frame must satisfy:

- Chamfer `<=0.8 m`;
- outlier fraction at 2 m `<=5%`;
- exact 10,000 output points;
- exact `8000/1700/300` range quotas;
- observed minimum pair distance `>=0.05 m`;
- no copying, padding, jitter, duplicate fallback, or best-of-k.

Failure stops Q-Local before training. This oracle cannot be reported as a
model result.

## Scorer

The fresh field and all its parameters are frozen and excluded from the
optimizer. For candidate `i`, the scorer receives:

1. refined normalized RAE coordinate;
2. detached fresh-parent logit;
3. detached global Fresh-WCE condition latents;
4. normalized local 64-bin log-power spectrum queried at the refined candidate;
5. local integrated log energy and spectrum entropy;
6. local finite-difference R/A/E energy gradients.

The scorer has at most 2,000,000 trainable parameters. Candidate-coordinate
Fourier features and local features are projected to one token, which
cross-attends the frozen global condition. A six-logit head predicts distance
mass at centers:

```text
[0.05, 0.175, 0.375, 0.75, 1.5, 3.0] m
```

The deterministic inference score is negative expected distance. The forward
API must not accept targets, future/test records, candidate-selection indices,
stochastic sample IDs, or best-of-k arguments.

## Same-coordinate intervention

Wrong-Cube evaluation must keep all of the following bit-identical to matched
inference:

- Q0/Q1 query rows;
- refined candidate coordinates and XYZ;
- fresh-parent confidence;
- target and exporter configuration.

Only global condition latents and local spectrum/energy/gradient features are
recomputed from the frozen wrong Cube at the same candidate coordinates. This
prevents a coordinate change from masquerading as condition dependence.

## Training target and loss

Target geometry is used only after the candidate/evidence field exists. Let
`d_i` be the nearest current target distance, clipped at 4 m. The target is a
two-bin linear distribution over the frozen centers. Coordinates, targets,
nearest indices, distances, and distribution labels are stop-gradient.

Each update samples 16,000 unique rows with quotas `12800/2720/480`. Sampling
is deterministic by source seed, frame identity, and update, with equal mass
from low-risk and remaining rows in each range.

Loss:

```text
L = mean_range distribution_cross_entropy
  + 0.5 * mean_range listwise_risk_ranking
```

The listwise term compares predicted expected risk with target clipped distance
inside each range. Binary occupancy BCE is prohibited.

## Evaluation and gate

Budget is at most 500 updates and at most 7200 measured GPU-seconds, whichever
comes first. Evaluate immutable checkpoints at updates 100, 200, 300, 400, and
500. Every evaluation covers all 12 frames and both matched/wrong arms.

Fit gate, required at two consecutive evaluations:

- mean Chamfer `<=1.0 m`;
- mean completeness `<=0.75 m`;
- mean outlier fraction at 2 m `<=10%`;
- all structural export checks pass.

At the later of those two checkpoints, unseen-train requirements are:

- mean Chamfer improves `>=30%` versus fresh-parent confidence;
- mean outlier decreases `>=5 percentage points`;
- matched Chamfer beats wrong Cube on at least 3/4 frames;
- wrong Cube worsens mean Chamfer `>=5%`;
- all matched and wrong exports pass exact-count, quotas, and spacing.

Passing status is `qlocal_f0_passed`. Update/time exhaustion is
`qlocal_f0_no_go`. Capacity failure is `qlocal_f0_capacity_no_go`. A source,
input, GPU, transaction, or API-boundary failure is implementation-invalid and
does not produce a scientific result.

## Provenance and transaction requirements

The run must close every open boundary from the Q1-R final static audit:

- independently reconstruct oracle exports and geometry from raw target and
  candidate tensors;
- enforce exact cohort identities, order, and wrong-pair map;
- recompute the complete immutable evaluation chain on resume;
- use staging plus atomic rename for normal and capacity-failure terminals;
- record physical GPU index, UUID, PCI bus ID, name, and visible-device map;
- add adversarial tests for raw-input tamper, dropped/duplicate frame, wrong
  pairing, forged pass history, capacity-failure crash recovery, and a real
  end-to-end source-bound invocation.

All scientific execution uses `wangning`, `/home/wangning/miniforge3/envs/hym_radar`,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, and physical H200 GPU 0 or 2. GPU 1 is forbidden.

## Unlock

A pass authorizes only a separately frozen 76-train/24-development Q-Local
geometry run. It does not pass the final geometry gate or unlock Doppler, cycle,
temporal, downstream, or test stages.
