# G1D: RaLD Query-Field Geometry Protocol

## Decision boundary

G1D is a new protocol created before any scientific G1C training result existed.
The first G1C queue failed before launching its child process because it resolved
the interpreter as a missing bare `python`; the repaired queue was stopped while
waiting for an H200 after a source-level RaLD audit exposed material architecture
differences. No G1C checkpoint, validation metric, or test result informed G1D.

G1D does not reopen G1, G1B, or the matched RaLD occupancy baseline. It uses no
failed occupancy checkpoint, CFAR helper, test frame, or LiDAR-derived inference
query. G1C remains an implemented deterministic control and must not be renamed
as a RaLD latent-diffusion model.

The audited upstream source is the Apache-2.0 RaLD release at commit
`ffec4b41241391734b1eda5c093de843c909eb8e`. G1D borrows the information flow
rather than only its dimensions. The claim-by-claim upstream mapping is frozen
in [`g1d_rald_source_map.md`](g1d_rald_source_map.md):

1. separate occupied and empty-space training queries;
2. static plus input-dependent dynamic set latents;
3. arbitrary spatial queries decoded through latent cross-attention;
4. radar-condition cross-attention in every latent Transformer block;
5. coarse candidate screening followed by local query refinement.

## Frozen representation

The input is the complete 64-bin Full-RAED Cube with train-only log-power
normalization. The radar encoder produces exactly 336 spatial tokens. G1D keeps
the Doppler spectrum; it does not copy RaLD's intensity-only condition.

The mixed latent encoder uses 512 working tokens:

```text
Q_dynamic = CrossAttn(Qd, radar-proposal tokens)
Z0 = Proj(Qs + Q_dynamic)
Z1 = Z0 + CrossAttn(Z0, radar-proposal tokens)
Z  = Z1 + FFN(Z1)
```

Twenty-four latent blocks then apply, in order:

```text
self-attention -> Full-RAED cross-attention -> feed-forward
```

with a residual connection around every operation. These are deterministic
`512 x 512` working tokens for the geometry gate, not RaLD's Gaussian
`512 x 32` VAE state. The physical VAE and EDM remain a later G3L-D gate.

## Training queries

Each frame uses exactly 10,000 arbitrary occupancy queries:

- 625 occupied queries (`6.25%`, matching RaLD's released protocol);
- 9,375 empty-space queries;
- occupied queries are sampled from radar-observable target cells with
  confidence weighting and independent `[-0.5, 0.5]` bin jitter;
- empty queries are sampled approximately equally from the near, middle, and
  far range thirds and receive the same independent `[-0.5, 0.5]` bin jitter;
- cells inside a `3 x 3 x 3` dilation of an occupied cell are ambiguous and are
  excluded from the empty set.

The fixed query split is a RaLD prior, while range stratification and the
ambiguous region are K-Radar adaptations for its 120 m view. Validation queries
are deterministically seeded by `(sequence, radar_index)` and are never
resampled for checkpoint comparison. Matching positive/empty jitter is a hard
gate: the fractional coordinate rate must be at least 90% in both classes and
differ by at most five percentage points, preventing coordinate phase from
leaking the occupancy label.

Occupancy supervision follows the released RaLD training implementation:

```text
L_occ = 0.1 * BCE_positive + 1.0 * BCE_empty.
```

RaLD takes separate class means and applies `vol_weight=0.1` to occupied
queries and `near_weight=1.0` to empty queries
(`engine_ae.py:48-50,79-86`;
`configs/ae/ae_indoor_cfg_aniso_mix_view_cone.yml:54-56`). The unweighted BCE
at `engine_ae.py:159,211` belongs only to evaluation and must not be copied as
the training objective. G1D preserves the official class weights and reports
both classwise BCE values, recall, and empty-space false-positive rate.

## Inference queries

Inference is deterministic and produces exactly 10,000 points:

1. integrate log energy over all 64 Doppler bins;
2. select 1,000 spatial seeds with deterministic plateau tie-breaking and
   strict local 3D suppression;
3. expand each seed with 32 fixed low-discrepancy RAE templates, yielding
   32,000 coarse queries;
4. decode arbitrary-query occupancy and confidence, then retain exactly 2,500
   queries by their fixed combined score;
5. expand each retained query with four fixed tetrahedral local templates;
6. decode the resulting 10,000 refined queries and predict bounded continuous
   RAE residuals and confidence.

The query feature contains normalized RAE, the complete normalized 64-bin local
spectrum, train-only standardized absolute log energy, and normalized range.
The raw energy is the sum of natural-log power over all 64 bins. Before the
query MLP, it is converted to mean `log10(1+x)`, standardized by the same
train-only center/scale as the radar encoder, and clipped to `[-4,4]`. This is
a fixed monotonic transform, not per-frame normalization, so cross-frame
absolute energy remains observable without overwhelming the spectrum and range
features. Decoding is chunked but mathematically identical across chunk sizes.
The geometry gate attaches the measured final-position Cube spectrum; it does
not claim learned Doppler generation before G2D.

The occupancy head uses the released RaLD decoder's default `nn.Linear`
initialization. The added confidence and offset heads remain zero initialized;
therefore the zero-offset control is exact at initialization without forcing
the occupancy field to begin as a constant logit.

## Frozen objective

The Stage A objective is

```text
L = L_occ
  + 1.00 L_chamfer
  + 0.25 L_outlier_hinge(2 m)
  + 0.10 L_existence
  + 0.02 L_offset
  + 0.02 L_global_knn_repulsion(0.10 m).
```

Global repulsion is evaluated across all generated points, not only points from
one seed. Confidence cannot reduce the geometry or outlier terms.

## Training schedule

- Stage A seed: `20260716`;
- 150 epochs;
- AdamW, base learning rate `1e-4`, weight decay `0.05`;
- five-epoch linear warmup, cosine decay to `1e-6`;
- gradient norm clipping at `10`;
- parameter EMA at `0.999`;
- one Cube per optimizer step;
- deterministic NMS flat indices may be cached per frame because they are a
  detached function of the frozen input Cube; cached and uncached forwards must
  be tensor-equivalent, while local spectra, absolute energy, radar tokens, and
  all differentiable query features are recomputed on every optimizer step;
- the EMA model is evaluated on the complete validation partition for every
  checkpoint-selection event;
- no architecture, query-count, loss-weight, or threshold sweep.

The preflight uses two train frames and two validation frames but exercises the
formal 32,000-to-10,000 query path. It may reduce model width/depth only when the
formal path cannot be instantiated without affecting tensor counts or source
coverage; the full H200 structure check must still instantiate all 24
conditioned blocks before Stage A. Validation condition shuffling uses a
deterministic derangement whose paired frames always have different sequence
IDs; the preflight and formal comparison both reject any same-sequence pair.

## Stage A gate

Stage A passes only when every check holds on the complete validation cohort:

- generated median Chamfer `<= 2.50 m`;
- generated mean outlier fraction at 2 m `<= 0.25`;
- median completeness mean distance `<= 0.65 m`;
- mean 60-120 m completeness mean distance `<= 8.0 m`;
- global duplicate fraction within `0.05 m` `<= 0.10`;
- mean confidence `>= 0.10`;
- positive occupancy recall `>= 0.80`;
- empty-query false-positive rate `<= 0.20`;
- the query field does not regress both Chamfer and outlier fraction relative
  to its frozen zero-offset radar-proposal control;
- cross-scene shuffled Full-RAED condition degrades occupancy BCE or Chamfer by
  at least `1%`;
- all 64 Cube channels, all 64 local-spectrum input columns, the mixed-latent
  encoder, and every one of the 24 radar-condition cross-attention blocks have
  finite nonzero gradients by the second optimizer step;
- exactly 336 radar tokens, 32,000 coarse queries, 2,500 selected queries, and
  10,000 final points are recorded for every validation frame;
- source commit, clean worktree, transitive source hashes, data hashes, split,
  normalization, seed, and untouched-test state all verify.

No bounded repair or threshold relaxation is permitted after a Stage A
scientific failure.

## Stage B gate

Only a passing Stage A authorizes the exact same configuration for seeds
`20260717` and `20260718`. Scene-first paired bootstrap upper 95% confidence
bounds must satisfy:

- Chamfer `<= 2.50 m`;
- outlier fraction `<= 0.25`;
- completeness `<= 0.65 m`;
- duplicate fraction `<= 0.10`;
- positive occupancy false-negative rate `<= 0.20`;
- empty-query false-positive rate `<= 0.20`.

The paired query-field-minus-zero-offset control must not have a positive upper
95% bound for both Chamfer and outlier fraction, and at least one must have an
upper bound below zero. Otherwise geometry may pass, but the RaLD query-field
contribution claim fails.

If G1D passes, its three checkpoints are the only eligible geometry family for
G2D, G3D, G3L-D, and G4L-D. If it fails, the current Cube-to-dense single-frame
program closes without releasing P5 test; only negative-result consolidation
and a separately proposed future data/model program remain.
