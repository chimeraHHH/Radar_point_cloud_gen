# Q-Local-F0R global-export capacity decision

Date: 2026-08-07

## Terminal decision

`qlocal_f0r_capacity_no_go`

Removing only the contradictory per-frame range quotas makes the unchanged
Fresh-WCE candidate field pass the aggregate per-frame geometry thresholds, but
does not preserve target-bearing middle/far strata. The global pointwise
GT-nearest oracle passed Chamfer, outlier, exact-count, uniqueness, and true
5 cm spacing on all 12 train-only frames, yet passed the preregistered
target-stratum retention gate on only 2/12 frames. Q-Local training is therefore
not authorized.

This is a scientific no-go, not an implementation failure. It closes pointwise
quality scoring followed by one unconstrained global export on this frozen
700k candidate field. It authorizes the separately frozen variable multi-return
renewal-hazard capacity route; it does not establish that the untrained
Q-Local scorer would fail.

## Bound execution

- source: `34579a245efecb0a10baeec3bcd1929a4744d502`;
- protocol: `g1_qlocal_f0r_global_export_capacity_v1`;
- host: `WHUServer-H200`;
- device: physical GPU 0, NVIDIA H200 NVL;
- GPU UUID: `GPU-74b9b73f-c405-55bc-bf76-f8aa85d1e7fe`;
- PCI bus: `00000000:01:00.0`;
- train-only cohort: eight fit plus four unseen-train frames;
- validation/test/future access: false/false/false;
- model trained: false;
- hard range quota used: false;
- elapsed GPU time: `456.3370 s`;
- peak allocated/reserved memory: `1,827,308,032 / 2,178,940,928` bytes;
- report SHA-256:
  `505644667ba3f73b58ee138608307e82632fe92bc3336f538c74767ab9c148ae`;
- run-log SHA-256:
  `9d10bc76ab8d1e3569e3c12e9e99f2be68909c13da1cd49f3c5acbb0ac2b5e39`.

All candidate/Q0/Q1/base-confidence reconstructions and archived fixed-oracle
reproductions matched their frozen hashes. The Q-Local model and loss Git blobs
were unchanged from source `bf1a166`. The Fresh-WCE parent checkpoint, metrics,
and run manifest were unchanged. No capacity exception occurred.

## Frozen gate result

| Check | Result |
|---|---:|
| exact 10,000 unique candidate rows | 12/12 |
| observed minimum spacing at least 5 cm | 12/12 |
| no copy, padding, jitter, or duplicate repair | 12/12 |
| per-frame Chamfer at most 0.8 m | 12/12 |
| per-frame 2 m outlier fraction at most 5% | 12/12 |
| all target-bearing strata retained | 2/12 |
| complete per-frame gate | 2/12 |

Every frame had zero 2 m outliers. Chamfer ranged from `0.2814 m` to
`0.7941 m`. The two complete passes were the old fixed-quota failure identities
`47:514` and `58:404`, confirming that the retired hard quotas caused those two
earlier failures.

| frame | group | CD (m) | selected near/mid/far | stratum retention | full gate |
|---|---|---:|---:|---:|---:|
| `1:232` | fit | 0.7941 | 9535/441/24 | fail | fail |
| `9:440` | fit | 0.3089 | 9777/219/4 | fail | fail |
| `21:402` | fit | 0.3936 | 9542/436/22 | fail | fail |
| `27:403` | fit | 0.2814 | 9967/32/1 | fail | fail |
| `29:399` | fit | 0.2814 | 9798/179/23 | fail | fail |
| `35:403` | fit | 0.3763 | 8506/1493/1 | fail | fail |
| `47:514` | fit | 0.4971 | 9904/96/0 | pass | pass |
| `58:404` | fit | 0.6548 | 10000/0/0 | pass | pass |
| `5:429` | unseen train | 0.4203 | 9858/140/2 | fail | fail |
| `26:403` | unseen train | 0.3878 | 9751/247/2 | fail | fail |
| `42:408` | unseen train | 0.4385 | 9693/276/31 | fail | fail |
| `50:405` | unseen train | 0.3996 | 9630/274/96 | fail | fail |

## Mechanism diagnosis

The pointwise nearest-target score is dominated by dense near-range surfaces.
After global sorting, the exact-10k budget is spent repeatedly on nearby
candidates while target-bearing middle/far strata receive too few returns. For
example, frame `1:232` selected only 24 far points and worsened far
completeness from the fixed oracle's `0.8063 m` to `3.5947 m`; frame `35:403`
selected one far point and reached `18.1071 m` far completeness. The same
collapse occurs despite excellent global Chamfer and zero outliers.

The failure therefore lies in set-level allocation and ordered return
representation, not in coarse aggregate candidate support. An independent
per-candidate score plus one global top-capacity operation has no mechanism to
renew allocation after a surface return or preserve multiple target layers
along a radar ray.

## Consequences

1. Do not run the 500-update Q-Local scorer training; no optimizer, learned
   checkpoint, or learned-result claim is authorized.
2. Close Q-Local scoring plus unconstrained global export on the frozen
   Fresh-WCE-20 field. Do not reopen it by weakening the target-stratum gate,
   deleting frames, or reporting aggregate Chamfer alone.
3. Keep the positive capacity evidence: the unchanged field supports exact
   10k, true 5 cm, low-outlier geometry, and the two old fixed-quota failures
   disappear after quota removal.
4. Start a separately named, train-only, zero-training capacity oracle for a
   variable multi-return renewal-hazard field. It must not reuse fixed
   `K=4/6`, hard `8000/1700/300` quotas, GT at inference, copy/jitter/padding,
   or best-of-k selection.
5. Sparse ray-range partial transport remains behind a separately frozen
   hard-rounding gate and is not authorized in parallel with the renewal-hazard
   capacity decision.
