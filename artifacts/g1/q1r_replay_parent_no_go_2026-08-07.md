# Q1-R Replay-Parent Certificate No-Go

Date: 2026-08-07

## Verdict

- Canonical action: `stop` the current Q1-R replay-parent route.
- Do not launch the eight-frame continuous-quality training run.
- Do not launch Q2 polar uncertainty or any Doppler, cycle, temporal, or test
  stage from this replay.
- Preserve the replay and diagnosis as evidence that the ranking bottleneck is
  reproducible, but do not identify the replay as the deleted original parent.

The source-equivalent replay passed source, data, runtime, query, diagnosis,
and evidence-boundary checks. It failed the preregistered formal endpoint
equivalence tolerance. Relaxing those tolerances after observing the replay is
not permitted.

A final independent static audit subsequently found three P1 and three P2
hardening gaps in the reusable authorization and resume implementation. These
do not alter this no-go because four endpoint comparisons fail directly, but
they mean `e159a22` must not be reused as an authorization baseline for a future
quality-ranking branch. See
`artifacts/g1/q1r_e159a22_final_static_audit_2026-08-07.md`.

## Replay

- Training source: `f2a9489d40323d1ef45d85de958f4aea8126e1c8`
- Certifier source: `e159a22c5343e81c0cf29351da07290a686edba5`
- Device: physical NVIDIA H200 GPU2
- Environment: Torch `2.12.1+cu130`
- Budget: 20 epochs, 76 train frames, 24 development frames
- Runtime: `7990.91 s`
- Peak allocated/reserved: `8.5145/9.0605 GiB`
- Replay checkpoint SHA-256:
  `c0ebb4c1509c3d36b6834bd39977cb1cda2fcaff1747eabc97ebb41af0f06ce4`
- Deleted original checkpoint SHA-256:
  `5be30e0f1ca23ea3b603abb0f5e330efd3599167362a8e23ab3a5967c411a2a0`
- Test, future Cube, CFAR, and Doppler head: not accessed or evaluated

## Frozen Endpoint Comparison

| Metric | Archived R-A1 | Replay | Absolute difference | Tolerance | Result |
|---|---:|---:|---:|---:|---|
| Mean Chamfer | 4.008953 m | 4.026066 m | 0.017113 m | 0.020000 m | Pass |
| Median completeness | 1.508053 m | 1.453179 m | 0.054874 m | 0.020000 m | Fail |
| Mean outlier fraction | 31.812916% | 31.785416% | 0.027500 pp | 0.200000 pp | Pass |
| Mean far completeness | 11.121106 m | 11.409670 m | 0.288564 m | 0.100000 m | Fail |
| Wrong-minus-matched Chamfer fraction | 12.987397% | 7.926609% | 5.060788 pp | 1.000000 pp | Fail |
| Matched-condition win fraction | 66.666667% | 79.166667% | 12.500000 pp | 4.166667 pp | Fail |

The certificate therefore records
`formal_stage0_decision_preserved=false` and
`q1r_tiny_authorized=false`.

## Replay-Bound Failure Diagnosis

The full 24-frame diagnosis was recomputed from per-frame reports. It passed
all count, runtime, hash-control, and forbidden-access checks:

- archived-vs-replay fixed Q0 hashes: `24/24`;
- archived-vs-replay learned Q1 hashes: `0/24`, expected for a newly trained
  occupancy-dependent replay;
- diagnosis-vs-replay Q0 hashes: `24/24`;
- diagnosis-vs-replay Q1 hashes: `24/24`;
- bit-exact formal export controls: `24/24`.

| Same replay pool and exporter | Mean Chamfer | Median completeness | Mean outlier | Mean far completeness |
|---|---:|---:|---:|---:|
| Current confidence | 4.026066 m | 1.453179 m | 31.785416% | 11.409670 m |
| Validation-GT nearest score | 0.637547 m | 0.154335 m | 2.277083% | 0.704566 m |

The diagnostic branch remains
`confidence_ranking_bottleneck_indicated`. This is a mechanism diagnosis, not
a deployable method result: the GT score is unattainable at inference and the
24-frame cohort has already informed route selection.

## Decision Boundary

The replay is a scientifically useful new parent candidate, but the current
Q1-R protocol was explicitly conditioned on reproducing the deleted R-A1
endpoint within fixed tolerances. Four of six endpoint comparisons failed.
Consequently:

1. The certificate is authoritative and Q1-R training is skipped.
2. Q2 cannot be used as a post-hoc repair because it was conditional on a Q1
   signal under the authorized parent.
3. The current fixed-10k geometry family still has no passing parent, so all
   downstream physical and temporal stages remain locked.
4. Any future continuous-quality experiment must be a newly named branch with
   a newly frozen baseline and untouched final test cohort; it cannot inherit
   the original-parent claim.
5. Before such a branch can authorize training, it must close the final static
   audit's raw-input oracle binding, exact eight-frame evaluation, full
   evaluation-chain, atomic capacity-failure, GPU identity, and adversarial-test
   requirements.

## Evidence

- `artifacts/g1/q1r_replay_parent_e159a22/certificate.json`
- `artifacts/g1/q1r_replay_parent_e159a22/full24.json`
- `artifacts/g1/q1r_replay_parent_e159a22/metrics_epoch020.json`
- `artifacts/g1/q1r_replay_parent_e159a22/run_manifest.json`
- `artifacts/g1/q1r_replay_parent_e159a22/summary.json`
- `artifacts/g1/q1r_e159a22_final_static_audit_2026-08-07.md`
- `docs/rald_wce_quality_tiny_protocol.md`

Artifact SHA-256 values:

- certificate: `0114cc6e0d4b443b1be69c0686eb709ed754572fbbe0e4f51a0f382b372743f4`
- diagnosis: `4f5324950c6fe2b806d697cca7953503a44f250927b6fa537d136e0fc1d9f1a9`
- replay metrics: `d9c0d6c41b74392b123b11e770bc122aba2ec30a835b45c157a21ad96fd14d7a`
- replay run manifest: `26ebab679d112e07c5bb364b8799cffc24b573c4abc6a71623705a4cfe962e9f`
- replay summary: `44c0f483117a543cf11984c51cb136f872abef3761d17aa6b2a5556d520b3120`
