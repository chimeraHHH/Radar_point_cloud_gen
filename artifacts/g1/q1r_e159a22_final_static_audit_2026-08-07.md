# Q1-R `e159a22` Final Static Audit

Date: 2026-08-07

## Disposition

- The replay-parent no-go remains unchanged. Four of six preregistered endpoint
  equivalence checks fail independently of the implementation gaps below.
- Q1-R tiny, conditional Q2, Doppler, cycle, temporal, and test stages remain
  unlaunched and locked.
- Commit `e159a22c5343e81c0cf29351da07290a686edba5` and its certificate may be
  cited as the implementation that recorded this no-go, but must not be reused
  as a general authorization baseline for a future quality-ranking run.
- The quality-head hypothesis remains untested. None of the findings below is a
  quality-model result.

The final independent review was static and read-only. It did not modify code,
rerun the 20-epoch replay, or access test data.

## P1 Findings

### 1. The GT-nearest diagnostic is not independently bound to raw inputs

The certifier recomputes diagnosis aggregates from the per-frame JSON, but the
oracle geometry and its aggregate can still be changed together. The current
source/input binding covers top-level frozen hashes and the current-confidence
control chain; it does not independently reconstruct the GT-nearest export and
geometry from the raw target and replay candidate pool. The positive test
fixture also authorizes a reduced diagnosis schema without the production
`current_inputs`, GT export report, or GT export hashes.

Relevant implementation:

- `code/scripts/certify_rald_wce_replay_parent.py:518`
- `code/scripts/certify_rald_wce_replay_parent.py:563`
- `code/scripts/certify_rald_wce_replay_parent.py:649`
- `code/tests/test_rald_wce_replay_parent.py:284`

### 2. Completed quality evaluations do not revalidate the frozen eight-frame set

`recompute_quality_metrics()` rebuilds aggregate values from frame reports, but
does not require exactly eight unique frames in the frozen identity order or
validate the frozen wrong-condition pairing. A self-consistent document with a
dropped hard frame or duplicated easy frame could therefore pass the current
resume validator.

Relevant implementation:

- `code/scripts/train_rald_wce_quality_tiny.py:1083`
- `code/scripts/train_rald_wce_quality_tiny.py:1362`

### 3. Terminal resume trusts historical consecutive-pass state

The terminal validator recomputes only the final evaluation and obtains the
prior consecutive-pass count from the final evaluation checkpoint. It does not
replay the complete immutable `100, 200, ..., final` evaluation chain. A forged
historical counter could therefore turn one passing evaluation into an apparent
two-pass early stop.

Relevant implementation:

- `code/scripts/train_rald_wce_quality_tiny.py:1390`

## P2 Findings

### 4. Capacity-failure initialization has a non-atomic branch

The normal initialization path stages and atomically renames the run directory.
The `ExactExportCapacityError` branch creates the final output directory before
writing the terminal failure JSON. A crash between those operations can leave an
empty directory that neither a fresh run nor `--resume` accepts.

Relevant implementation:

- `code/scripts/train_rald_wce_quality_tiny.py:1794`

### 5. Physical GPU provenance lacks UUID or PCI bus identity

The runtime guard enforces `CUDA_DEVICE_ORDER=PCI_BUS_ID`, one visible device,
`CUDA_VISIBLE_DEVICES` equal to `0` or `2`, and the exact H200 model name. It
does not record an independently checked PCI bus ID or GPU UUID, so the physical
GPU0/GPU2 claim remains an environment-policy record rather than complete device
provenance.

Relevant implementation:

- `code/scripts/diagnose_rald_wce_failure_factors.py:63`
- `code/scripts/train_rald_wce_quality_tiny.py:362`
- `code/scripts/certify_rald_wce_replay_parent.py:597`

### 6. Regression tests do not exercise the full transaction and tamper surface

The trainer-certificate integration test substitutes the real certifier. The
staging test manually renames prepared files, and the terminal test covers only
one final artifact. The suite lacks dropped/duplicated evaluation-frame cases,
forged historical pass chains, the capacity-failure crash path, and an
end-to-end real-certifier call.

Relevant implementation:

- `code/tests/test_rald_wce_quality.py:360`
- `code/tests/test_rald_wce_quality.py:664`
- `code/tests/test_rald_wce_quality.py:679`

## Control Status After Audit

| Control | Status |
|---|---|
| Formal metric and condition-effect recomputation | Closed |
| Current-confidence hash/geometry control | Closed for this archived no-go |
| GT-nearest raw-input binding | Open, P1 |
| Frozen eight-frame evaluation contract | Open, P1 |
| Full terminal evaluation-chain validation | Open, P1 |
| Normal atomic initialization | Closed |
| Capacity-failure atomic initialization | Open, P2 |
| H200 runtime guard | Closed |
| Physical GPU UUID/PCI provenance | Open, P2 |
| Realistic adversarial transaction tests | Incomplete, P2 |

## Requirement For Any Future Branch

A newly named continuous-quality branch must fail closed until it adds all of
the following: independent raw-input reconstruction or equivalent external
binding for the GT diagnostic; exact frozen eight-frame identities, order, and
wrong-condition pairs; full evaluation-chain recomputation; atomic capacity
failure commits; GPU UUID or PCI bus provenance; and adversarial tests covering
each crash and tamper boundary. These repairs cannot retroactively authorize the
current replay under the already failed endpoint-equivalence gate.
