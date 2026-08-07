# Q-Local-F0 source-bound preflight decision

Date: 2026-08-07

## Terminal decision

`qlocal_f0_capacity_no_go`

Q-Local-F0 training is not authorized. The scorer, optimizer, and learned
checkpoint were not created. This is a valid parent-field/output-contract
capacity no-go, not a learned geometric-risk result.

## Bound execution

- source: `bf1a166c48be60f82c5283b5937edf0cf6ad64c2`;
- protocol: `g1_qlocal_distributional_risk_f0_preflight_v1`;
- Fresh-WCE-20 checkpoint SHA-256:
  `c0ebb4c1509c3d36b6834bd39977cb1cda2fcaff1747eabc97ebb41af0f06ce4`;
- physical GPU: H200 index 2,
  UUID `GPU-000b6236-3632-a001-9667-1f02cbb61c8b`,
  PCI `00000000:D1:00.0`;
- elapsed: `444.5814 s`;
- peak allocated/reserved CUDA bytes: `2432197120/2657091584`;
- report SHA-256:
  `b1b740dff2515df274dfbf6fdb15ece53c1c3129d7847c6e803abc568c97fef4`;
- log SHA-256:
  `95863701e7da651c4f8e05b0dc48f4a54b8e2826fdbb07c4b4071a6d7924c0d3`.

The run accessed all and only the frozen eight fit and four unseen-train frame
identities. Validation, test, future Cube, best-of-k, and target-at-inference
access remained false.

## Capacity result

Ten of twelve frame-wise GT-nearest exact-10k oracles passed. Two fit frames
failed the immutable `CD <=0.8 m` and `outlier <=5%` gate:

| frame | target count | Chamfer | completeness | outlier | min spacing |
|---|---:|---:|---:|---:|---:|
| `47:514` | 1,404 | `1.4250 m` | `0.0878 m` | `7.0%` | `0.050004 m` |
| `58:404` | 285 | `2.2598 m` | `0.1057 m` | `20.0%` | `0.050008 m` |

Every frame still passed exact 10,000, `8000/1700/300`, unique-row, no
copy/padding/jitter, and true 5 cm spacing checks. The no-go is therefore not an
export-structure failure.

## Mechanism checks that passed

- parameter count: `380818`, below the 2M ceiling;
- all six gradient groups were finite and nonzero over two updates;
- losses were finite (`5.3242021`, `5.3219166`);
- same-coordinate wrong-Cube kept candidate coordinates bit-identical while
  changing local spectrum, global condition, and the selection score;
- Fresh-WCE state SHA-256 was unchanged before and after the run;
- source, manifest, split, normalization, raw Cube, target cache, target tensor,
  checkpoint, metrics, run manifest, physical GPU, and output transaction were
  bound in the terminal report.

These checks validate the implementation boundary but cannot override the
capacity prerequisite.

## New failure localization

Frame `58:404` has target radial range `3.071--27.877 m` with counts
`285/0/0` in `[0,30)/[30,60)/[60,120)` m. The fixed exporter nevertheless
forces `1700+300=2000` predictions above 30 m. By the reverse triangle
inequality, every such prediction is more than 2 m from every target, so the
hard-quota contract alone imposes a 20% outlier lower bound on this frame.

This observation does not retroactively change the Q-Local protocol or result.
It triggers the separately frozen 76-train-frame audit in
`docs/fixed_range_quota_feasibility_audit.md`. A variable multi-return
ray-hazard fallback must not inherit the same hard quotas unless that audit
passes.
