# Claim-Evidence Ledger

> Updated: 2026-07-19 01:35 CST
> Rule: a claim is paper-eligible only when its frozen gate is complete and the authoritative artifact is recorded here.

## Claim Matrix

| ID | Candidate claim | Required evidence | Current evidence | Status | Allowed wording now |
|---|---|---|---|---|---|
| C0 | The K-Radar Cube-to-dense data path has reliable axes, synchronization, timing choice, and radar-observable targets. | G0 100-frame audit, all checks, no frame errors, frozen hashes. | `artifacts/g0/g0_repair_100_pass_2026-07-18.md`; 100/100, 11/11. | Passed | “The development cohort passed the preregistered data audit.” |
| C1 | Preserving the full 64-bin Doppler spectrum improves dense geometry over a matched Doppler-collapsed encoder. | E2 vs E1, three seeds, scene-first paired CI; Doppler-sensitive endpoint improves and overall Chamfer does not regress beyond tolerance. | First formal G1 failed: E2 Chamfer +28.3% and far completeness +232.6% vs E1. One zero-initialized residual-spectrum redesign is deployed at `0e5fe84`; H200 preflight is queued behind unrelated compute. | Recovery pending | “We test whether the full Doppler spectrum improves geometry.” |
| C2 | Per-point circular Doppler distributions are more useful and faithful than scalar regression. | E4 vs E3 on NLL/KL, circular W1, calibration, PCE, CD-Doppler, and geometry tolerance. | `0e5fe84` G2/G3 queue waits for the recovery G1 comparison and exits without training unless G1 passes. | Pending | “We parameterize Doppler as a circular distribution.” |
| C3 | Generated points can explain the measured Cube through a differentiable cycle without confidence or coverage collapse. | C0-C3 ablation; cycle metric gain plus independent geometry/physical/downstream gain; confidence, coverage, ECE, and saturation pass. | Renderer tests and protocol exist; formal G3 not run. | Pending | “We introduce a differentiable point-to-Cube cycle and evaluate anti-collapse criteria.” |
| C4 | A warped historical prediction improves temporal consistency while the current Cube refreshes geometry and Doppler. | T0/T3/T4-T6, current-frame accuracy, radial error, flicker, refresh, 25-step rollout, three seeds. | Manifest 10/10; 2,160-frame data download active. | Pending | “We study a current-observation temporal extension.” |
| C5 | The frozen model improves object radial-velocity estimation and generalizes across operating slices. | G4 family frozen; untouched P5 test, downstream report, scene-first uncertainty, slices. | Test manifest intentionally absent. | Pending | No result claim. |
| C6 | The system has practical H200 latency and memory. | Matched CUDA benchmark on frozen models and fixed point count. | Queue implemented, not released. | Pending | No efficiency claim. |

## Rejected or Restricted Claims

| ID | Rejected/restricted statement | Evidence | Required treatment |
|---|---|---|---|
| R1 | “Start-time LiDAR deskew improves K-Radar alignment.” | Formal train evidence selected no-deskew; start-minus-none was negative. | State no-deskew was frozen; list scan-origin convention as a limitation. |
| R2 | “An analytic static-Doppler prior is valid on the current cohort.” | Train selected positive-ego, but validation error `1.020193 m/s` was worse than random `0.966296 m/s`; bounded SNR recovery failed. | E5 omitted. May report as a negative result only. |
| R3 | “P5 sign calibration validates the static physics prior.” | P5 sign-only mapping is restricted to descriptive box range-rate alignment. | Record `sign_only_calibration=true` and `physics_prior_claim_enabled=false`. |
| R4 | “TruckScenes FlowRadar results validate the K-Radar Cube model.” | Different input, target, data, and architecture. | Use only as temporal design motivation. |
| R5 | “The method is the first radar densification model.” | Existing Cube/spectrum-to-point generation work. | Claim only the verified combination and complete a submission-time literature rescan. |
| R6 | “G4 is successful because temporal output is smoother.” | Static or copied clouds can reduce flicker. | Require current-frame geometry, Doppler refresh, coverage, and rollout stability jointly. |

## Gate-to-Artifact Map

| Gate | Authoritative artifact or expected path | Frozen decision output |
|---|---|---|
| G0 | `artifacts/g0/g0_repair_100_pass_2026-07-18.md` | Passed |
| Static audit | `artifacts/g2/static_doppler_failure_recovery_2026-07-18.md` | Failed; E5 removed |
| G1 | `artifacts/g1/g1_formal_failure_decision_2026-07-19.md`; first comparison in `formal_a7d06db`; recovery expected at server `formal_0e5fe84_g1_recovery/g1_comparison_0e5fe843.json` | First run failed; one bounded recovery queued on H200 |
| G2/G3 | Server: `formal_0e5fe84_g2_g3/g2_g3_queue_summary_0e5fe843.json`; parent comparison and runs are resolved from `formal_0e5fe84_g1_recovery/` | Hard-gated on recovery G1; E5 remains omitted |
| G4 data | `artifacts/g4/g4_temporal_manifest_a7d06db1.json`; download summary | Manifest passed; download pending |
| G4 | Expected server queue will be rebound to the frozen `0e5fe84` G2/G3 summary after that gate closes; no current formal queue is authoritative | Data download active; training pending G2/G3 |
| P5 | Expected server queue will be rebound only after G4 family freeze | Test locked |

## Figure and Table Evidence Contracts

| Paper item | Claim IDs | Minimum content before release |
|---|---|---|
| Figure 1 method overview | C2, C3, C4 | Method structure only; mark temporal path optional until G4 passes. |
| Figure 2 spectrum and renderer | C2, C3 | Fixed frames, selected locations, target/predicted spectrum, rendered Cube, confidence. |
| Figure 3 qualitative main result | C1-C3 | Same frames and color ranges for all methods; include worst failures. |
| Figure 4 mechanism analysis | C3 | Cycle metrics, geometry, confidence, coverage, ECE, saturation together. |
| Table 1 geometry | C1 | CFAR, E1, E2, three-seed scene-first uncertainty. |
| Table 2 Doppler/cycle | C2, C3 | Q0, E3, E4, C0-C3; no E5 success row. |
| Table 3 temporal | C4 | T0/T3/T4-T6 and rollout; main text only if G4 passes. |
| Table 4 frozen test | C5, C6 | Downstream velocity, slices, latency, memory; no model selection. |

## Reviewer Objection Map

| Objection | Required answer | Blocking gate |
|---|---|---|
| “This is RAE densification with a velocity head.” | Matched Full-RAED vs RAE-Max, scalar vs distribution, and independent cycle ablation. | G1-G3 |
| “LiDAR points do not have true Doppler.” | Cube-local spectrum supervision, radar observability mask, confidence, Q0 query baseline. | G2 |
| “Cycle loss cheats by suppressing confidence.” | Confidence/coverage retention, ECE, offset saturation, fixed point count. | G3 |
| “Temporal gains are history accumulation.” | Current Cube present in every arm, T3 aggregation baseline, Doppler refresh and current-frame metrics. | G4 |
| “Results are development-set artifacts.” | Eight untouched test sequences and scene-first uncertainty. | P5 |
| “Full Cube encoding is impractical.” | Parameter, latency, memory, and fixed-output benchmark. | P5 |

## Update Procedure

1. Copy only final aggregate values from an authoritative artifact.
2. Record commit, manifest/split hashes, seeds, and pass/fail decision.
3. Update the allowed wording before updating the abstract or contribution list.
4. Preserve failed branches; do not delete rows or retroactively weaken gates.
5. Never use test results to reopen model or family selection.
