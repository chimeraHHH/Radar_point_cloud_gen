# Claim-Evidence Ledger

> Updated: 2026-07-19 21:15 CST
> Rule: a claim is paper-eligible only when its frozen gate is complete and the authoritative artifact is recorded here.

## Claim Matrix

| ID | Candidate claim | Required evidence | Current evidence | Status | Allowed wording now |
|---|---|---|---|---|---|
| C0 | The K-Radar Cube-to-dense data path has reliable axes, synchronization, timing choice, and radar-observable targets. | G0 100-frame audit, all checks, no frame errors, frozen hashes. | `artifacts/g0/g0_repair_100_pass_2026-07-18.md`; 100/100, 11/11. | Passed | “The development cohort passed the preregistered data audit.” |
| C1 | Preserving the full 64-bin Doppler spectrum in the early occupancy encoder improves dense geometry over a matched Doppler-collapsed encoder. | E2 vs E1, three seeds, scene-first paired CI; Doppler-sensitive endpoint improves and overall Chamfer does not regress beyond tolerance. | Bounded recovery failed: Full-RAED Chamfer `3.1024 m` vs RAE-Max `2.9306 m` (`+5.86%`, 95% CI `+0.78%` to `+14.69%`); no Doppler-sensitive endpoint improved confidently. | Rejected | “Early Full-RAED fusion did not improve geometry under the frozen protocol.” |
| C2 | Per-point circular Doppler distributions are more useful and faithful than scalar regression. | G2R: matched cycle-free scalar/distribution RaLD arms, same-position final-coordinate direct Cube query, NLL/KL, circular W1, calibration, CD-Doppler, geometry tolerance, and anti-collapse gates. | Training/comparison/queue code exists but no eligible G2R run has completed. The rejected static-ego PCE convention is excluded. | New gate required | “The candidate parameterizes Doppler as a circular distribution and evaluates it against scalar and direct-query controls.” |
| C3 | Generated points can explain the measured Cube through a differentiable cycle without confidence or coverage collapse. | G3R: four arms forked from the exact passing cycle-free G2R checkpoint; local/marginal/full cycle, two metric classes, confidence, coverage, ECE, saturation, renderer, and robustness gates. | Training/comparison/robustness/queue code exists but no eligible G3R run has completed. | New gate required | “The candidate includes a differentiable point-to-Cube cycle with anti-collapse checks.” |
| C4 | A warped historical prediction improves temporal consistency while the current Cube refreshes geometry and Doppler. | G4R: zero-init identity; matched RaLD token/latent/query fusion; single-frame and history-aggregation controls; current-frame geometry and circular Doppler; ego-aligned matching/flicker; 25-step confidence/coverage; three seeds. | Manifest 10/10; data download is active. The old G4 route is closed. RaLD-native G4R core and checkpoint contract are implemented but preflight/training remain locked behind passing G3R. | Pending, reroute implemented | “We study historical evidence at three RaLD representation levels while retaining the current Cube.” |
| C5 | The frozen model improves object radial-velocity estimation and generalizes across operating slices. | G4 family frozen; untouched P5 test, downstream report, scene-first uncertainty, slices. | Test manifest intentionally absent. | Pending | No result claim. |
| C6 | The system has practical H200 latency and memory. | Matched CUDA benchmark on frozen models and fixed point count. | Queue implemented, not released. | Pending | No efficiency claim. |
| C7 | RaLD mixed set latents improve anchor-level geometry and physical attributes without losing long-range coverage. | RH1 one-frame gradient/anti-collapse gate, then RH2 development comparison against an independently authorized frozen occupancy parent. | RH0 passed on H200. Original G1 and RAE-Max parent routes both failed; RH now waits for an independently passing G1B Stage B candidate. Full-RAED tokens still carry all 64 bins into 512 mixed latents, and the parent remains frozen. No trained RH comparison exists. | Pending | “We implement a Full-RAED-conditioned RaLD anchor-latent candidate behind an independently gated geometry parent.” |

## Rejected or Restricted Claims

| ID | Rejected/restricted statement | Evidence | Required treatment |
|---|---|---|---|
| R1 | “Start-time LiDAR deskew improves K-Radar alignment.” | Formal train evidence selected no-deskew; start-minus-none was negative. | State no-deskew was frozen; list scan-origin convention as a limitation. |
| R2 | “An analytic static-Doppler prior is valid on the current cohort.” | Train selected positive-ego, but validation error `1.020193 m/s` was worse than random `0.966296 m/s`; bounded SNR recovery failed. | E5 omitted. May report as a negative result only. |
| R3 | “P5 sign calibration validates the static physics prior.” | P5 sign-only mapping is restricted to descriptive box range-rate alignment. | Record `sign_only_calibration=true` and `physics_prior_claim_enabled=false`. |
| R4 | “TruckScenes FlowRadar results validate the K-Radar Cube model.” | Different input, target, data, and architecture. | Use only as temporal design motivation. |
| R5 | “The method is the first radar densification model.” | Existing Cube/spectrum-to-point generation work. | Claim only the verified combination and complete a submission-time literature rescan. |
| R6 | “G4 is successful because temporal output is smoother.” | Static or copied clouds can reduce flicker. | Require current-frame geometry, Doppler refresh, coverage, and rollout stability jointly. |
| R7 | “The official or matched RaLD checkpoint is a competitive K-Radar main baseline.” | The official checkpoint is ColoRadar/intensity-only and not protocol-matched. The from-scratch matched AE failed its frozen one-frame Chamfer gate after one bounded repair (`9.1444 m` vs `<= 5.0 m`). | Cite RaLD as related work and architecture motivation. Preserve the matched run as a no-go; do not train its EDM or report it as a headline quantitative baseline. |

## Gate-to-Artifact Map

| Gate | Authoritative artifact or expected path | Frozen decision output |
|---|---|---|
| G0 | `artifacts/g0/g0_repair_100_pass_2026-07-18.md` | Passed |
| Static audit | `artifacts/g2/static_doppler_failure_recovery_2026-07-18.md` | Failed; E5 removed |
| G1 | `artifacts/g1/g1_bounded_recovery_failure_2026-07-19.md`; archived comparison `artifacts/g1/g1_bounded_recovery_comparison_0e5fe843.json`; SHA-256 `ef8eb17b...` | Failed and closed; RAE-Max also failed the fixed parent gate |
| G1B | Server Stage A: `formal_3fa7ae8_g1b_screen/g1b_screen_3fa7ae88.json`; Stage B is authorized only by that report | Independent branch waiting for an H200; does not reopen G1 |
| G2/G3 | Original `formal_28d69a0_g2_g3` queue | Not unlocked and permanently closed after G1 failure; any successor must be named G2R/G3R |
| G4 data | `artifacts/g4/g4_temporal_manifest_a7d06db1.json`; download summary | Manifest passed; download pending |
| G4/G4R | Old server route `formal_206ffeb_g4` is closed. The source- and hash-bound RaLD-native cache/train/preflight/baseline/rollout/compare/queue chain is implemented. | Old queue must not train; new G4R queue awaits passing G3R and 45/45 verified sequences |
| P5 | Server: `launch_logs/206ffeb/p5_queue.log` and `p5_download_gate.log`; both wait for `formal_206ffeb_g4/g4_queue_summary_206ffeba.json` | Test locked until G4 family freeze |
| RaLD hybrid | `artifacts/baselines/rald/anchor_hybrid_rh0_a1c862a.json`; `artifacts/baselines/rald/anchor_hybrid_training_chain_2468acd.json`; `artifacts/baselines/rald/anchor_hybrid_late_fusion_d69be57.json`; protocol `docs/rald_anchor_hybrid_protocol.md` | RH0 passed; G1B-to-RH provenance routing and independent RH/G2R/G3R chain implemented; RH0.5/RH1/RH2/G2R/G3R remain result-pending |

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
