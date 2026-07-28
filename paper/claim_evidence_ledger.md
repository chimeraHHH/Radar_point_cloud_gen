# Claim-Evidence Ledger

> Updated: 2026-07-28 19:30 CST
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
| C7 | A RaLD-structured arbitrary query field improves dense geometry without losing long-range coverage. | G1D: occupied/empty arbitrary queries, exact mixed-latent topology, per-layer Full-RAED condition, coarse-to-refine fixed-10k inference, control comparison, gradients, geometry, confidence, and coverage gates. | Original G1/G1B failed and RH1/RH2 were skipped. G1C produced no scientific result. Two G1D runs were invalidated before gate completion for source-parity, scale, label-leakage, and control-pairing defects. G1D v2 source `4c6150cd` passed expanded preflight and is training; no scientific G1D result exists yet. | Rerouted to G1D v2 | “We evaluate a RaLD-structured query field under an independent geometry gate.” |
| C8 | RaLD's central latent-diffusion mechanism can model dense physical point states when query allocation is independently validated. | G1E/G3L: source-faithful `512 x 32` target posterior, coordinate-only implicit decoder, 24-layer Full-RAED-conditioned EDM, official 18-step sampler, condition-shuffle control, and no best-of-k. | G1E-D0 failed: proposal support changed R1-fidelity Chamfer from `10.9985` to `10.7680 m` and R1-KRadar from `9.9612` to `11.4948 m`, far outside the frozen gate. Independent occupancy VAE/EDM training is closed. The structurally verified physical G3L remains locked behind a passing geometry parent. | D0 failed; downstream mechanism locked | “RaLD's independent occupancy latent path did not pass our K-Radar representation gate; no latent-diffusion performance claim is made.” |
| C9 | Separating occupancy training queries from radar-proposal inference queries reduces outliers while retaining fixed-count dense geometry. | G1D: 6.25% occupied and range-stratified reliable-empty queries; RaLD classwise `0.1/1.0` BCE; 32k coarse radar proposals; 2.5k selection; 10k local refinement; absolute and paired control gates over three seeds. | G1D v2 source `4c6150cd` passed expanded preflight; Stage A is running. Earlier runs are audit-only and cannot support the claim. | New independent gate | “We test coarse-to-refine radar query fields as an independently named alternative to failed occupancy ranking.” |
| C10 | Full-RAED radar-observable state generation jointly produces fixed-count dense geometry, a circular pointwise Doppler distribution, and confidence under spatial and temporal physical closure. | A geometry parent passing the frozen gate; generated Doppler rather than measured-value attachment; Cube reprojection; displacement-Doppler consistency; Radar-Mamba/RadarMP/DoppDrive controls; three seeds and untouched test. | Literature boundary and four Stage-0 geometry candidates frozen on 2026-07-28. No geometry parent has passed, so Doppler and temporal claims remain locked. | Pending, scoped novelty only | “We target joint radar-observable state generation; existing multi-frame enhancement and aggregation are explicit related-work baselines, not claimed as absent.” |
| C11 | The frozen G1D measurement proposal pool contains enough support for a learned allocation repair. | G1F-F0 GT coverage oracle over exact 32k-to-10k support under the complete geometry gate. | Oracle source `ca60d76` failed Chamfer, completeness, far completeness, and duplicate gates; only outlier passed. | Rejected | “The frozen proposal pool did not support a selector-only repair; balanced-transport training on that pool was not run.” |
| C12 | Wide continuous spatial queries or Doppler-warped history provide support that is absent from the frozen 32k current-frame pool. | Corrected G1T temporal support; corrected global-support R-A1 initial-query diagnostic; no learned selector claim; complete far-target frame accounting. | Legacy G1T is archived but used a censored far metric. Corrected G1T is running. Initial G1A/G1R implementations are blocked by independent code review and are being repaired before H200 preflight. | Pending, parallel support diagnostics | “We separately test temporal and wide-query support before training a new generator.” |

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
| R8 | “Archived `8.x m` far-completeness values represent all far-target validation frames.” | The old evaluator omitted frames that had 60--120 m GT but no same-bin prediction. Corrected G1D epoch-15 covers 23/23 far-target frames and gives `46.9407 m`, versus the censored `8.1239 m`. | Do not cite any old far-completeness value. Re-evaluate every parent/control with source `1561ac3` or later before freezing a new absolute far gate. |

## Gate-to-Artifact Map

| Gate | Authoritative artifact or expected path | Frozen decision output |
|---|---|---|
| G0 | `artifacts/g0/g0_repair_100_pass_2026-07-18.md` | Passed |
| Static audit | `artifacts/g2/static_doppler_failure_recovery_2026-07-18.md` | Failed; E5 removed |
| G1 | `artifacts/g1/g1_bounded_recovery_failure_2026-07-19.md`; archived comparison `artifacts/g1/g1_bounded_recovery_comparison_0e5fe843.json`; SHA-256 `ef8eb17b...` | Failed and closed; RAE-Max also failed the fixed parent gate |
| G1B | `artifacts/g1/g1b_final/g1b_screen_3fa7ae88.json`; Stage B and downstream queue summaries in the same directory | Failed: no Stage A survivor; Stage B/RH1/RH2/G2R/G3R skipped |
| G1C | `docs/g1c_rald_guided_query_protocol.md`; server logs under `launch_logs/08c0a01` and `launch_logs/0145004` | No scientific result: first queue failed before child launch; repaired queue was stopped while waiting after source audit |
| G1D | `artifacts/g1/g1d_epoch150_d1_d2_d3_22f4aa6.json`; `artifacts/g1/g1d_formal_failure_and_factors_2026-07-29.md`; protocol, source map, invalid-run records, and preflight artifacts | Formal 150-epoch Stage A completed. Epoch 15 remained selected; epoch 150 worsened corrected median Chamfer from `4.4609` to `5.4221 m`, outlier from `17.8875%` to `24.8396%`, and duplicate from `26.9721%` to `30.7279%`. D1/D2/D3 authorize neither offset clipping nor same-pool ranking repair. G1D is closed as a geometry parent. |
| G1E | `docs/g1e_rald_latent_edm_protocol.md`; `artifacts/g1/g1e_d0_da6f8a5f.json`; archived R1 checkpoints and gate records under `artifacts/baselines/rald/` | D0 failed both arms while preserving source, frame, query counts, latent-only decoding, and the test lock. E1/E2 are not authorized. |
| G1F | `artifacts/idea/pre_idea_drafts/g1f_balanced_transport.md`; `artifacts/g1/g1f_f0_ca60d76.json` | Explicitly unattainable GT coverage oracle failed the complete geometry gate; F1 selector training is closed. |
| Corrected G1D control | `artifacts/g1/g1d_epoch15_corrected_geometry_control_1561ac3.json`; SHA-256 `56a0343745f29bb7ecb2f9176b4527435db97f14bd510ed2f03e5d0c9d0e3fd7` | Frozen epoch-15 EMA, same 24 validation frames and data bytes; completeness median `3.5637 m`; 23/23 far-target frames, far mean `46.9407 m`. This is the only matched Stage-0 control. |
| G1G | `artifacts/g1/g1g_formal_stage0_decision_c2a0ccb.json`; `artifacts/g1/g1g_formal_failure_2026-07-29.md`; earlier smoke artifacts | Formal 20-epoch H200 Stage-0 completed and failed. Anti-bypass gradients passed, but condition shuffle was `-0.1211%`, duplicate `70.8108%`, and outlier `84.6854%`; the route is closed without an epoch extension. |
| G1T | `artifacts/g1/g1t_temporal_proposal_corrected_1561ac3.json`; `artifacts/g1/g1t_corrected_decision_1561ac3.json` | Corrected evaluator covers all required far-target frames. Doppler history changed geometry by less than `0.05%` while all arms had about `15.9 m` Chamfer and `87.6%` outliers; no learned follow-up is authorized from this proposal mechanism. |
| P-RF | `docs/polar_rectified_flow_target_lifting_protocol.md`; `artifacts/g1/p_rf_target_lifting_failure_2026-07-29.md`; earlier max-frame preflight `artifacts/g1/p_rf_preflight_0c433a7.json` | The max-frame model preflight passed, but the real-distribution target adapter failed: only 47/57 sparse train/validation frames reached exact 10k under observed-cell and 5 cm spacing constraints. Long training is not authorized; no general rectified-flow failure claim is made. |
| G2/G3 | Original `formal_28d69a0_g2_g3` queue | Not unlocked and permanently closed after G1 failure; any successor must be named G2R/G3R |
| G4 data | `artifacts/g4/g4_temporal_manifest_a7d06db1.json`; server download summary `g4_temporal_download_w48/manifests/summary.json` | 45/45 requested sequences downloaded, no failures, about 601 GB; final source-bound CRC audit pending |
| G4/G4R | Old server route `formal_206ffeb_g4` is closed. The source- and hash-bound RaLD-native cache/train/preflight/baseline/rollout/compare/queue chain is implemented. | Old queue must not train; new G4R queue awaits passing G3R and 45/45 verified sequences |
| P5 | Server: `launch_logs/206ffeb/p5_queue.log` and `p5_download_gate.log`; G4 gate recorded `PARENT_FAILED:G1` | Stale waiting queue stopped; no P5 output or test access; test remains locked until a G1D-derived G4 family freezes |
| RaLD hybrid | `artifacts/baselines/rald/anchor_hybrid_rh0_a1c862a.json`; `artifacts/g1/g1b_final/rh1_queue_summary_94eba97e.json`; `artifacts/g1/g1b_final/rh2_queue_summary_94eba97e.json`; `artifacts/g1/g1b_final/g2r_g3r_summary_94eba97e.json` | Structural RH0 passed, but empirical RH/G2R/G3R were skipped after G1B no-go; no method claim |

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
