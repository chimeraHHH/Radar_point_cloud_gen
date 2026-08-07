# VRH-F0 paired capacity decision

> Date: 2026-08-07  
> Terminal status: `vrh_f0_capacity_no_go`  
> Scientific exit code: `2`  
> Test/validation/future/Cube/CFAR access: false

## Frozen identity

- execution source: `380f3eaef336bee5a019eff5121eeacceb6e2066`
- protocol freeze source: `d034dcaba1d3c015239a9623b281c5b01fafe2c6`
- protocol SHA-256: `78c62f29e8f325287d79eefea141bf2b014f1388448d4ed1a3b5365071cd7946`
- formal support: `512 x 214 x 74`, 8,108,032 cells, 15,836 rays
- support SHA-256: `c1819265c69a402939262ec36b5a5be1a3cb25d6337b7376a6316e40d88e4927`
- GPU: physical GPU0, NVIDIA H200 NVL, UUID `GPU-74b9b73f-c405-55bc-bf76-f8aa85d1e7fe`
- elapsed time: 325.073 s
- peak host RSS: 1,913,389,056 bytes
- peak CUDA allocated/reserved: 95,247,360 / 115,343,360 bytes

The implementation and resource gates passed. The formal run completed both
arms on all 12 frozen train-only frames. The result is therefore a scientific
no-go rather than an implementation- or resource-invalid run.

## Formal paired result

| Frame | Split role | Sequential CD (m) | Flat CD (m) | Sequential count | Sequential strata | First/later gate |
|---|---|---:|---:|---:|---|---|
| 1:232 | fit | 1.5405 | 0.1267 | 10,000 | fail | fail |
| 9:440 | fit | 1.0889 | 0.0698 | 10,000 | fail | fail |
| 21:402 | fit | 0.4890 | 0.0962 | 10,000 | fail | fail |
| 27:403 | fit | 0.3087 | 0.1026 | 10,000 | fail | fail |
| 29:399 | fit | 0.2643 | 0.1085 | 10,000 | fail | fail |
| 35:403 | fit | 0.3613 | 0.0564 | 10,000 | fail | fail |
| 47:514 | fit | 0.1375 | 0.1311 | 4,359 | pass | fail |
| 58:404 | fit | 0.1695 | 0.1645 | 1,602 | pass | fail |
| 5:429 | unseen train | 1.0433 | 0.0803 | 10,000 | fail | fail |
| 26:403 | unseen train | 0.6376 | 0.0678 | 10,000 | fail | fail |
| 42:408 | unseen train | 0.3772 | 0.1391 | 10,000 | fail | fail |
| 50:405 | unseen train | 0.1752 | 0.1708 | 9,993 | pass | fail |

Both arms had zero 2 m outliers on every frame. Aggregate gate counts were:

| Gate | Sequential frontier | Flat exposure |
|---|---:|---:|
| CD <= 0.8 m | 9/12 | 12/12 |
| outlier <= 5% | 12/12 | 12/12 |
| exact-10k and all structural checks | 9/12 | 9/12 |
| all target-bearing strata retained | 3/12 | 4/12 |
| all first/later classes passed | 0/12 | 0/12 |
| passed first/later classes | 1/66 | 1/66 |

Sequential-frontier mean/median CD was `0.5494/0.3692 m`; flat-exposure
mean/median CD was `0.1095/0.1055 m`. Sequential renewal therefore worsened,
rather than improved, the geometry of the shared fitted mark field.

## Mechanism verdict

Renewal activity is certified:

- both arms used the same canonical stream hash on all 12 frames;
- selected-cell hashes differed on at least one frame, in fact on all 12;
- the sequential arm accepted depth-at-least-two events on all frames.

Renewal utility is not certified:

- the decision arm did not pass all per-frame gates;
- no corresponding gate was passed by the decision arm and failed by the
  flat control;
- the flat control was strictly stronger on aggregate geometry and retained
  one more frame's target-bearing strata;
- three sparse frames could not produce exact 10k points after the true 5 cm
  check, despite the fitted fanout.

This closes the frozen GT-fitted `2R x 2A x 2E` renewal/frontier recipe. It does
not prove that all ray-native renewal models are impossible. It does prohibit
training a target-free renewal model from this capacity result and does not
unlock Doppler, cycle, temporal, validation, or test evaluation.

## Route

Per the preregistered decision boundary, the next eligible geometry experiment
is a separately frozen sparse ray-range partial-transport hard-rounding oracle.
It must couple set allocation without a dense candidate-by-target matrix, must
remain train-only and target-free at deployment, and must pass a zero-training
hard-rounding/resource gate before any learned transport model is implemented.

## Artifacts

- authoritative full report on H200: `preflight.json`, 79,156,059 bytes,
  SHA-256 `5393576d905bc8344384fcf4c9e194334c26ee1b3ed58e3e61460142181c5054`
- Git-sized compact report: `preflight_compact.json`, with only the large
  per-ray assignment payloads removed, SHA-256
  `944b3e338823498ae61c545b067bf92dbd0d3600f1aef6aa099eb7dc498c1eed`
- formal run log: `run.log`, SHA-256
  `5d46a21d60940682ffce65aabda342c9baaa3ed72e10cd3c848d58cdb991eb94`
