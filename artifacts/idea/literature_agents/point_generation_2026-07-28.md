# Cube -> fixed 10k point generation: transferable mechanisms and Stage-0 routes

> Date: 2026-07-28
> Scope: read-only literature and code audit; no scientific computation was run.
> Task boundary: this report concerns the single-frame geometry parent only.
> Doppler-distribution generation, Cube cycle closure, temporal consistency, and
> untouched-test evaluation remain locked behind a passing geometry family.

## 1. Executive decision

The present failure is not one generic "point completion" problem. Repository
evidence separates it into three coupled failures:

1. **support failure:** even an unattainable GT-assisted selector could not
   extract a passing 10k subset from the frozen 32k G1D proposal pool;
2. **allocation failure:** G1D spends too many output slots in crowded local
   regions, producing roughly 30% duplicates in active intermediate evidence;
3. **condition-path failure:** shuffling the global Full-RAED condition changes
   G1D geometry by far less than the frozen 1% threshold because current local
   Cube features provide a bypass.

This rules out a selector-only repair on the old 32k pool. It does **not** rule
out optimal transport, learned sampling, hierarchical point growth, implicit
fields, or conditional generation when they are attached to a new support
representation and a causal condition path.

The four model classes should be treated as mutually exclusive Stage-0
experiments:

| Class | What it changes | Main advantage | Main risk | Stage-0 priority |
|---|---|---|---|---|
| A. Direct coordinate regression | query/seed allocation and local growth | lowest implementation and H200 cost; G1G is already close | can still collapse and learn an object-shape prior rather than radar support | run/finish first as the cheapest causal test |
| B. Expanded candidate set + resampling | support domain and fixed-budget selection | explicit control of far mass and duplicates | cannot recover geometry absent from the expanded pool | run the no-train oracle before any learning |
| C. Implicit occupancy/density field + sampling | continuous support representation | queries the full radar frustum and decouples representation from 10k export | sparse labels and threshold/calibration can create empty-space mass | strongest deterministic replacement if C0 sampling is sound |
| D. Cube-projected conditional flow/diffusion | the complete point distribution | generates support beyond a fixed proposal pool and can force condition use at every step | highest cost and easiest route to hallucination or stochastic metric gaming | use flow matching as the first generative Stage-0 |

**Recommended frontier:** keep G1G/A as the low-cost direct-regression
diagnostic; immediately run the B0 expanded-support oracle; prepare C and D in
parallel, with C preferred over D for the first full implementation because
continuous RAE queries preserve radar observability more directly.

## 2. Repository-grounded failure diagnosis

### 2.1 Frozen objective

Every route must use the existing scene-held-out 76/24 split and export exactly
10,000 points. The authoritative final gate is:

| Metric | Required |
|---|---:|
| median Chamfer | `<= 2.50 m` |
| mean outlier fraction at 2 m | `<= 25%` |
| median completeness | `<= 0.65 m` |
| mean far completeness | `<= 8.0 m` |
| mean duplicate fraction within 5 cm | `<= 10%` |
| output count | exactly `10,000` |
| cross-scene learned-condition shuffle degradation | `>= 1%` |

No Stage-0 result may weaken these final thresholds. A screening threshold can
authorize a longer frozen run, but cannot support a paper claim.

### 2.2 What has already been falsified

The formal G1F-F0 oracle selected 10k points from the exact frozen 32k G1D pool
with GT access, yet obtained:

| Metric | G1F-F0 oracle | Gate |
|---|---:|---:|
| median Chamfer | `2.8863 m` | `<= 2.50 m` |
| mean outlier fraction | `24.853%` | `<= 25%` |
| median completeness | `1.6513 m` | `<= 0.65 m` |
| mean far completeness | `8.6533 m` | `<= 8.0 m` |
| mean duplicate fraction | `14.015%` | `<= 10%` |
| far-range GT recall from all proposals at 2 m | `17.80%` | diagnostic |

Therefore:

- hard top-k is not the sole cause;
- SampleNet, Sinkhorn, FPS, or blue-noise selection applied **only** to this
  pool cannot pass the frozen geometry gate;
- a new candidate route must first enlarge or continuously redefine support.

G1D's best early control is also informative: epoch-15 completeness is
`3.5811 m` and far completeness is `8.1239 m`. Later active evidence improves
occupancy recall but not completeness, while duplicates approach 30% and the
global-condition shuffle remains near zero. This is consistent with local
Cube-driven allocation crowding, not merely insufficient training time.

### 2.3 Radar-specific compatibility contract

A transferable mechanism is compatible only if it preserves all of:

1. queries remain inside the calibrated RAE frustum;
2. current Full-RAED values, not LiDAR or CFAR at inference, determine support;
3. far-range point mass is explicit rather than an accidental consequence of
   Cartesian density;
4. one point cannot consume multiple fixed-count slots within 5 cm;
5. all 10k coordinates can later receive generated circular Doppler and
   confidence and participate in point-to-Cube reprojection;
6. cross-scene condition shuffling changes predictions measurably;
7. no stochastic best-of-k selection is used.

Object-level completion checkpoints are not valid radar baselines. Their useful
content is architectural mechanism, not reported ShapeNet/PCN performance.

## 3. Mechanism comparison

### 3.1 Completion and direct coordinate generation

| Work and official code | Mechanism, not just model name | Support expansion | Duplicate / coverage behavior | Condition dependence | Radar-Cube transfer verdict |
|---|---|---|---|---|---|
| [PoinTr paper](https://arxiv.org/abs/2108.08839), [official code](https://github.com/yuxumin/PoinTr) | Converts partial points into local point proxies and performs set-to-set Transformer translation from observed proxies to missing proxies; geometry-aware attention adds local inductive bias. | Learned missing proxies can leave the input point support. | Coarse-to-fine generation helps coverage, but CD-based coordinate heads do not by themselves enforce unique slots or range mass. | Decoder cross-attention provides a condition path, but there is no anti-bypass guarantee. | **Medium-low.** Reuse proxy/query translation, but replace the point-input encoder with Full-RAED tokens and RAE-aware losses. |
| [AdaPoinTr paper](https://arxiv.org/abs/2301.04545), [official code](https://github.com/yuxumin/PoinTr) | Generates adaptive queries from the observed/missing structure and adds denoising queries during training instead of relying only on fixed learned queries. | Better than fixed templates because scene-dependent queries can move to missing regions. | Denoising queries regularize query semantics, but do not impose exact range quotas or minimum point separation. | Adaptive queries make the input more causally useful than fixed queries. | **High for G1G.** Add adaptive center queries and query denoising while preserving the rule that no local Cube lookup occurs before center allocation. |
| [SeedFormer paper](https://arxiv.org/abs/2207.10315), [official code](https://github.com/hrzhou2/seedformer) | Learns patch seeds carrying both position and regional feature, then uses an Upsample Transformer to grow local patches while sharing neighboring semantic/geometric information. | Seeds generate new coordinates outside observed points. | Regional seeds reduce global-code information loss; local growth can still crowd unless child radius/diversity are bounded. | Seed features are condition-derived, but an unconditional seed template can dominate without shuffle controls. | **High for hierarchical growth.** Closest precedent for 2.5k centers x 4 children, with RAE-aware seed allocation and bounded offsets. |
| [SnowflakeNet paper](https://arxiv.org/abs/2108.04444), [official code](https://github.com/AllenXiangX/SnowflakeNet) | Snowflake Point Deconvolution recursively splits parent points; Skip-Transformer carries previous splitting patterns into later growth stages. | Recursive splitting expands local detail around coarse parents. | Parent-child structure makes the source of duplicates observable; compact local patches help smoothness but cannot repair wrong parent placement. | Global shape feature and previous splitting condition growth, but no explicit condition-shuffle test. | **Medium-high.** Reuse parent-child bookkeeping and splitting controls, not unrestricted recursive depth. One bounded 4-child stage is safer for radar. |
| [PU-Flow paper](https://arxiv.org/abs/2107.05893), [official code](https://github.com/unknownue/puflow) | Maps neighboring points into an invertible latent space and generates dense samples by learned weighted combinations of local latent neighbors. | Interpolates the underlying local surface; weak at truly unobserved regions. | Designed for uniform upsampling and proximity to a surface. | Primarily conditioned on sparse local point geometry. | **Low alone.** Radar Cube is not a surface point set, and G1F proves the old point support is insufficient. Useful only after a new support representation exists. |

### 3.2 Sampling, blue noise, and optimal transport

| Work and official code | Mechanism | What it can fix | What it cannot fix | Radar-Cube transfer verdict |
|---|---|---|---|---|
| [SampleNet paper](https://arxiv.org/abs/1912.03663), [official code](https://github.com/itailang/SampleNet) | Learns task-specific simplified points and differentiably projects them onto mixtures/neighborhoods of the input cloud before hard inference sampling. | Gives sampling coordinates task gradients and can outperform task-agnostic FPS. | Projection remains tied to candidate support; on the frozen 32k G1D pool it is upper-bounded by failed G1F-F0. | **Conditional.** Use only on a newly expanded frustum pool, never as G1F-F1 under another name. |
| [SampleNet FPS discussion](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lang_SampleNet_Differentiable_Point_Cloud_Sampling_CVPR_2020_paper.pdf), [PyTorch3D FPS implementation](https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/sample_farthest_points.html) | Greedily selects the point farthest from the selected set; each prefix is a coverage-oriented subset. | Exact count, deterministic coverage, and fewer duplicates over a valid support pool. | Ignores radar confidence and cannot create missing far support. Cartesian FPS also overweights large physical distances at far range. | **Medium as an export control.** Use weighted RAE distance or per-range-bin FPS, not raw Cartesian FPS. |
| [Progressive blue-noise/FPS paper](https://onlinelibrary.wiley.com/doi/10.1111/cgf.13848), [maximal Poisson-disk paper](https://doi.org/10.1145/2010324.1964944) | Enforces a conflict radius while maintaining coverage; progressive order allows an exact-count prefix. | Directly attacks 5 cm duplicates and makes fixed-count spatial coverage explicit. | A fixed conflict radius is wrong across radar range and cannot infer occupancy. | **High as deterministic post-processing/control.** Use a range-adaptive RAE metric and confidence-aware tie-breaking. |
| [Unbalanced Sinkhorn paper](https://arxiv.org/abs/1910.12958), [GeomLoss code](https://github.com/jeanfeydy/geomloss), [POT code](https://github.com/PythonOT/POT) | Entropic OT sends prediction mass to target mass; unbalanced OT can create/destroy mass with a penalty and is robust to outliers. Debiased Sinkhorn avoids entropic support shrinkage. | Gives all candidate masses gradients, couples precision and completeness, and supports explicit range-bin marginals. | OT cannot invent coordinates outside the source support; dense 10k x target cost matrices require chunking/online kernels. | **High after support expansion.** Use train-only GT for the loss, learned mass at inference, and range-stratified marginals. |
| [Gumbel Subset Sampling paper](https://arxiv.org/abs/1904.03375) | Uses a differentiable Gumbel relaxation to select representative subsets rather than heuristic FPS. | Makes hard subset allocation trainable. | Stochastic subset selection can be unstable and remains support-limited. | **Medium-low.** Suitable only as a paired selector ablation; deterministic export and fixed seeds are mandatory. |

### 3.3 Implicit occupancy and continuous fields

| Work and official code | Mechanism | Support / duplicate effect | Condition dependence | Radar-Cube transfer verdict |
|---|---|---|---|---|
| [Occupancy Networks paper](https://arxiv.org/abs/1812.03828), [official code](https://github.com/autonomousvision/occupancy_networks) | Represents a surface as the continuous decision boundary of a coordinate-conditioned occupancy classifier. | Can query arbitrary coordinates, so support is not restricted to a proposal list; fixed 10k sampling is a separate operation. | A single global latent may wash out local scene evidence and repeat G1E/G1D bypass behavior. | **Medium.** Continuous querying is valuable, but a global latent-only decoder is explicitly stale for this project. |
| [ConvONet paper](https://arxiv.org/abs/2003.04618), [official code](https://github.com/autonomousvision/convolutional_occupancy_networks) | Stores local condition features on convolutional planes or volumes and interpolates them at arbitrary query coordinates before implicit decoding. | Full-frustum continuous support plus spatially structured local evidence; adaptive sampling can enforce exact count. | Local feature interpolation ties each query to the observation, while a global branch can carry scene context. | **High.** Replace XYZ planes with calibrated RAE/tri-plane or sparse RAE volume features and retain all 64 Doppler bins in the encoder. |
| [POCO paper](https://arxiv.org/abs/2201.01831), [official code](https://github.com/valeoai/POCO) | Places latent features at observed points and interpolates occupancy from learned nearest-neighbor weights, concentrating capacity near surfaces. | Good for thin surfaces and scalable queries, but support remains centered on observed point features. | Strong local dependence on input points. | **Low-medium.** Useful if Cube cells become pseudo-points first; otherwise it introduces an unnecessary and possibly support-limiting Cube-to-point bottleneck. |
| [RangeLDM paper](https://arxiv.org/abs/2403.10094), [official code](https://github.com/WoodwindHu/RangeLDM) | Corrects sensor projection, models compact range images with a VAE, then runs latent diffusion with a range-guided discriminator; conditional upsampling/inpainting is supported. | One organized pixel/ray gives structured coverage and largely avoids same-pixel duplicates. | Conditional generation can act in a compact sensor-aligned latent. | **Medium-high representation lesson, low direct reuse.** Radar has multi-return RAE occupancy and Doppler spectra, not one LiDAR return per beam. Use range/ray factorization, not the LiDAR range-image target unchanged. |

### 3.4 Diffusion, point-voxel diffusion, and flow

| Work and official code | Mechanism | Support expansion / duplicate behavior | Condition dependence | Radar-Cube transfer verdict |
|---|---|---|---|---|
| [DiffusionPoint paper](https://arxiv.org/abs/2103.01458), [official code](https://github.com/luost26/diffusion-point-cloud) | Treats points as particles and learns a latent-conditioned reverse Markov chain from Gaussian noise to exactly `N` points. | Generates beyond a fixed pool and supports arbitrary fixed `N`; independent particle heads do not guarantee diversity. | A shared shape latent conditions every reverse step, but a weak latent can be ignored. | **Medium.** Exact-cardinality particle diffusion is useful, but replace object latent with Full-RAED projection conditioning and add set interactions/range mass. |
| [PVD paper](https://arxiv.org/abs/2104.03670), [official code](https://github.com/alexzhou907/PVD) | Combines point updates with voxel features in a denoising diffusion model for generation and multimodal completion. | Voxel context adds local/global structure and supports new geometry; diffusion remains expensive at 10k points. | Conditional completion shows that partial observations can guide a probabilistic generator. | **Medium-high architecture lesson.** A sparse RAE point-voxel denoiser is more compatible than object PointNet conditioning, but must remain inside radar support. |
| [SPVD paper/code](https://github.com/JohnRomanelis/SPVD) | Uses sparse point-voxel U-Net diffusion for scalable generation, completion, and super-resolution. | Sparse voxels reduce dense 3D compute and expose local occupancy neighborhoods. | Provides conditional-generation variants, but not radar-calibrated conditioning. | **Medium-high.** Useful implementation reference for an H200-feasible sparse RAE denoiser. |
| [PointFlow paper](https://arxiv.org/abs/1906.12320), [official code](https://github.com/stevenygd/PointFlow) | Learns a distribution over shapes and a conditional continuous normalizing flow over points, allowing an arbitrary number of point samples. | Samples new coordinates without a fixed pool, but conditionally i.i.d. samples can crowd; the paper reports difficulty on rare/thin structures. | Shape-level latent is the only observation path in the original model. | **Low-medium.** Arbitrary count is attractive, but global-latent CNF is too weak for long-range radar allocation without local projection and repulsion. |
| [PUDM paper](https://arxiv.org/abs/2312.02719), [official code](https://github.com/QWTforGithub/PUDM) | Treats sparse-to-dense upsampling as conditional DDPM; the sparse condition enters every reverse step, with a rate prior and dual mapping. It avoids CD as the sole training objective. | Noise-to-dense generation expands local support and improves uniformity, but published sampling is multi-step and arbitrary-scale output is a stated limitation. | Stronger than one-shot latent conditioning because `c` participates at every step. | **High mechanism, medium practicality.** Use current Cube at every step; do not inherit fixed object upsampling rates or 30-step inference. |
| [PC2 paper](https://arxiv.org/abs/2302.10668), [official code](https://github.com/lukemelas/projection-conditioned-point-cloud-diffusion) | Projects local image features onto each partially denoised 3D point at every diffusion step, producing geometrically aligned conditioning. | No fixed candidate support; local projected features constrain where noisy points may move. | The strongest direct analogue for sampling Full-RAED features at each noisy RAE coordinate. | **High.** Replace camera projection with differentiable RAE trilinear lookup; shuffle the entire Cube condition, not only global tokens. |
| [PUFM paper](https://arxiv.org/abs/2501.15286), [official code](https://github.com/Holmes-Alan/PUFM) | Uses EMD pre-alignment and flow matching to learn a short, coherent path from an interpolated sparse point set to the dense target instead of starting from Gaussian noise. | Fewer solver steps than DDPM; EMD alignment reduces unordered correspondence noise. Midpoint interpolation alone remains input-support limited. | Sparse input conditions the flow trajectory. | **High for a bounded Stage-0.** Start from a full-frustum radar prior rather than only old proposals, then use OT alignment and Cube-projected velocity fields. |

### 3.5 Terminology note on "PCDiff / PointDiffusion"

The reviewed literature does not support treating `PCDiff` as one canonical
pre-2025 point-completion architecture. The name is overloaded. This report
therefore uses the identifiable primary sources above: DiffusionPoint,
Point-Voxel Diffusion, PUDM, PC2, and PUFM. No conclusion is attributed to an
ambiguous `PCDiff` label.

## 4. Cross-mechanism conclusions

### 4.1 How to enlarge proposal support

Mechanisms that genuinely enlarge support:

- adaptive learned queries (AdaPoinTr);
- coarse patch seeds placed by the global condition (SeedFormer);
- arbitrary continuous occupancy queries (ConvONet);
- Gaussian/prior-to-point diffusion (DiffusionPoint/PVD/PUDM);
- full-frustum prior-to-target flow (PUFM adaptation);
- sensor-aligned ray/range fields (RangeLDM representation lesson).

Mechanisms that do **not** enlarge support by themselves:

- SampleNet projection;
- FPS or Poisson-disk sampling;
- Sinkhorn/EMD;
- PU-Flow interpolation;
- any new score head over the frozen G1D 32k candidates.

### 4.2 How to avoid duplicates

No reviewed neural generator makes the 5 cm duplicate gate automatic.
The most defensible combination is:

1. parent/child bookkeeping with bounded child radius;
2. a loss on nearest-other distance or density-aware repulsion;
3. balanced or unbalanced OT so multiple outputs cannot cheaply explain one
   target region;
4. deterministic range-adaptive Poisson/FPS export;
5. explicit reporting of unique center cells and child collapse.

Chamfer alone is insufficient because repeated predictions can preserve the
prediction-to-target term while wasting point budget.

### 4.3 How to preserve far-range coverage

Cartesian uniformity is not radar uniformity. Far coverage should be enforced
by:

- RAE-space support covering the full calibrated frustum;
- train-time range-bin target marginals in OT/field/flow losses;
- inference-time range-bin mass floors derived only from the current Cube
  prediction, not LiDAR;
- per-range completeness and 2 m GT recall diagnostics;
- an ablation with the range constraint disabled.

Range mass must not be fixed to the GT at inference. The model predicts mass;
GT only supervises it during training.

### 4.4 How to force condition dependence

Merely adding cross-attention or observing nonzero gradients is not enough.
Each route needs a causal bottleneck:

- direct regression: no local Cube feature before global center allocation;
- resampling: candidate logits depend on Full-RAED features, and the complete
  condition is shuffled across scenes;
- implicit field: query coordinates alone are insufficient; field values
  require interpolated current-Cube features plus global context;
- flow/diffusion: current-Cube projection enters every velocity/denoising step,
  with condition dropout and a fixed guidance setting.

The shuffled control must replace **all learned Cube-derived context belonging
to a frame as one unit**. Shuffling only global tokens while retaining current
local Cube samples repeats the G1D bypass.

## 5. Four mutually exclusive Stage-0 candidates

All budgets below are planning estimates for one physical H200, not measured
results. Scientific execution remains restricted to H200 GPU 0 or 2.

### Candidate A: adaptive condition-exclusive patch generator

**Class:** direct coordinate regression.
**Closest sources:** AdaPoinTr + SeedFormer + SnowflakeNet.
**Relationship to current work:** a bounded upgrade/evaluation of G1G, not a
new name for G1D.

#### Data flow

```text
Full-RAED Cube
  -> existing Full-RAED encoder -> 336 global spatial tokens
  -> adaptive query generator -> 2,500 scene-dependent center queries
  -> global cross-attention center allocator -> 2,500 RAE centers
  -> only now: local 64-bin Cube spectrum lookup at each center
  -> one bounded patch split -> 4 children per center
  -> exactly 10,000 XYZ points
```

Add AdaPoinTr-style noisy center queries during training, but remove them at
inference. Keep the child radius at one preregistered Cube-cell diagonal.

#### Code-level change surface

- extend `code/models/g1g_hierarchical_allocator.py` with an adaptive center
  query generator and training-only denoising queries;
- extend `code/losses/g1g_hierarchy.py` with denoising-query matching while
  retaining center coverage, center repulsion, child diversity, and 2 m hinge;
- extend `code/scripts/train_g1g_hierarchy.py` with fixed adaptive-query and
  query-denoising ablations;
- reuse `code/eval/dense_geometry.py` unchanged.

No modification to dataset, split, target cache, or metric code is justified.

#### Stage-0 and H200 budget

- one seed `20260716`, exactly 20 epochs, evaluation every 5 epochs;
- estimated peak memory: `<= 16 GB` (current structural preflight was about
  `8.63 GB`);
- estimated wall time cap: `4 H200-hours`;
- stop at epoch 10 if condition shuffle is `<0.5%` and completeness has not
  improved by 15% over the matched control.

#### Baselines and ablations

- G1D epoch-15 matched control;
- current fixed-query G1G;
- adaptive queries without query denoising;
- query denoising without adaptive queries;
- zero local refinement;
- child-collapse and zero-offset controls;
- cross-scene full condition shuffle.

#### Numerical promotion gate

All must pass:

- condition-shuffle Chamfer degradation `>= 1%`;
- duplicate fraction `<= 15%`;
- median completeness `<= 2.5068 m` (30% below `3.5811 m`);
- outlier fraction `<= 25%`;
- far completeness `<= 8.1239 m`;
- at least 80% of centers occupy unique 5 cm cells;
- exactly 10k finite in-frustum outputs.

Promotion only authorizes a full frozen run; the final universal geometry gate
still applies.

#### Failure interpretation

- shuffle failure: adaptive queries still behave like unconditional templates;
- duplicate failure: patch splitting wastes fixed point slots;
- completeness failure with good outlier: direct coordinate regression lacks
  support, so proceed to C/D rather than tuning losses;
- far failure only: global allocation or RAE positional encoding is biased
  toward near range.

### Candidate B: full-frustum candidate field + mass-constrained resampling

**Class:** candidate-set resampling.
**Closest sources:** SampleNet + unbalanced Sinkhorn + progressive blue noise.
**Critical distinction from G1F:** support is rebuilt; the frozen 32k pool is
not reused as the complete candidate universe.

#### Data flow

```text
calibrated RAE frustum
  -> 128k deterministic low-discrepancy/range-stratified coordinates
Full-RAED Cube
  -> existing encoder + chunked local feature lookup at all 128k coordinates
  -> learned occupancy mass / confidence per coordinate
training: range-marginal unbalanced Sinkhorn to radar-observable GT
inference: range-stratified confidence-weighted progressive FPS/Poisson export
  -> exactly 10,000 unique points
```

Stage B0 forbids learned coordinate offsets. This makes it a clean test of
whether full-frustum discretized support plus allocation is sufficient.

#### Code-level change surface

- add a low-discrepancy full-frustum query generator beside
  `code/models/rald_query_field.py`;
- add chunked mass prediction and a range-marginal Sinkhorn loss under
  `code/losses/`;
- add deterministic weighted progressive FPS/Poisson export under `code/eval/`;
- add a source-bound B0 oracle and B1 trainer under `code/scripts/`;
- do not change `code/eval/dense_geometry.py` or G1F artifacts.

#### Stage-0 and H200 budget

- **B0:** no-train GT coverage oracle on all 24 validation frames, `<=1`
  H200-hour;
- **B1:** only if B0 passes, one seed, at most 10 epochs;
- 128k candidate scores evaluated in chunks of 8k-16k;
- estimated peak memory `<= 32 GB`, estimated B1 wall time `<= 4 H200-hours`.

#### Baselines and ablations

- full-frustum hard Cube-energy top-10k;
- confidence top-10k without spacing;
- weighted FPS without OT;
- balanced versus unbalanced Sinkhorn;
- no range-marginal constraint;
- cross-scene complete-Cube shuffle.

#### Numerical promotion gate

B0 must first pass the complete universal geometry gate and improve far-range
2 m GT recall from `17.80%` to at least `40%`. If not, B1 is forbidden.

B1 must then:

- pass the complete universal geometry gate;
- degrade by `>=1%` under complete-Cube shuffle;
- improve completeness by `>=20%` and duplicates by `>=30%` relative to the
  hard-score top-10k control;
- keep outlier fraction `<=25%`;
- export exactly 10k unique indices with no GT at inference.

#### Failure interpretation

- B0 failure: even full-frustum discretization is too coarse or the target is
  not representable as a subset; selector work stops;
- B0 pass/B1 failure: support is sufficient but mass learning/export is wrong;
- OT helps loss but not hard export: soft-to-hard mismatch;
- spacing helps duplicates but hurts completeness: conflict radius or RAE
  metric is miscalibrated;
- shuffle failure: local scalar energy dominates the learned condition.

### Candidate C: Cube-conditioned continuous RAE occupancy/density field

**Class:** implicit occupancy field + adaptive sampling.
**Closest sources:** ConvONet + Occupancy Networks + RangeLDM's sensor-aligned
representation.
**Critical distinction from failed G1E:** local structured Full-RAED feature
fields and adaptive continuous queries replace a global latent-only occupancy
decoder and fixed proposal support.

#### Data flow

```text
Full-RAED Cube
  -> compact RAE 3D U-Net or R/A, R/E, A/E tri-plane features
  -> global Full-RAED context token
continuous query q=(r,a,e)
  -> trilinear feature interpolation at q + global context + q encoding
  -> occupancy density, existence confidence
adaptive inference:
  coarse full-frustum cells -> subdivide high-mass/high-uncertainty cells
  -> range-marginal mass allocation -> range-adaptive blue-noise samples
  -> exactly 10,000 XYZ points
```

The field predicts a measure, not a binary top-k grid. Exact point count belongs
to the sampler, which prevents the representation from learning the arbitrary
10k cardinality.

#### Code-level change surface

- add `code/models/cube_implicit_field.py` using the existing Full-RAED
  normalization and a compact structured encoder;
- add `code/losses/implicit_field.py` for occupied/empty query BCE or focal
  loss, range-mass calibration, and optional Sinkhorn sample loss;
- add `code/eval/implicit_field_sampler.py` for deterministic adaptive queries
  and exact-count export;
- add a separate source-bound trainer/comparator under `code/scripts/`;
- reuse the current dataset/cache and geometry evaluator unchanged.

#### Stage-0 and H200 budget

- one seed, 15 epochs (hard maximum 20);
- 8k-16k mixed train queries per frame; validation evaluates the field in
  deterministic chunks;
- estimated peak memory `<= 28 GB`;
- estimated wall time `<= 6 H200-hours`;
- early stop after epoch 8 if occupied recall is `<30%` or complete-Cube
  shuffle degradation is `<0.5%`.

#### Baselines and ablations

- existing discrete `CubeOccupancyNet` / strongest G1B geometry control;
- global latent-only field;
- local structured field without global context;
- fixed-grid top-10k versus adaptive field sampling;
- adaptive sampling without blue-noise spacing;
- no range-marginal calibration;
- full Cube condition shuffle.

#### Numerical promotion gate

All universal final geometry thresholds must pass by epoch 15, plus:

- complete-Cube shuffle degradation `>=1%`;
- positive-query recall `>=60%`;
- reliable-empty false-positive rate `<=20%`;
- duplicate fraction `<=10%`;
- no more than 5% of exported points may come from sampler fallback;
- exactly 10k finite in-frustum outputs.

#### Failure interpretation

- good field recall but poor sampled geometry: sampler/mass calibration failure;
- poor field recall: representation or sparse occupancy supervision failure;
- good completeness but high outlier: occupancy threshold/mass creation is too
  permissive;
- shuffle failure: coordinate/range prior dominates the Cube;
- far failure: adaptive subdivision or range mass remains near-biased.

### Candidate D: Cube-projected OT flow matching

**Class:** conditional diffusion / flow matching.
**Closest sources:** PC2 + PUDM + PVD/SPVD + PUFM.
**Stage-0 choice:** use flow matching, not a long DDPM, to stay within the
bounded experiment budget.

#### Data flow

```text
10k range-stratified low-discrepancy prior points over the calibrated frustum
  + train-only OT/EMD alignment to 10k target points
Full-RAED Cube -> global 336 tokens
for each flow time t:
  noisy/interpolated RAE points
  -> point self-attention / sparse point-voxel context
  -> project current 64-bin Cube feature at every current point
  -> cross-attend global Cube tokens
  -> predict coordinate velocity field
4-8 fixed ODE steps -> exactly 10,000 XYZ points
```

The prior spans the full frustum, so this is not PUFM interpolation from the
failed 32k pool. Training uses one OT alignment, endpoint geometry/range loss,
and point-separation regularization. Inference uses a fixed prior seed and no
GT.

#### Code-level change surface

- add `code/models/cube_projected_point_flow.py`;
- factor reusable timestep embedding/self- and cross-attention from
  `code/models/point_diffusion.py`, but replace its LiDAR encoder;
- add RAE projection lookup shared with the existing Cube query field;
- add `code/losses/point_flow_matching.py` with OT-aligned velocity,
  endpoint/range, and repulsion terms;
- add a fixed-step sampler and source-bound Stage-0 trainer/comparator;
- do not reuse old LiDAR-conditioned P1 checkpoints.

#### Stage-0 and H200 budget

- one seed, 20 epochs, fixed 4-step and 8-step validation;
- point interactions use local kNN/sparse point-voxel blocks rather than dense
  10k x 10k attention;
- estimated peak memory `<= 48 GB`;
- estimated wall time `<= 8 H200-hours`;
- stop after epoch 10 if complete-Cube shuffle is `<0.5%`, endpoint
  completeness is `>3.0 m`, or any solver step leaves the calibrated frustum.

#### Baselines and ablations

- deterministic candidate A output;
- same network trained as one-shot coordinate regression;
- Gaussian-noise DDPM-style prior versus full-frustum low-discrepancy prior;
- global Cube tokens only;
- projected local Cube features only;
- no OT pre-alignment;
- no range-mass term;
- 4 versus 8 solver steps;
- complete-Cube shuffle and zero-Cube controls.

No best-of-k sample may be reported. Model selection uses one preregistered
prior seed; three additional seeds are reported only as stability diagnostics.

#### Numerical promotion gate

All universal final geometry thresholds must pass by epoch 20, plus:

- complete-Cube shuffle Chamfer degradation `>=1%`;
- 4-step Chamfer may be at most 5% worse than 8-step Chamfer;
- duplicate fraction `<=10%` for every reported prior seed, not only the mean;
- far completeness `<=8.0 m`;
- zero out-of-frustum points after every solver step;
- exactly 10k finite outputs without confidence masking.

#### Failure interpretation

- low training velocity loss but poor endpoint geometry: OT path or ODE solver
  mismatch;
- shuffle failure: the prior/scene distribution dominates observation
  conditioning;
- high seed variance: probabilistic ambiguity is uncontrolled and cannot be
  hidden with best-of-k;
- high duplicates: point interactions or endpoint repulsion are insufficient;
- near-good geometry but far failure: prior and OT cost need RAE/range
  normalization, not more diffusion steps;
- all geometry fails within 20 epochs: stop rather than scaling the denoiser;
  return to the deterministic implicit field.

## 6. Unified experiment order

The four routes should not be fused before independent evidence exists.

1. **Finish Candidate A/G1G Stage-0** because implementation and anti-bypass
   tests already exist and its cost is lowest.
2. **Run Candidate B0 concurrently.** It is a no-train support test and decides
   whether any candidate-resampling family deserves training.
3. **Implement Candidate C and Candidate D in parallel**, but run C first when
   only one H200 slot is available.
4. **Do not combine A centers, C field scores, or D refinement** unless each
   component has a positive paired ablation under its own frozen protocol.
5. If A fails support, B0 fails, and C fails field recall, the remaining
   problem is target/observability representation rather than point-generator
   capacity; do not respond by enlarging D.

### Decision table

| Result | Decision |
|---|---|
| A passes | freeze direct hierarchical family; then test Doppler/cycle on this parent |
| A shuffle fails | close direct regression regardless of geometry |
| B0 fails | close all subset/resampling routes on that discretization |
| B0 passes, B1 fails | keep B0 as diagnostic; selector objective/export is at fault |
| C field passes but sampler fails | repair sampler only under a new frozen C1 protocol |
| C field itself fails | close implicit route; do not relabel it as latent diffusion |
| D passes only with best-of-k or high seed variance | fail D |
| D passes all gates | freeze one prior seed and solver; release full multi-seed run |
| more than one route passes | select by complete gate, condition effect, then H200 cost; do not average models |

## 7. Concrete recommendation

The literature does not justify replacing G1G immediately with a large
diffusion model. It justifies a broader, parallel frontier:

- **lowest-risk test:** AdaPoinTr/SeedFormer-style adaptive hierarchical G1G;
- **cheapest support diagnosis:** new full-frustum B0 oracle;
- **best radar-compatible representation:** ConvONet-style continuous RAE
  measure with exact deterministic sampling;
- **best generative alternative:** PC2-conditioned, PUFM-trained short flow
  from a full-frustum prior.

The key migration is not "use a newer point-cloud model." It is:

> replace the old support-limited hard top-k pipeline with either a
> full-frustum continuous measure or an observation-conditioned transport
> process, while making fixed-count allocation, far-range mass, duplicate
> suppression, and complete-Cube dependence explicit and independently
> falsifiable.
