# G1G Stage-0 独立怀疑主义代码审查

> 审查日期：2026-07-29  
> 报告文件沿用任务指定日期后缀 `2026-07-28`  
> 审查对象：HEAD `2ea3cf0`  
> 审查方式：只读静态审查；未运行训练、pytest 或科学计算

## 结论

**当前实现存在 2 个 P0 阻塞项，不应直接启动正式 20 epoch 并据此判定
G1G 通过或失败。**

1. `60--120 m` completeness 会删去“有远距 GT、但没有远距预测”的灾难帧，
   Stage-0 又没有检查该指标是否覆盖完整 24 帧；硬编码的 G1D `8.1239 m`
   也来自同一指标实现，必须用修正口径重算。
2. G1G 只把 manifest/split/normalization 三个 JSON 锁成 G1D 合同；Cube、dense
   target cache、坐标轴以及 G1D epoch-15 控制指标均未与一个冻结的 G1D artifact
   做强一致性校验。当前代码可以在输入文件被替换后仍声称是 matched comparison。

此外有 5 个 P1 问题会使 20 epoch 的结果难以解释：EMA 对随机初始化残留很大、
最终 10k 重复率没有对应训练项、center score 不参与输出分配、child BCE 仍消费
autocast 内生成的概率、以及若干正则项在量纲和可达域上几乎不起作用。即使修复
P0，正式运行前也应至少关闭这些解释性缺口。

## Findings

### [P0] 1. 远距 completeness 存在按模型输出删失，能让失败模型获得虚假的较好均值

**证据**

- `code/eval/dense_geometry.py:120-124` 只有在某个距离段同时存在 prediction 和
  target 时才生成该段的所有指标。
- `code/scripts/train_g1g_hierarchy.py:924-945` 读取聚合后的 far mean，但只在整个
  aggregate 完全缺少 far key 时 fail closed。
- `code/scripts/train_g1g_hierarchy.py:952-969` 不检查 far 指标的 `sample_count`
  是否等于 24。
- 该缺口已在仓库正式 artifact 中出现：`artifacts/g1/g1f_f0_ca60d76.json`
  的 `60--120 m` completeness 只有 `sample_count=14`，而总验证帧为 24。
- G1D 的 `geometry_report` 调用路径相同：
  `code/scripts/train_rald_query_field.py:418-438`。因此硬编码控制值
  `8.1239 m` 不能直接视为完整 24 帧的 far mean。

**影响**

若某帧有远距 GT、但模型把所有点都放在 60 m 内，当前实现不会给该帧一个很差的
far completeness，而是直接不计入 far mean。不同模型的 far `sample_count`
还可能不同，因此即使都调用同一 evaluator，也不是 matched comparison。
这会直接改变 `far_completeness_no_worse_than_g1d_epoch15` 的通过与否。

**最小修复**

1. 在 `geometry_report` 中，只要 `target_mask.any()` 就必须计算该段
   completeness；不能以 `prediction_mask.any()` 作为 completeness 的前提。
2. precision/fscore 可独立处理无该段预测的情况，但不得删除 target-side 指标。
3. Stage-0 强制 far completeness 的 `sample_count == 24`，scene-first 也应记录
   每个 scene 的有效帧数。
4. 用修正后的 evaluator 从冻结 G1D epoch-15 checkpoint 重新评估完整 24 帧，
   替换 `8.1239 m`，同时记录旧值与新值的差异。

**必须新增的测试**

- 一帧含 `80 m` GT、全部 prediction 位于 `20 m`：far completeness 必须存在且
  为有限的大误差，而不是缺 key。
- 24 帧中 10 帧无 far prediction：聚合 `sample_count` 必须仍为 24。
- `stage0_decision` 收到 far `sample_count != 24` 时必须 fail closed。
- 对同一组 frame records 验证 G1D 与 G1G 的 far endpoint 和 sample count 完全一致。

### [P0] 2. “matched G1D control”没有被 artifact 和逐文件数据合同强绑定

**证据**

- `code/scripts/train_g1g_hierarchy.py:284-303` 只校验 manifest、scene split 和
  normalization 三个固定 SHA。
- `code/scripts/train_g1g_hierarchy.py:322-355` 会计算 Cube/cache SHA，但只是把
  当前文件哈希写入 provenance，没有与冻结 G1D run 的哈希集合比较。
- `code/scripts/train_g1g_hierarchy.py:1172-1184` 对 RAE/Doppler axis 同样只记录
  当前 SHA，不检查预期值。
- `code/scripts/train_g1g_hierarchy.py:50-56` 把 G1D epoch-15 指标写成两个浮点
  常量；`1256-1261` 只声明 `model_loaded=False`，没有控制 artifact 的路径、
  SHA、source commit、checkpoint SHA、frame identity 或 endpoint sample count。
- `code/scripts/train_g1g_hierarchy.py:991` 将口径标记为
  `matched_frame_first_G1D_epoch15_24frame_aggregation`，但代码没有验证上述
  “matched”条件。
- `code/tests/test_train_g1g_hierarchy.py:231-270` 只锁定两个常量和阈值边界，
  未证明常量来自指定 G1D artifact。

**影响**

以下情况当前都会被接受：

- dense target cache 被重新生成或替换；
- `info_arr.mat` / `arr_doppler.mat` 被替换；
- G1D 控制值抄自另一个 checkpoint、另一个 evaluator source 或不完整 far cohort；
- G1D 与 G1G 使用相同 manifest identity，但实际 Cube/cache bytes 不同。

结果仍可复现“本次 G1G 输入”，却不能支持“相对 matched G1D 改善 30%”这一核心
Stage-0 结论。

**最小修复**

1. 新增冻结的 G1D epoch-15 control JSON，并锁定 artifact SHA、source commit
   `4c6150c...`、checkpoint SHA、evaluator source SHA、24 个 frame identity、
   每个 endpoint 的 sample count 和逐帧数值。
2. G1G 启动时读取并验证该 artifact，不再手抄 `3.5811/8.1239`。
3. 从 G1D control 或独立冻结 data-contract artifact 读取 100 个 Cube/cache SHA
   和两个 axis SHA；G1G 当前文件必须逐项相等。
4. 校验 cache 内嵌 schema、source manifest SHA、LiDAR time reference 和构建
   source，而不只是校验外层文件 bytes。

**必须新增的测试**

- 改动任意一个 cache byte、Cube byte 或 axis file 后，formal contract 必须拒绝。
- G1D control 的 source commit、epoch、frame identity、sample count、artifact SHA
  任一不符时必须拒绝。
- G1D control 值只能由已验证 artifact 派生，测试中不得直接构造两个裸浮点常量。

### [P1] 3. EMA 在 20 epoch 预算内仍显著保留随机初始化，且没有 raw-model 对照

**证据**

- `code/scripts/train_g1g_hierarchy.py:126` 固定 `ema_decay=0.999`。
- `code/scripts/train_g1g_hierarchy.py:568-583` 从初始模型的完整拷贝开始做常数
  decay EMA，没有 update-count warmup。
- formal 一共只有 `76 * 20 = 1520` 次更新。初始随机参数在 EMA 中的系数约为
  `0.999^1520 = 21.8%`；epoch 5 时约为 `68.4%`。
- `code/scripts/train_g1g_hierarchy.py:1457-1486` 只评估 EMA，raw model 从未进入
  checkpoint selection。
- G1D epoch-15 控制约 1140 次更新，初始残留约 `32.0%`。两者接近但不相同，
  不能据此证明当前 decay 对短程 Stage-0 合理。

**影响**

早期 checkpoint 主要衡量“随机初始模型与已训练模型的重混合”，不是纯粹的当前
优化状态。若 raw model 在 20 epoch 内已经改善而 EMA 尚未跟上，会造成假阴性；
若 raw model 后期退化，滞后的 EMA 又可能造成假阳性。当前 best-epoch 选择无法
区分这两种情况。

**最小修复**

- 在结果选择前冻结一个短程 EMA 方案：例如 update-count ramp，或使用使最终
  有效窗口与 1520 updates 匹配的 decay。
- 每个评估点同时记录 raw 与 EMA 指标；正式 gate 使用哪一个必须在运行前冻结。
- 不应在看到 G1G 指标后再选择 decay。

**必须新增的测试**

- 用常数参数序列验证 epoch 5/20 的 EMA 权重与解析值一致。
- resume 前后 EMA update count 和数值连续，不能重新 warmup。
- smoke 同时输出 raw/EMA，明确 smoke 不用于判断正式 decay 的科学有效性。

### [P1] 4. 最终 10k 重复率是硬门，但训练目标没有跨中心的 final-point repulsion

**证据**

- G1G 只对 2500 个 centers 做 repulsion：
  `code/losses/g1g_hierarchy.py:231-235`。
- child diversity 只比较同一 parent 内的四个 children：
  `code/losses/g1g_hierarchy.py:236-240`。
- 两个相邻 center 的 children 完全重合时，上述两项都可能很小或为零。
- 正式 duplicate gate 却作用于完整 10k：
  `code/scripts/train_g1g_hierarchy.py:804` 和 `936-958`。
- matched G1D 明确在完整生成点集上计算 global kNN repulsion：
  `code/losses/rald_query_field.py:573-587`。
- G1G 设计稿称使用“same geometry losses and metrics as G1D”，但当前训练目标
  与 G1D 在这个直接对应 gate 的项上并不等价。

**影响**

若 G1G 因 duplicate `>15%` 失败，不能把结果干净归因于 condition-exclusive
hierarchy；训练目标本身没有直接优化该门。center repulsion 和 sibling diversity
不能替代跨 parent 的 final-point duplicate penalty。

**最小修复**

- 恢复与 G1D 同定义的 10k global kNN repulsion，或在运行前将“无 final-point
  repulsion”冻结为一个明确消融臂，不能只跑该臂后关闭 G1G。
- 保持 geometry/outlier/confidence 不受预测 confidence 加权，避免靠低置信度
  隐藏重复点。

**必须新增的测试**

- 构造两个相距大于 0.1 m 的 centers，但让不同 parent 的 child 坐标重合；loss
  必须增加。
- 训练项使用的 duplicate 距离阈值与 evaluator 的 `<0.05 m` 定义必须有一致性测试。

### [P1] 5. center score 不参与点分配或导出，child/center confidence 也没有校准门

**证据**

- center score 由独立 head 产生：
  `code/models/g1g_hierarchical_allocator.py:183-185,413`。
- `refine_children` 在 `416-507` 只读取 center coordinates 和 features；不读取
  `center_score_logit`。每个 center 无条件导出四个 children。
- center score 只进入辅助 BCE：
  `code/losses/g1g_hierarchy.py:224-230`，以及均值报告：
  `code/scripts/train_g1g_hierarchy.py:650-660`。
- child confidence 同样不影响固定 10k geometry，评估只记录 mean：
  `code/scripts/train_g1g_hierarchy.py:842-844,912-917`，没有 Brier/ECE/AUPRC、
  matched-vs-unmatched calibration 或 confidence-conditioned precision。
- center/child existence target 由“距任意 target 是否小于 1 m”二值化：
  `code/losses/g1g_hierarchy.py:214-230`；target 的 radar-observability confidence
  大小没有进入该标签。

**影响**

center score 的数值不会改变最终点集，最多通过共享 trunk 的辅助梯度间接影响
coordinates。它不能支持“学习了中心占用并据此分配点数”的表述。现有 Stage-0
仍可评几何，但不能据此验证 per-point confidence 或有效点数主张。

**最小修复**

- 若 Stage-0 只评几何：明确把 center score 降级为 auxiliary diagnostic，不作
  输出机制主张。
- 若要保留 confidence 主张：至少将 parent center score 合入 child existence
  logit，并增加冻结的 calibration 指标；若 score 要影响几何，则需预注册固定
  10k 下的 score-aware allocation/resampling，而不是事后筛点。

**必须新增的测试**

- 固定 coordinates/features，仅改变 center score；若设计声称 score 影响输出，
  导出 confidence 或 allocation 必须随之变化。
- target confidence 高低变化时，existence supervision 的预期行为必须被显式测试。
- matched/unmatched child 的 calibration metric 必须在 24 帧完整输出。

### [P1] 6. BCE 已移出 autocast，但 child BCE 仍消费 autocast 内生成的 sigmoid 概率

**证据**

- 修正后的训练脚本确实在 autocast 块外调用 loss：
  `code/scripts/train_g1g_hierarchy.py:1358-1388`。
- 但 child probability 在模型 forward 内先做 sigmoid：
  `code/models/g1g_hierarchical_allocator.py:474-503`。该 forward 正在 bfloat16
  autocast 中执行。
- loss 随后读取 `output["confidence"]` 而不是 float32 logit：
  `code/losses/g1g_hierarchy.py:163,214-218`。
- `existence_confidence_loss` 再对该概率 clamp 后做 `binary_cross_entropy`：
  `code/losses/cube_cycle.py:120-138`。
- 当前 AST 测试 `code/tests/test_train_g1g_hierarchy.py:102-125` 只能证明 loss
  函数调用位于 autocast 外，不能证明 sigmoid 概率是在 float32 中生成。

**影响**

这次修复解决了 probability BCE 的 autocast 执行限制，但未消除 bfloat16 sigmoid
的量化/饱和路径。概率若已舍入到 0 或 1，转成 float32 和 clamp 不能恢复信息，
clamp 饱和区还可能给出零梯度。center BCE 使用 float32 logits，child BCE 应采用
同样稳定的实现。

**最小修复**

- 在 autocast 外对 `confidence_logit.float()` 使用
  `binary_cross_entropy_with_logits`。它与 sigmoid+BCE 数学等价，但数值稳定。
- `output["confidence"]` 只用于导出/报告，不作为训练 BCE 的输入。

**必须新增的测试**

- H200 autocast 下用大正/大负 child logits，BCE 必须有限且梯度非零、方向正确。
- 测试 loss 读取 `confidence_logit`，并拒绝只有 probability、没有 logit 的 formal output。

### [P1] 7. 正则项量纲未校准，child-bound 项对模型可达输出恒为零

**证据**

- child offset 已被逐轴硬限制在半个 bin：
  `code/models/g1g_hierarchical_allocator.py:450-454`。
- `center_cell_diagonal_m` 是同一 cell 八个角点的最大两两距离：
  `339-355`。任何由上述半-bin box 产生的 child 都不可能离 center 超过这个
  full diagonal。
- 因此 `physical_child_bound_loss`（`code/losses/g1g_hierarchy.py:110-125`）对
  正常模型输出恒为零；现有测试只用模型不可能产生的 `1.5 m` synthetic offset
  激活它：`code/tests/test_g1g_hierarchy_loss.py:175-186`。
- center repulsion 未加权上界是 `0.1^2=0.01`，乘 `0.05` 后最多 `0.0005`；
  child diversity 未加权上界是 `0.2^2=0.04`，乘 `0.05` 后最多 `0.002`。
  几何项则以米计，通常为数米。当前没有 component-gradient scale 审计。

**影响**

配置表面上包含强度为 `1.0` 的 physical bound 和两个 anti-collapse regularizer，
但前者是 dead loss，后两者在 loss value 上最多为千分量级。不能仅凭“组件存在”
推断它们足以约束 10k 输出。若 collapse/duplicate 失败，解释会混淆架构与损失尺度。

**最小修复**

- 删除恒零 soft bound，或改成对可达域内部有意义的 margin/offset regularizer。
- 在看正式指标前，用固定的初始/前两步 preflight 记录每个 component 的 value、
  weighted value 和对 center/child heads 的 gradient norm，据此一次性冻结权重。

**必须新增的测试**

- 对真实 model forward 输出，physical-bound loss 必须被证明恒零；若保留该项，
  新定义必须能由合法模型输出激活。
- component-scale preflight 必须拒绝零梯度或相对主几何梯度低于预注册下限的
  anti-collapse 项。

### [P2] 8. center collapse gate 使用 5 cm voxel unique，不等价于 5 cm 最近邻重复

**证据**

- `code/scripts/train_g1g_hierarchy.py:633-650` 通过
  `round(center_xyz / 0.05)` 后数 unique voxel。
- final duplicate gate 则使用严格欧氏最近邻 `<0.05 m`：
  `code/eval/rald_guided_query.py:28-37`。

**影响**

两个相距小于 5 cm 的 center 只要跨过 round voxel 边界，仍会被计为两个 unique
cells；同一 voxel 内两点的距离又可能超过 5 cm。该指标可以作为 voxel occupancy，
但作为“center collapse”保护不够稳健。

**最小修复**

- 同时报告 center nearest-neighbor duplicate fraction，并以该值作为 collapse gate；
  voxel unique 保留为结构诊断。

**必须新增的测试**

- 构造跨 voxel 边界但欧氏距离 `<0.05 m` 的 centers，collapse gate 必须识别。

### [P2] 9. formal smoke 没有覆盖最坏显存/时间路径，存在完成性风险

**证据**

- 每次 local query 都重新对完整 4D Cube 计算 `log1p`：
  `code/models/cube_doppler.py:64-66`。
- 一个 forward 先查询 2500 center spectra，再查询 10k final spectra：
  `code/models/g1g_hierarchical_allocator.py:441-442,470-473`。后者在 Stage-0
  loss/evaluator 中未使用。
- 所有 6 层 center features 被 stack 并保留：
  `code/models/g1g_hierarchical_allocator.py:391-410`，正式 loss 只用最后一层。
- 训练中的双向 10k-to-target cdist 和 2500-center all-pairs 都保留 autograd：
  `code/losses/g1g_hierarchy.py:199-235`。
- smoke 只使用两个均匀抽取的 train frames：
  `code/scripts/train_g1g_hierarchy.py:218-233,1220-1225`，不保证包含 target count
  最大的帧，也不覆盖完整 24-frame evaluation 的累计时间。

**影响**

已有结构 preflight 的约 8.6 GB 峰值不包含正式 geometry loss 的最坏 target
cardinality。smoke 通过不能证明 formal 不会在某个高密度 target 帧 OOM，也不能
给出完整评估墙钟时间。该问题不改变已完成 run 的数值，但可能导致 20 epoch
无法按计划完成。

**最小修复**

- 在 formal 前对 target count 最大的 train frame 做一次完整 forward+loss+backward。
- 只计算一次 normalized log Cube 并复用；Stage-0 允许关闭未消费的 final
  point spectrum 输出。
- 不需要报告中间层时，不 stack `center_layer_features`。
- 记录单 update、单 validation frame 和完整 24-frame eval 的峰值显存/墙钟估计。

**必须新增的测试**

- 可选关闭 final spectrum 时，geometry/confidence 输出与原路径逐元素一致。
- smoke 必须包含最大 target-count 帧并执行与 formal 相同的完整 loss。

### [P2] 10. resume 基本完整，但“clean/reproducible”声明强于实际保证

**证据**

- `code/scripts/train_g1g_hierarchy.py:177-184` 使用
  `git status --untracked-files=no`，随后 provenance 固定写
  `worktree_clean=True`（`1243-1245`）。未跟踪 Python 文件不会使 formal 拒绝。
- `code/scripts/train_g1g_hierarchy.py:1191` 开启 `cudnn.benchmark=True`，但没有
  deterministic algorithm 约束；RNG state 恢复
  `199-215,1282-1300` 不能保证中断续训与不中断训练逐 bit 等价。
- 当前测试没有覆盖真实 checkpoint round trip、epoch 边界中断或 best/last
  一致性；`code/tests/test_train_g1g_hierarchy.py` 主要覆盖静态合同与阈值。

**影响**

checkpoint/resume 的模型、EMA、optimizer 和 RNG 字段设计总体合理，但当前证据
只支持“可继续训练”，不支持“bitwise deterministic replay”或“完全 clean tree”。

**最小修复**

- formal snapshot 要求完整 `git status --porcelain` 为空，或明确列出唯一允许的
  untracked output 路径。
- 关闭 benchmark 并启用可支持的 deterministic 设置；若性能原因不做，则在
  artifact 中明确 `bitwise_reproducible=False`。
- checkpoint load 时额外验证 `protocol`、`artifact_type` 和 anti-bypass 字段。

**必须新增的测试**

- 在固定小模型上比较连续两 epoch 与 epoch 1 后 resume 的 model/EMA/optimizer、
  update count、best epoch 和 JSONL 行号。
- 篡改 checkpoint protocol/artifact type/anti-bypass 后必须拒绝 resume。

## 已核对且当前方向正确的部分

- normalization SHA 已切换到逐 Doppler-bin Full-RAED 文件：
  `code/scripts/train_g1g_hierarchy.py:63-65`。
- loss 调用本身已移到 autocast 块外：
  `code/scripts/train_g1g_hierarchy.py:1358-1388`；Finding 6 是更深一层的
  probability 生成精度问题。
- condition shuffle 的方向正确：foreign condition 仍配 current measured Cube，
  degradation 定义为 `shuffled/current - 1`，正值表示条件被打乱后变差：
  `code/scripts/train_g1g_hierarchy.py:753-766,811-837`。
- cross-scene mapping 是 deterministic permutation，且显式拒绝 same-sequence：
  `code/scripts/train_g1g_hierarchy.py:236-250`。
- Stage-0 gate 使用 frame-first aggregation，与当前硬编码 G1D control 的声明
  一致；scene-first 另行报告，没有暗中替换 gate 口径：
  `code/scripts/train_g1g_hierarchy.py:886-911,921-1008`。但 far missingness 会同时
  污染两种 aggregation，必须先修 Finding 1。
- checkpoint 已保存 model、EMA、optimizer、epoch、RNG、gradient audit 和 record：
  `code/scripts/train_g1g_hierarchy.py:1078-1112`；主要缺口是端到端 resume 测试和
  deterministic 声明。

## 正式运行前的最小放行条件

1. 修复 far completeness missingness，并重算、冻结 G1D epoch-15 control artifact。
2. 把 G1D control、100 帧 Cube/cache 和 axis SHA 做逐项强绑定。
3. 将 child existence 改为 autocast 外的 float32 logits BCE。
4. 冻结 raw/EMA 选择规则，至少同时报告二者。
5. 为 final 10k 增加跨 parent duplicate 对应训练项，或把无该项明确设为消融而非
   唯一 G1G 判定臂。
6. 在最大 target-count frame 上完成 formal-size forward/loss/backward 预飞。

在这六项完成前，工程 smoke 可以用于检查脚本能否执行，但不得授权科学
20-epoch run，更不得用其结果关闭 G1G 路线。
