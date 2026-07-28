# P-RF Continuous Target Lifting Protocol

状态：H200 preflight no-go，禁止长训练
适用范围：P-RF 单帧训练 target endpoint，不适用于推理期点云后处理
固定输出：`N = 10,000`

## 1. 阻塞与审计结论

P-RF 原 `canonical_fixed_target` 只允许从原始 target unique XYZ 中无放回选
10,000 点。K-Radar 100 帧缓存中有 57 帧低于该容量，因此原 adapter 不能覆盖
真实训练分布。

只在同一 RAE cell 的观测点凸包内插值不能解决阻塞。最稀疏帧只有 102 个
occupied RAE cells，同-cell 三角插值在 5 cm 网格上仅提供约 778 个位置。该机制
容量不足，冻结为 no-go。

本轮接受审查的最小候选机制是：

```text
observed target point
  -> local weighted-PCA tangent plane
  -> bounded deterministic tangent offsets
  -> endpoint must remain in a RAE cell already occupied by this frame's target
  -> global 5 cm Euclidean spacing gate
  -> exact 10,000 or hard failure
```

最终 source `259663c` 的全量 H200 preflight 表明，该机制只能让 57 个稀疏帧中
的 47 帧达到 exact 10,000；其余 10 帧的认证容量为 `3,410--9,901`。因此该
候选在进入模型梯度、NFE 和显存检查前 no-go。完整失败表见
`artifacts/g1/p_rf_target_lifting_failure_2026-07-29.md`。

## 2. 冻结构造

对每个 exact-unique target 点：

1. 在 XYZ 中取最多 12 个、距离不超过 1.5 m 的邻点。
2. 至少需要 6 个邻点；否则该点不产生 lifted candidates。
3. 用距离加权 PCA 拟合局部切平面。
4. 要求第三与第二主值之比不超过 `0.25`，否则拒绝该局部面。
5. 在两个切向基上用 2.5 cm 确定性格点提出 candidates。
6. patch 半径为第 12 邻点距离的 `0.6` 倍，且最大不超过 0.6 m。
7. candidate 必须落入本帧原始 target 已占据的 RAE cell。
8. 全局确定性空间哈希只接受与已有 endpoint 至少相距 5 cm 的 candidate。
9. 达到 10,000 点立即停止；耗尽后仍不足则报容量错误。

该构造不会复制坐标，不会打开新的 RAE cell，也不会从其他帧借 target。它仍然
包含“局部 target 点采样自连续表面”的监督假设，因此所有 lifted 点必须在归档中
显式标记，不能当成新增传感器观测。

## 3. 逐点来源

每个 endpoint 点必须保存：

- `source_indices`：原始 target 中的锚点；
- `is_lifted`：原始点或 lifted 点；
- `tangent_source_indices`：拟合局部切平面的原始 target 邻点；
- `tangent_offset_uv_m`：切平面内的确定性偏移；
- `rae_indices`：对最终 float32 XYZ 重新量化得到的 Cube 查询 cell。

confidence 继承锚点的 radar-observability confidence。Doppler distribution
不从锚点复制，而是用最终点的 `rae_indices` 查询同一当前帧 Full-RAED Cube：

```text
q_i(v) = normalize(log(1 + C_t[:, r_i, a_i, e_i]))
```

因此 Doppler 标签来源是当前帧 Cube cell，可逐点追踪；target 几何仅用于监督
endpoint，不决定 Doppler bin。

## 4. 无泄漏边界

- 参数在 train 最稀疏帧容量审计后冻结。
- preflight 可以只读检查 train/validation 的 adapter 容量，但不能据 validation
  几何调整半径、planarity ratio 或间距。
- 不访问 test。
- 不使用类别、3D box、track ID、未来帧、预测 checkpoint 或评估 GT selector。
- validation target 只用于验证监督 adapter 能否按同一冻结规则构造 endpoint。
- 推理期模型仅输入当前 Cube；continuous lifting 不进入部署路径。

## 5. Dense 兼容性

当原始 exact-unique XYZ 已达到 10,000 时，`continuous_fixed_target` 必须直接
调用旧 `canonical_fixed_target` 并原样返回 XYZ、confidence、source indices 和
hashes。新 5 cm spacing gate 只约束 sparse lifting 分支，不能事后改写已归档的
dense endpoint，否则旧 P-RF preflight 将失去字节级可比性。

因此两条分支的证据语义不同：

- dense compatibility：旧版 exact-unique、无放回、Morton 均匀采样；
- sparse lifting：exact-10k、至少 5 cm 欧氏间距、occupied-cell support。

论文和实验表必须披露这一分支差异。

## 6. Preflight 门

正式训练前必须在 H200 通过：

1. 100 帧 train/validation 中所有 `<10k` 帧均构造 exact 10,000；
2. 最稀疏 train 帧确实走 lifting 分支；
3. selected sparse endpoint 的最小点间距不小于 5 cm；
4. 所有 endpoint RAE cells 属于该帧原始 occupied target cells；
5. Doppler target shape 为 `10,000 x 64`，且来源为当前帧 Cube 查询；
6. 最大 dense 帧与旧 adapter 的 XYZ、confidence、indices、hashes 字节一致；
7. 原 P-RF 的 one-to-one transport、梯度、wrong-Cube sensitivity、NFE=4 和
   H200 显存检查继续通过。

任一 sparse 帧容量不足即 no-go。不得通过复制点、降低 5 cm 间距、打开未占据
RAE cell、读取未来帧或按 validation 结果扩大 patch 来补救。

## 7. 证据边界

该协议本轮未通过。现有结果只说明机制在 47/57 个稀疏帧上具备受约束容量，
不能说明 P-RF target representation 可覆盖完整训练分布，更不能证明 lifted
surface 是真实未采样表面或 P-RF 几何优于现有模型。它不授权长训练，也不改变
G1D、G1G、RaLD-WCE 或时序路线的任何结论。
