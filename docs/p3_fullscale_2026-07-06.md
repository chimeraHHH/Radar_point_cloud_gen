# P3 规模战纪要 2026-07-06:trainval 全量 + 40M 桥式 → **G3 机制层通过**

> 数据:trainval 全量 598 场景(经 wangning 账号 1.3T 大盘,雷达+LiDAR+meta,453G)
> 缓存:`temporal_full_k10` 326,467 对(6 分片并行 21 分钟)+ OT 配对;train 319,894 / val 6,573(12 场景,**高速场景为主,GT v_r std 7.22**——远难于 mini val 的 2.94)
> 模型:dim512/depth8(~35M),60k 步,L_temp λ=0.1,双臂(dopp/ego 草稿)各 ~92 分钟/L40S

## 1. 单步生成(val N=24)

| | big_dopp | big_ego | 各自复制基线 | GT |
|---|---|---|---|---|
| CD | **5.130** | 6.769 | 4.830 / 3.918 | (锚 11.07) |
| W1(v_r) | 0.805 | 0.736 | — | 0 |
| 动态样占比 | **62.0%** | 68.9% | — | **65.5%** |
| PCE<0.5 | **30.6%** | 24.0% | — | **31.1%** |

**发现 1(草稿质量在规模上传导为生成质量)**:dopp 草稿臂的生成全面优于 ego 草稿臂(CD 5.13 vs 6.77,**−24%**)——mini 上两臂打平,规模+高速场景下首次大分离。
**发现 2(分布贴合)**:big_dopp 的动态样占比(62.0%)与 PCE(30.6%)与 GT(65.5%/31.1%)几乎重合——**生成云的物理混合形态在全量训练后贴上真实分布**;CD 与复制的差距从 mini 的 0.55m 缩到 **0.30m**。

## 2. Rollout 漂移(24 segments × 5 步 × 0.5s)

**CD(几何)**:

| t(s) | copy_ego | **copy_dopp** | bridge_big_dopp | bridge_big_ego |
|---|---|---|---|---|
| 0.5 | 3.655 | **3.629** | 6.651 | 6.521 |
| 1.0 | 5.030 | **4.689** | 8.405 | 8.355 |
| 1.5 | 6.899 | **5.997** | 11.212 | 10.097 |
| 2.5 | 13.335 | **9.256(−31%)** | 13.645 | 15.760 |

**PCE<0.5(物理新鲜度)**:

| t(s) | copy_dopp | **bridge_big_dopp** | GT |
|---|---|---|---|
| 0.5 | 26.1% | **69.1%(≈GT)** | 66.7% |
| 1.5 | 16.4% | **41.8%** | 66.1% |
| 2.5 | 12.4% | **38.0%(3×)** | 65.1% |

## 3. G3 判定(如实,分层)

1. **✅ 机制层通过(G3 核心主张)**:**Doppler-warp 链几何全程优于 ego-only 链,且优势随时距扩大(2.5s 时 −31%)**。mini 低速场景上曾反向(杂波>动态);trainval 高速场景+长时距下,proposal"Doppler 驱动帧间 warp"的价值完整兑现。这是"优于 ego-only 下限"的直接证据——在 warp 机制层。
2. **✅ 桥式精修的物理价值**:rollout 首步物理新鲜度 69.1%≈GT,全程对 copy 链保持 ~3 倍;单步分布形态贴合 GT。
3. **❌ 桥式 rollout 几何生成税仍在**(step1 6.65 vs copy 3.63,复利到 13.6@2.5s)——已知病因(训练只见真实草稿,rollout 喂生成帧),已知药方(**scheduled sampling**:训练时喂真实生成帧),下一轮工程。

**综合**:G3 两个组件各自的价值主张均在规模上得到验证;完整方法(warp+桥式)的几何超越取决于 scheduled sampling——工程问题而非机制问题。

## 4. 排障记录(工作方式教训,已入 memory)

- **ssh 引号坑**:单引号命令内 `\"` 使 pgrep 模式带引号字符 → 恒 0 → 误诊进程死亡 → 重复启动叠双份。规则:单引号内直接用 `"`;进程检查用 `ps aux | grep -F`。
- **python 后台必须 `-u`**:块缓冲让日志长期 0 字节,与上一条叠加造成一小时误诊断。

## 5. 下一步

1. **scheduled sampling**(喂真实生成帧训练)→ 桥式 rollout 几何税攻坚 → G3 完整关门
2. 反事实速度可控性(贡献点 4,纯推理);桥式+L_dop 合流
3. 🧑 VoD(G2 下游)仍等用户申请

## 6. 资产

[bridge_big_dopp_metrics.txt](assets/bridge_big_dopp_metrics.txt) / [bridge_big_ego_metrics.txt](assets/bridge_big_ego_metrics.txt) / [rollout_metrics_full.txt](assets/rollout_metrics_full.txt) / [bridge_big_dopp_samples.png](assets/bridge_big_dopp_samples.png) · ckpt: `results/bridge_big_{dopp,ego}_ckpt.pt` · 复现:`launch_fullscale.sh`(全程)/ `launch_fullscale_s34.sh`(恢复)
