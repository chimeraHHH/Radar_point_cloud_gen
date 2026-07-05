#!/usr/bin/env python
"""诊断: 时序配对中 draft 与 GT 的 v_r 符号/量级一致性(排查桥式样本图色相不符)."""
import json
import os

import numpy as np

P = os.path.expanduser("~/data/radar_gen/truckscenes/temporal_mini_k10")
mani = json.load(open(f"{P}/manifest.json"))
scenes = sorted({m["scene"] for m in mani["pairs"]})
va = [m for m in mani["pairs"] if m["scene"] in set(scenes[-2:])]
eidx = np.linspace(0, len(va) - 1, 24).astype(int)

print(f"{'j':>3s} {'scene':>14s} {'channel':>18s} {'dt':>5s} {'|v_ego|':>7s} "
      f"{'vr_draft(med)':>13s} {'vr_GT(med)':>10s} {'符号一致%':>8s}")
for j in [0, 3, 6, 9, 12, 15, 18, 21]:
    m = va[eidx[j]]
    z = np.load(os.path.join(P, m["file"]))
    d, g = z["cond_dopp"], z["radar"]
    # NN 匹配后逐点符号一致率(阈值 0.5 内视为同号)
    from scipy.spatial import cKDTree
    _, nn = cKDTree(d[:, :3]).query(g[:, :3], k=1)
    a, b = g[:, 3], d[nn, 3]
    m_valid = (np.abs(a) > 0.5) & (np.abs(b) > 0.5)
    agree = float((np.sign(a[m_valid]) == np.sign(b[m_valid])).mean()) if m_valid.sum() else np.nan
    print(f"{j:>3d} {m['scene'][-14:]:>14s} {m['channel']:>18s} {m['dt']:>5.2f} "
          f"{m['v_ego_norm']:>7.2f} {np.median(d[:, 3]):>13.2f} {np.median(g[:, 3]):>10.2f} "
          f"{agree*100:>7.1f}%")
print("\n若某行符号一致率低且 vr 中位数反号 → 配对/传输 bug;若都正常 → 图上的色差是个别动态点/色标问题")
