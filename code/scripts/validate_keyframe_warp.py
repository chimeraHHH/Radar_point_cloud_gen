#!/usr/bin/env python
"""B 线预演 v2: 关键帧对(Δt≈0.5s) + 框标注圈定真动态点的 Doppler-warp 对比.

v1(sweeps, 残差门限圈动态)教训: 未加运动分割门控时, 32% 被圈点多为杂波,
径向推进放大噪声 → K=5 反而变差. 本脚本用 GT 框做门控, 并加 oracle 上限:
  (a) identity    (b) ego-warp    (c) ego + 残差Doppler·Δt·r̂    (d) ego + v_obj·Δt(oracle 全速度向量)
仅在 label==2(框内运动)点上评估; 背景静态点作对照(b≈c 应成立).

产出: results/keyframe_warp_validation.txt / .png
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.truckscenes_loader import TruckScenesRadar, label_static_dynamic  # noqa: E402

from truckscenes import TruckScenes  # noqa: E402

DATAROOT = os.path.expanduser("~/data/radar_gen/truckscenes/man-truckscenes")
VER = "v1.2-mini"
RES = os.path.expanduser("~/Workspace/radar_gen/results")
os.makedirs(RES, exist_ok=True)
CLIP = 15.0

tsc = TruckScenes(VER, DATAROOT, verbose=False)
ldr = TruckScenesRadar(tsc)

acc_dyn = {s: [] for s in "abcd"}
acc_bg = {s: [] for s in "abc"}
disp_dyn = []          # 动态点真实位移幅度(oracle 向量模)
n_pairs = 0

for scene in tsc.scene:
    tok = scene["first_sample_token"]
    prev = None
    while tok:
        smp = tsc.get("sample", tok)
        if prev is not None:
            for ch in [c for c in smp["data"] if c.startswith("RADAR")]:
                f0 = ldr.load_frame(prev["data"][ch])
                f1 = ldr.load_frame(smp["data"][ch], load_boxes=False)
                if len(f0["xyz"]) < 20 or len(f1["xyz"]) < 20:
                    continue
                dt = (f1["timestamp"] - f0["timestamp"]) / 1e6
                if not (0.2 < dt < 0.8):
                    continue
                labels, v_obj = label_static_dynamic(f0)
                dyn = (labels == 2) & np.all(np.isfinite(v_obj), axis=1)
                bg = labels == 0
                tree = cKDTree(f1["xyz"])
                res_dopp = f0["v_r"] - f0["pred_static_vr"]

                def to_next(p):
                    g = p @ f0["R_gs"].T + f0["t_gs"]
                    return (g - f1["t_gs"]) @ f1["R_gs"]

                pa = f0["xyz"]
                pb = to_next(pa)
                pc = to_next(pa + (res_dopp * dt)[:, None] * f0["rhat"])
                pd = to_next(pa + np.nan_to_num(v_obj) * dt)

                if dyn.any():
                    for s, p in (("a", pa), ("b", pb), ("c", pc), ("d", pd)):
                        acc_dyn[s].append(np.clip(tree.query(p[dyn], k=1)[0], 0, CLIP))
                    disp_dyn.append(np.linalg.norm(v_obj[dyn], axis=1) * dt)
                if bg.any():
                    sub = np.where(bg)[0][::5]        # 背景抽 1/5 控制量
                    for s, p in (("a", pa), ("b", pb), ("c", pc)):
                        acc_bg[s].append(np.clip(tree.query(p[sub], k=1)[0], 0, CLIP))
                n_pairs += 1
        prev = smp
        tok = smp["next"]

print(f"== 关键帧对(×通道)数: {n_pairs}, Δt≈0.5s")
D = np.concatenate(disp_dyn)
print(f"== 动态点真实位移 |v_obj|·Δt: med={np.median(D):.2f} m, P90={np.percentile(D,90):.2f} m")

names = dict(a="identity", b="ego-warp", c="dopp-warp (radial)", d="oracle (full v_obj)")
for tag, A in (("动态点(GT框, 真运动)", acc_dyn), ("背景静态点(对照)", acc_bg)):
    print(f"\n   -- {tag} --")
    for s in "abcd":
        if s not in A or not A[s]:
            continue
        d = np.concatenate(A[s])
        print(f"   {names[s]:>18s}: N={len(d):>7d}  NN med={np.median(d):.3f} m  "
              f"mean={d.mean():.3f}  <1m:{np.mean(d < 1)*100:5.1f}%  <2m:{np.mean(d < 2)*100:5.1f}%")

fig, ax = plt.subplots(figsize=(7.5, 4.6))
for s, color in (("a", "gray"), ("b", "tab:orange"), ("c", "tab:blue"), ("d", "tab:green")):
    d = np.concatenate(acc_dyn[s])
    ax.hist(d, bins=120, range=(0, 10), histtype="step", lw=1.7, color=color,
            label=f"{names[s]} (med={np.median(d):.2f}m)")
ax.set_title("GT-box dynamic points, keyframe pairs (dt=0.5s)")
ax.set_xlabel("NN distance to real next frame (m)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{RES}/keyframe_warp_validation.png", dpi=130)
print(f"\n== 图已存: {RES}/keyframe_warp_validation.png")
print("== DONE")
