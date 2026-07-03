#!/usr/bin/env python
"""P1 验证: 3D 框静/动分离后, 分别检验主线 A 的两条物理关系(TruckScenes mini).

  静态(背景+静止目标): v_r ≈ -(v_ego+ω×p)·r̂    [解析硬约束; 对照无 ω×r 修正]
  动态(运动目标):      v_r ≈ (v_obj - v_plat)·r̂  [一致性软约束, v_obj 取自框速度]

产出: results/static_dynamic_validation.txt / .png
"""
import os
import sys
from collections import Counter

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.truckscenes_loader import TruckScenesRadar, label_static_dynamic  # noqa: E402
from eval.physics import (static_residual, dynamic_residual,                 # noqa: E402
                          consistency_report, fmt_report)

from truckscenes import TruckScenes  # noqa: E402

DATAROOT = os.path.expanduser("~/data/radar_gen/truckscenes/man-truckscenes")
VER = "v1.2-mini"
RES = os.path.expanduser("~/Workspace/radar_gen/results")
os.makedirs(RES, exist_ok=True)

tsc = TruckScenes(VER, DATAROOT, verbose=False)
ldr = TruckScenesRadar(tsc)

res_bg, res_bg_noomega, res_statobj, res_dyn = [], [], [], []
dyn_cats, label_counts = Counter(), Counter()
n_frames = n_skip_slow = 0

for scene in tsc.scene:
    tok = scene["first_sample_token"]
    while tok:
        smp = tsc.get("sample", tok)
        for ch in [c for c in smp["data"] if c.startswith("RADAR")]:
            fr = ldr.load_frame(smp["data"][ch])
            if fr["v_ego_norm"] < 2.0:
                n_skip_slow += 1
                continue
            fr_no = ldr.load_frame(smp["data"][ch], yaw_rate_correction=False)
            labels, v_obj = label_static_dynamic(fr)
            label_counts.update(labels.tolist())

            bg = labels == 0
            so = labels == 1
            dy = labels == 2
            res_bg.append(static_residual(fr["v_r"][bg], fr["pred_static_vr"][bg]))
            res_bg_noomega.append(static_residual(fr_no["v_r"][bg], fr_no["pred_static_vr"][bg]))
            res_statobj.append(static_residual(fr["v_r"][so], fr["pred_static_vr"][so]))
            if dy.any():
                res_dyn.append(dynamic_residual(fr["v_r"][dy], fr["rhat"][dy],
                                                v_obj[dy], fr["v_plat_s"][dy]))
                for b in fr["boxes"]:
                    if b["v_obj_sensor"] is not None and np.linalg.norm(b["v_obj_sensor"]) > 0.5:
                        dyn_cats[b["name"]] += 1
            n_frames += 1
        tok = smp["next"]

cat = {k: np.concatenate(v) if v else np.array([]) for k, v in
       dict(bg=res_bg, bg_no=res_bg_noomega, so=res_statobj, dyn=res_dyn).items()}

total = sum(label_counts.values())
print(f"== 帧数: {n_frames}(跳过低速 {n_skip_slow})  总点数: {total}")
print(f"== 静/动比例: 背景 {label_counts[0]/total*100:.1f}% | 框内-静止 "
      f"{label_counts[1]/total*100:.1f}% | 框内-运动 {label_counts[2]/total*100:.1f}%")
print(f"== 运动目标类别 Top8: {dyn_cats.most_common(8)}")
print()

reports = [
    consistency_report(cat["bg"], "背景静态 (含 ω×r 修正)"),
    consistency_report(cat["bg_no"], "背景静态 (无 ω×r, 对照)"),
    consistency_report(cat["so"], "框内静止目标"),
    consistency_report(cat["dyn"], "运动目标 vs (v_obj-v_plat)·r̂"),
]
for r in reports:
    print(fmt_report(r))

# 图: 三类残差直方图
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, key, title in [(axes[0], "bg", "static background"),
                       (axes[1], "so", "static objects (in-box)"),
                       (axes[2], "dyn", "moving objects vs box-velocity")]:
    r = cat[key]
    r = r[np.isfinite(r)]
    ax.hist(np.clip(r, -6, 6), bins=150)
    ax.axvline(0, color="r", lw=0.8)
    med = np.median(r) if len(r) else float("nan")
    mad = np.median(np.abs(r - med)) if len(r) else float("nan")
    ax.set_title(f"{title}\nN={len(r)}, med={med:+.3f}, MAD={mad:.3f}")
    ax.set_xlabel("residual m/s")
fig.tight_layout()
fig.savefig(f"{RES}/static_dynamic_validation.png", dpi=130)
print(f"\n== 图已存: {RES}/static_dynamic_validation.png")
print("== DONE")
