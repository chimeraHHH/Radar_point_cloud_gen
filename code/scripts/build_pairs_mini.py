#!/usr/bin/env python
"""构建 mini 的 LiDAR→Radar 配对缓存(npz)+ manifest(P1 基线数据面)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.truckscenes_loader import TruckScenesRadar            # noqa: E402
from data.pairs_truckscenes import load_lidars_global, build_pair  # noqa: E402

from truckscenes import TruckScenes                             # noqa: E402

DATAROOT = os.path.expanduser("~/data/radar_gen/truckscenes/man-truckscenes")
VER = "v1.2-mini"
OUT = os.path.expanduser("~/data/radar_gen/truckscenes/pairs_mini")
os.makedirs(OUT, exist_ok=True)

tsc = TruckScenes(VER, DATAROOT, verbose=False)
ldr = TruckScenesRadar(tsc)
rng = np.random.default_rng(42)

manifest, n_skip = [], 0
for si, scene in enumerate(tsc.scene):
    tok = scene["first_sample_token"]
    while tok:
        smp = tsc.get("sample", tok)
        lidar_g = load_lidars_global(tsc, smp)
        for ch in sorted(c for c in smp["data"] if c.startswith("RADAR")):
            pair = build_pair(tsc, ldr, smp, ch, lidar_g, rng=rng)
            if pair is None:
                n_skip += 1
                continue
            name = f"{smp['token'][:12]}_{ch}.npz"
            np.savez_compressed(os.path.join(OUT, name),
                                lidar=pair["lidar"], radar=pair["radar"],
                                pred_static_vr=pair["pred_static_vr"],
                                v_ego_norm=pair["v_ego_norm"])
            manifest.append(dict(file=name, channel=ch, scene=scene["name"],
                                 sample_token=smp["token"],
                                 v_ego_norm=float(pair["v_ego_norm"])))
        tok = smp["next"]
    print(f"scene {si+1}/10 done, pairs={len(manifest)}")

# 归一化统计(供训练端)
rad = np.concatenate([np.load(os.path.join(OUT, m["file"]))["radar"] for m in manifest[::10]])
lid_int = np.concatenate([np.load(os.path.join(OUT, m["file"]))["lidar"][:, 3] for m in manifest[::10]])
stats = dict(
    radar_xyz_scale=60.0, vr_scale=15.0,
    rcs_mean=float(rad[:, 4].mean()), rcs_std=float(rad[:, 4].std()),
    lidar_xyz_scale=60.0,
    intensity_mean=float(lid_int.mean()), intensity_std=float(lid_int.std()),
)
json.dump(dict(pairs=manifest, stats=stats), open(os.path.join(OUT, "manifest.json"), "w"))
print(f"== 完成: {len(manifest)} 对(跳过 {n_skip}), 输出 {OUT}")
print(f"== stats: {stats}")
