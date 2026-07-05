#!/usr/bin/env python
"""构建 mini sweeps 时序配对缓存 (t, t+K), K=5≈0.25s, 双条件云(P3 主线 B)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.truckscenes_loader import TruckScenesRadar     # noqa: E402
from data.temporal_pairs import build_temporal_pair      # noqa: E402

from truckscenes import TruckScenes                      # noqa: E402

DATAROOT = os.path.expanduser("~/data/radar_gen/truckscenes/man-truckscenes")
VER = "v1.2-mini"
import sys as _sys
K = int(_sys.argv[1]) if len(_sys.argv) > 1 else 5
STRIDE = 2
OUT = os.path.expanduser(f"~/data/radar_gen/truckscenes/temporal_mini_k{K}")
os.makedirs(OUT, exist_ok=True)

tsc = TruckScenes(VER, DATAROOT, verbose=False)
ldr = TruckScenesRadar(tsc)
rng = np.random.default_rng(42)

manifest, n_skip = [], 0
for si, scene in enumerate(tsc.scene):
    smp0 = tsc.get("sample", scene["first_sample_token"])
    for ch in sorted(c for c in smp0["data"] if c.startswith("RADAR")):
        chain, tok = [], smp0["data"][ch]
        while tok:
            chain.append(tok)
            tok = tsc.get("sample_data", tok)["next"]
        frames = {}
        for i in range(0, len(chain) - K, STRIDE):
            for j in (i, i + K):
                if j not in frames:
                    frames[j] = ldr.load_frame(chain[j], load_boxes=False)
            pair = build_temporal_pair(frames[i], frames[i + K], rng=rng)
            if pair is None:
                n_skip += 1
                continue
            name = f"s{si}_{ch}_{i:04d}.npz"
            np.savez_compressed(os.path.join(OUT, name), **pair)
            manifest.append(dict(file=name, scene=scene["name"], channel=ch,
                                 v_ego_norm=float(pair["v_ego_norm"]),
                                 dt=float(pair["dt"])))
        frames.clear()
    print(f"scene {si+1}/10 done, pairs={len(manifest)}", flush=True)

json.dump(dict(pairs=manifest, K=K, stride=STRIDE),
          open(os.path.join(OUT, "manifest.json"), "w"))
print(f"== 完成: {len(manifest)} 对(跳过 {n_skip}), 输出 {OUT}")
