#!/usr/bin/env python
"""trainval 全量时序配对构建(分片并行). 用法: python build_temporal_full.py <shard> <num_shards> [stride=4] [K=10]

场景按名排序后 scenes[shard::num_shards];输出 npz + manifest_shard<i>.json(由编排器合并)。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.truckscenes_loader import TruckScenesRadar     # noqa: E402
from data.temporal_pairs import build_temporal_pair      # noqa: E402

from truckscenes import TruckScenes                      # noqa: E402

SHARD, NSHARD = int(sys.argv[1]), int(sys.argv[2])
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 4
K = int(sys.argv[4]) if len(sys.argv) > 4 else 10
DATAROOT = os.environ.get(
    "TSROOT", "/storage/data/metaiot_data/wangning_radar/truckscenes/man-truckscenes")
VER = os.environ.get("TSVER", "v1.2-trainval")
OUT = os.path.expanduser(f"~/data/radar_gen/truckscenes/temporal_full_k{K}")
os.makedirs(OUT, exist_ok=True)

tsc = TruckScenes(VER, DATAROOT, verbose=False)
ldr = TruckScenesRadar(tsc)
rng = np.random.default_rng(1000 + SHARD)
scenes = sorted(tsc.scene, key=lambda s: s["name"])[SHARD::NSHARD]
print(f"shard {SHARD}/{NSHARD}: {len(scenes)} scenes", flush=True)

manifest, n_skip = [], 0
for si, scene in enumerate(scenes):
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
            name = f"{scene['name'][-12:]}_{ch}_{i:04d}.npz"
            np.savez_compressed(os.path.join(OUT, name), **pair)
            manifest.append(dict(file=name, scene=scene["name"], channel=ch,
                                 v_ego_norm=float(pair["v_ego_norm"]),
                                 dt=float(pair["dt"])))
        frames.clear()
    if (si + 1) % 10 == 0:
        print(f"shard{SHARD}: {si+1}/{len(scenes)} scenes, pairs={len(manifest)}", flush=True)

json.dump(manifest, open(os.path.join(OUT, f"manifest_shard{SHARD}.json"), "w"))
print(f"shard{SHARD} 完成: {len(manifest)} 对(跳过 {n_skip})", flush=True)
