#!/usr/bin/env python
"""三元组缓存(scheduled sampling 用): (t-K, t, t+K). 用法: python build_triples_full.py <shard> <nshard> [stride=4] [K=10]

存: cond0_dopp(t-K→t 草稿, t 系) | ego_t | A,b(t→t+K 传感器变换, 行向量 p1=p@A.T+b) | dt2
    radar1(目标) | ego1 | cond1_dopp/cp1(OT对齐) | p01(cond1_ego[perm], L_temp 锚)
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

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
OUT = os.path.expanduser(f"~/data/radar_gen/truckscenes/triples_full_k{K}")
os.makedirs(OUT, exist_ok=True)

tsc = TruckScenes(VER, DATAROOT, verbose=False)
ldr = TruckScenesRadar(tsc)
rng = np.random.default_rng(2000 + SHARD)
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
        for i in range(K, len(chain) - K, STRIDE):
            for j in (i - K, i, i + K):
                if j not in frames:
                    frames[j] = ldr.load_frame(chain[j], load_boxes=False)
            f0, f1, f2 = frames[i - K], frames[i], frames[i + K]
            pA = build_temporal_pair(f0, f1, rng=rng)     # t-K → t
            pB = build_temporal_pair(f1, f2, rng=rng)     # t → t+K
            if pA is None or pB is None:
                n_skip += 1
                continue
            # OT 配对(真实分支)
            cost = cdist(pB["radar"][:, :3], pB["cond_dopp"][:, :3])
            _, col = linear_sum_assignment(np.clip(cost, 0, 50.0))
            # t→t+K 传感器变换(行向量): p1 = p @ A.T + b
            A = f2["R_gs"].T @ f1["R_gs"]
            b = f2["R_gs"].T @ (f1["t_gs"] - f2["t_gs"])
            ego_t = np.concatenate([f1["v_ego_s"], f1["omega_s"], f1["t_s"]]).astype(np.float32)
            name = f"{scene['name'][-12:]}_{ch}_{i:04d}.npz"
            np.savez_compressed(os.path.join(OUT, name),
                                cond0=pA["cond_dopp"], ego_t=ego_t,
                                A=A.astype(np.float32), b=b.astype(np.float32),
                                dt2=pB["dt"],
                                radar1=pB["radar"], ego1=pB["ego"],
                                cond1=pB["cond_dopp"],
                                cp1=pB["cond_dopp"][col],
                                p01=pB["cond_ego"][col])
            manifest.append(dict(file=name, scene=scene["name"], channel=ch,
                                 v_ego_norm=float(pB["v_ego_norm"]), dt=float(pB["dt"])))
        frames.clear()
    if (si + 1) % 10 == 0:
        print(f"shard{SHARD}: {si+1}/{len(scenes)}, triples={len(manifest)}", flush=True)

json.dump(manifest, open(os.path.join(OUT, f"manifest_shard{SHARD}.json"), "w"))
print(f"shard{SHARD} 完成: {len(manifest)} 三元组(跳过 {n_skip})", flush=True)
