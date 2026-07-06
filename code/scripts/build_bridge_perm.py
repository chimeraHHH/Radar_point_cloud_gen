#!/usr/bin/env python
"""为时序配对缓存补充 OT 配对 sidecar(桥式扩散用). 用法: python build_bridge_perm.py [pairs_dir]

对每对 (draft, GT) 在 xyz 上做匈牙利指派, 存 perm_{ego,dopp}: GT 第 i 行对应 draft 的行号.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

PDIR = sys.argv[1] if len(sys.argv) > 1 else "temporal_mini_k10"
SHARD = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NSHARD = int(sys.argv[3]) if len(sys.argv) > 3 else 1
PAIRS = os.path.expanduser(f"~/data/radar_gen/truckscenes/{PDIR}")

mani = json.load(open(f"{PAIRS}/manifest.json"))
todo = mani["pairs"][SHARD::NSHARD]
t0 = time.time()
for n, m in enumerate(todo):
    out = os.path.join(PAIRS, m["file"].replace(".npz", ".perm.npz"))
    if os.path.exists(out):
        continue
    z = np.load(os.path.join(PAIRS, m["file"]))
    gt = z["radar"][:, :3]
    perms = {}
    for c in ("ego", "dopp"):
        cost = cdist(gt, z[f"cond_{c}"][:, :3])          # (Ngt, Ndraft)
        _, col = linear_sum_assignment(np.clip(cost, 0, 50.0))
        perms[f"perm_{c}"] = col.astype(np.int16)
    np.savez_compressed(out, **perms)
    if (n + 1) % 2000 == 0:
        print(f"shard{SHARD}: {n+1}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
print(f"== shard{SHARD} 完成 {len(todo)} 对 ({time.time()-t0:.0f}s)")
