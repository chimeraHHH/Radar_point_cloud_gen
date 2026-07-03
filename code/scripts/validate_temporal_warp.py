#!/usr/bin/env python
"""B 线物理内核预演: sweeps(~20Hz) 上验证 Doppler 驱动的帧间 warp(TruckScenes mini).

把第 t 帧点云推进到 t+K 帧, 对比三种策略与真实 t+K 帧的最近邻(NN)距离:
  (a) identity   : 不作任何补偿(运动幅度参照)
  (b) ego-warp   : 仅按自车位姿刚体变换(proposal 的"稳健下限")
  (c) dopp-warp  : ego-warp + 残差 Doppler 径向推进 p += (v_r - pred_static)·Δt·r̂
若 (c) 在动态点(|残差 Doppler|>1 m/s)上优于 (b), 则 v_r·Δt ≈ Δrange 的时序耦合可用于生成.

产出: results/temporal_warp_validation.txt / .png
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.truckscenes_loader import TruckScenesRadar  # noqa: E402

from truckscenes import TruckScenes  # noqa: E402

DATAROOT = os.path.expanduser("~/data/radar_gen/truckscenes/man-truckscenes")
VER = "v1.2-mini"
RES = os.path.expanduser("~/Workspace/radar_gen/results")
os.makedirs(RES, exist_ok=True)

KS = (1, 5)           # 跨 K 个 sweep(K=1 ≈ 57ms, K=5 ≈ 0.29s)
STRIDE = 3            # 起始帧步长(控制运行时长)
DYN_THR = 1.0         # 残差 Doppler 判动态阈值 m/s
CLIP = 10.0           # NN 距离截断 m

tsc = TruckScenes(VER, DATAROOT, verbose=False)
ldr = TruckScenesRadar(tsc)

# 收集每 scene × radar 通道的完整 sweep 链
chains = []
for scene in tsc.scene:
    smp0 = tsc.get("sample", scene["first_sample_token"])
    for ch in [c for c in smp0["data"] if c.startswith("RADAR")]:
        chain, tok = [], smp0["data"][ch]
        while tok:
            chain.append(tok)
            tok = tsc.get("sample_data", tok)["next"]
        chains.append((scene["name"], ch, chain))

print(f"== 链数: {len(chains)}, sweep 总帧数: {sum(len(c[2]) for c in chains)}")

# 逐链预载(不带框, 快), 再做 K 步 warp 对比
acc = {K: {s: [] for s in ("a", "b", "c")} for K in KS}        # 全部点
acc_dyn = {K: {s: [] for s in ("a", "b", "c")} for K in KS}    # 仅动态点
dts = {K: [] for K in KS}
n_pairs = 0

for sname, ch, chain in chains:
    frames = []
    for tok in chain:
        fr = ldr.load_frame(tok, load_boxes=False)
        frames.append(fr)
    for K in KS:
        for i in range(0, len(frames) - K, STRIDE):
            f0, f1 = frames[i], frames[i + K]
            if len(f0["xyz"]) < 20 or len(f1["xyz"]) < 20:
                continue
            dt = (f1["timestamp"] - f0["timestamp"]) / 1e6
            if not (0 < dt < 0.15 * K + 0.1):
                continue
            dts[K].append(dt)
            tree = cKDTree(f1["xyz"])
            res_dopp = f0["v_r"] - f0["pred_static_vr"]
            dyn = np.abs(res_dopp) > DYN_THR

            # (a) identity
            da = tree.query(f0["xyz"], k=1)[0]
            # (b) ego-warp: sensor_t -> global -> sensor_{t+K}
            pg = f0["xyz"] @ f0["R_gs"].T + f0["t_gs"]
            pb = (pg - f1["t_gs"]) @ f1["R_gs"]
            db = tree.query(pb, k=1)[0]
            # (c) ego + Doppler 残差径向推进(传感器 t 系内先推进)
            padv = f0["xyz"] + (res_dopp * dt)[:, None] * f0["rhat"]
            pg2 = padv @ f0["R_gs"].T + f0["t_gs"]
            pc = (pg2 - f1["t_gs"]) @ f1["R_gs"]
            dc = tree.query(pc, k=1)[0]

            for s, d in (("a", da), ("b", db), ("c", dc)):
                acc[K][s].append(np.clip(d, 0, CLIP))
                if dyn.any():
                    acc_dyn[K][s].append(np.clip(d[dyn], 0, CLIP))
            n_pairs += 1

print(f"== 帧对数: {n_pairs}")


def stats(dlist):
    if not dlist:
        return None
    d = np.concatenate(dlist)
    return dict(n=len(d), med=float(np.median(d)), mean=float(d.mean()),
                lt1=float(np.mean(d < 1.0)), lt2=float(np.mean(d < 2.0)))


for K in KS:
    dt_med = np.median(dts[K]) if dts[K] else float("nan")
    print(f"\n== K={K} (Δt med={dt_med*1000:.0f} ms) "
          f"===============================================")
    for tag, A in (("全部点", acc[K]), ("动态点(|res_dopp|>1)", acc_dyn[K])):
        print(f"   -- {tag} --")
        for s, name in (("a", "identity"), ("b", "ego-warp"), ("c", "dopp-warp")):
            st = stats(A[s])
            if st is None:
                print(f"   {name:>10s}: (无)")
                continue
            print(f"   {name:>10s}: N={st['n']:>8d}  NN med={st['med']:.3f} m  "
                  f"mean={st['mean']:.3f}  <1m:{st['lt1']*100:5.1f}%  <2m:{st['lt2']*100:5.1f}%")

# 图: 动态点 NN 距离分布(每个 K 一行)
fig, axes = plt.subplots(1, len(KS), figsize=(6 * len(KS), 4))
axes = np.atleast_1d(axes)
for ax, K in zip(axes, KS):
    for s, name, color in (("a", "identity", "gray"),
                           ("b", "ego-warp", "tab:orange"),
                           ("c", "dopp-warp", "tab:blue")):
        if not acc_dyn[K][s]:
            continue
        d = np.concatenate(acc_dyn[K][s])
        ax.hist(d, bins=100, range=(0, 8), histtype="step", lw=1.6,
                label=f"{name} (med={np.median(d):.2f}m)", color=color)
    ax.set_title(f"dynamic points, K={K}")
    ax.set_xlabel("NN distance to real next frame (m)")
    ax.legend()
fig.tight_layout()
fig.savefig(f"{RES}/temporal_warp_validation.png", dpi=130)
print(f"\n== 图已存: {RES}/temporal_warp_validation.png")
print("== DONE")
