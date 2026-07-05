#!/usr/bin/env python
"""G3 判据实验: 多步自回归 rollout 漂移测试(val 场景, 步距 K=10≈0.5s × 5 步 = 2.5s).

四臂对照(每步 vs 真实帧的 CD / PCE):
  copy_ego   : 初始真实帧逐步纯 ego-warp(G3 的 "ego-only 下限")
  copy_dopp  : 每步门控 Doppler 推进 + ego-warp(v_r 通道陈旧)
  bridge_ego : 每步 ego-warp 草稿 → 桥式模型精修(v_r 每步更新)
  bridge_dopp: 每步 dopp-warp 草稿 → 桥式模型精修
产出: results/rollout_metrics.txt
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.truckscenes_loader import TruckScenesRadar           # noqa: E402
from models.point_diffusion import RadarPointDenoiser          # noqa: E402
from eval.gen_metrics import chamfer                           # noqa: E402
from losses.physics import pce_report                          # noqa: E402

from truckscenes import TruckScenes                            # noqa: E402

DATAROOT = os.path.expanduser("~/data/radar_gen/truckscenes/man-truckscenes")
RES = os.path.expanduser("~/Workspace/radar_gen/results")
K, T_STEPS, START, NPTS, ODE = 10, 5, 40, 384, 50
dev = torch.device("cuda")
rng = np.random.default_rng(0)

tsc = TruckScenes("v1.2-mini", DATAROOT, verbose=False)
ldr = TruckScenesRadar(tsc)
scenes = sorted(tsc.scene, key=lambda s: s["name"])
val_scenes = scenes[-2:]

# ---- 收集 12 个 segment(2 场景 × 6 通道), 每个 T_STEPS+1 帧 ----
segs = []
for scene in val_scenes:
    smp0 = tsc.get("sample", scene["first_sample_token"])
    for ch in sorted(c for c in smp0["data"] if c.startswith("RADAR")):
        chain, tok = [], smp0["data"][ch]
        while tok:
            chain.append(tok)
            tok = tsc.get("sample_data", tok)["next"]
        idxs = [START + m * K for m in range(T_STEPS + 1)]
        if idxs[-1] >= len(chain):
            continue
        frames = [ldr.load_frame(chain[i], load_boxes=False) for i in idxs]
        if any(len(f["xyz"]) < 50 for f in frames):
            continue
        segs.append(frames)
print(f"segments={len(segs)}")


def np_pred_static(xyz, ego):
    v = ego[:3][None] + np.cross(np.broadcast_to(ego[3:6], xyz.shape), xyz + ego[6:9][None])
    rhat = xyz / (np.linalg.norm(xyz, axis=1, keepdims=True) + 1e-6)
    return -(v * rhat).sum(1)


def ego_vecs(fr):
    return np.concatenate([fr["v_ego_s"], fr["omega_s"], fr["t_s"]]).astype(np.float32)


def make_draft(cloud5, ego_prev, fr_prev, fr_next, mode):
    """cloud5 在 fr_prev 传感器系; 返回 fr_next 传感器系的草稿(5ch)."""
    xyz, vr = cloud5[:, :3], cloud5[:, 3]
    dt = (fr_next["timestamp"] - fr_prev["timestamp"]) / 1e6
    if mode == "dopp":
        res = vr - np_pred_static(xyz, ego_prev)
        adv = np.where(np.abs(res) > 1.0, res, 0.0)
        rhat = xyz / (np.linalg.norm(xyz, axis=1, keepdims=True) + 1e-6)
        xyz = xyz + adv[:, None] * dt * rhat
    g = xyz @ fr_prev["R_gs"].T + fr_prev["t_gs"]
    p1 = (g - fr_next["t_gs"]) @ fr_next["R_gs"]
    return np.concatenate([p1, cloud5[:, 3:5]], 1).astype(np.float32)


def sample_pts(fr):
    idx = rng.choice(len(fr["xyz"]), NPTS, replace=len(fr["xyz"]) < NPTS)
    return np.concatenate([fr["xyz"][idx], fr["v_r"][idx, None], fr["rcs"][idx, None]], 1).astype(np.float32)


# ---- 加载桥式模型(argv 指定 tag 列表) ----
TAGS = sys.argv[1:] if len(sys.argv) > 1 else ["br_ego", "br_dopp"]
models = {}
for tag in TAGS:
    ck = torch.load(f"{RES}/bridge_{tag}_ckpt.pt", map_location="cpu", weights_only=False)
    m = RadarPointDenoiser(dim=256, depth=6, heads=8, pt_ch=5, lidar_ch=5).to(dev)
    m.load_state_dict(ck["ema"]); m.eval()
    models[tag] = (m, ck)


def bridge_step(tag, drafts, egos):
    """drafts (B,N,5) 物理单位 → 精修后 (B,N,5)."""
    m, ck = models[tag]
    R_MU, R_SD, E_MU, E_SD = ck["r_mu"], ck["r_sd"], ck["e_mu"], ck["e_sd"]
    x = torch.tensor((drafts - R_MU) / R_SD, dtype=torch.float32, device=dev)
    condt = x.clone()
    egoN = torch.tensor((egos - E_MU) / E_SD, dtype=torch.float32, device=dev)
    with torch.no_grad():
        dt = 1.0 / ODE
        for k in range(ODE):
            t = torch.full((len(x),), k * dt, device=dev)
            x = x + dt * m(x, t * 999, condt, None, egoN)
    return x.cpu().numpy() * R_SD + R_MU


# ---- rollout ----
ARMS = ["copy_ego", "copy_dopp"] + [f"bridge_{t}" for t in TAGS]
MODE = {f"bridge_{t}": models[t][1]["cond"] for t in TAGS}
MODE.update(copy_ego="ego", copy_dopp="dopp")
state = {a: [sample_pts(s[0]) for s in segs] for a in ARMS}
rows = []
pce_rows = []
for step in range(1, T_STEPS + 1):
    # 各臂推进
    for arm in state:
        mode = MODE[arm]
        drafts = np.stack([
            make_draft(state[arm][i], ego_vecs(segs[i][step - 1]), segs[i][step - 1], segs[i][step], mode)
            for i in range(len(segs))])
        if arm.startswith("bridge"):
            egos = np.stack([ego_vecs(segs[i][step]) for i in range(len(segs))])
            out = bridge_step(arm[len("bridge_"):], drafts, egos)
            state[arm] = [out[i] for i in range(len(segs))]
        else:
            state[arm] = [drafts[i] for i in range(len(segs))]
    # 指标
    cds = {}
    for arm in state:
        cds[arm] = float(np.median([
            chamfer(state[arm][i], sample_pts(segs[i][step])) for i in range(len(segs))]))
    rows.append((step, cds))
    egos_t = torch.tensor(np.stack([ego_vecs(segs[i][step]) for i in range(len(segs))]), dtype=torch.float32)
    pces = {}
    for arm in ["copy_dopp"] + [f"bridge_{t}" for t in TAGS]:
        cl = torch.tensor(np.stack(state[arm]), dtype=torch.float32)
        pces[arm] = pce_report(cl, egos_t[:, :3], egos_t[:, 3:6], egos_t[:, 6:9])["frac<0.5"]
    gtc = torch.tensor(np.stack([sample_pts(segs[i][step]) for i in range(len(segs))]), dtype=torch.float32)
    pces["GT"] = pce_report(gtc, egos_t[:, :3], egos_t[:, 3:6], egos_t[:, 6:9])["frac<0.5"]
    pce_rows.append((step, pces))

hdr = " ".join(f"{a:>12s}" for a in ARMS)
lines = [f"Rollout 漂移测试 (K=10≈0.5s/步, {len(segs)} segments, val 场景)",
         f"{'step':>4s} {'t(s)':>5s} {hdr}   [CD med, m]"]
for step, cds in rows:
    vals = " ".join(f"{cds[a]:>12.3f}" for a in ARMS)
    lines.append(f"{step:>4d} {step*0.5:>5.1f} {vals}")
lines.append("")
pcearms = ["copy_dopp"] + [f"bridge_{t}" for t in TAGS] + ["GT"]
lines.append("PCE<0.5:  " + " ".join(f"{a:>12s}" for a in pcearms))
for step, p in pce_rows:
    lines.append(f"step{step}:  " + " ".join(f"{p[a]*100:>11.1f}%" for a in pcearms))
report = "\n".join(lines)
print(report)
open(f"{RES}/rollout_metrics_v2.txt", "w").write(report + "\n")
print("== DONE")
