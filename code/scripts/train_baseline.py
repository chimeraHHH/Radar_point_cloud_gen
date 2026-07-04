#!/usr/bin/env python
"""P1 基线正式训练(G1): 全量配对 + 场景划分 train/val + CFG + EMA + 完整指标.

产出: results/baseline_{log.txt,ckpt.pt,metrics.txt,samples.png,loss.png}
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.point_diffusion import RadarPointDenoiser  # noqa: E402
from eval.gen_metrics import full_report, chamfer      # noqa: E402

from diffusers import DDPMScheduler                    # noqa: E402

PAIRS = os.path.expanduser("~/data/radar_gen/truckscenes/pairs_mini")
RES = os.path.expanduser("~/Workspace/radar_gen/results")
os.makedirs(RES, exist_ok=True)
STEPS, BS, LR = 40000, 64, 3e-4
CFG_DROP, CFG_W, EMA_DECAY = 0.1, 2.0, 0.999
N_EVAL = 24
torch.manual_seed(0)
np.random.seed(0)

# ---- 数据: 按场景划分(后 2 个场景为 val, 防泄漏) ----
mani = json.load(open(f"{PAIRS}/manifest.json"))
scenes = sorted({m["scene"] for m in mani["pairs"]})
val_scenes = set(scenes[-2:])
tr = [m for m in mani["pairs"] if m["scene"] not in val_scenes]
va = [m for m in mani["pairs"] if m["scene"] in val_scenes]
print(f"scenes={len(scenes)}  train={len(tr)}  val={len(va)}  val_scenes={sorted(val_scenes)}")


def load(ms):
    r = np.stack([np.load(f"{PAIRS}/{m['file']}")["radar"] for m in ms])
    l = np.stack([np.load(f"{PAIRS}/{m['file']}")["lidar"] for m in ms])
    return r, l


r_tr, l_tr = load(tr)
r_va, l_va = load(va)
R_MU = r_tr.reshape(-1, 5).mean(0); R_SD = r_tr.reshape(-1, 5).std(0) + 1e-6
L_MU = l_tr.reshape(-1, 4).mean(0); L_SD = l_tr.reshape(-1, 4).std(0) + 1e-6
print("radar mu:", R_MU.round(2), "sd:", R_SD.round(2))

dev = torch.device("cuda")
Rtr = torch.tensor((r_tr - R_MU) / R_SD, dtype=torch.float32, device=dev)
Ltr = torch.tensor((l_tr - L_MU) / L_SD, dtype=torch.float32, device=dev)
Rva = torch.tensor((r_va - R_MU) / R_SD, dtype=torch.float32, device=dev)
Lva = torch.tensor((l_va - L_MU) / L_SD, dtype=torch.float32, device=dev)

model = RadarPointDenoiser(dim=256, depth=6, heads=8).to(dev)
print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
opt = torch.optim.AdamW(model.parameters(), lr=LR)
lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

log, t0 = [], time.time()
for step in range(1, STEPS + 1):
    idx = torch.randint(0, len(tr), (BS,), device=dev)
    x0, cond = Rtr[idx], Ltr[idx]
    t = torch.randint(0, 1000, (BS,), device=dev)
    noise = torch.randn_like(x0)
    xt = sched.add_noise(x0, noise, t)
    drop = torch.rand(BS, device=dev) < CFG_DROP
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = F.mse_loss(model(xt, t, cond, drop), noise)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step(); lr_sched.step()
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                ema[k].mul_(EMA_DECAY).add_(v, alpha=1 - EMA_DECAY)
            else:
                ema[k].copy_(v)
    if step % 1000 == 0 or step == 1:
        with torch.no_grad():
            vi = torch.randint(0, len(va), (BS,), device=dev)
            vt = torch.randint(0, 1000, (BS,), device=dev)
            vn = torch.randn_like(Rva[vi])
            vloss = F.mse_loss(model(sched.add_noise(Rva[vi], vn, vt), vt, Lva[vi]), vn)
        log.append((step, loss.item(), vloss.item()))
        print(f"step {step:6d}  train {loss.item():.4f}  val {vloss.item():.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

torch.save(dict(ema=ema, r_mu=R_MU, r_sd=R_SD, l_mu=L_MU, l_sd=L_SD,
                val_scenes=sorted(val_scenes)), f"{RES}/baseline_ckpt.pt")
model.load_state_dict(ema)
model.eval()

# ---- CFG 采样 N_EVAL 个 val 条件 ----
eidx = np.linspace(0, len(va) - 1, N_EVAL).astype(int)
cond = Lva[eidx]
samp = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
samp.set_timesteps(1000)
with torch.no_grad():
    x = torch.randn(N_EVAL, Rva.shape[1], 5, device=dev)
    dropF = torch.zeros(N_EVAL, dtype=torch.bool, device=dev)
    dropT = torch.ones(N_EVAL, dtype=torch.bool, device=dev)
    for t in samp.timesteps:
        tb = t.expand(N_EVAL).to(dev)
        e_c = model(x, tb, cond, dropF)
        e_u = model(x, tb, cond, dropT)
        x = samp.step(e_u + CFG_W * (e_c - e_u), t, x).prev_sample
gen = x.cpu().numpy() * R_SD + R_MU
gt = r_va[eidx]

# ---- 指标 ----
reps = [full_report(gen[i], gt[i]) for i in range(N_EVAL)]
cd_anchor = [chamfer(gt[i], gt[(i + N_EVAL // 2) % N_EVAL]) for i in range(N_EVAL)]
med = lambda key: float(np.median([r[key] for r in reps]))
lines = [
    f"val 指标(N={N_EVAL}, 中位):",
    f"  CD         = {med('cd'):.3f} m   (GT 互比锚: {np.median(cd_anchor):.3f} m)",
    f"  CD_Doppler = {med('cd_dopp'):.3f}",
    f"  MMD-RBF    = {med('mmd'):.5f}",
    f"  JSD        = {med('jsd'):.4f}",
    f"  v_r std    gen={np.median([r['vr_std_gen'] for r in reps]):.2f}"
    f"  gt={np.median([r['vr_std_gt'] for r in reps]):.2f}",
]
report = "\n".join(lines)
print("\n" + report)
open(f"{RES}/baseline_metrics.txt", "w").write(report + "\n")

# ---- 图 ----
ls = np.array(log)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ls[:, 0], ls[:, 1], label="train")
ax.plot(ls[:, 0], ls[:, 2], label="val")
ax.set_yscale("log"); ax.legend(); ax.set_title("baseline eps-MSE")
fig.tight_layout(); fig.savefig(f"{RES}/baseline_loss.png", dpi=120)

fig, axes = plt.subplots(2, 4, figsize=(17, 7))
for k in range(4):
    for row, cloud, tag in ((0, gt[k * (N_EVAL // 4)], "GT"), (1, gen[k * (N_EVAL // 4)], "gen")):
        a = axes[row, k]
        s = a.scatter(cloud[:, 0], cloud[:, 1], c=np.clip(cloud[:, 3], -15, 15), s=4, cmap="coolwarm")
        a.set_title(f"{tag} val#{k * (N_EVAL // 4)} (color=v_r)"); a.set_aspect("equal")
fig.tight_layout(); fig.savefig(f"{RES}/baseline_samples.png", dpi=120)
print("== DONE")
