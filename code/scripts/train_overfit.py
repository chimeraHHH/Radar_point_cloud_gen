#!/usr/bin/env python
"""P1 基线冒烟: 在 64 个配对上过拟合最小点扩散, 验证"loss 降 + 可采样 + 指标可算".

产出: results/overfit_{log.txt,loss.png,samples.png,ckpt.pt}
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

from diffusers import DDPMScheduler, DDIMScheduler     # noqa: E402

PAIRS = os.path.expanduser("~/data/radar_gen/truckscenes/pairs_mini")
RES = os.path.expanduser("~/Workspace/radar_gen/results")
os.makedirs(RES, exist_ok=True)
N_PAIRS, STEPS, BS, LR = 64, 3000, 16, 2e-4
torch.manual_seed(0)

mani = json.load(open(f"{PAIRS}/manifest.json"))
st = mani["stats"]
sel = [m for m in mani["pairs"] if m["v_ego_norm"] > 2.0][:N_PAIRS]
print(f"pairs={len(sel)} (要求 {N_PAIRS})")


def norm_radar(r):
    out = r.copy()
    out[:, :3] /= st["radar_xyz_scale"]
    out[:, 3] /= st["vr_scale"]
    out[:, 4] = (out[:, 4] - st["rcs_mean"]) / (st["rcs_std"] + 1e-6)
    return out


def denorm_radar(r):
    out = r.copy()
    out[:, :3] *= st["radar_xyz_scale"]
    out[:, 3] *= st["vr_scale"]
    out[:, 4] = out[:, 4] * (st["rcs_std"] + 1e-6) + st["rcs_mean"]
    return out


def norm_lidar(l):
    out = l.copy()
    out[:, :3] /= st["lidar_xyz_scale"]
    out[:, 3] = (out[:, 3] - st["intensity_mean"]) / (st["intensity_std"] + 1e-6)
    return out


radar = np.stack([norm_radar(np.load(f"{PAIRS}/{m['file']}")["radar"]) for m in sel])
lidar = np.stack([norm_lidar(np.load(f"{PAIRS}/{m['file']}")["lidar"]) for m in sel])
dev = torch.device("cuda")
radar_t = torch.tensor(radar, dtype=torch.float32, device=dev)
lidar_t = torch.tensor(lidar, dtype=torch.float32, device=dev)

model = RadarPointDenoiser().to(dev)
n_par = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=LR)
sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
print(f"model params={n_par/1e6:.2f}M")

log = []
t0 = time.time()
for step in range(1, STEPS + 1):
    idx = torch.randint(0, len(sel), (BS,), device=dev)
    x0, cond = radar_t[idx], lidar_t[idx]
    t = torch.randint(0, 1000, (BS,), device=dev)
    noise = torch.randn_like(x0)
    xt = sched.add_noise(x0, noise, t)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = F.mse_loss(model(xt, t, cond), noise)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % 100 == 0 or step == 1:
        log.append((step, loss.item()))
        print(f"step {step:5d}  loss {loss.item():.4f}  ({time.time()-t0:.0f}s)")

torch.save(dict(model=model.state_dict(), stats=st), f"{RES}/overfit_ckpt.pt")

# ---- DDIM 采样 8 个(条件取训练集前 8 对) ----
model.eval()
ddim = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
ddim.set_timesteps(100)
with torch.no_grad():
    x = torch.randn(8, radar_t.shape[1], 5, device=dev)
    cond = lidar_t[:8]
    for t in ddim.timesteps:
        eps = model(x, t.expand(8).to(dev), cond)
        x = ddim.step(eps, t, x).prev_sample
gen = np.stack([denorm_radar(g) for g in x.cpu().numpy()])
gt = np.stack([denorm_radar(g) for g in radar[:8]])


def chamfer_xyz(a, b):
    d = torch.cdist(torch.tensor(a[:, :3]), torch.tensor(b[:, :3]))
    return float(d.min(1).values.mean() + d.min(0).values.mean()) / 2


cds = [chamfer_xyz(gen[i], gt[i]) for i in range(8)]
cds_rand = [chamfer_xyz(gen[i], gt[(i + 3) % 8]) for i in range(8)]  # 错配对照
print(f"\nChamfer(gen vs 配对GT):  med={np.median(cds):.2f} m  {['%.1f' % c for c in cds]}")
print(f"Chamfer(gen vs 错配GT):  med={np.median(cds_rand):.2f} m (应明显更大)")
print(f"v_r 分布: GT std={gt[:, :, 3].std():.2f}  gen std={gen[:, :, 3].std():.2f}")

# ---- 图: loss 曲线 + BEV 对比 ----
fig, axes = plt.subplots(2, 5, figsize=(19, 7))
ax = axes[0, 0]
ls = np.array(log)
ax.plot(ls[:, 0], ls[:, 1]); ax.set_yscale("log"); ax.set_title("train loss (MSE eps)")
for k in range(4):
    for row, cloud, tag in ((0, gt[k], "GT"), (1, gen[k], "gen")):
        a = axes[row, k + 1]
        s = a.scatter(cloud[:, 0], cloud[:, 1], c=np.clip(cloud[:, 3], -15, 15),
                      s=4, cmap="coolwarm")
        a.set_title(f"{tag} #{k} (color=v_r)"); a.set_aspect("equal")
axes[1, 0].axis("off")
fig.colorbar(s, ax=axes[1, 0], fraction=0.4)
fig.tight_layout()
fig.savefig(f"{RES}/overfit_samples.png", dpi=120)
print(f"== 图已存: {RES}/overfit_samples.png")
print("== DONE")
