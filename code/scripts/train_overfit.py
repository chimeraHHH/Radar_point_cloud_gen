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
N_PAIRS, STEPS, BS, LR = 64, 20000, 32, 3e-4
torch.manual_seed(0)

mani = json.load(open(f"{PAIRS}/manifest.json"))
cand = [m for m in mani["pairs"] if m["v_ego_norm"] > 2.0]
sel = cand[:: max(1, len(cand) // N_PAIRS)][:N_PAIRS]          # 跨场景/通道抽样
# 评估对: 8 个不同 scene 各取一对(都在训练集内, 仍是过拟合测试)
eval_idx, seen = [], set()
for i, m in enumerate(sel):
    if m["scene"] not in seen:
        seen.add(m["scene"]); eval_idx.append(i)
    if len(eval_idx) == 8:
        break
E = len(eval_idx)
print("eval scenes:", [sel[i]["scene"][:14] for i in eval_idx])
print(f"pairs={len(sel)} (要求 {N_PAIRS})")

radar_raw = np.stack([np.load(f"{PAIRS}/{m['file']}")["radar"] for m in sel])
lidar_raw = np.stack([np.load(f"{PAIRS}/{m['file']}")["lidar"] for m in sel])

# 逐通道标准化(零均值/单位方差)——教训: 固定尺度除法留下 x 均值~1.3、v_r std~0.13, 扩散学不动
R_MU = radar_raw.reshape(-1, 5).mean(0)
R_SD = radar_raw.reshape(-1, 5).std(0) + 1e-6
L_MU = lidar_raw.reshape(-1, 4).mean(0)
L_SD = lidar_raw.reshape(-1, 4).std(0) + 1e-6
print("radar mu:", R_MU.round(2), "sd:", R_SD.round(2))


def denorm_radar(r):
    return r * R_SD + R_MU


radar = (radar_raw - R_MU) / R_SD
lidar = (lidar_raw - L_MU) / L_SD
dev = torch.device("cuda")
radar_t = torch.tensor(radar, dtype=torch.float32, device=dev)
lidar_t = torch.tensor(lidar, dtype=torch.float32, device=dev)

model = RadarPointDenoiser(dim=256, depth=6, heads=8).to(dev)
n_par = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=LR)
lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
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
    lr_sched.step()
    if step % 500 == 0 or step == 1:
        log.append((step, loss.item()))
        print(f"step {step:5d}  loss {loss.item():.4f}  ({time.time()-t0:.0f}s)")

torch.save(dict(model=model.state_dict(), r_mu=R_MU, r_sd=R_SD, l_mu=L_MU, l_sd=L_SD),
           f"{RES}/overfit_ckpt.pt")

# ---- DDIM 采样 8 个(条件取训练集前 8 对) ----
model.eval()
ddim = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
ddim.set_timesteps(1000)
with torch.no_grad():
    x = torch.randn(E, radar_t.shape[1], 5, device=dev)
    cond = lidar_t[eval_idx]
    for t in ddim.timesteps:
        eps = model(x, t.expand(E).to(dev), cond)
        x = ddim.step(eps, t, x).prev_sample
gen = np.stack([denorm_radar(g) for g in x.cpu().numpy()])
gt = np.stack([denorm_radar(g) for g in radar[eval_idx]])


def chamfer_xyz(a, b):
    d = torch.cdist(torch.tensor(a[:, :3]), torch.tensor(b[:, :3]))
    return float(d.min(1).values.mean() + d.min(0).values.mean()) / 2


cds = [chamfer_xyz(gen[i], gt[i]) for i in range(E)]
cds_rand = [chamfer_xyz(gen[i], gt[(i + 3) % E]) for i in range(E)]      # 跨场景错配
cds_gtgt = [chamfer_xyz(gt[i], gt[(i + 3) % E]) for i in range(E)]       # GT 间跨场景基准
print(f"\nChamfer(gen vs 配对GT):    med={np.median(cds):.2f} m  {['%.1f' % c for c in cds]}")
print(f"Chamfer(gen vs 跨场景GT):  med={np.median(cds_rand):.2f} m (条件若生效应明显更大)")
print(f"Chamfer(GT vs 跨场景GT):   med={np.median(cds_gtgt):.2f} m (场景间本底差异)")
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
