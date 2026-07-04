#!/usr/bin/env python
"""P2 消融: 基线 vs 基线+自门控静态物理约束. 用法: python train_ablate.py <lam> <tag>

lam=0 → 纯基线(对照); lam>0 → + L_phys(x̂0 桥接, ᾱ_t 加权).
产出: results/ablate_<tag>_{log.txt,ckpt.pt,metrics.txt,samples.png}
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
from models.point_diffusion import RadarPointDenoiser          # noqa: E402
from eval.gen_metrics import full_report, chamfer              # noqa: E402
from losses.physics import self_gated_static_loss, pce_report  # noqa: E402

from diffusers import DDPMScheduler                            # noqa: E402

LAM = float(sys.argv[1])
TAG = sys.argv[2]
PAIRS = os.path.expanduser("~/data/radar_gen/truckscenes/pairs_mini_v2")
RES = os.path.expanduser("~/Workspace/radar_gen/results")
os.makedirs(RES, exist_ok=True)
STEPS, BS, LR = 40000, 64, 3e-4
CFG_DROP, CFG_W, EMA_DECAY = 0.1, 2.0, 0.999
N_EVAL = 24
torch.manual_seed(0)
np.random.seed(0)
print(f"== ablate tag={TAG} lam={LAM}")

mani = json.load(open(f"{PAIRS}/manifest.json"))
scenes = sorted({m["scene"] for m in mani["pairs"]})
val_scenes = set(scenes[-2:])
tr = [m for m in mani["pairs"] if m["scene"] not in val_scenes]
va = [m for m in mani["pairs"] if m["scene"] in val_scenes]
print(f"train={len(tr)} val={len(va)}")


def load(ms):
    r, l, e = [], [], []
    for m in ms:
        z = np.load(f"{PAIRS}/{m['file']}")
        r.append(z["radar"]); l.append(z["lidar"])
        e.append(np.concatenate([z["v_ego_s"], z["omega_s"], z["t_s"]]))
    return np.stack(r), np.stack(l), np.stack(e)


r_tr, l_tr, e_tr = load(tr)
r_va, l_va, e_va = load(va)
R_MU = r_tr.reshape(-1, 5).mean(0); R_SD = r_tr.reshape(-1, 5).std(0) + 1e-6
L_MU = l_tr.reshape(-1, 4).mean(0); L_SD = l_tr.reshape(-1, 4).std(0) + 1e-6

dev = torch.device("cuda")
Rtr = torch.tensor((r_tr - R_MU) / R_SD, dtype=torch.float32, device=dev)
Ltr = torch.tensor((l_tr - L_MU) / L_SD, dtype=torch.float32, device=dev)
Etr = torch.tensor(e_tr, dtype=torch.float32, device=dev)      # (B,9)=v_ego|omega|t_s
Rva = torch.tensor((r_va - R_MU) / R_SD, dtype=torch.float32, device=dev)
Lva = torch.tensor((l_va - L_MU) / L_SD, dtype=torch.float32, device=dev)
Eva = torch.tensor(e_va, dtype=torch.float32, device=dev)
MU_t = torch.tensor(R_MU, dtype=torch.float32, device=dev)
SD_t = torch.tensor(R_SD, dtype=torch.float32, device=dev)
E_MU = e_tr.mean(0); E_SD = e_tr.std(0) + 1e-3
EtrN = torch.tensor((e_tr - E_MU) / E_SD, dtype=torch.float32, device=dev)
EvaN = torch.tensor((e_va - E_MU) / E_SD, dtype=torch.float32, device=dev)

model = RadarPointDenoiser(dim=256, depth=6, heads=8).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
acp = sched.alphas_cumprod.to(dev)
ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

t0 = time.time()
for step in range(1, STEPS + 1):
    idx = torch.randint(0, len(tr), (BS,), device=dev)
    x0, cond, ego = Rtr[idx], Ltr[idx], Etr[idx]
    egoN = EtrN[idx]
    t = torch.randint(0, 1000, (BS,), device=dev)
    noise = torch.randn_like(x0)
    xt = sched.add_noise(x0, noise, t)
    drop = torch.rand(BS, device=dev) < CFG_DROP
    with torch.autocast("cuda", dtype=torch.bfloat16):
        eps = model(xt, t, cond, drop, egoN)
        loss_mse = F.mse_loss(eps, noise)
        loss = loss_mse
        if LAM > 0:
            ab = acp[t]
            x0_hat = (xt - (1 - ab).sqrt()[:, None, None] * eps) / ab.sqrt()[:, None, None]
            x0_phys = x0_hat * SD_t + MU_t
            loss_phys = self_gated_static_loss(x0_phys, ego[:, :3], ego[:, 3:6],
                                               ego[:, 6:9], step_w=ab)
            loss = loss_mse + LAM * loss_phys
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step(); lr_sched.step()
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                ema[k].mul_(EMA_DECAY).add_(v, alpha=1 - EMA_DECAY)
            else:
                ema[k].copy_(v)
    if step % 2000 == 0 or step == 1:
        lp = float(loss_phys) if LAM > 0 else 0.0
        print(f"step {step:6d}  mse {float(loss_mse):.4f}  phys {lp:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

torch.save(dict(ema=ema, r_mu=R_MU, r_sd=R_SD, e_mu=E_MU, e_sd=E_SD, lam=LAM), f"{RES}/ablate_{TAG}_ckpt.pt")
model.load_state_dict(ema)
model.eval()

# ---- 采样(与基线同协议: CFG w=2, DDPM-1000)----
eidx = np.linspace(0, len(va) - 1, N_EVAL).astype(int)
cond, egoE = Lva[eidx], Eva[eidx]
egoEN = EvaN[eidx]
samp = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
samp.set_timesteps(1000)
with torch.no_grad():
    x = torch.randn(N_EVAL, Rva.shape[1], 5, device=dev)
    dF = torch.zeros(N_EVAL, dtype=torch.bool, device=dev)
    dT = torch.ones(N_EVAL, dtype=torch.bool, device=dev)
    for t in samp.timesteps:
        tb = t.expand(N_EVAL).to(dev)
        e_c = model(x, tb, cond, dF, egoEN)
        e_u = model(x, tb, cond, dT, egoEN)
        x = samp.step(e_u + CFG_W * (e_c - e_u), t, x).prev_sample
gen = x.cpu().numpy() * R_SD + R_MU
gt = r_va[eidx]

# ---- 指标: 生成质量 + PCE(物理一致性)----
reps = [full_report(gen[i], gt[i]) for i in range(N_EVAL)]
cd_anchor = [chamfer(gt[i], gt[(i + N_EVAL // 2) % N_EVAL]) for i in range(N_EVAL)]
gen_t = torch.tensor(gen, dtype=torch.float32)
gt_t = torch.tensor(gt, dtype=torch.float32)
egoC = egoE.cpu()
pce_gen = pce_report(gen_t, egoC[:, :3], egoC[:, 3:6], egoC[:, 6:9])
pce_gt = pce_report(gt_t, egoC[:, :3], egoC[:, 3:6], egoC[:, 6:9])
med = lambda k: float(np.median([r[k] for r in reps]))
report = "\n".join([
    f"tag={TAG} lam={LAM}  val N={N_EVAL} 中位:",
    f"  CD={med('cd'):.3f} m (锚 {np.median(cd_anchor):.3f})  CD_Doppler={med('cd_dopp'):.3f}",
    f"  MMD={med('mmd'):.5f}  JSD={med('jsd'):.4f}",
    f"  v_r std gen={np.median([r['vr_std_gen'] for r in reps]):.2f} gt={np.median([r['vr_std_gt'] for r in reps]):.2f}",
    f"  PCE(gen): med|r|={pce_gen['med_abs']:.3f}  <0.5: {pce_gen['frac<0.5']*100:.1f}%",
    f"  PCE(GT):  med|r|={pce_gt['med_abs']:.3f}  <0.5: {pce_gt['frac<0.5']*100:.1f}%",
])
print("\n" + report)
open(f"{RES}/ablate_{TAG}_metrics.txt", "w").write(report + "\n")

fig, axes = plt.subplots(2, 4, figsize=(17, 7))
for k in range(4):
    j = k * (N_EVAL // 4)
    for row, cloud, tag2 in ((0, gt[j], "GT"), (1, gen[j], TAG)):
        a = axes[row, k]
        a.scatter(cloud[:, 0], cloud[:, 1], c=np.clip(cloud[:, 3], -15, 15), s=4, cmap="coolwarm")
        a.set_title(f"{tag2} val#{j}"); a.set_aspect("equal")
fig.tight_layout(); fig.savefig(f"{RES}/ablate_{TAG}_samples.png", dpi=120)
print("== DONE")
