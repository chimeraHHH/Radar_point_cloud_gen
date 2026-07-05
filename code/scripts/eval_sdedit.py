#!/usr/bin/env python
"""SDEdit 式部分扩散评估(纯推理): 从 warp 条件云加噪至 t_start 再去噪.

用法: python eval_sdedit.py <cond: ego|dopp> <ckpt_tag> [pairs_dir=temporal_mini_k5]
t_start=0 即"复制条件"基线; t_start=1000 即从头生成; 中间为部分扩散。
产出: results/sdedit_<ckpt_tag>_metrics.txt
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.point_diffusion import RadarPointDenoiser          # noqa: E402
from eval.gen_metrics import chamfer, cd_doppler               # noqa: E402
from losses.physics import pce_report, static_pred_vr          # noqa: E402

from diffusers import DDPMScheduler                            # noqa: E402

COND = sys.argv[1]
TAG = sys.argv[2]
PDIR = sys.argv[3] if len(sys.argv) > 3 else "temporal_mini_k5"
PAIRS = os.path.expanduser(f"~/data/radar_gen/truckscenes/{PDIR}")
RES = os.path.expanduser("~/Workspace/radar_gen/results")
T_STARTS = [0, 200, 400, 600, 1000]
N_EVAL, CFG_W = 24, 2.0

ck = torch.load(f"{RES}/temporal_{TAG}_ckpt.pt", map_location="cpu", weights_only=False)
R_MU, R_SD, E_MU, E_SD = ck["r_mu"], ck["r_sd"], ck["e_mu"], ck["e_sd"]
dev = torch.device("cuda")
model = RadarPointDenoiser(dim=256, depth=6, heads=8, pt_ch=5, lidar_ch=5).to(dev)
model.load_state_dict(ck["ema"])
model.eval()

mani = json.load(open(f"{PAIRS}/manifest.json"))
scenes = sorted({m["scene"] for m in mani["pairs"]})
va = [m for m in mani["pairs"] if m["scene"] in set(scenes[-2:])]
eidx = np.linspace(0, len(va) - 1, N_EVAL).astype(int)
r, c, e = [], [], []
for i in eidx:
    z = np.load(f"{PAIRS}/{va[i]['file']}")
    r.append(z["radar"]); c.append(z[f"cond_{COND}"]); e.append(z["ego"])
gt = np.stack(r); cnd = np.stack(c); ego = np.stack(e)
cond_t = torch.tensor((cnd - R_MU) / R_SD, dtype=torch.float32, device=dev)
egoN = torch.tensor((ego - E_MU) / E_SD, dtype=torch.float32, device=dev)
egoC = torch.tensor(ego, dtype=torch.float32)

samp = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
samp.set_timesteps(1000)
dF = torch.zeros(N_EVAL, dtype=torch.bool, device=dev)
dT = torch.ones(N_EVAL, dtype=torch.bool, device=dev)


def dynfrac(cl):
    ct = torch.tensor(cl, dtype=torch.float32)
    rr = ct[..., 3] - static_pred_vr(ct[..., :3], egoC[:, :3], egoC[:, 3:6], egoC[:, 6:9])
    return float((rr.abs() > 1.0).float().mean())


q = np.linspace(0, 1, 200)
lines = [f"SDEdit 扫描 tag={TAG} cond={COND} pairs={PDIR}  (val N={N_EVAL} 中位)",
         f"{'t_start':>8s} {'CD':>7s} {'CD_dopp':>8s} {'PCE<0.5':>8s} {'W1vr':>6s} {'dyn%':>6s}"]
for ts in T_STARTS:
    torch.manual_seed(0)
    with torch.no_grad():
        if ts == 0:
            gen = cnd.copy()
        else:
            if ts >= 1000:
                x = torch.randn_like(cond_t)
            else:
                noise = torch.randn_like(cond_t)
                tt = torch.full((N_EVAL,), ts - 1, dtype=torch.long, device=dev)
                x = samp.add_noise(cond_t, noise, tt)
            for t in samp.timesteps:
                if int(t) >= ts:
                    continue
                tb = t.expand(N_EVAL).to(dev)
                e_c = model(x, tb, cond_t, dF, egoN)
                e_u = model(x, tb, cond_t, dT, egoN)
                x = samp.step(e_u + CFG_W * (e_c - e_u), t, x).prev_sample
            gen = x.cpu().numpy() * R_SD + R_MU
    cd = float(np.median([chamfer(gen[i], gt[i]) for i in range(N_EVAL)]))
    cdd = float(np.median([cd_doppler(gen[i], gt[i]) for i in range(N_EVAL)]))
    pce = pce_report(torch.tensor(gen, dtype=torch.float32),
                     egoC[:, :3], egoC[:, 3:6], egoC[:, 6:9])
    w1 = float(np.abs(np.quantile(gen[:, :, 3], q) - np.quantile(gt[:, :, 3], q)).mean())
    lines.append(f"{ts:>8d} {cd:>7.3f} {cdd:>8.3f} {pce['frac<0.5']*100:>7.1f}% "
                 f"{w1:>6.3f} {dynfrac(gen)*100:>5.1f}%")
pce_gt = pce_report(torch.tensor(gt, dtype=torch.float32), egoC[:, :3], egoC[:, 3:6], egoC[:, 6:9])
lines.append(f"{'GT参照':>8s} {'-':>7s} {'-':>8s} {pce_gt['frac<0.5']*100:>7.1f}% "
             f"{0.0:>6.3f} {dynfrac(gt)*100:>5.1f}%")
report = "\n".join(lines)
print(report)
open(f"{RES}/sdedit_{TAG}_metrics.txt", "w").write(report + "\n")
print("== DONE")
