#!/usr/bin/env python
"""P3 桥式扩散(rectified-flow, draft→GT 传输). 用法: python train_bridge.py <cond: ego|dopp> <tag> [pairs_dir]

x_t = (1-t)·draft_π + t·GT + σ(1-t)·ε,  模型预测位移场 v ≈ GT − draft_π;
推理从草稿出发 Euler 积分 N=50 步。复制基线是 v≡0 特例 → 构造上不应差于复制。
静态物理约束施加在 x̂1 = x_t + (1-t)·v̂ 上(λ=0.1, 权重 t)。
产出: results/bridge_<tag>_{log.txt,ckpt.pt,metrics.txt,samples.png}
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
from losses.physics import (self_gated_static_loss, pce_report,  # noqa: E402
                            static_pred_vr)

COND = sys.argv[1]
TAG = sys.argv[2]
PDIR = sys.argv[3] if len(sys.argv) > 3 else "temporal_mini_k10"
AUG = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
LAM_TEMP = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
DIM = int(sys.argv[6]) if len(sys.argv) > 6 else 256
DEPTH = int(sys.argv[7]) if len(sys.argv) > 7 else 6
assert COND in ("ego", "dopp")
PAIRS = os.path.expanduser(f"~/data/radar_gen/truckscenes/{PDIR}")
RES = os.path.expanduser("~/Workspace/radar_gen/results")
os.makedirs(RES, exist_ok=True)
STEPS = int(os.environ.get("STEPS", 40000))
BS, LR, LAM, SIGMA = 64, 3e-4, 0.1, 0.05
EMA_DECAY, N_EVAL, ODE_STEPS = 0.999, 24, 50
torch.manual_seed(0)
np.random.seed(0)
print(f"== bridge tag={TAG} cond={COND} pairs={PDIR} aug={AUG} lam_temp={LAM_TEMP}")

mani = json.load(open(f"{PAIRS}/manifest.json"))
scenes = sorted({m["scene"] for m in mani["pairs"]})
val_scenes = set(scenes[-int(os.environ.get("VAL_N", "2")):])
tr = [m for m in mani["pairs"] if m["scene"] not in val_scenes]
va = [m for m in mani["pairs"] if m["scene"] in val_scenes]
print(f"train={len(tr)} val={len(va)}")


def load(ms):
    r, c, cp, p0, e, dt = [], [], [], [], [], []
    for m in ms:
        z = np.load(f"{PAIRS}/{m['file']}")
        zp = np.load(f"{PAIRS}/{m['file'].replace('.npz', '.perm.npz')}")
        perm = zp[f"perm_{COND}"]
        r.append(z["radar"]); c.append(z[f"cond_{COND}"])
        cp.append(z[f"cond_{COND}"][perm])                   # 与 GT 行对齐的草稿
        p0.append(z["cond_ego"][perm])                       # 同一批 prev 点的纯 ego-warp 位置(L_temp 锚)
        e.append(z["ego"]); dt.append(z["dt"])
    return (np.stack(r), np.stack(c), np.stack(cp), np.stack(p0),
            np.stack(e), np.array(dt, dtype=np.float32))


r_tr, c_tr, cp_tr, p0_tr, e_tr, dt_tr = load(tr)
r_va, c_va, cp_va, p0_va, e_va, dt_va = load(va)
R_MU = r_tr.reshape(-1, 5).mean(0); R_SD = r_tr.reshape(-1, 5).std(0) + 1e-6
E_MU = e_tr.mean(0); E_SD = e_tr.std(0) + 1e-3

dev = torch.device("cuda")
nrm = lambda a: torch.tensor((a - R_MU) / R_SD, dtype=torch.float32, device=dev)
Rtr, Ctr, CPtr = nrm(r_tr), nrm(c_tr), nrm(cp_tr)
Rva, Cva, CPva = nrm(r_va), nrm(c_va), nrm(cp_va)
Etr = torch.tensor(e_tr, dtype=torch.float32, device=dev)
EtrN = torch.tensor((e_tr - E_MU) / E_SD, dtype=torch.float32, device=dev)
Eva = torch.tensor(e_va, dtype=torch.float32, device=dev)
EvaN = torch.tensor((e_va - E_MU) / E_SD, dtype=torch.float32, device=dev)
MU_t = torch.tensor(R_MU, dtype=torch.float32, device=dev)
SD_t = torch.tensor(R_SD, dtype=torch.float32, device=dev)
P0tr = torch.tensor(p0_tr, dtype=torch.float32, device=dev)          # 物理单位
DTtr = torch.tensor(dt_tr, dtype=torch.float32, device=dev)
DM_tr = torch.tensor(np.linalg.norm(r_tr[:, :, :3] - cp_tr[:, :, :3], axis=-1),
                     dtype=torch.float32, device=dev)                # GT-草稿匹配距离(持久点门控)

model = RadarPointDenoiser(dim=DIM, depth=DEPTH, heads=8, pt_ch=5, lidar_ch=5).to(dev)
print(f"model dim={DIM} depth={DEPTH} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M steps={STEPS}")
opt = torch.optim.AdamW(model.parameters(), lr=LR)
lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

t0 = time.time()
for step in range(1, STEPS + 1):
    idx = torch.randint(0, len(tr), (BS,), device=dev)
    gt, cond, draft, ego, egoN = Rtr[idx], Ctr[idx], CPtr[idx], Etr[idx], EtrN[idx]
    if AUG:   # rollout-aware: 以概率 0.5 给"草稿/条件"加噪, 模拟生成帧作为输入(治分布偏移)
        on = (torch.rand(BS, 1, 1, device=dev) < 0.5).float()
        sig = torch.rand(BS, 1, 1, device=dev) * torch.tensor(
            [1.0, 1.0, 0.3, 0.5, 0.5], device=dev) / SD_t   # 物理量级→归一化空间
        cond = cond + on * sig * torch.randn_like(cond)
        draft = draft + on * sig * torch.randn_like(draft)
    t = torch.rand(BS, device=dev)
    tb = t[:, None, None]
    xt = (1 - tb) * draft + tb * gt + SIGMA * (1 - tb) * torch.randn_like(gt)
    v_tgt = gt - draft
    with torch.autocast("cuda", dtype=torch.bfloat16):
        v_hat = model(xt, (t * 999), cond, None, egoN)
        loss_mse = F.mse_loss(v_hat, v_tgt)
        x1_hat = xt + (1 - tb) * v_hat
        x1_phys = x1_hat * SD_t + MU_t
        loss_phys = self_gated_static_loss(x1_phys, ego[:, :3], ego[:, 3:6],
                                           ego[:, 6:9], step_w=t)
        loss = loss_mse + LAM * loss_phys
        loss_temp = x1_phys.new_zeros(())
        if LAM_TEMP > 0:   # L_temp = |Δrange − v̄_r·Δt|, 持久点门控(GT-草稿匹配距离)
            p0 = P0tr[idx]
            rng0 = p0[..., :3].norm(dim=-1)
            rng1 = x1_phys[..., :3].norm(dim=-1)
            vbar = 0.5 * (p0[..., 3] + x1_phys[..., 3])
            rr = (rng1 - rng0) - vbar * DTtr[idx][:, None]
            wm = torch.exp(-(DM_tr[idx] ** 2) / (2 * 2.0 ** 2)) * t[:, None]
            hub = F.huber_loss(rr, torch.zeros_like(rr), delta=0.5, reduction="none")
            loss_temp = (wm * hub).sum() / (wm.sum() + 1e-6)
            loss = loss + LAM_TEMP * loss_temp
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
        print(f"step {step:6d}  v-mse {float(loss_mse):.4f}  phys {float(loss_phys):.4f}  "
              f"temp {float(loss_temp):.4f}  ({time.time()-t0:.0f}s)", flush=True)

torch.save(dict(ema=ema, r_mu=R_MU, r_sd=R_SD, e_mu=E_MU, e_sd=E_SD, cond=COND, aug=AUG, lam_temp=LAM_TEMP, dim=DIM, depth=DEPTH),
           f"{RES}/bridge_{TAG}_ckpt.pt")
model.load_state_dict(ema)
model.eval()

# ---- 推理: 从草稿 Euler 积分 ----
eidx = np.linspace(0, len(va) - 1, N_EVAL).astype(int)
cond, egoEN = Cva[eidx], EvaN[eidx]
egoC = Eva[eidx].cpu()
with torch.no_grad():
    x = Cva[eidx].clone()               # 从(未对齐)草稿出发 —— 推理无需配对
    dt = 1.0 / ODE_STEPS
    for k in range(ODE_STEPS):
        t = torch.full((N_EVAL,), k * dt, device=dev)
        x = x + dt * model(x, t * 999, cond, None, egoEN)
gen = x.cpu().numpy() * R_SD + R_MU
gt = r_va[eidx]
cnd = c_va[eidx]

reps = [full_report(gen[i], gt[i]) for i in range(N_EVAL)]
cd_anchor = [chamfer(gt[i], gt[(i + N_EVAL // 2) % N_EVAL]) for i in range(N_EVAL)]
cd_copy = [chamfer(cnd[i], gt[i]) for i in range(N_EVAL)]
pce_gen = pce_report(torch.tensor(gen, dtype=torch.float32), egoC[:, :3], egoC[:, 3:6], egoC[:, 6:9])
pce_gt = pce_report(torch.tensor(gt, dtype=torch.float32), egoC[:, :3], egoC[:, 3:6], egoC[:, 6:9])
q = np.linspace(0, 1, 200)
w1_vr = float(np.abs(np.quantile(gen[:, :, 3], q) - np.quantile(gt[:, :, 3], q)).mean())


def dynfrac(c):
    ct = torch.tensor(c, dtype=torch.float32)
    r = ct[..., 3] - static_pred_vr(ct[..., :3], egoC[:, :3], egoC[:, 3:6], egoC[:, 6:9])
    return float((r.abs() > 1.0).float().mean())


med = lambda k: float(np.median([r[k] for r in reps]))
report = "\n".join([
    f"tag={TAG} cond={COND} bridge(RF, ODE{ODE_STEPS})  val N={N_EVAL} 中位:",
    f"  CD={med('cd'):.3f} m (锚 {np.median(cd_anchor):.3f} | 复制条件 {np.median(cd_copy):.3f})",
    f"  CD_Doppler={med('cd_dopp'):.3f}  MMD={med('mmd'):.5f}  JSD={med('jsd'):.4f}",
    f"  v_r std gen={np.median([r['vr_std_gen'] for r in reps]):.2f} gt={np.median([r['vr_std_gt'] for r in reps]):.2f}",
    f"  PCE(gen) med|r|={pce_gen['med_abs']:.3f} <0.5:{pce_gen['frac<0.5']*100:.1f}%"
    f"  | PCE(GT) med|r|={pce_gt['med_abs']:.3f} <0.5:{pce_gt['frac<0.5']*100:.1f}%",
    f"  W1(v_r)={w1_vr:.3f}   动态样占比 gen={dynfrac(gen)*100:.1f}% / GT={dynfrac(gt)*100:.1f}%",
])
print("\n" + report)
open(f"{RES}/bridge_{TAG}_metrics.txt", "w").write(report + "\n")

fig, axes = plt.subplots(3, 4, figsize=(17, 10))
for k in range(4):
    j = k * (N_EVAL // 4)
    for row, cloud, tt in ((0, cnd[j], "draft"), (1, gt[j], "GT"), (2, gen[j], "bridge")):
        a = axes[row, k]
        a.scatter(cloud[:, 0], cloud[:, 1], c=np.clip(cloud[:, 3], -15, 15), s=4, cmap="coolwarm")
        a.set_title(f"{tt} #{j}"); a.set_aspect("equal")
fig.tight_layout(); fig.savefig(f"{RES}/bridge_{TAG}_samples.png", dpi=120)
print("== DONE")
