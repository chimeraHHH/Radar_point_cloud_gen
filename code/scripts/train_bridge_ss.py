#!/usr/bin/env python
"""桥式 + scheduled sampling(G3 关门): 以概率 P_SS 用模型自生成的上一帧构造草稿训练.

用法: python train_bridge_ss.py <tag> [triples_dir=triples_full_k10]
scheduled 分支: cond0 --ODE_short(no-grad)--> gen_t --门控dopp-warp+位姿--> draft_ss
真实分支: 与 train_bridge 相同(cond1/cp1/p01)。配对: scheduled 用 GPU-NN, 真实用预存 OT。
产出: results/bridge_<tag>_{log.txt,ckpt.pt,metrics.txt}
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.point_diffusion import RadarPointDenoiser              # noqa: E402
from eval.gen_metrics import full_report, chamfer                  # noqa: E402
from losses.physics import (self_gated_static_loss, pce_report,    # noqa: E402
                            static_pred_vr)

TAG = sys.argv[1]
PDIR = sys.argv[2] if len(sys.argv) > 2 else "triples_full_k10"
PAIRS = os.path.expanduser(f"~/data/radar_gen/truckscenes/{PDIR}")
RES = os.path.expanduser("~/Workspace/radar_gen/results")
STEPS = int(os.environ.get("STEPS", 60000))
P_SS, ODE_SHORT = 0.4, 10
BS, LR, LAM, LAM_TEMP, SIGMA = 64, 3e-4, 0.1, 0.1, 0.05
EMA_DECAY, N_EVAL, ODE_STEPS = 0.999, 24, 50
DIM, DEPTH = 512, 8
torch.manual_seed(0)
np.random.seed(0)
print(f"== bridge_ss tag={TAG} P_SS={P_SS} ODE_short={ODE_SHORT}", flush=True)

mani = json.load(open(f"{PAIRS}/manifest.json"))
scenes = sorted({m["scene"] for m in mani["pairs"]})
val_scenes = set(scenes[-int(os.environ.get("VAL_N", "12")):])
tr = [m for m in mani["pairs"] if m["scene"] not in val_scenes]
va = [m for m in mani["pairs"] if m["scene"] in val_scenes]
print(f"train={len(tr)} val={len(va)}", flush=True)

KEYS = ["cond0", "ego_t", "A", "b", "radar1", "ego1", "cond1", "cp1", "p01"]


def load(ms):
    n = len(ms)
    out = dict(cond0=np.empty((n, 384, 5), np.float32), ego_t=np.empty((n, 9), np.float32),
               A=np.empty((n, 3, 3), np.float32), b=np.empty((n, 3), np.float32),
               dt2=np.empty(n, np.float32),
               radar1=np.empty((n, 384, 5), np.float32), ego1=np.empty((n, 9), np.float32),
               cond1=np.empty((n, 384, 5), np.float32), cp1=np.empty((n, 384, 5), np.float32),
               p01=np.empty((n, 384, 5), np.float32))
    for i, m in enumerate(ms):
        z = np.load(f"{PAIRS}/{m['file']}")
        for k in KEYS:
            out[k][i] = z[k]
        out["dt2"][i] = z["dt2"]
    return out


D_tr = load(tr)
D_va = load(va)
R_MU = D_tr["radar1"].reshape(-1, 5).mean(0); R_SD = D_tr["radar1"].reshape(-1, 5).std(0) + 1e-6
E_MU = D_tr["ego1"].mean(0); E_SD = D_tr["ego1"].std(0) + 1e-3
dev = torch.device("cuda")
T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
G = {k: T(v) for k, v in D_tr.items()}
MU_t, SD_t = T(R_MU), T(R_SD)
EMU_t, ESD_t = T(E_MU), T(E_SD)
nrm = lambda x: (x - MU_t) / SD_t
egoN = lambda e: (e - EMU_t) / ESD_t

model = RadarPointDenoiser(dim=DIM, depth=DEPTH, heads=8, pt_ch=5, lidar_ch=5).to(dev)
print(f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M steps={STEPS}", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
ema = {k: v.detach().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def ode_gen(draft_n, cond_n, ego_n, steps):
    x = draft_n.clone()
    dt = 1.0 / steps
    for k in range(steps):
        t = torch.full((len(x),), k * dt, device=dev)
        x = x + dt * model(x, t * 999, cond_n, None, ego_n)
    return x


t0 = time.time()
for step in range(1, STEPS + 1):
    idx = torch.randint(0, len(tr), (BS,), device=dev)
    gt_n = nrm(G["radar1"][idx])
    ego1 = G["ego1"][idx]
    use_ss = float(torch.rand(())) < P_SS
    if use_ss:
        # 1) 自生成上一帧 t(no-grad, 短 ODE)
        c0n = nrm(G["cond0"][idx])
        gen_t = ode_gen(c0n, c0n, egoN(G["ego_t"][idx]), ODE_SHORT) * SD_t + MU_t
        # 2) 门控 dopp-warp + 位姿 → t+K 系草稿
        xyz, vr = gen_t[..., :3], gen_t[..., 3]
        pred = static_pred_vr(xyz, G["ego_t"][idx][:, :3], G["ego_t"][idx][:, 3:6], G["ego_t"][idx][:, 6:9])
        res = vr - pred
        adv = torch.where(res.abs() > 1.0, res, torch.zeros_like(res))
        rhat = xyz / (xyz.norm(dim=-1, keepdim=True) + 1e-6)
        p_adv = xyz + (adv * G["dt2"][idx][:, None])[..., None] * rhat
        p1 = torch.einsum("bij,bnj->bni", G["A"][idx], p_adv) + G["b"][idx][:, None, :]
        draft_phys = torch.cat([p1, gen_t[..., 3:5]], -1)
        # 3) GPU-NN 配对到 GT 行
        d2 = torch.cdist(G["radar1"][idx][..., :3], p1)
        nn = d2.argmin(-1)
        draft_pi = torch.gather(draft_phys, 1, nn[..., None].expand(-1, -1, 5))
        p0_phys = torch.cat([torch.einsum("bij,bnj->bni", G["A"][idx],
                                          gen_t[..., :3]) + G["b"][idx][:, None, :],
                             gen_t[..., 3:5]], -1)
        p0_pi = torch.gather(p0_phys, 1, nn[..., None].expand(-1, -1, 5))
        dm = torch.gather(d2, 2, nn[..., None]).squeeze(-1).sqrt() if False else d2.gather(2, nn[..., None]).squeeze(-1)
        cond_n = nrm(draft_phys)
        draft_n = nrm(draft_pi)
    else:
        cond_n = nrm(G["cond1"][idx])
        draft_n = nrm(G["cp1"][idx])
        p0_pi = G["p01"][idx]
        dm = (G["radar1"][idx][..., :3] - G["cp1"][idx][..., :3]).norm(dim=-1)
    t = torch.rand(BS, device=dev)
    tb = t[:, None, None]
    xt = (1 - tb) * draft_n + tb * gt_n + SIGMA * (1 - tb) * torch.randn_like(gt_n)
    v_tgt = gt_n - draft_n
    with torch.autocast("cuda", dtype=torch.bfloat16):
        v_hat = model(xt, t * 999, cond_n, None, egoN(ego1))
        loss_mse = F.mse_loss(v_hat, v_tgt)
        x1_phys = (xt + (1 - tb) * v_hat) * SD_t + MU_t
        loss_phys = self_gated_static_loss(x1_phys, ego1[:, :3], ego1[:, 3:6], ego1[:, 6:9], step_w=t)
        rr = (x1_phys[..., :3].norm(dim=-1) - p0_pi[..., :3].norm(dim=-1)) \
            - 0.5 * (p0_pi[..., 3] + x1_phys[..., 3]) * G["dt2"][idx][:, None]
        wm = torch.exp(-(dm ** 2) / 8.0) * t[:, None]
        loss_temp = (wm * F.huber_loss(rr, torch.zeros_like(rr), delta=0.5, reduction="none")).sum() / (wm.sum() + 1e-6)
        loss = loss_mse + LAM * loss_phys + LAM_TEMP * loss_temp
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
        print(f"step {step:6d} ss={int(use_ss)} v-mse {float(loss_mse):.4f} "
              f"phys {float(loss_phys):.4f} temp {float(loss_temp):.4f} ({time.time()-t0:.0f}s)", flush=True)

torch.save(dict(ema=ema, r_mu=R_MU, r_sd=R_SD, e_mu=E_MU, e_sd=E_SD,
                cond="dopp", dim=DIM, depth=DEPTH, p_ss=P_SS),
           f"{RES}/bridge_{TAG}_ckpt.pt")
model.load_state_dict(ema)
model.eval()

# ---- 单步 val 评估(与 train_bridge 同协议) ----
eidx = np.linspace(0, len(va) - 1, N_EVAL).astype(int)
Va = {k: T(v[eidx]) for k, v in D_va.items()}
with torch.no_grad():
    x = nrm(Va["cond1"])
    condv = x.clone()
    dtc = 1.0 / ODE_STEPS
    for k in range(ODE_STEPS):
        t = torch.full((N_EVAL,), k * dtc, device=dev)
        x = x + dtc * model(x, t * 999, condv, None, egoN(Va["ego1"]))
gen = (x * SD_t + MU_t).cpu().numpy()
gt = D_va["radar1"][eidx]
cnd = D_va["cond1"][eidx]
egoC = torch.tensor(D_va["ego1"][eidx], dtype=torch.float32)
reps = [full_report(gen[i], gt[i]) for i in range(N_EVAL)]
cd_copy = [chamfer(cnd[i], gt[i]) for i in range(N_EVAL)]
pce_gen = pce_report(torch.tensor(gen, dtype=torch.float32), egoC[:, :3], egoC[:, 3:6], egoC[:, 6:9])
pce_gt = pce_report(torch.tensor(gt, dtype=torch.float32), egoC[:, :3], egoC[:, 3:6], egoC[:, 6:9])
med = lambda k: float(np.median([r[k] for r in reps]))
report = "\n".join([
    f"tag={TAG} scheduled-sampling(P={P_SS})  val N={N_EVAL} 中位:",
    f"  CD={med('cd'):.3f} (复制 {np.median(cd_copy):.3f})  CD_Doppler={med('cd_dopp'):.3f}",
    f"  MMD={med('mmd'):.5f}  JSD={med('jsd'):.4f}",
    f"  PCE(gen)<0.5:{pce_gen['frac<0.5']*100:.1f}% | PCE(GT)<0.5:{pce_gt['frac<0.5']*100:.1f}%",
])
print("\n" + report)
open(f"{RES}/bridge_{TAG}_metrics.txt", "w").write(report + "\n")
print("== DONE")
