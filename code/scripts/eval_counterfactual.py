#!/usr/bin/env python
"""反事实速度可控性(proposal 贡献点 4): 干预 ego 条件, 检验生成 v_r 的可预测响应.

用 P2 的 ego_dyn 模型(LiDAR+ego 条件, 从噪声生成——无草稿 v_r 干扰, 干预最干净):
对 α ∈ {0, 0.5, 1, 1.5, 2} 缩放 v_ego 条件, 生成后从点云稳健反解"隐含平台速度",
剂量响应斜率 ≈1 即完美可控。产出: results/counterfactual_{metrics.txt,curve.png}
"""
import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.point_diffusion import RadarPointDenoiser          # noqa: E402

from diffusers import DDPMScheduler                            # noqa: E402

PAIRS = os.path.expanduser("~/data/radar_gen/truckscenes/pairs_mini_v3")
RES = os.path.expanduser("~/Workspace/radar_gen/results")
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
N_EVAL, CFG_W = 16, 2.0
dev = torch.device("cuda")
torch.manual_seed(0)

ck = torch.load(f"{RES}/ablate_ego_dyn_ckpt.pt", map_location="cpu", weights_only=False)
R_MU, R_SD, E_MU, E_SD = ck["r_mu"], ck["r_sd"], ck["e_mu"], ck["e_sd"]
model = RadarPointDenoiser(dim=256, depth=6, heads=8).to(dev)
model.load_state_dict(ck["ema"])
model.eval()

mani = json.load(open(f"{PAIRS}/manifest.json"))
scenes = sorted({m["scene"] for m in mani["pairs"]})
va = [m for m in mani["pairs"] if m["scene"] in set(scenes[-2:]) and m["v_ego_norm"] > 3.0]
sel = va[:: max(1, len(va) // N_EVAL)][:N_EVAL]
print(f"val pairs(|v_ego|>3)={len(va)}, 使用 {len(sel)}")

lidar, ego = [], []
for m in sel:
    z = np.load(f"{PAIRS}/{m['file']}")
    lidar.append(z["lidar"])
    ego.append(np.concatenate([z["v_ego_s"], z["omega_s"], z["t_s"]]))
lidar = np.stack(lidar)
ego = np.stack(ego)
L_MU = lidar.reshape(-1, 4).mean(0); L_SD = lidar.reshape(-1, 4).std(0) + 1e-6
Lt = torch.tensor((lidar - L_MU) / L_SD, dtype=torch.float32, device=dev)

samp = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
samp.set_timesteps(1000)
dF = torch.zeros(len(sel), dtype=torch.bool, device=dev)
dT = torch.ones(len(sel), dtype=torch.bool, device=dev)


def implied_platform_speed(cloud, ego_vec):
    """从生成点云稳健反解隐含平台速度(两轮修剪最小二乘): v_r ≈ -(v+ω×(p+t))·r̂ 解 v."""
    xyz, vr = cloud[:, :3], cloud[:, 3]
    rhat = xyz / (np.linalg.norm(xyz, axis=1, keepdims=True) + 1e-6)
    lever = np.cross(np.broadcast_to(ego_vec[3:6], xyz.shape), xyz + ego_vec[6:9][None])
    y = -(vr + (lever * rhat).sum(1))          # y ≈ v·r̂
    A = rhat
    mask = np.ones(len(y), bool)
    v = np.zeros(3)
    for _ in range(3):
        if mask.sum() < 30:
            break
        v, *_ = np.linalg.lstsq(A[mask], y[mask], rcond=None)
        r = np.abs(A @ v - y)
        mask = r < np.maximum(1.0, np.median(r[mask]) * 3)
    return v


rows = []
per_alpha_pts = []
for a in ALPHAS:
    ego_i = ego.copy()
    ego_i[:, :3] = a * ego[:, :3]              # 只干预 v_ego, ω/t_s 不动
    egoN = torch.tensor((ego_i - E_MU) / E_SD, dtype=torch.float32, device=dev)
    torch.manual_seed(1)
    with torch.no_grad():
        x = torch.randn(len(sel), 384, 5, device=dev)
        for t in samp.timesteps:
            tb = t.expand(len(sel)).to(dev)
            e_c = model(x, tb, Lt, dF, egoN)
            e_u = model(x, tb, Lt, dT, egoN)
            x = samp.step(e_u + CFG_W * (e_c - e_u), t, x).prev_sample
    gen = x.cpu().numpy() * R_SD + R_MU
    cmd, imp = [], []
    for i in range(len(sel)):
        v_imp = implied_platform_speed(gen[i], ego_i[i])
        v_cmd = ego_i[i, :3]
        # 投影到指令方向(α=0 时用原方向)
        d = ego[i, :3] / (np.linalg.norm(ego[i, :3]) + 1e-6)
        cmd.append(float(v_cmd @ d))
        imp.append(float(v_imp @ d))
    rows.append((a, float(np.mean(cmd)), float(np.mean(imp)), float(np.std(imp))))
    per_alpha_pts.append((np.array(cmd), np.array(imp)))
    print(f"alpha={a}: 指令速度均值={rows[-1][1]:.2f}  隐含速度均值={rows[-1][2]:.2f}±{rows[-1][3]:.2f}", flush=True)

C = np.concatenate([c for c, _ in per_alpha_pts])
I = np.concatenate([i for _, i in per_alpha_pts])
slope, intercept = np.polyfit(C, I, 1)
r = float(np.corrcoef(C, I)[0, 1])
lines = [f"反事实速度可控性(ego_dyn 模型, N={len(sel)}/α, DDPM-1000, CFG {CFG_W})",
         f"{'alpha':>6s} {'指令v̄':>8s} {'隐含v̄':>8s} {'±std':>6s}"]
for a, c, i, s in rows:
    lines.append(f"{a:>6.1f} {c:>8.2f} {i:>8.2f} {s:>6.2f}")
lines.append(f"剂量响应: slope={slope:.3f} (理想=1)  intercept={intercept:.2f}  Pearson r={r:.3f}")
report = "\n".join(lines)
print("\n" + report)
open(f"{RES}/counterfactual_metrics.txt", "w").write(report + "\n")

fig, ax = plt.subplots(figsize=(6, 5))
for (c, i), a in zip(per_alpha_pts, ALPHAS):
    ax.scatter(c, i, s=18, label=f"α={a}")
xs = np.linspace(C.min(), C.max(), 10)
ax.plot(xs, xs, "k--", lw=1, label="ideal (slope=1)")
ax.plot(xs, slope * xs + intercept, "r-", lw=1.5, label=f"fit slope={slope:.2f}")
ax.set_xlabel("commanded ego speed (m/s)")
ax.set_ylabel("implied ego speed from generated v_r (m/s)")
ax.legend(); ax.set_title("Counterfactual ego-velocity controllability")
fig.tight_layout(); fig.savefig(f"{RES}/counterfactual_curve.png", dpi=130)
print("== DONE")
