"""可微 Doppler 物理约束(P2 主线 A 核心, torch 版).

静态解析关系(传感器系, RAW 口径, 经 P0/P1 实测验证):
    v_r = -v_plat(p)·r̂ ,  v_plat(p) = v_ego_s + ω_s × (p + t_s)

设计要点(吸收 2026-07-04 时序预演教训"未门控推进放大杂波"):
- **自门控**: w = exp(-r²/2τ²)(r detach)——只把已接近静态流形的点拉到严格一致,
  真动态点/杂波残差大 → 权重≈0 → 不受惩罚, 无需运动分割标注。
- **x̂0 桥接**: 扩散训练中在预测的干净样本 x̂0 上施加物理约束(反归一化到物理单位),
  并按 ᾱ_t 加权——高噪声步 x̂0 不可靠时约束自动趋零。
"""
import torch
import torch.nn.functional as F


def static_pred_vr(xyz, v_ego_s, omega_s, t_s, lever=True):
    """在任意点 xyz (B,N,3, 米) 上计算静态 v_r 解析预测 (B,N).

    v_ego_s/omega_s/t_s: (B,3) 传感器系 ego 速度/角速度/安装平移(R_se^T t_se).
    """
    rhat = xyz / (xyz.norm(dim=-1, keepdim=True) + 1e-6)
    v = v_ego_s[:, None, :].expand_as(xyz)
    if lever:
        v = v + torch.cross(omega_s[:, None, :].expand_as(xyz),
                            xyz + t_s[:, None, :], dim=-1)
    return -(v * rhat).sum(-1)


def self_gated_static_loss(x0_phys, v_ego_s, omega_s, t_s,
                           tau=1.0, delta=0.5, step_w=None):
    """自门控静态一致性损失.

    x0_phys: (B,N,5) 物理单位的(预测)干净样本 [x,y,z,v_r,rcs]
    step_w:  (B,) 各样本的时间步权重(如 ᾱ_t), None 则全 1
    返回标量 loss.
    """
    xyz, vr = x0_phys[..., :3], x0_phys[..., 3]
    pred = static_pred_vr(xyz, v_ego_s, omega_s, t_s)
    r = vr - pred
    w = torch.exp(-(r.detach() ** 2) / (2 * tau ** 2))          # 自门控(detach 防塌缩)
    if step_w is not None:
        w = w * step_w[:, None]
    hub = F.huber_loss(r, torch.zeros_like(r), delta=delta, reduction="none")
    return (w * hub).sum() / (w.sum() + 1e-6)


@torch.no_grad()
def pce_report(cloud_phys, v_ego_s, omega_s, t_s, thr=(0.25, 0.5, 1.0)):
    """物理一致性误差 PCE(评估指标): 点云对静态解析关系的残差统计.

    cloud_phys: (N,5) 或 (B,N,5) 物理单位. 返回 dict(med_abs, frac<thr...).
    注意: 该指标混合静态+动态点, 应与 GT 云的同指标对照解读(GT ≈ 静态占比的上限).
    """
    if cloud_phys.dim() == 2:
        cloud_phys = cloud_phys[None]
        v_ego_s, omega_s, t_s = v_ego_s[None], omega_s[None], t_s[None]
    r = (cloud_phys[..., 3] -
         static_pred_vr(cloud_phys[..., :3], v_ego_s, omega_s, t_s)).flatten()
    out = {"med_abs": float(r.abs().median())}
    for t in thr:
        out[f"frac<{t}"] = float((r.abs() < t).float().mean())
    return out
