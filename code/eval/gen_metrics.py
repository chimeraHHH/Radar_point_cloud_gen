"""生成质量指标(P1 基线评估套件, 物理单位): Chamfer / CD_Doppler / MMD-RBF / JSD.

约定输入为 numpy (N,5): x,y,z,v_r,rcs(米 / m/s / dBsm), 内部用 torch 计算.
对标 4D-RaDiff / RadarGen 的指标口径; CD_Doppler 在 (x,y,z,λ·v_r) 四维空间做 Chamfer.
"""
import numpy as np
import torch


def _t(a):
    return torch.as_tensor(np.asarray(a, dtype=np.float32))


def chamfer(a, b, dims=slice(0, 3)):
    """对称 Chamfer(均值形式, 米)."""
    d = torch.cdist(_t(a[:, dims]), _t(b[:, dims]))
    return float(d.min(1).values.mean() + d.min(0).values.mean()) / 2


def cd_doppler(a, b, lam=1.0):
    """4D Chamfer: (x,y,z,λ·v_r) —— 同时惩罚几何与 Doppler 偏差."""
    fa = np.concatenate([a[:, :3], lam * a[:, 3:4]], 1)
    fb = np.concatenate([b[:, :3], lam * b[:, 3:4]], 1)
    d = torch.cdist(_t(fa), _t(fb))
    return float(d.min(1).values.mean() + d.min(0).values.mean()) / 2


def mmd_rbf(a, b, sigma=10.0, dims=slice(0, 3)):
    """RBF-核 MMD^2(xyz, σ 米)."""
    x, y = _t(a[:, dims]), _t(b[:, dims])
    k = lambda d2: torch.exp(-d2 / (2 * sigma ** 2))
    xx = k(torch.cdist(x, x) ** 2).mean()
    yy = k(torch.cdist(y, y) ** 2).mean()
    xy = k(torch.cdist(x, y) ** 2).mean()
    return float(xx + yy - 2 * xy)


def jsd_hist(a, b, bins=24, lim=150.0):
    """xyz 3D 直方图分布的 Jensen-Shannon 散度(平方)."""
    from scipy.spatial.distance import jensenshannon
    ha = np.histogramdd(a[:, :3], bins=bins, range=[(-lim, lim)] * 3)[0].ravel() + 1e-9
    hb = np.histogramdd(b[:, :3], bins=bins, range=[(-lim, lim)] * 3)[0].ravel() + 1e-9
    return float(jensenshannon(ha / ha.sum(), hb / hb.sum()) ** 2)


def full_report(gen, gt, lam=1.0):
    """单对样本的全套指标 dict."""
    return dict(
        cd=chamfer(gen, gt),
        cd_dopp=cd_doppler(gen, gt, lam),
        mmd=mmd_rbf(gen, gt),
        jsd=jsd_hist(gen, gt),
        vr_std_gen=float(gen[:, 3].std()), vr_std_gt=float(gt[:, 3].std()),
    )
