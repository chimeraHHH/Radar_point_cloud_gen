"""时序配对构建(P3 主线 B 数据面): sweeps 上 (t, t+K) 帧对 + 双条件云.

条件云两种口径(对照 B 线核心主张):
  cond_ego : 上一帧点仅按自车位姿刚体变换到目标帧传感器系(稳健下限)
  cond_dopp: 上一帧点先做**门控 Doppler 径向推进**(|残差|>thr 的点按 res·Δt·r̂ 前推,
             吸收 07-04 教训: 未门控会放大杂波), 再 ego 变换
特征 5 维 (x,y,z,v_r,rcs); v_r 携带原始值(径向方向在新帧略有偏差, 作条件特征可接受)。
"""
import numpy as np


def sample_n(arr, n, rng):
    if len(arr) >= n:
        idx = rng.choice(len(arr), n, replace=False)
    else:
        idx = rng.choice(len(arr), n, replace=True)
    return idx


def build_temporal_pair(fr0, fr1, n_pts=384, dyn_thr=1.0, min_pts=50, rng=None):
    """fr0/fr1: TruckScenesRadar.load_frame 输出(同通道相邻 sweep). 返回 dict 或 None."""
    rng = rng or np.random.default_rng(0)
    if len(fr0["xyz"]) < min_pts or len(fr1["xyz"]) < min_pts:
        return None
    dt = (fr1["timestamp"] - fr0["timestamp"]) / 1e6
    if not (0.01 < dt < 1.0):
        return None

    def to1(p):
        g = p @ fr0["R_gs"].T + fr0["t_gs"]
        return (g - fr1["t_gs"]) @ fr1["R_gs"]

    res = fr0["v_r"] - fr0["pred_static_vr"]
    adv = np.where(np.abs(res) > dyn_thr, res, 0.0)          # 门控
    p_dopp = fr0["xyz"] + adv[:, None] * dt * fr0["rhat"]

    i0 = sample_n(fr0["xyz"], n_pts, rng)
    feat = lambda p: np.concatenate(
        [p[i0], fr0["v_r"][i0, None], fr0["rcs"][i0, None]], 1).astype(np.float32)
    cond_ego = feat(to1(fr0["xyz"]))
    cond_dopp = feat(to1(p_dopp))

    i1 = sample_n(fr1["xyz"], n_pts, rng)
    radar = np.concatenate(
        [fr1["xyz"][i1], fr1["v_r"][i1, None], fr1["rcs"][i1, None]], 1).astype(np.float32)

    ego1 = np.concatenate([fr1["v_ego_s"], fr1["omega_s"], fr1["t_s"]]).astype(np.float32)
    return dict(radar=radar, cond_ego=cond_ego, cond_dopp=cond_dopp,
                ego=ego1, dt=np.float32(dt),
                v_ego_norm=np.float32(fr1["v_ego_norm"]))
