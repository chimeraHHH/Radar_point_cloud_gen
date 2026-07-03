"""Doppler 物理一致性指标(proposal §5 "新指标" 的参考实现, numpy 版).

核心关系(RAW 口径, 传感器系):
  静态点:  v_r = -v_plat·r̂,  v_plat = v_ego + ω×p        (解析硬约束)
  动态点:  v_r = (v_obj - v_plat)·r̂                       (一致性软约束)
训练期的可微 torch 版在 losses/ 中实现(P2)。
"""
import numpy as np


def static_residual(v_r, pred_static_vr):
    """静态点残差: v_r - (-v_plat·r̂). 理想为 0."""
    return v_r - pred_static_vr


def dynamic_residual(v_r, rhat, v_obj_sensor, v_plat_s):
    """动态点残差: v_r - (v_obj - v_plat)·r̂. 理想为 0.

    v_obj_sensor: (N,3) 逐点目标速度(传感器系); v_plat_s: (N,3) 平台速度(传感器系)
    """
    pred = np.einsum("ij,ij->i", v_obj_sensor - v_plat_s, rhat)
    return v_r - pred


def consistency_report(residual, name="", thr=(0.25, 0.5, 1.0)):
    """残差统计: 内点率(多阈值)/中位/MAD/截断RMS."""
    r = residual[np.isfinite(residual)]
    if len(r) == 0:
        return dict(name=name, n=0)
    med = float(np.median(r))
    return dict(
        name=name, n=int(len(r)), med=med,
        mad=float(np.median(np.abs(r - med))),
        rms_c=float(np.sqrt(np.mean(np.clip(r, -5, 5) ** 2))),
        inlier={t: float(np.mean(np.abs(r) < t)) for t in thr},
    )


def fmt_report(rep):
    if rep.get("n", 0) == 0:
        return f"   {rep.get('name', ''):>34s}: (无有效点)"
    inl = " ".join(f"|r|<{t}:{v * 100:5.1f}%" for t, v in rep["inlier"].items())
    return (f"   {rep['name']:>34s}: N={rep['n']:>8d}  med={rep['med']:+.3f}  "
            f"MAD={rep['mad']:.3f}  RMSc={rep['rms_c']:.3f}  {inl}")
