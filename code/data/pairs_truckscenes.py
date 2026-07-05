"""LiDAR→Radar 配对样本构建(P1 基线数据面).

每个 (keyframe sample × 雷达通道) 生成一个配对:
  lidar (M,4): 6 路 LiDAR 融合 -> 全局系 -> 该雷达传感器系, 按雷达 FoV 裁剪, 采样 M 点
  radar (N,5): (x,y,z,v_r,rcs) 传感器系, 采样 N 点(不足有放回)
  pred_static_vr (N,): 静态解析预测(P2 物理损失用)
坐标留在雷达传感器系: v_r=vrel·r̂ 的物理语义在此系下最干净.
"""
import os

import numpy as np

from truckscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion


def _sensor_pose_global(tsc, sd):
    """sample_data -> (R_gs, t_gs): 传感器 -> 全局."""
    cs = tsc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ep = tsc.get("ego_pose", sd["ego_pose_token"])
    R_se = Quaternion(cs["rotation"]).rotation_matrix
    t_se = np.asarray(cs["translation"])
    R_eg = Quaternion(ep["rotation"]).rotation_matrix
    t_eg = np.asarray(ep["translation"])
    return R_eg @ R_se, R_eg @ t_se + t_eg


def load_lidars_global(tsc, sample, channels=None):
    """融合某 sample 的多路 LiDAR 到全局系. 返回 (P,4): xyz_global + intensity."""
    if channels is None:
        channels = [c for c in sample["data"] if c.startswith("LIDAR")]
    outs = []
    for ch in channels:
        sd = tsc.get("sample_data", sample["data"][ch])
        pc = LidarPointCloud.from_file(tsc.get_sample_data_path(sample["data"][ch]))
        pts = pc.points[:4].T.astype(np.float64)           # (P,4) sensor 系
        R_gs, t_gs = _sensor_pose_global(tsc, sd)
        pts[:, :3] = pts[:, :3] @ R_gs.T + t_gs
        outs.append(pts)
    return np.concatenate(outs, 0)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def crop_to_radar_fov(lidar_sensor, radar_xyz, az_pad=0.26, rng_pad=20.0, min_range=1.0):
    """按该帧雷达返回的方位扇区 + 距离范围裁剪 LiDAR(雷达传感器系)."""
    az_r = np.arctan2(radar_xyz[:, 1], radar_xyz[:, 0])
    # 圆均值作为通道朝向, 处理 ±π 环绕
    c = np.exp(1j * az_r).mean()
    az0 = np.angle(c)
    half = np.abs(_wrap(az_r - az0)).max() + az_pad
    rmax = np.linalg.norm(radar_xyz, axis=1).max() + rng_pad

    az_l = np.arctan2(lidar_sensor[:, 1], lidar_sensor[:, 0])
    rng_l = np.linalg.norm(lidar_sensor[:, :3], axis=1)
    m = (np.abs(_wrap(az_l - az0)) <= half) & (rng_l <= rmax) & (rng_l >= min_range)
    return lidar_sensor[m]


def sample_n(arr, n, rng):
    """采样到固定 n 行(多降采、少有放回)."""
    if len(arr) >= n:
        idx = rng.choice(len(arr), n, replace=False)
    else:
        idx = rng.choice(len(arr), n, replace=True)
    return arr[idx]


def build_pair(tsc, ldr, sample, radar_ch, lidar_global,
               n_lidar=4096, n_radar=384, min_radar_pts=50, rng=None,
               v_moving_thr=1.0):
    """构建单个配对; 点数不足或无 FoV 交集返回 None.

    v3: 附带逐点静/动标签与运动目标速度(动态一致性损失用);
    v_moving_thr=1.0 抬高判动门限, 缓解框速度差分噪声误标(R3)。
    """
    from data.truckscenes_loader import label_static_dynamic
    rng = rng or np.random.default_rng(0)
    fr = ldr.load_frame(sample["data"][radar_ch], load_boxes=True)
    if len(fr["xyz"]) < min_radar_pts:
        return None
    labels, v_obj = label_static_dynamic(fr, v_moving_thr=v_moving_thr)
    # LiDAR: 全局 -> 该雷达传感器系
    lid = lidar_global.copy()
    lid[:, :3] = (lid[:, :3] - fr["t_gs"]) @ fr["R_gs"]
    lid = crop_to_radar_fov(lid, fr["xyz"])
    if len(lid) < 256:
        return None
    lid = sample_n(lid, n_lidar, rng)

    idx = rng.choice(len(fr["xyz"]), n_radar, replace=len(fr["xyz"]) < n_radar)
    radar = np.concatenate([fr["xyz"][idx], fr["v_r"][idx, None], fr["rcs"][idx, None]], 1)
    return dict(lidar=lid.astype(np.float32), radar=radar.astype(np.float32),
                pred_static_vr=fr["pred_static_vr"][idx].astype(np.float32),
                label=labels[idx].astype(np.int8),
                v_obj_s=np.nan_to_num(v_obj[idx]).astype(np.float32),
                v_ego_s=fr["v_ego_s"].astype(np.float32),
                omega_s=fr["omega_s"].astype(np.float32),
                t_s=fr["t_s"].astype(np.float32),
                v_ego_norm=np.float32(fr["v_ego_norm"]),
                channel=radar_ch, sample_token=sample["token"])
