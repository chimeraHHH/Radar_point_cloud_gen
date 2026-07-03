"""TruckScenes 雷达帧读取与坐标/速度对齐(P1 数据管线核心).

约定(经 2026-07-03 自洽性校验确认):
- 逐点字段 x,y,z,vrel_{x,y,z},rcs;vrel 为纯径向向量 -> 标量 v_r = vrel·r̂(RAW, 含自车)
- ego 速度源 ego_motion_chassis(ego/chassis 系), 与雷达帧对齐 ~1ms
- 静态点物理关系: v_r = -(v_ego + ω×p_ego)·r̂ (ω×r 为自车角速度杠杆项)
"""
import json
import os

import numpy as np
from pyquaternion import Quaternion

_TMAP = {("F", 4): "f4", ("F", 8): "f8", ("I", 1): "i1", ("I", 2): "i2",
         ("I", 4): "i4", ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4"}


def read_pcd(path):
    """通用 PCD 读取(binary/ascii), 返回 (fields, structured array)."""
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == "DATA":
                fmt = parts[1]
                break
        fields = header["FIELDS"]
        sizes = list(map(int, header["SIZE"]))
        types = header["TYPE"]
        counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))
        n = int(header["POINTS"][0]) if "POINTS" in header else \
            int(header["WIDTH"][0]) * int(header["HEIGHT"][0])
        dt = [(fn, _TMAP[(t, s)]) if c == 1 else (fn, _TMAP[(t, s)], (c,))
              for fn, s, t, c in zip(fields, sizes, types, counts)]
        if fmt == "binary":
            buf = f.read(n * sum(s * c for s, c in zip(sizes, counts)))
            arr = np.frombuffer(buf, dtype=np.dtype(dt), count=n)
        elif fmt == "ascii":
            arr = np.loadtxt(f, dtype=np.dtype(dt), max_rows=n)
        else:
            raise ValueError(f"unsupported pcd DATA={fmt}: {path}")
    return fields, arr


class TruckScenesRadar:
    """雷达帧加载器: 点云 + ego 运动 + 3D 框(含逐框速度), 全部对齐到传感器系."""

    def __init__(self, tsc):
        self.tsc = tsc
        ego = json.load(open(os.path.join(tsc.dataroot, tsc.version, "ego_motion_chassis.json")))
        ts = np.array([e["timestamp"] for e in ego], dtype=np.int64)
        order = np.argsort(ts)
        self.ego_ts = ts[order]
        self.ego = [ego[i] for i in order]

    def ego_motion_at(self, timestamp):
        """最近邻 ego_motion_chassis 记录, 返回 (record, 时间间隙秒)."""
        i = int(np.clip(np.searchsorted(self.ego_ts, timestamp), 1, len(self.ego_ts) - 1))
        j = i if abs(int(self.ego_ts[i]) - timestamp) <= abs(int(self.ego_ts[i - 1]) - timestamp) else i - 1
        return self.ego[j], abs(int(self.ego_ts[j]) - timestamp) / 1e6

    def load_frame(self, sd_token, min_range=1.0, yaw_rate_correction=True, load_boxes=True):
        """加载一个雷达 sample_data(sweeps 亦可, 此时建议 load_boxes=False 提速).

        返回 dict:
          xyz (N,3) 传感器系 | v_r (N,) 标量 raw Doppler | rcs (N,)
          rhat (N,3) | pred_static_vr (N,) 静态点解析预测 | v_plat_s (N,3) 平台速度(传感器系)
          boxes: list of dict(center, wlh, rot(Quaternion), name, ann_token,
                              v_obj_sensor(3,) 或 None); load_boxes=False 时为 []
          R_gs/t_gs: 传感器 -> 全局 位姿 | v_ego_norm, ego_gap_s, channel, timestamp
        """
        tsc = self.tsc
        sd = tsc.get("sample_data", sd_token)
        cs = tsc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        R_se = Quaternion(cs["rotation"]).rotation_matrix          # sensor -> ego
        t_se = np.asarray(cs["translation"])
        ep = tsc.get("ego_pose", sd["ego_pose_token"])
        R_eg = Quaternion(ep["rotation"]).rotation_matrix          # ego -> global
        t_eg = np.asarray(ep["translation"])

        e, gap = self.ego_motion_at(sd["timestamp"])
        v_ego = np.array([e["vx"], e["vy"], e["vz"]], float)        # ego 系
        omega = np.array([e.get("roll_rate", 0.0), e.get("pitch_rate", 0.0),
                          e.get("yaw_rate", 0.0)], float)

        if load_boxes:
            path, boxes_raw, _ = tsc.get_sample_data(sd_token)      # boxes 已在传感器系
        else:
            path, boxes_raw = tsc.get_sample_data_path(sd_token), []
        _, arr = read_pcd(path)
        xyz = np.stack([np.asarray(arr["x"], float), np.asarray(arr["y"], float),
                        np.asarray(arr["z"], float)], 1)
        vrel = np.stack([np.asarray(arr["vrel_x"], float), np.asarray(arr["vrel_y"], float),
                         np.asarray(arr["vrel_z"], float)], 1)
        rcs = np.asarray(arr["rcs"], float)

        rng = np.linalg.norm(xyz, axis=1)
        keep = rng > min_range
        xyz, vrel, rcs, rng = xyz[keep], vrel[keep], rcs[keep], rng[keep]
        rhat = xyz / rng[:, None]
        v_r = np.einsum("ij,ij->i", vrel, rhat)

        # 静态预测: 平台在点处速度 = v_ego + ω×p_ego(杠杆项)
        p_ego = xyz @ R_se.T + t_se
        v_plat_ego = v_ego[None, :] + (np.cross(np.broadcast_to(omega, p_ego.shape), p_ego)
                                       if yaw_rate_correction else 0.0)
        v_plat_s = v_plat_ego @ R_se                                # 逐点, R_se^T v
        pred_static_vr = -np.einsum("ij,ij->i", v_plat_s, rhat)

        boxes = []
        for b in boxes_raw:
            try:
                v_glob = tsc.box_velocity(b.token)                  # 全局系
            except Exception:
                v_glob = np.full(3, np.nan)
            v_obj_sensor = None
            if np.all(np.isfinite(v_glob)):
                v_obj_sensor = (v_glob @ R_eg) @ R_se               # global->ego->sensor
            boxes.append(dict(center=b.center, wlh=b.wlh, rot=b.orientation,
                              name=b.name, ann_token=b.token, v_obj_sensor=v_obj_sensor))

        # 传感器 -> 全局 位姿(时序 warp 用)
        R_gs = R_eg @ R_se
        t_gs = R_eg @ t_se + t_eg

        return dict(xyz=xyz, v_r=v_r, rcs=rcs, rhat=rhat, rng=rng,
                    pred_static_vr=pred_static_vr, v_plat_s=v_plat_s, boxes=boxes,
                    R_gs=R_gs, t_gs=t_gs,
                    v_ego_norm=float(np.linalg.norm(v_ego)), ego_gap_s=gap,
                    channel=sd["channel"], timestamp=sd["timestamp"])


def points_in_box_mask(xyz, center, wlh, rot, margin=1.0):
    """点是否落在 3D 框内(框坐标系: x=length, y=width, z=height); margin 略放大框."""
    p = (xyz - np.asarray(center)) @ rot.rotation_matrix            # R^T(p-c), 行向量形式
    w, l, h = wlh
    return (np.abs(p[:, 0]) <= l / 2 * margin) & \
           (np.abs(p[:, 1]) <= w / 2 * margin) & \
           (np.abs(p[:, 2]) <= h / 2 * margin)


def label_static_dynamic(frame, v_moving_thr=0.5, margin=1.15):
    """逐点标注: 0=背景(视为静态) 1=框内-静止目标 2=框内-运动目标.

    返回 (labels(N,), v_obj_per_point(N,3) 运动目标点的目标速度(传感器系), 其余 NaN)
    """
    xyz = frame["xyz"]
    labels = np.zeros(len(xyz), dtype=np.int8)
    v_obj = np.full((len(xyz), 3), np.nan)
    for b in frame["boxes"]:
        m = points_in_box_mask(xyz, b["center"], b["wlh"], b["rot"], margin)
        if not m.any():
            continue
        vs = b["v_obj_sensor"]
        if vs is not None and np.linalg.norm(vs) > v_moving_thr:
            labels[m] = 2
            v_obj[m] = vs
        else:
            labels[m] = np.maximum(labels[m], 1)
    return labels, v_obj
