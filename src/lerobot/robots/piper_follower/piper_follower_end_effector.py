# !/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time
from typing import Any
import numpy as np
import pdb
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.model.kinematics import RobotKinematics

from .piper_follower import PiperFollower
from .config_piper_follower_end_effector import PiperFollowerEndEffectorConfig

logger = logging.getLogger(__name__)


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]], dtype=np.float64)


def _exp_so3(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-9:
        return np.eye(3, dtype=np.float64) + _skew(w)
    K = _skew(w / theta)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


class PiperFollowerEndEffector(PiperFollower):
    """Piper 双臂末端 6-DoF 控制：仅平移时 IK 只约束位置；显式转动时才启用姿态权重。"""

    config_class = PiperFollowerEndEffectorConfig
    name = "piper_follower_end_effector"

    def __init__(self, config: PiperFollowerEndEffectorConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config

        if self.config.urdf_path is None:
            raise ValueError("必须提供 urdf_path 才能进行末端控制。")

        # 可被 config 覆盖的默认参数
        if not hasattr(self.config, "end_effector_rot_step_sizes"):
            self.config.end_effector_rot_step_sizes = {"rx": 0.003, "ry": 0.003, "rz": 0.003}
        if not hasattr(self.config, "ik_orientation_weight"):
            self.config.ik_orientation_weight = 1.0
        if not hasattr(self.config, "ik_position_weight"):
            self.config.ik_position_weight = 1.0
        if not hasattr(self.config, "orientation_activation_eps"):
            self.config.orientation_activation_eps = 1e-9

        kinL = RobotKinematics(urdf_path=self.config.urdf_path, target_frame_name=self.config.target_frame_name)
        kinR = RobotKinematics(urdf_path=self.config.urdf_path, target_frame_name=self.config.target_frame_name)

        model_joint_order = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7","joint8"]
        for kin in (kinL, kinR):
            if hasattr(kin, "joint_names"):
                try:
                    kin.joint_names = model_joint_order
                except Exception:
                    pass
        pdb.set_trace()
        self.cameras = make_cameras_from_configs(config.cameras)

        self.sides = {
            "left": {
                "bus": self.bus_left, "kin": kinL,
                "arm_keys": [f"left_joint_{i}" for i in range(1, 7)],
                "grip_key": "left_gripper",
                "q6": None, "width": None, "T": None,
            },
            "right": {
                "bus": self.bus_right, "kin": kinR,
                "arm_keys": [f"right_joint_{i}" for i in range(1, 7)],
                "grip_key": "right_gripper",
                "q6": None, "width": None, "T": None,
            },
        }

        self._urdf_finger_max = 0.04

    @property
    def action_features(self) -> dict[str, Any]:
        return {
            "dtype": "float32",
            "shape": (14,),
            "names": {
                "left_delta_x": 0, "left_delta_y": 1, "left_delta_z": 2,
                "left_delta_rx": 3, "left_delta_ry": 4, "left_delta_rz": 5,
                "left_gripper": 6,
                "right_delta_x": 7, "right_delta_y": 8, "right_delta_z": 9,
                "right_delta_rx": 10, "right_delta_ry": 11, "right_delta_rz": 12,
                "right_gripper": 13,
            },
        }

    # ---------- 工具函数 ----------

    def _present_to_q8(self, side: str, present: dict) -> tuple[np.ndarray, float, np.ndarray]:
        arm_keys = self.sides[side]["arm_keys"]
        grip_key = self.sides[side]["grip_key"]
        q6 = np.array([float(present[k]) for k in arm_keys], dtype=np.float64)
        width = float(present.get(grip_key, 0.0))
        d = float(np.clip(width * 0.5, 0.0, self._urdf_finger_max))
        q8 = np.concatenate([q6, np.array([d, d], dtype=np.float64)], axis=0)
        return q6, width, q8

    def _clip_xyz(self, side: str, p: np.ndarray) -> np.ndarray:
        b = self.config.end_effector_bounds_left if side == "left" else self.config.end_effector_bounds_right
        if b:
            p = np.clip(p, b["min"], b["max"])
        return p

    def _grip_next_width(self, cur_w: float, cmd: float) -> float:
        return float(np.clip(cur_w + (float(cmd) - 1.0) * self.config.max_gripper_pos,
                             0.0, self.config.max_gripper_pos))

    def _apply_se3_increment(self, side: str, T_cur: np.ndarray,
                             d_xyz_base: np.ndarray, d_rpy_base: np.ndarray,
                             active_rot: bool) -> np.ndarray:
        """
        基坐标下应用 SE(3) 增量。
        - 平移：总是应用（并做边界裁剪）
        - 旋转：仅当 active_rot=True（显式转动）时：R_tar = R_cur @ exp(Δr)
               否则保持 R_tar = R_cur（只约束位置）
        """
        T_tar = np.eye(4, dtype=np.float64)
        # 平移
        p_tar = self._clip_xyz(side, T_cur[:3, 3] + d_xyz_base)
        # 姿态
        if active_rot:
            R_inc = _exp_so3(d_rpy_base.astype(np.float64))
            R_tar = T_cur[:3, :3] @ R_inc
        else:
            R_tar = T_cur[:3, :3].copy()

        T_tar[:3, :3] = R_tar
        T_tar[:3, 3] = p_tar
        return T_tar

    # ---------- 主流程 ----------

    def send_action(self, action: dict[str, Any] | np.ndarray) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        fields = (
            "left_delta_x","left_delta_y","left_delta_z",
            "left_delta_rx","left_delta_ry","left_delta_rz","left_gripper",
            "right_delta_x","right_delta_y","right_delta_z",
            "right_delta_rx","right_delta_ry","right_delta_rz","right_gripper",
        )
        a14 = np.zeros(14, dtype=np.float64)
        if isinstance(action, dict):
            missing = [k for k in fields if k not in action]
            if missing:
                raise ValueError(f"缺少必要动作字段: {missing}")
            for i, k in enumerate(fields):
                a14[i] = float(action[k])
        else:
            arr = np.asarray(action, dtype=np.float64).ravel()
            if arr.shape != (14,):
                raise ValueError(f"动作应为形状 (14,) 的数组，收到 {arr.shape}")
            a14[:] = arr

        sx, sy, sz = (float(self.config.end_effector_step_sizes[k]) for k in ("x","y","z"))
        srx, sry, srz = (float(self.config.end_effector_rot_step_sizes[k]) for k in ("rx","ry","rz"))

        dL_xyz = np.array([a14[0]*sx, a14[1]*sy, a14[2]*sz], dtype=np.float64)
        dL_rpy = np.array([a14[3]*srx, a14[4]*sry, a14[5]*srz], dtype=np.float64)
        gL_cmd = float(a14[6])

        dR_xyz = np.array([a14[7]*sx, a14[8]*sy, a14[9]*sz], dtype=np.float64)
        dR_rpy = np.array([a14[10]*srx, a14[11]*sry, a14[12]*srz], dtype=np.float64)
        gR_cmd = float(a14[13])

        out = {}
        for side, d_xyz, d_rpy, g_cmd in (
            ("left",  dL_xyz, dL_rpy, gL_cmd),
            ("right", dR_xyz, dR_rpy, gR_cmd),
        ):
            bus   = self.sides[side]["bus"]
            kin   = self.sides[side]["kin"]
            armks = self.sides[side]["arm_keys"]
            gkey  = self.sides[side]["grip_key"]

            present = bus.sync_read("Present_Position")
            q6, width, q8 = self._present_to_q8(side, present)
            T_cur = kin.forward_kinematics(q8)

            # 显式姿态控制检测：只有当 Δrpy 的范数超过阈值时才启用姿态约束
            active_rot = (np.linalg.norm(d_rpy) > float(self.config.orientation_activation_eps))

            # 目标位姿
            T_tar = self._apply_se3_increment(side, T_cur, d_xyz, d_rpy, active_rot)

            # IK：位置权重固定；姿态权重按需启用
            q8_tar = kin.inverse_kinematics(
                q8,
                T_tar,
                position_weight=float(self.config.ik_position_weight),
                orientation_weight=(float(self.config.ik_orientation_weight) if active_rot else 0.0),
            )
            q6_tar = np.asarray(q8_tar[:6], dtype=np.float64)

            # 夹爪
            width_tar = self._grip_next_width(width, g_cmd)

            for i, name in enumerate(armks):
                out[f"{name}.pos"] = float(q6_tar[i])
            out[f"{gkey}.pos"] = width_tar

            self.sides[side]["q6"]    = q6_tar
            self.sides[side]["width"] = width_tar
            self.sides[side]["T"]     = T_tar

        return super().send_action(out)

    def reset(self):
        for side in ("left", "right"):
            self.sides[side]["q6"] = None
            self.sides[side]["width"] = None
            self.sides[side]["T"] = None

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position from both arms
        start = time.perf_counter()
        
        # 左臂位置
        left_positions = self.bus_left.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in left_positions.items()}
        
        # 右臂位置
        right_positions = self.bus_right.sync_read("Present_Position")
        obs_dict.update({f"{motor}.pos": val for motor, val in right_positions.items()})
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read dual-arm state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            image = cam.async_read()
            if image is not None:
                obs_dict[cam_key] = image
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict