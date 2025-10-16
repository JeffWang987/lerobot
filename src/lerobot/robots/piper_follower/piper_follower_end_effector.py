# !/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError
from lerobot.model.kinematics import RobotKinematics

from .piper_follower import PiperFollower
from .config_piper_follower_end_effector import PiperFollowerEndEffectorConfig

logger = logging.getLogger(__name__)


class PiperFollowerEndEffector(PiperFollower):
    """
    Piper 双臂末端空间控制。
    把 EE 空间动作 (Δx,Δy,Δz,gripper) → 关节空间，再交给父类 send_action 下发。
    - URDF: 6 个旋转关节 + 2 个平动手指 (joint7/joint8)
    - 硬件: 6 个旋转关节 + 1 个 gripper 宽度(米)
    - 映射: joint7 = joint8 = width/2（并裁剪到[0, 0.04]m）
    """

    config_class = PiperFollowerEndEffectorConfig
    name = "piper_follower_end_effector"

    def __init__(self, config: PiperFollowerEndEffectorConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config

        # 末端运动学（左右臂各一份）
        if self.config.urdf_path is None:
            raise ValueError(
                "必须提供 urdf_path 才能进行末端控制。请在 PiperFollowerEndEffectorConfig 中设置 urdf_path。"
            )
        self.kinematics_left = RobotKinematics(
            urdf_path=self.config.urdf_path,
            target_frame_name=self.config.target_frame_name,
        )
        self.kinematics_right = RobotKinematics(
            urdf_path=self.config.urdf_path,
            target_frame_name=self.config.target_frame_name,
        )

        # —— 关键：显式指定模型关节顺序（与你的 URDF 对应）——
        # joint7/joint8 为两指的平动关节
        self.model_joint_order = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"]
        # 让 FK/IK 都按这套顺序 set/get（如果 RobotKinematics 支持）
        try:
            self.kinematics_left.joint_names = self.model_joint_order
            self.kinematics_right.joint_names = self.model_joint_order
        except Exception:
            pass

        # 状态缓存（硬件 6 关节 + gripper 宽度；以及模型的 8 维 q 的 EE 齐次变换）
        self.current_q_left = None         # 左臂 6x1（rad）
        self.current_q_right = None        # 右臂 6x1（rad）
        self.current_width_left = None     # 左夹爪宽度（m）
        self.current_width_right = None    # 右夹爪宽度（m）
        self.current_ee_T_left = None      # 左 EE 4x4
        self.current_ee_T_right = None     # 右 EE 4x4

        # 相机（与父类一致）
        self.cameras = make_cameras_from_configs(config.cameras)

        # 硬件侧“臂关节”键名（不含 gripper）
        self.arm_joint_keys_left = [f"left_joint_{i}" for i in range(1, 7)]
        self.arm_joint_keys_right = [f"right_joint_{i}" for i in range(1, 7)]
        self.left_gripper_key = "left_gripper"
        self.right_gripper_key = "right_gripper"

        # URDF 中两指的最大行程（各自）
        self._urdf_finger_max = 0.04  # joint7/joint8 上限（米），来自你的 URDF

    # === 动作特征：左(Δx,Δy,Δz,gripper) + 右(Δx,Δy,Δz,gripper) ===
    @property
    def action_features(self) -> dict[str, Any]:
        return {
            "dtype": "float32",
            "shape": (8,),
            "names": {
                "left_delta_x": 0, "left_delta_y": 1, "left_delta_z": 2, "left_gripper": 3,
                "right_delta_x": 4, "right_delta_y": 5, "right_delta_z": 6, "right_gripper": 7,
            },
        }

    # ===== 工具：硬件读数 → 模型 8 维 q（joint1..joint8）=====
    def _hw_to_model_q(self, side: str, present_pos: dict) -> np.ndarray:
        if side == "left":
            arm_keys = self.arm_joint_keys_left
            g_key = self.left_gripper_key
        else:
            arm_keys = self.arm_joint_keys_right
            g_key = self.right_gripper_key

        q = []
        # 1..6 旋转关节
        for k in arm_keys:
            q.append(float(present_pos[k]))  # Python float→C++ double

        # gripper 宽度（m）→ 两指各自位移
        width = float(present_pos.get(g_key, 0.0))
        d = np.clip(width * 0.5, 0.0, self._urdf_finger_max)
        q.append(float(d))  # joint7
        q.append(float(d))  # joint8

        return np.asarray(q, dtype=np.float64)  # FK/IK 用 float64

    # ===== 从缓存的硬件 6 关节 + width 组合出模型 8 维 q（供 IK 初值或 FK 使用）=====
    def _compose_model_q_from_cache(self, side: str) -> np.ndarray:
        if side == "left":
            assert self.current_q_left is not None and self.current_width_left is not None
            width = self.current_width_left
            q6 = self.current_q_left
        else:
            assert self.current_q_right is not None and self.current_width_right is not None
            width = self.current_width_right
            q6 = self.current_q_right

        d = np.clip(float(width) * 0.5, 0.0, self._urdf_finger_max)
        q = np.concatenate([np.asarray(q6, dtype=np.float64), np.array([d, d], dtype=np.float64)], axis=0)
        return q

    # === 读取一次当前 q/EE（内部工具） ===
    def _read_current_q_ee_once(self):
        # 左臂 present
        left_present = self.bus_left.sync_read("Present_Position")
        self.current_q_left = np.array([left_present[n] for n in self.arm_joint_keys_left], dtype=np.float64)
        self.current_width_left = float(left_present.get(self.left_gripper_key, 0.0))
        q_left_model = self._hw_to_model_q("left", left_present)
        self.current_ee_T_left = self.kinematics_left.forward_kinematics(q_left_model)

        # 右臂 present
        right_present = self.bus_right.sync_read("Present_Position")
        self.current_q_right = np.array([right_present[n] for n in self.arm_joint_keys_right], dtype=np.float64)
        self.current_width_right = float(right_present.get(self.right_gripper_key, 0.0))
        q_right_model = self._hw_to_model_q("right", right_present)
        self.current_ee_T_right = self.kinematics_right.forward_kinematics(q_right_model)

    def send_action(self, action: dict[str, Any] | np.ndarray) -> dict[str, Any]:
        """
        输入（dict 或 ndarray）:
        - dict: 包含 left/right_delta_x/y/z，left/right_gripper（可缺省，默认 1.0）
        - ndarray shape=(8,): [lx,ly,lz,lg, rx,ry,rz,rg]
        其中 Δx/Δy/Δz 会乘以 config.end_effector_step_sizes 的步长（米）。
        gripper∈[0,2]，会映射到 [-max,+max] 的增量（米），与当前宽度叠加，并裁剪到 [0, max_gripper_pos]（米）。
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # ---- 解析动作为 ndarray ----
        if isinstance(action, dict):
            lgr = float(action.get("left_gripper", 1.0))
            rgr = float(action.get("right_gripper", 1.0))
            arr = np.array([
                float(action.get("left_delta_x", 0.0)),
                float(action.get("left_delta_y", 0.0)),
                float(action.get("left_delta_z", 0.0)),
                lgr,
                float(action.get("right_delta_x", 0.0)),
                float(action.get("right_delta_y", 0.0)),
                float(action.get("right_delta_z", 0.0)),
                rgr,
            ], dtype=np.float32)
            action = arr
        else:
            action = np.asarray(action, dtype=np.float32)
            if action.shape != (8,):
                logger.warning(f"Expected action shape (8,), got {action.shape}. Zeroing.")
                action = np.zeros(8, dtype=np.float32)

        # === 修改开始：每一步都从硬件重读 present，并用 FK 同步当前 EE（闭环增量） ===
        left_present = self.bus_left.sync_read("Present_Position")
        right_present = self.bus_right.sync_read("Present_Position")

        self.current_q_left = np.array([left_present[n] for n in self.arm_joint_keys_left], dtype=np.float64)
        self.current_q_right = np.array([right_present[n] for n in self.arm_joint_keys_right], dtype=np.float64)
        self.current_width_left = float(left_present.get(self.left_gripper_key, 0.0))
        self.current_width_right = float(right_present.get(self.right_gripper_key, 0.0))

        q_left_model = self._hw_to_model_q("left", left_present)
        q_right_model = self._hw_to_model_q("right", right_present)
        self.current_ee_T_left = self.kinematics_left.forward_kinematics(q_left_model)
        self.current_ee_T_right = self.kinematics_right.forward_kinematics(q_right_model)
        # === 修改结束 ===

        # ---- 步长缩放（米） ----
        sx = float(self.config.end_effector_step_sizes["x"])
        sy = float(self.config.end_effector_step_sizes["y"])
        sz = float(self.config.end_effector_step_sizes["z"])
        l_delta = np.array([action[0]*sx, action[1]*sy, action[2]*sz], dtype=np.float64)
        r_delta = np.array([action[4]*sx, action[5]*sy, action[6]*sz], dtype=np.float64)

        # ---- 目标 EE 位姿（保持当前姿态，只改平移） ----
        Tl = np.eye(4, dtype=np.float64); Tl[:3, :3] = self.current_ee_T_left[:3, :3]
        Tr = np.eye(4, dtype=np.float64); Tr[:3, :3] = self.current_ee_T_right[:3, :3]
        Tl[:3, 3] = self.current_ee_T_left[:3, 3] + l_delta
        Tr[:3, 3] = self.current_ee_T_right[:3, 3] + r_delta

        # ---- 边界裁剪（米）----
        if self.config.end_effector_bounds_left:
            Tl[:3, 3] = np.clip(
                Tl[:3, 3],
                self.config.end_effector_bounds_left["min"],
                self.config.end_effector_bounds_left["max"],
            )
        if self.config.end_effector_bounds_right:
            Tr[:3, 3] = np.clip(
                Tr[:3, 3],
                self.config.end_effector_bounds_right["min"],
                self.config.end_effector_bounds_right["max"],
            )

        # ---- IK：用“模型 8 维 q（含手指）”作为初值，但只关心前 6 个旋转关节的结果 ----
        ql_init8 = self._compose_model_q_from_cache("left")
        qr_init8 = self._compose_model_q_from_cache("right")

        ql_target8 = self.kinematics_left.inverse_kinematics(ql_init8, Tl)
        qr_target8 = self.kinematics_right.inverse_kinematics(qr_init8, Tr)

        # 取前 6 维作为要下发的关节角
        ql_target6 = np.asarray(ql_target8[:6], dtype=np.float64)
        qr_target6 = np.asarray(qr_target8[:6], dtype=np.float64)

        # ---- 夹爪（增量映射：0..2 → [-1, +1]，乘最大行程再与当前叠加，最后裁剪到 [0, max] 米）----
        l_new_width = float(np.clip(self.current_width_left  + (float(action[3]) - 1.0) * self.config.max_gripper_pos,
                                    0.0, self.config.max_gripper_pos))
        r_new_width = float(np.clip(self.current_width_right + (float(action[7]) - 1.0) * self.config.max_gripper_pos,
                                    0.0, self.config.max_gripper_pos))

        # ---- 组装关节空间动作（键名要与父类解析一致）----
        left_joint_names  = self.arm_joint_keys_left
        right_joint_names = self.arm_joint_keys_right
        joint_action = {}
        for i, name in enumerate(left_joint_names):
            joint_action[f"{name}.pos"] = float(ql_target6[i])
        for i, name in enumerate(right_joint_names):
            joint_action[f"{name}.pos"] = float(qr_target6[i])

        joint_action[f"{self.left_gripper_key}.pos"]  = l_new_width
        joint_action[f"{self.right_gripper_key}.pos"] = r_new_width

        # ---- 更新缓存 ----
        self.current_q_left  = ql_target6.copy()
        self.current_q_right = qr_target6.copy()
        self.current_width_left  = l_new_width
        self.current_width_right = r_new_width
        self.current_ee_T_left  = Tl.copy()
        self.current_ee_T_right = Tr.copy()

        # 交给父类做安全裁剪（max_relative_target）并通过总线下发
        return super().send_action(joint_action)

    # 可选：覆写 reset 以清缓存
    def reset(self):
        self.current_q_left = None
        self.current_q_right = None
        self.current_width_left = None
        self.current_width_right = None
        self.current_ee_T_left = None
        self.current_ee_T_right = None
