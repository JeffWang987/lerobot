# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from typing import Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError
from lerobot.model.kinematics import RobotKinematics
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.piper import (
    PiperMotorsBus,
)

from . import PiperFollower
from .config_piper_follower import PiperFollowerEndEffectorConfig

logger = logging.getLogger(__name__)


class PiperFollowerEndEffector(PiperFollower):
    """
    PiperFollower robot with end-effector space control.

    This robot inherits from PiperFollower but transforms actions from
    end-effector space to joint space before sending them to the motors.
    """

    config_class = PiperFollowerEndEffectorConfig
    name = "piper_follower_end_effector"

    def __init__(self, config: PiperFollowerEndEffectorConfig):
        super().__init__(config)
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100 # TODO check 下含义
        self.bus_left = PiperMotorsBus(
            port=self.config.port_left,  # TODO 设置为"can0"和"can1"，还没找到在哪里设置
            motors={
                "joint_1": Motor(1, "agilex_piper", norm_mode_body),  # TODO check norm
                "joint_2": Motor(2, "agilex_piper", norm_mode_body),
                "joint_3": Motor(3, "agilex_piper", norm_mode_body),
                "joint_4": Motor(4, "agilex_piper", norm_mode_body),
                "joint_5": Motor(5, "agilex_piper", norm_mode_body),
                "joint_6": Motor(6, "agilex_piper", norm_mode_body),
                "gripper": Motor(7, "agilex_piper", MotorNormMode.RANGE_0_100),  # TODO check norm
            },
        )
        self.bus_right = PiperMotorsBus(
            port=self.config.port_right,  # TODO 设置为"can0"和"can1"，还没找到在哪里设置
            motors={
                "joint_1": Motor(1, "agilex_piper", norm_mode_body),  # TODO check norm
                "joint_2": Motor(2, "agilex_piper", norm_mode_body),
                "joint_3": Motor(3, "agilex_piper", norm_mode_body),
                "joint_4": Motor(4, "agilex_piper", norm_mode_body),
                "joint_5": Motor(5, "agilex_piper", norm_mode_body),
                "joint_6": Motor(6, "agilex_piper", norm_mode_body),
                "gripper": Motor(7, "agilex_piper", MotorNormMode.RANGE_0_100),  # TODO check norm
            },
        )

        self.cameras = make_cameras_from_configs(config.cameras)

        self.config = config

        # Initialize the kinematics module for the so100 robot
        if self.config.urdf_path is None:
            raise ValueError(
                "urdf_path must be provided in the configuration for end-effector control. "
                "Please set urdf_path in your SO100FollowerEndEffectorConfig."
            )

        self.kinematics = RobotKinematics(
            urdf_path=self.config.urdf_path,
            target_frame_name=self.config.target_frame_name,
        )

        # Store the bounds for end-effector position
        self.end_effector_bounds_left = self.config.end_effector_bounds_left
        self.end_effector_bounds_right = self.config.end_effector_bounds_right

        self.current_ee_pos_left = None
        self.current_joint_pos_left = None
        self.current_ee_pos_right = None
        self.current_joint_pos_right = None

    @property
    def action_features(self) -> dict[str, Any]:
        """
        Define action features for end-effector control.
        Returns dictionary with dtype, shape, and names.
        """
        return {
            "dtype": "float32",
            "shape": (8,),
            # TODO check
            "names": {"left_delta_x": 0, "left_delta_y": 1, "left_delta_z": 2, "left_gripper": 3, "right_delta_x": 4, "right_delta_y": 5, "right_delta_z": 6, "right_gripper": 7},
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Transform action from end-effector space to joint space and send to motors.

        Args:
            action: Dictionary with keys 'delta_x', 'delta_y', 'delta_z' for end-effector control
                   or a numpy array with [delta_x, delta_y, delta_z]

        Returns:
            The joint-space action that was sent to the motors
        """

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Convert action to numpy array if not already
        if isinstance(action, dict):
            if all(k in action for k in ["left_delta_x", "left_delta_y", "left_delta_z", "right_delta_x", "right_delta_y", "right_delta_z"]):
                delta_ee = np.array(
                    [
                        action["left_delta_x"] * self.config.end_effector_step_sizes["x"],
                        action["left_delta_y"] * self.config.end_effector_step_sizes["y"],
                        action["left_delta_z"] * self.config.end_effector_step_sizes["z"],
                        action["right_delta_x"] * self.config.end_effector_step_sizes["x"],
                        action["right_delta_y"] * self.config.end_effector_step_sizes["y"],
                        action["right_delta_z"] * self.config.end_effector_step_sizes["z"],
                    ],
                    dtype=np.float32,
                )
                if "left_gripper" not in action:
                    action["left_gripper"] = [1.0]
                if "right_gripper" not in action:
                    action["right_gripper"] = [1.0]
                action = np.append(delta_ee, action["left_gripper"])
                action = np.append(action, action["right_gripper"])
            else:
                logger.warning(
                    f"Expected action keys 'left_delta_x', 'left_delta_y', 'left_delta_z', 'right_delta_x', 'right_delta_y', 'right_delta_z', got {list(action.keys())}"
                )
                action = np.zeros(8, dtype=np.float32)

        if self.current_joint_pos_left is None and self.current_joint_pos_right is None:
            # Read current joint positions
            current_joint_pos_left = self.bus_left.sync_read("Present_Position")
            current_joint_pos_right = self.bus_right.sync_read("Present_Position")
            self.current_joint_pos_left = np.array([current_joint_pos_left[name] for name in self.bus_left.motors])
            self.current_joint_pos_right = np.array([current_joint_pos_right[name] for name in self.bus_right.motors])

        # Calculate current end-effector position using forward kinematics
        if self.current_ee_pos_left is None and self.current_ee_pos_right is None:
            self.current_ee_pos_left = self.kinematics.forward_kinematics(self.current_joint_pos_left)
            self.current_ee_pos_right = self.kinematics.forward_kinematics(self.current_joint_pos_right)

        # Set desired end-effector position by adding delta
        desired_ee_pos_left = np.eye(4)
        desired_ee_pos_left[:3, :3] = self.current_ee_pos_left[:3, :3]  # Keep orientation
        desired_ee_pos_right = np.eye(4)
        desired_ee_pos_right[:3, :3] = self.current_ee_pos_right

        # Add delta to position and clip to bounds
        desired_ee_pos_left[:3, 3] = self.current_ee_pos_left[:3, 3] + action[:3]
        desired_ee_pos_right[:3, 3] = self.current_ee_pos_right[:3, 3] + action[4:7]
        if self.end_effector_bounds_left is not None:
            desired_ee_pos_left[:3, 3] = np.clip(
                desired_ee_pos_left[:3, 3],
                self.end_effector_bounds_left["min"],
                self.end_effector_bounds_left["max"],
            )
        if self.end_effector_bounds_right is not None:
            desired_ee_pos_right[:3, 3] = np.clip(
                desired_ee_pos_right[:3, 3],
                self.end_effector_bounds_right["min"],
                self.end_effector_bounds_right["max"],
            )

        # Compute inverse kinematics to get joint positions
        target_joint_values_in_degrees_left = self.kinematics.inverse_kinematics(
            self.current_joint_pos_left, desired_ee_pos_left
        )
        target_joint_values_in_degrees_right = self.kinematics.inverse_kinematics(
            self.current_joint_pos_right, desired_ee_pos_right
        )

        # Create joint space action dictionary
        joint_action_left = {
            f"{key}.pos": target_joint_values_in_degrees_left[i] for i, key in enumerate(self.bus_left.motors.keys())
        }
        joint_action_right = {
            f"{key}.pos": target_joint_values_in_degrees_right[i] for i, key in enumerate(self.bus_right.motors.keys())
        }

        # Handle gripper separately if included in action
        # Gripper delta action is in the range 0 - 2,
        # We need to shift the action to the range -1, 1 so that we can expand it to -Max_gripper_pos, Max_gripper_pos
        # TODO check piper gripper 范围
        joint_action_left["gripper.pos"] = np.clip(
            self.current_joint_pos_left[-1] + (action[3] - 1) * self.config.max_gripper_pos,
            5,
            self.config.max_gripper_pos,
        )
        joint_action_right["gripper.pos"] = np.clip(
            self.current_joint_pos_right[-1] + (action[7] - 1) * self.config.max_gripper_pos,
            5,
            self.config.max_gripper_pos,
        )

        self.current_ee_pos_left = desired_ee_pos_left.copy()
        self.current_joint_pos_left = target_joint_values_in_degrees_left.copy()
        self.current_joint_pos_left[-1] = joint_action_left["gripper.pos"]
        self.current_ee_pos_right = desired_ee_pos_right.copy()
        self.current_joint_pos_right = target_joint_values_in_degrees_right.copy()
        self.current_joint_pos_right[-1] = joint_action_right["gripper.pos"]


        # Send joint space action to parent class
        # return super().send_action(joint_action)
        return super().send_action({**joint_action_left, **joint_action_right})
    

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        obs_dict_left = self.bus_left.sync_read("Present_Position")
        obs_dict_left = {f"{motor}.pos": val for motor, val in obs_dict_left.items()}
        obs_dict_right = self.bus_right.sync_read("Present_Position")
        obs_dict_right = {f"{motor}.pos": val for motor, val in obs_dict_right.items()}
        obs_dict = {}
        obs_dict['left_arm'] = obs_dict_left # TODO check left or right写法是否正确
        obs_dict['right_arm'] = obs_dict_right # TODO check left or right写法是否正确
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def reset(self):
        self.current_ee_pos_left = None
        self.current_joint_pos_left = None
        self.current_ee_pos_right = None
        self.current_joint_pos_right = None
