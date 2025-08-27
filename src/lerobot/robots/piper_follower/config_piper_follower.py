#!/usr/bin/env python

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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras.dabai import OrbbecDabaiCameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("piper_follower")
@dataclass
class PiperFollowerConfig(RobotConfig):
    # Port to connect to the arm
    port_left: str
    port_right: str
    target_frame_name: str = "gripper"
    urdf_path: str = "local_assets/piper.urdf"

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "head": OrbbecDabaiCameraConfig( # TODO check 头部相机是否能用realsense
                type='dabai',
                serial_number_or_name="0123456789", # TODO Replace with actual SN
                # use_depth=True,  # TODO check depth 使用
                fps=30, 
                width=640,  # TODO check  dabai相机分辨率
                height=480,
            ),
            "wrist_left": RealSenseCameraConfig(
                type='intelrealsense',
                serial_number_or_name="0123456789", # Replace with actual SN
                # use_depth=True,
                fps=30,
                width=640,
                height=480,
            ),
            "wrist_right": RealSenseCameraConfig(
                type='intelrealsense',
                serial_number_or_name="0123456789", # Replace with actual SN
                # use_depth=True,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )


    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False


@RobotConfig.register_subclass("piper_follower_end_effector")
@dataclass
class PiperFollowerEndEffectorConfig(PiperFollowerConfig):
    """Configuration for the PiperFollowerEndEffector robot."""

    # Path to URDF file for kinematics
    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo:
    # https://github.com/agilexrobotics/piper_ros/tree/noetic/src/piper_description/urdf
    urdf_path: str = "local_assets/piper.urdf"

    # End-effector frame name in the URDF
    target_frame_name: str = "gripper"

    # Default bounds for the end-effector position (in meters)
    end_effector_bounds_left: dict[str, list[float]] = field(
        default_factory=lambda: {
            "min": [-1.0, -1.0, -1.0],  # min x, y, z
            "max": [1.0, 1.0, 1.0],  # max x, y, z
        }
    )
    end_effector_bounds_right: dict[str, list[float]] = field(
        default_factory=lambda: {
            "min": [-1.0, -1.0, -1.0],  # min x, y, z
            "max": [1.0, 1.0, 1.0],  # max x, y, z
        }
    )

    max_gripper_pos: float = 50

    end_effector_step_sizes: dict[str, float] = field(
        default_factory=lambda: {
            "x": 0.02,
            "y": 0.02,
            "z": 0.02,
        }
    )
