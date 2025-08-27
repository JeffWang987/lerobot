#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.piper import (
    PiperMotorsBus,
)

from ..teleoperator import Teleoperator
from .config_piper_leader import PiperLeaderConfig

logger = logging.getLogger(__name__)


class PiperLeader(Teleoperator):
    """
    [Piper Leader Arm](https://github.com/agilexrobotics/piper_sdk)
    """

    config_class = PiperLeaderConfig
    name = "piper_leader"

    def __init__(self, config: PiperLeaderConfig):
        super().__init__(config)
        self.config = config
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

    @property
    def action_features(self) -> dict[str, type]:
        # return {f"{motor}.pos": float for motor in self.bus.motors}
        left_motors =  {f"{motor}.pos": float for motor in self.bus_left.motors}
        right_motors =  {f"{motor}.pos": float for motor in self.bus_right.motors}
        return {**left_motors, **right_motors} # TODO check left or right写法是否正确

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus_left.is_connected and self.bus_right.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus_left.connect(enable=True)
        self.bus_right.connect(enable=True)
        self.calibrate()
        logger.info(f"{self} connected.")
        
    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus_left.safe_disconnect()
        self.bus_right.safe_disconnect()
        self.bus_left.connect(enable=False)
        self.bus_right.connect(enable=False)

        logger.info(f"{self} disconnected.")
    
    def calibrate(self) -> None:
        """move piper to the home position"""
        if not self.is_connected:
            raise ConnectionError()
        
        self.bus_left.apply_calibration()
        self.bus_right.apply_calibration()

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action_left = self.bus_left.sync_read("Present_Position")
        action_right = self.bus_right.sync_read("Present_Position")
        action_left = {f"{motor}.pos": val for motor, val in action_left.items()}
        action_right = {f"{motor}.pos": val for motor, val in action_right.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return {**action_left, **action_right} # TODO check 能否接收到

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Write the predicted actions from policy to the motors"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # send to motors, torch to list
        target_joints = action.tolist()
        self.bus_left.write(target_joints[:7])
        self.bus_right.write(target_joints[7:])

        return action
    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

