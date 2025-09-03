#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Piper机械臂独立实现，用于LeRobot框架

这个模块包含了完整的Piper机械臂实现，包括：
- PiperMotorsBus: 电机总线控制
- PiperFollowerConfig/PiperFollowerEndEffectorConfig: 机器人配置
- PiperFollower: 双臂机器人控制接口
- PiperFollowerEndEffector: 双臂末端执行器控制接口
- PiperLeaderConfig: 领导臂配置
- PiperLeader: 双臂领导臂遥操控接口

适用于强化学习脚本：find_joint_limits.py, gym_manipulator.py, actor.py, learner.py
"""

import logging
import time
from typing import Dict, Any, Optional, List
import numpy as np
import torch
from pathlib import Path
from functools import cached_property
from dataclasses import dataclass, field

# LeRobot框架导入
from lerobot.robots.robot import Robot
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.cameras import CameraConfig
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.robots.utils import ensure_safe_goal_position

# Piper SDK导入 - 使用V2版本接口
try:
    from third_party.piper_sdk.piper_sdk.interface.piper_interface_v2 import C_PiperInterface_V2
except ImportError:
    logging.error("Piper SDK V2 not found. Please ensure third_party/piper_sdk is available.")
    C_PiperInterface_V2 = None


# =============================================================================
# 电机总线实现
# =============================================================================

class PiperMotorsBus:
    """
    Piper机械臂电机总线独立实现 - 使用V2接口

    提供类似MotorsBus的接口，但直接使用Piper SDK V2的CAN总线通信
    适配固件版本 V1.5-2 及以上（V1.6-5完全支持）
    """

    def __init__(self, can_name: str, baud_rate: int = 1000000):
        """
        初始化Piper电机总线

        Args:
            can_name: CAN接口名称 (如 "can0", "can_left", "can_right")
            baud_rate: CAN波特率 (通常为1000000)
        """
        self.can_name = can_name
        self.baud_rate = baud_rate
        self.interface = None
        self.is_connected = False

        # Piper机械臂关节定义 (基于常见6DOF+夹爪配置)
        self.motors = [
            "joint_1",  # 底座旋转
            "joint_2",  # 肩部俯仰
            "joint_3",  # 肘部俯仰
            "joint_4",  # 腕部旋转
            "joint_5",  # 腕部俯仰
            "joint_6",  # 腕部滚转
            "gripper"   # 夹爪
        ]

        # 关节限位 (弧度) - 基于Piper官方文档
        self.joint_limits = {
            "joint_1": (-2.6179, 2.6179),      # [-150°, 150°]
            "joint_2": (0, 3.14),              # [0°, 180°]
            "joint_3": (-2.967, 0),            # [-170°, 0°]
            "joint_4": (-1.745, 1.745),        # [-100°, 100°]
            "joint_5": (-1.22, 1.22),          # [-70°, 70°]
            "joint_6": (-2.09439, 2.09439),    # [-120°, 120°]
            "gripper": (0.0, 0.1)              # 夹爪开合度 [0-100mm]
        }

        # 当前状态缓存
        self._current_positions = {motor: 0.0 for motor in self.motors}
        self._current_velocities = {motor: 0.0 for motor in self.motors}
        self._current_currents = {motor: 0.0 for motor in self.motors}

        logging.info(f"Initialized PiperMotorsBus V2 with {len(self.motors)} motors")
        logging.info(f"CAN interface: {can_name}, Baud rate: {baud_rate}")

    def connect(self) -> bool:
        """连接到Piper机械臂"""
        if C_PiperInterface_V2 is None:
            logging.error("Piper SDK V2 not available")
            return False

        try:
            # 创建V2接口实例
            self.interface = C_PiperInterface_V2(
                can_name=self.can_name,
                judge_flag=True,
                can_auto_init=True,
                dh_is_offset=0x01,              # 启用DH参数偏移
                start_sdk_joint_limit=True,     # 启用软件关节限位
                start_sdk_gripper_limit=True    # 启用软件夹爪限位
            )

            # 连接CAN端口
            self.interface.ConnectPort(
                can_init=True,
                piper_init=True,
                start_thread=True
            )

            if self.interface.get_connect_status():
                self.is_connected = True
                logging.info(f"Successfully connected to Piper robot on {self.can_name}")
                self._initialize_robot()
                return True
            else:
                logging.error(f"Failed to connect to Piper robot on {self.can_name}")
                return False

        except Exception as e:
            logging.error(f"Exception during Piper connection: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        if self.interface and self.is_connected:
            try:
                self.interface.DisconnectPort()
                self.is_connected = False
                logging.info("Disconnected from Piper robot")
            except Exception as e:
                logging.error(f"Error disconnecting from Piper robot: {e}")

    def _initialize_robot(self):
        """初始化机器人状态"""
        try:
            # 等待连接稳定
            time.sleep(0.1)

            # 使能机械臂 (V2版本方法) - 循环等待直到成功
            max_retries = 100
            retry_count = 0
            while retry_count < max_retries:
                if self.interface.EnablePiper():  # 使用便捷方法
                    break
                time.sleep(0.01)
                retry_count += 1

            if retry_count >= max_retries:
                raise Exception("Failed to enable Piper robot after maximum retries")

            # 设置控制模式为CAN命令控制
            self.interface.ModeCtrl(
                ctrl_mode=0x01,     # CAN命令控制模式
                move_mode=0x01,     # MOVE J (关节)模式
                move_spd_rate_ctrl=50,  # 50%速度
                is_mit_mode=0x00    # 位置-速度模式
            )

            # 初始化夹爪参数 (V2版本特有功能)
            self.interface.GripperTeachingPendantParamConfig(
                teaching_range_per=100,     # 100%行程系数
                max_range_config=70,        # 小夹爪70mm最大行程
                teaching_friction=1         # 示教摩擦力
            )

            # 设置碰撞保护等级 (安全考虑)
            self.interface.CrashProtectionConfig(
                joint_1_protection_level=3,
                joint_2_protection_level=3,
                joint_3_protection_level=3,
                joint_4_protection_level=3,
                joint_5_protection_level=3,
                joint_6_protection_level=3
            )

            # 读取初始位置
            self._update_positions()

            logging.info("Piper robot V2 initialized successfully")

        except Exception as e:
            logging.error(f"Failed to initialize Piper robot: {e}")
            raise

    def _update_positions(self):
        """更新关节位置"""
        try:
            # 获取关节角度信息 (返回0.001度单位)
            joint_msgs = self.interface.GetArmJointMsgs()
            if joint_msgs is not None:
                # 转换为弧度并更新前6个关节
                for i in range(6):
                    motor_name = self.motors[i]
                    # 从0.001度转换为弧度
                    angle_deg = joint_msgs.joint_state[i] / 1000.0
                    angle_rad = np.deg2rad(angle_deg)
                    self._current_positions[motor_name] = angle_rad

            # 获取夹爪信息
            gripper_msgs = self.interface.GetArmGripperMsgs()
            if gripper_msgs is not None:
                # 夹爪位置单位为0.001mm，转换为米
                gripper_pos_m = gripper_msgs.grippers_angle / 1000000.0  # 0.001mm -> m
                self._current_positions["gripper"] = gripper_pos_m

        except Exception as e:
            logging.error(f"Failed to update positions: {e}")

    def sync_read(self, parameter: str) -> Dict[str, float]:
        """
        同步读取参数

        Args:
            parameter: 参数名称 ("Present_Position", "Present_Velocity", "Present_Current")

        Returns:
            电机名称到值的字典
        """
        try:
            if parameter == "Present_Position":
                self._update_positions()
                return self._current_positions.copy()

            elif parameter == "Present_Velocity":
                # 获取高速反馈信息（包含速度）
                high_spd_msgs = self.interface.GetArmHighSpdInfoMsgs()
                if high_spd_msgs is not None:
                    for i in range(6):
                        motor_name = self.motors[i]
                        # 速度单位为rad/s
                        self._current_velocities[motor_name] = high_spd_msgs.joint_speed[i]

                    # 夹爪速度 (假设为0，或根据实际反馈计算)
                    self._current_velocities["gripper"] = 0.0

                return self._current_velocities.copy()

            elif parameter == "Present_Current":
                # 获取高速反馈信息（包含电流）
                high_spd_msgs = self.interface.GetArmHighSpdInfoMsgs()
                if high_spd_msgs is not None:
                    for i in range(6):
                        motor_name = self.motors[i]
                        # 电流单位为0.001A，转换为A
                        current_a = high_spd_msgs.joint_current[i] / 1000.0
                        self._current_currents[motor_name] = current_a

                    # 夹爪电流
                    gripper_msgs = self.interface.GetArmGripperMsgs()
                    if gripper_msgs is not None:
                        # 夹爪力矩单位为0.001N·m
                        gripper_effort = gripper_msgs.grippers_effort / 1000.0
                        self._current_currents["gripper"] = gripper_effort

                return self._current_currents.copy()

            else:
                logging.warning(f"Unsupported parameter: {parameter}")
                return {}

        except Exception as e:
            logging.error(f"Failed to sync_read {parameter}: {e}")
            return {}

    def sync_write(self, parameter: str, values: Dict[str, float] | float):
        """
        同步写入参数

        Args:
            parameter: 参数名称 ("Goal_Position", "Torque_Enable")
            values: 值字典或单个值
        """
        try:
            if not self.interface or not self.interface.get_connect_status():
                logging.warning("Interface not connected, cannot write")
                return

            if parameter == "Goal_Position":
                if isinstance(values, dict):
                    # 分离关节角度和夹爪
                    joint_angles = []

                    # 提取关节角度（弧度转换为0.001度）
                    for i, motor in enumerate(self.motors[:-1]):  # 除了夹爪
                        if motor in values:
                            angle_rad = values[motor]
                        else:
                            # 使用当前位置
                            angle_rad = self._current_positions[motor]
                        joint_angles.append(angle_rad)

                    # 验证关节角度
                    validated_angles = self._validate_joint_angles(joint_angles)

                    # 转换为0.001度单位
                    joint_angles_millideg = [int(np.rad2deg(angle) * 1000)
                                           for angle in validated_angles]

                    # 发送关节指令 (V2版本方法)
                    if len(joint_angles_millideg) == 6:
                        self.interface.JointCtrl(
                            joint_1=joint_angles_millideg[0],
                            joint_2=joint_angles_millideg[1],
                            joint_3=joint_angles_millideg[2],
                            joint_4=joint_angles_millideg[3],
                            joint_5=joint_angles_millideg[4],
                            joint_6=joint_angles_millideg[5]
                        )

                    # 处理夹爪
                    if "gripper" in values:
                        gripper_pos_m = self._validate_gripper_position(values["gripper"])
                        # 米转换为0.001mm
                        gripper_angle_micro = int(gripper_pos_m * 1000000)

                        self.interface.GripperCtrl(
                            gripper_angle=gripper_angle_micro,
                            gripper_effort=1000,        # 1N·m 默认力矩
                            gripper_code=0x01,          # 使能
                            set_zero=0x00               # 不设零点
                        )

            elif parameter == "Torque_Enable":
                enable = int(values) if not isinstance(values, dict) else 1
                if enable:
                    self.interface.EnableArm(motor_num=7, enable_flag=0x02)
                    # 使能后更新位置状态，确保数据同步
                    time.sleep(0.1)  # 等待使能生效
                    self._update_positions()
                else:
                    self.interface.DisableArm(motor_num=7, enable_flag=0x01)

            else:
                logging.warning(f"Unsupported write parameter: {parameter}")

        except Exception as e:
            logging.error(f"Failed to sync_write {parameter}: {e}")

    def write(self, parameter: str, motor: str, value: float):
        """
        写入单个电机参数

        Args:
            parameter: 参数名称
            motor: 电机名称
            value: 值
        """
        self.sync_write(parameter, {motor: value})

    def _validate_joint_angles(self, angles: List[float]) -> List[float]:
        """
        验证并限制关节角度

        Args:
            angles: 关节角度列表(弧度)

        Returns:
            限制后的关节角度列表
        """
        validated_angles = []
        for i, angle in enumerate(angles):
            if i < len(self.motors) - 1:  # 排除夹爪
                motor_name = self.motors[i]
                min_limit, max_limit = self.joint_limits[motor_name]
                clamped_angle = np.clip(angle, min_limit, max_limit)

                if abs(clamped_angle - angle) > 0.001:  # 1mrad tolerance
                    logging.warning(f"{motor_name} angle {angle:.4f} clamped to [{min_limit:.4f}, {max_limit:.4f}]")

                validated_angles.append(clamped_angle)
            else:
                validated_angles.append(angle)

        return validated_angles

    def _validate_gripper_position(self, position: float) -> float:
        """
        验证并限制夹爪位置

        Args:
            position: 夹爪位置(米)

        Returns:
            限制后的夹爪位置
        """
        min_limit, max_limit = self.joint_limits["gripper"]
        clamped_pos = np.clip(position, min_limit, max_limit)

        if abs(clamped_pos - position) > 0.0001:  # 0.1mm tolerance
            logging.warning(f"Gripper position {position:.4f} clamped to [{min_limit:.4f}, {max_limit:.4f}]")

        return clamped_pos


# =============================================================================
# 配置类定义
# =============================================================================

@RobotConfig.register_subclass("piper_follower")
@dataclass
class PiperFollowerConfig(RobotConfig):
    """Piper双臂机器人配置"""

    # 双臂CAN接口配置
    can_name_left: str = "can0"        # 左臂CAN接口名称
    can_name_right: str = "can1"       # 右臂CAN接口名称
    baud_rate: int = 1000000          # CAN波特率 1Mbps

    # 安全配置
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | None = None  # 最大相对位移限制

    # V2版本特有配置
    enable_sdk_joint_limit: bool = True      # 启用软件关节限位
    enable_sdk_gripper_limit: bool = True    # 启用软件夹爪限位
    dh_offset: bool = True                   # 启用DH参数偏移
    control_mode: str = "joint"              # 控制模式：joint/cartesian
    speed_rate: int = 50                     # 运动速度百分比

    # 相机配置
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # 兼容性设置
    use_degrees: bool = False                # 向后兼容


@RobotConfig.register_subclass("piper_follower_end_effector")
@dataclass
class PiperFollowerEndEffectorConfig(PiperFollowerConfig):
    """Piper双臂末端执行器配置"""

    # 运动学配置
    urdf_path: str | None = None
    target_frame_name: str = "end_effector"

    # 末端执行器边界 (米)
    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "min": [-0.8, -0.8, 0.0],    # min x, y, z
            "max": [0.8, 0.8, 0.8],      # max x, y, z
        }
    )

    # 夹爪配置
    max_gripper_pos: float = 0.07            # 小夹爪70mm最大行程

    # 末端执行器步长配置
    end_effector_step_sizes: dict[str, float] = field(
        default_factory=lambda: {
            "delta_x": 0.005,            # 5mm步长
            "delta_y": 0.005,
            "delta_z": 0.005,
        }
    )


@TeleoperatorConfig.register_subclass("piper_leader")
@dataclass
class PiperLeaderConfig(TeleoperatorConfig):
    """Piper双臂领导臂配置"""

    # 双臂CAN接口配置
    can_name_left: str = "can2"         # 左臂CAN接口名称
    can_name_right: str = "can3"        # 右臂CAN接口名称
    baud_rate: int = 1000000           # CAN波特率

    # 主从模式配置
    enable_master_slave: bool = True    # 启用主从模式
    linkage_config: int = 0xFA         # 设为示教输入臂

    # V2版本特有配置
    enable_sdk_joint_limit: bool = True
    enable_sdk_gripper_limit: bool = True
    speed_rate: int = 50


# =============================================================================
# 双臂机器人实现
# =============================================================================

class PiperFollower(Robot):
    """Piper双臂跟随机器人实现 - 继承Robot抽象基类"""

    # 必须的类属性
    config_class = PiperFollowerConfig
    name = "piper_follower"

    def __init__(self, config: PiperFollowerConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.config = config

        # 初始化双臂电机总线
        self.bus_left = PiperMotorsBus(
            can_name=config.can_name_left,
            baud_rate=config.baud_rate
        )
        self.bus_right = PiperMotorsBus(
            can_name=config.can_name_right,
            baud_rate=config.baud_rate
        )

        # 初始化相机系统
        self.cameras = make_cameras_from_configs(config.cameras)

        # 运动学模型 (如果有URDF)
        self.kinematics = None
        if hasattr(config, 'urdf_path') and config.urdf_path and Path(config.urdf_path).exists():
            try:
                from lerobot.model.kinematics import RobotKinematics
                self.kinematics = RobotKinematics(
                    urdf_path=config.urdf_path,
                    target_frame_name=getattr(config, 'target_frame_name', 'end_effector')
                )
            except Exception as e:
                logging.warning(f"Failed to load kinematics: {e}")

        logging.info("PiperFollower (dual-arm) initialized")

    # 必须实现的抽象属性
    @property
    def _motors_ft(self) -> dict[str, type]:
        """双臂电机特征字典"""
        left_motors = {f"left_{motor}.pos": float for motor in self.bus_left.motors}
        right_motors = {f"right_{motor}.pos": float for motor in self.bus_right.motors}
        return {**left_motors, **right_motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """相机特征字典"""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """观测特征描述 - Robot抽象属性"""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """动作特征描述 - Robot抽象属性"""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """检查连接状态 - Robot抽象属性"""
        return (self.bus_left.is_connected and self.bus_right.is_connected and
                all(cam.is_connected for cam in self.cameras.values()))

    @property
    def is_calibrated(self) -> bool:
        """检查校准状态 - Robot抽象属性"""
        # Piper 机械臂通常不需要传统意义的校准，检查是否已正确初始化
        return self.is_connected

    # 必须实现的抽象方法
    def connect(self, calibrate: bool = True) -> None:
        """连接到双臂机器人 - Robot抽象方法"""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 连接双臂
        left_success = self.bus_left.connect()
        right_success = self.bus_right.connect()

        if not (left_success and right_success):
            raise ConnectionError(f"Failed to connect to {self}")

        # 连接相机
        for cam in self.cameras.values():
            cam.connect()

        # 如果需要校准
        if not self.is_calibrated and calibrate:
            self.calibrate()

        self.configure()
        logging.info(f"{self} connected.")

    def calibrate(self) -> None:
        """校准机器人 - Robot抽象方法"""
        logging.info(f"Calibrating {self}")
        # Piper 机械臂的校准逻辑：重置到安全位置
        try:
            self.reset()
            logging.info("Piper dual-arm robot calibration completed")
        except Exception as e:
            logging.error(f"Calibration failed: {e}")

    def configure(self) -> None:
        """配置机器人 - Robot抽象方法"""
        logging.info(f"Configuring {self}")
        # Piper 机械臂的配置在 bus.connect() 中的 _initialize_robot() 已完成
        # 这里可以添加额外的双臂协调配置
        pass

    def get_observation(self) -> Dict[str, Any]:
        """获取观测数据 - Robot抽象方法"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        observation = {}

        # 获取双臂关节位置
        start = time.perf_counter()

        # 左臂位置
        left_positions = self.bus_left.sync_read("Present_Position")
        for motor_name in self.bus_left.motors:
            observation[f"left_{motor_name}.pos"] = left_positions.get(motor_name, 0.0)

        # 右臂位置
        right_positions = self.bus_right.sync_read("Present_Position")
        for motor_name in self.bus_right.motors:
            observation[f"right_{motor_name}.pos"] = right_positions.get(motor_name, 0.0)

        dt_ms = (time.perf_counter() - start) * 1e3
        logging.debug(f"{self} read dual-arm state: {dt_ms:.1f}ms")

        # 获取图像
        for camera_name, camera in self.cameras.items():
            start = time.perf_counter()
            image = camera.async_read()
            if image is not None:
                observation[camera_name] = image
            dt_ms = (time.perf_counter() - start) * 1e3
            logging.debug(f"{self} read {camera_name}: {dt_ms:.1f}ms")

        return observation

    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """发送动作指令 - Robot抽象方法"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 分离左臂和右臂的动作
        left_goal_pos = {}
        right_goal_pos = {}

        for key, val in action.items():
            if key.endswith(".pos"):
                motor = key.removesuffix(".pos")
                if motor.startswith("left_"):
                    actual_motor = motor.removeprefix("left_")
                    left_goal_pos[actual_motor] = val
                elif motor.startswith("right_"):
                    actual_motor = motor.removeprefix("right_")
                    right_goal_pos[actual_motor] = val

        # 安全限制 (如果配置了)
        if self.config.max_relative_target is not None:
            # 左臂安全检查
            if left_goal_pos:
                left_present_pos = self.bus_left.sync_read("Present_Position")
                left_goal_present_pos = {key: (g_pos, left_present_pos[key]) for key, g_pos in left_goal_pos.items()}
                left_goal_pos = ensure_safe_goal_position(left_goal_present_pos, self.config.max_relative_target)

            # 右臂安全检查
            if right_goal_pos:
                right_present_pos = self.bus_right.sync_read("Present_Position")
                right_goal_present_pos = {key: (g_pos, right_present_pos[key]) for key, g_pos in right_goal_pos.items()}
                right_goal_pos = ensure_safe_goal_position(right_goal_present_pos, self.config.max_relative_target)

        try:
            # 发送双臂指令
            if left_goal_pos:
                self.bus_left.sync_write("Goal_Position", left_goal_pos)
            if right_goal_pos:
                self.bus_right.sync_write("Goal_Position", right_goal_pos)

            # 返回实际发送的动作
            sent_action = {}
            for motor, val in left_goal_pos.items():
                sent_action[f"left_{motor}.pos"] = val
            for motor, val in right_goal_pos.items():
                sent_action[f"right_{motor}.pos"] = val

            return sent_action

        except Exception as e:
            logging.error(f"Failed to send dual-arm action: {e}")
            return {}

    def disconnect(self) -> None:
        """断开连接 - Robot抽象方法"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 断开双臂连接
        self.bus_left.disconnect()
        self.bus_right.disconnect()

        # 断开相机连接
        for camera in self.cameras.values():
            camera.disconnect()

        logging.info(f"{self} disconnected.")

    def reset(self):
        """重置双臂机器人到初始位置"""
        try:
            # 重置到零位
            zero_positions = {motor: 0.0 for motor in self.bus_left.motors}
            zero_positions["gripper"] = 0.5  # 夹爪半开

            # 双臂同时重置
            self.bus_left.sync_write("Goal_Position", zero_positions)
            self.bus_right.sync_write("Goal_Position", zero_positions)

            # 等待到达位置
            time.sleep(3.0)

            logging.info("Dual-arm robot reset to initial position")

        except Exception as e:
            logging.error(f"Failed to reset dual-arm robot: {e}")


# =============================================================================
# 双臂末端执行器实现
# =============================================================================

class PiperFollowerEndEffector(Robot):
    """Piper双臂末端执行器实现 - 专门控制夹爪"""

    # 必须的类属性
    config_class = PiperFollowerEndEffectorConfig
    name = "piper_follower_end_effector"

    def __init__(self, config: PiperFollowerEndEffectorConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.config = config

        # 初始化双臂电机总线 (只关注夹爪)
        self.bus_left = PiperMotorsBus(
            can_name=config.can_name_left,
            baud_rate=config.baud_rate
        )
        self.bus_right = PiperMotorsBus(
            can_name=config.can_name_right,
            baud_rate=config.baud_rate
        )

        # 初始化相机系统
        self.cameras = make_cameras_from_configs(config.cameras)

        # 运动学模型
        self.kinematics = None
        if config.urdf_path and Path(config.urdf_path).exists():
            try:
                from lerobot.model.kinematics import RobotKinematics
                self.kinematics = RobotKinematics(
                    urdf_path=config.urdf_path,
                    target_frame_name=config.target_frame_name
                )
            except Exception as e:
                logging.warning(f"Failed to load kinematics: {e}")

        # 末端执行器边界
        self.end_effector_bounds = {
            "min": np.array(config.end_effector_bounds["min"]),
            "max": np.array(config.end_effector_bounds["max"])
        }

        logging.info("PiperFollowerEndEffector (dual-arm) initialized")

    # 必须实现的抽象属性
    @property
    def _motors_ft(self) -> dict[str, type]:
        """双臂末端执行器特征字典"""
        return {
            "left_gripper.pos": float,
            "right_gripper.pos": float,
            "delta_x": float,
            "delta_y": float,
            "delta_z": float
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """相机特征字典"""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """观测特征描述"""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """动作特征描述"""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """检查连接状态"""
        return (self.bus_left.is_connected and self.bus_right.is_connected and
                all(cam.is_connected for cam in self.cameras.values()))

    @property
    def is_calibrated(self) -> bool:
        """检查校准状态"""
        return self.is_connected

    # 必须实现的抽象方法
    def connect(self, calibrate: bool = True) -> None:
        """连接到双臂末端执行器"""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 连接双臂
        left_success = self.bus_left.connect()
        right_success = self.bus_right.connect()

        if not (left_success and right_success):
            raise ConnectionError(f"Failed to connect to {self}")

        # 连接相机
        for cam in self.cameras.values():
            cam.connect()

        if not self.is_calibrated and calibrate:
            self.calibrate()

        self.configure()
        logging.info(f"{self} connected.")

    def calibrate(self) -> None:
        """校准末端执行器"""
        logging.info(f"Calibrating {self}")
        try:
            self.reset()
            logging.info("Dual-arm end-effector calibration completed")
        except Exception as e:
            logging.error(f"End-effector calibration failed: {e}")

    def configure(self) -> None:
        """配置末端执行器"""
        logging.info(f"Configuring {self}")
        pass

    def get_observation(self) -> Dict[str, Any]:
        """获取末端执行器观测数据"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        observation = {}

        # 获取夹爪位置
        start = time.perf_counter()
        left_positions = self.bus_left.sync_read("Present_Position")
        right_positions = self.bus_right.sync_read("Present_Position")

        observation["left_gripper.pos"] = left_positions.get("gripper", 0.0)
        observation["right_gripper.pos"] = right_positions.get("gripper", 0.0)

        # 末端执行器位置 (如果有运动学模型)
        if self.kinematics:
            try:
                # 计算当前末端执行器位置作为观测
                left_joint_pos = np.array([left_positions[f"joint_{i}"] for i in range(1, 7)])
                left_ee_pos = self.kinematics.forward_kinematics(left_joint_pos)[:3, 3]
                observation["left_ee_x"] = left_ee_pos[0]
                observation["left_ee_y"] = left_ee_pos[1]
                observation["left_ee_z"] = left_ee_pos[2]

                right_joint_pos = np.array([right_positions[f"joint_{i}"] for i in range(1, 7)])
                right_ee_pos = self.kinematics.forward_kinematics(right_joint_pos)[:3, 3]
                observation["right_ee_x"] = right_ee_pos[0]
                observation["right_ee_y"] = right_ee_pos[1]
                observation["right_ee_z"] = right_ee_pos[2]
            except Exception as e:
                logging.warning(f"Failed to compute forward kinematics: {e}")

        dt_ms = (time.perf_counter() - start) * 1e3
        logging.debug(f"{self} read end-effector state: {dt_ms:.1f}ms")

        # 获取图像
        for camera_name, camera in self.cameras.items():
            start = time.perf_counter()
            image = camera.async_read()
            if image is not None:
                observation[camera_name] = image
            dt_ms = (time.perf_counter() - start) * 1e3
            logging.debug(f"{self} read {camera_name}: {dt_ms:.1f}ms")

        return observation

    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """发送末端执行器动作指令"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            # 处理夹爪动作
            if "left_gripper.pos" in action:
                self.bus_left.sync_write("Goal_Position", {"gripper": action["left_gripper.pos"]})
            if "right_gripper.pos" in action:
                self.bus_right.sync_write("Goal_Position", {"gripper": action["right_gripper.pos"]})

            # 处理末端执行器增量动作
            if any(key in action for key in ["delta_x", "delta_y", "delta_z"]):
                self._send_ee_delta_action(action)

            return action

        except Exception as e:
            logging.error(f"Failed to send end-effector action: {e}")
            return {}

    def disconnect(self) -> None:
        """断开末端执行器连接"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus_left.disconnect()
        self.bus_right.disconnect()

        for camera in self.cameras.values():
            camera.disconnect()

        logging.info(f"{self} disconnected.")

    def reset(self):
        """重置末端执行器到初始位置"""
        try:
            # 夹爪半开
            gripper_positions = {"gripper": 0.5}

            self.bus_left.sync_write("Goal_Position", gripper_positions)
            self.bus_right.sync_write("Goal_Position", gripper_positions)

            time.sleep(2.0)
            logging.info("Dual-arm end-effector reset completed")

        except Exception as e:
            logging.error(f"Failed to reset end-effector: {e}")

    def _send_ee_delta_action(self, action: Dict[str, float]):
        """发送末端执行器增量动作"""
        if not self.kinematics:
            logging.warning("No kinematics model available for end-effector control")
            return

        # 获取当前关节位置
        left_current = self.bus_left.sync_read("Present_Position")
        right_current = self.bus_right.sync_read("Present_Position")

        # 计算当前末端执行器位置
        left_joint_pos = np.array([left_current[f"joint_{i}"] for i in range(1, 7)])
        right_joint_pos = np.array([right_current[f"joint_{i}"] for i in range(1, 7)])

        left_ee_pos = self.kinematics.forward_kinematics(left_joint_pos)[:3, 3]
        right_ee_pos = self.kinematics.forward_kinematics(right_joint_pos)[:3, 3]

        # 应用增量
        delta_x = action.get("delta_x", 0.0)
        delta_y = action.get("delta_y", 0.0)
        delta_z = action.get("delta_z", 0.0)

        left_target_ee_pos = left_ee_pos + np.array([delta_x, delta_y, delta_z])
        right_target_ee_pos = right_ee_pos + np.array([delta_x, delta_y, delta_z])

        # 边界限制
        left_target_ee_pos = np.clip(left_target_ee_pos,
                                   self.end_effector_bounds["min"],
                                   self.end_effector_bounds["max"])
        right_target_ee_pos = np.clip(right_target_ee_pos,
                                    self.end_effector_bounds["min"],
                                    self.end_effector_bounds["max"])

        # 逆运动学解算
        try:
            left_target_joint_pos = self.kinematics.inverse_kinematics(left_target_ee_pos)
            right_target_joint_pos = self.kinematics.inverse_kinematics(right_target_ee_pos)

            if left_target_joint_pos is not None:
                left_goal_positions = {f"joint_{i}": left_target_joint_pos[i-1] for i in range(1, 7)}
                self.bus_left.sync_write("Goal_Position", left_goal_positions)

            if right_target_joint_pos is not None:
                right_goal_positions = {f"joint_{i}": right_target_joint_pos[i-1] for i in range(1, 7)}
                self.bus_right.sync_write("Goal_Position", right_goal_positions)

        except Exception as e:
            logging.error(f"Inverse kinematics failed: {e}")


# =============================================================================
# 双臂领导机器人实现
# =============================================================================

class PiperLeader(Teleoperator):
    """Piper双臂领导机器人实现 - 继承Teleoperator抽象基类"""

    # 必须的类属性
    config_class = PiperLeaderConfig
    name = "piper_leader"

    def __init__(self, config: PiperLeaderConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.config = config

        # 初始化双臂电机总线
        self.bus_left = PiperMotorsBus(
            can_name=config.can_name_left,
            baud_rate=config.baud_rate
        )
        self.bus_right = PiperMotorsBus(
            can_name=config.can_name_right,
            baud_rate=config.baud_rate
        )

        logging.info("PiperLeader (dual-arm) initialized")

    # 必须实现的抽象属性
    @property
    def action_features(self) -> dict[str, type]:
        """动作特征描述 - Teleoperator抽象属性"""
        left_motors = {f"left_{motor}.pos": float for motor in self.bus_left.motors}
        right_motors = {f"right_{motor}.pos": float for motor in self.bus_right.motors}
        return {**left_motors, **right_motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        """反馈特征描述 - Teleoperator抽象属性"""
        return {}  # Piper暂不支持力反馈

    @property
    def is_connected(self) -> bool:
        """检查连接状态 - Teleoperator抽象属性"""
        return self.bus_left.is_connected and self.bus_right.is_connected

    @property
    def is_calibrated(self) -> bool:
        """检查校准状态"""
        return self.is_connected

    # 必须实现的抽象方法
    def connect(self, calibrate: bool = True) -> None:
        """连接到双臂领导机器人 - Teleoperator抽象方法"""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 连接双臂
        left_success = self.bus_left.connect()
        right_success = self.bus_right.connect()

        if not (left_success and right_success):
            raise ConnectionError(f"Failed to connect to {self}")

        if not self.is_calibrated and calibrate:
            self.calibrate()

        self.configure()
        logging.info(f"{self} connected.")

    def calibrate(self) -> None:
        """校准领导机器人"""
        logging.info(f"Calibrating {self}")
        try:
            # 设置为主从模式
            if self.config.enable_master_slave:
                self.bus_left.interface.MasterSlaveConfig(
                    linkage_config=self.config.linkage_config,
                    feedback_offset=0x00,
                    ctrl_offset=0x00,
                    linkage_offset=0x00
                )
                self.bus_right.interface.MasterSlaveConfig(
                    linkage_config=self.config.linkage_config,
                    feedback_offset=0x00,
                    ctrl_offset=0x00,
                    linkage_offset=0x00
                )
            logging.info("Dual-arm leader calibration completed")
        except Exception as e:
            logging.error(f"Leader calibration failed: {e}")

    def configure(self) -> None:
        """配置领导机器人"""
        logging.info(f"Configuring {self}")
        pass

    def get_action(self) -> Dict[str, float]:
        """获取遥操控动作 - Teleoperator抽象方法"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            start = time.perf_counter()

            # 读取双臂位置作为动作
            left_action = self.bus_left.sync_read("Present_Position")
            right_action = self.bus_right.sync_read("Present_Position")

            # 格式化动作
            action = {}
            for motor_name in self.bus_left.motors:
                action[f"left_{motor_name}.pos"] = left_action.get(motor_name, 0.0)
            for motor_name in self.bus_right.motors:
                action[f"right_{motor_name}.pos"] = right_action.get(motor_name, 0.0)

            dt_ms = (time.perf_counter() - start) * 1e3
            logging.debug(f"{self} read dual-arm action: {dt_ms:.1f}ms")

            return action

        except Exception as e:
            logging.error(f"Failed to get leader action: {e}")
            return {f"left_{motor}.pos": 0.0 for motor in self.bus_left.motors} | \
                   {f"right_{motor}.pos": 0.0 for motor in self.bus_right.motors}

    def send_feedback(self, feedback: Dict[str, float]) -> None:
        """发送反馈 - Teleoperator抽象方法"""
        # Piper机械臂暂不支持力反馈
        raise NotImplementedError("Piper does not support force feedback yet")

    def disconnect(self) -> None:
        """断开连接 - Teleoperator抽象方法"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus_left.disconnect()
        self.bus_right.disconnect()

        logging.info(f"{self} disconnected.")


# =============================================================================
# 工厂函数
# =============================================================================

def make_piper_follower(config_path: Optional[str] = None, **config_kwargs) -> PiperFollower:
    """
    创建Piper双臂跟随机器人实例

    Args:
        config_path: 配置文件路径
        **config_kwargs: 额外配置参数

    Returns:
        PiperFollower实例
    """
    if config_path:
        import json
        with open(config_path, 'r') as f:
            file_config = json.load(f)
        config = PiperFollowerConfig(**{**file_config, **config_kwargs})
    else:
        config = PiperFollowerConfig(**config_kwargs)

    return PiperFollower(config)


def make_piper_follower_end_effector(config_path: Optional[str] = None, **config_kwargs) -> PiperFollowerEndEffector:
    """
    创建Piper双臂末端执行器实例

    Args:
        config_path: 配置文件路径
        **config_kwargs: 额外配置参数

    Returns:
        PiperFollowerEndEffector实例
    """
    if config_path:
        import json
        with open(config_path, 'r') as f:
            file_config = json.load(f)
        config = PiperFollowerEndEffectorConfig(**{**file_config, **config_kwargs})
    else:
        config = PiperFollowerEndEffectorConfig(**config_kwargs)

    return PiperFollowerEndEffector(config)


def make_piper_leader(config_path: Optional[str] = None, **config_kwargs) -> PiperLeader:
    """
    创建Piper双臂领导机器人实例

    Args:
        config_path: 配置文件路径
        **config_kwargs: 额外配置参数

    Returns:
        PiperLeader实例
    """
    if config_path:
        import json
        with open(config_path, 'r') as f:
            file_config = json.load(f)
        config = PiperLeaderConfig(**{**file_config, **config_kwargs})
    else:
        config = PiperLeaderConfig(**config_kwargs)

    return PiperLeader(config)


# =============================================================================
# 框架注册 (可选)
# =============================================================================

def register_piper_components():
    """向LeRobot框架注册Piper组件"""
    try:
        from lerobot.robots import ROBOT_CONFIG_CLASSES, ROBOT_CLASSES
        from lerobot.teleoperators import TELEOP_CONFIG_CLASSES, TELEOP_CLASSES

        # 注册双臂机器人
        ROBOT_CONFIG_CLASSES["piper_follower"] = PiperFollowerConfig
        ROBOT_CLASSES["piper_follower"] = PiperFollower

        # 注册双臂末端执行器
        ROBOT_CONFIG_CLASSES["piper_follower_end_effector"] = PiperFollowerEndEffectorConfig
        ROBOT_CLASSES["piper_follower_end_effector"] = PiperFollowerEndEffector

        # 注册双臂遥操控
        TELEOP_CONFIG_CLASSES["piper_leader"] = PiperLeaderConfig
        TELEOP_CLASSES["piper_leader"] = PiperLeader

        logging.info("Piper dual-arm components registered successfully")

    except ImportError:
        logging.warning("Could not register Piper components - framework classes not found")


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 创建双臂机器人配置
    config = PiperFollowerConfig(
        can_name_left="can0",
        can_name_right="can1",
        cameras={
            "top": CameraConfig(device_id=0, fps=30, width=640, height=480)
        }
    )

    # 创建双臂机器人实例
    robot = PiperFollower(config)

    print("Piper dual-arm robot created successfully!")
    print(f"Robot type: {config.robot_type}")
    print(f"Left arm motors: {robot.bus_left.motors}")
    print(f"Right arm motors: {robot.bus_right.motors}")

    # 测试连接 (需要实际硬件)
    # if robot.connect():
    #     print("Dual-arm robot connected successfully!")
    #     obs = robot.get_observation()
    #     print(f"Observation keys: {list(obs.keys())}")
    #     robot.disconnect()
