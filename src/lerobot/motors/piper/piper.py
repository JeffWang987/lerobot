import logging
import numpy as np
from typing import Dict, Literal
from traceback import format_exc
# Piper SDK导入 - 使用V2版本接口
from piper_sdk import C_PiperInterface_V2


class PiperMotorsBus:
    """
    Piper机械臂电机总线独立实现 - 使用V2接口

    提供类似MotorsBus的接口，但直接使用Piper SDK V2的CAN总线通信
    适配固件版本 V1.5-2 及以上（V1.6-5完全支持）
    """

    def __init__(
        self, can_name: str,
        baud_rate: int = 1000000,
        motor_prefix: Literal['left', 'right'] = 'left'
        ):
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
        self.motors = [f"{motor_prefix}_{motor}" for motor in self.motors]

        # 关节限位 (弧度) - 基于Piper官方文档
        self.joint_limits = {
            "joint_1": (-2.6179, 2.6179),      # [-150°, 150°]
            "joint_2": (0, 3.14),              # [0°, 180°]
            "joint_3": (-2.967, 0),            # [-170°, 0°]
            "joint_4": (-1.745, 1.745),        # [-100°, 100°]
            "joint_5": (-1.22, 1.22),          # [-70°, 70°]
            "joint_6": (-2.09439, 2.09439),    # [-120°, 120°]
            "gripper": (0.0, 0.07)              # 夹爪开合度 [0-70mm]
        }

        # 复位位置
        if 'left' in self.can_name:
            self.arm_reset_rad_pos = [
                -0.00133514404296875, 0.00209808349609375, 0.01583099365234375,
                -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 0.06
            ]
        elif 'right' in self.can_name:
            self.arm_reset_rad_pos = [
                -0.00133514404296875, 0.00438690185546875, 0.034523963928222656,
                -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 0.06
            ]

        # TODO: 断开位置
        # 当前状态缓存
        self._current_positions = {motor: 0.0 for motor in self.motors}
        self._current_velocities = {motor: 0.0 for motor in self.motors}
        self._current_currents = {motor: 0.0 for motor in self.motors}

        logging.info(f"Initialized PiperMotorsBus V2 with {len(self.motors)} motors")
        logging.info(f"CAN interface: {can_name}, Baud rate: {baud_rate}")

    def connect(
        self,
        start_sdk_joint_limit: bool = True,
        start_sdk_gripper_limit: bool = True,
        move_spd_rate_ctrl: int = 50
    ) -> bool:
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
                dh_is_offset=0x01,  # 启用DH参数偏移
                start_sdk_joint_limit=start_sdk_joint_limit,  # 启用软件关节限位
                start_sdk_gripper_limit=start_sdk_gripper_limit,  # 启用软件夹爪限位
                logger_level=logging.DEBUG,  # 日志配置
                log_to_file=False,
                log_file_path=None
            )

            # 连接CAN端口
            result = self.interface.ConnectPort(
                can_init=True,
                piper_init=True,
                start_thread=True
            )

            if self.interface.get_connect_status():
                self.is_connected = True
                logging.info(f"Successfully connected to Piper robot on {self.can_name}")
                self._initialize_robot(move_spd_rate_ctrl=move_spd_rate_ctrl)
                return True
            else:
                logging.error(f"Failed to connect to Piper robot on {self.can_name}")
                return False

        except Exception as e:
            logging.error(f"Exception during Piper connection: {format_exc()}")
            return False

    def disconnect(self, reset_pos: bool = False, disable_torque: bool = True):
        """断开连接"""
        if self.interface and self.is_connected:
            try:
                if reset_pos:
                    self.reset_pos()
                if disable_torque:
                    self.sync_write("Torque_Enable", 0)
                self.interface.DisconnectPort()
                self.is_connected = False
                logging.info("Disconnected from Piper robot")
            except Exception as e:

                logging.error(f"Error disconnecting from Piper robot: {format_exc()}")

    def _initialize_robot(self, move_spd_rate_ctrl: int = 50):
        """初始化机器人状态"""
        try:
            # 使能机械臂 (V2版本方法)
            self.interface.EnableArm(
                motor_num=7,       # 7表示所有电机
                enable_flag=0x02   # 0x02表示使能
            )

            # 设置控制模式为CAN命令控制
            self.interface.ModeCtrl(
                ctrl_mode=0x01,     # CAN命令控制模式
                move_mode=0x01,     # MOVE J (关节)模式
                                    # 0x00: MOVE P (Position)
                                    # 0x01: MOVE J (Joint)
                                    # 0x02: MOVE L (Linear)
                                    # 0x03: MOVE C (Circular)
                move_spd_rate_ctrl=move_spd_rate_ctrl,  # 50%速度
                is_mit_mode=0x00    # 位置-速度模式
            )

            # 初始化夹爪参数 (V2版本特有功能)
            self.interface.GripperTeachingPendantParamConfig(
                teaching_range_per=100,     # 100%行程系数
                max_range_config=70,        # 小夹爪70mm最大行程
                teaching_friction=1         # 示教摩擦力
            )

            # 使能夹爪
            self.interface.GripperCtrl(
                gripper_angle=0,
                gripper_effort=1000,
                gripper_code=0x01,      # 使能
                set_zero=0x00
            )

            # 复位
            self.reset_pos()

            # 读取初始位置
            self._update_positions()

            logging.info("Piper robot V2 initialized successfully")

        except Exception as e:
            logging.error(f"Failed to initialize Piper robot: {format_exc()}")

    def reset_pos(self):
        if self.arm_reset_rad_pos:
            self.sync_write("Goal_Position", {
                    motor: pos for motor, pos in zip(self.motors, self.arm_reset_rad_pos)
                })
            logging.info(f"Robot {self.can_name} reset position")
        else:
            logging.warning(f"No reset position defined for this arm {self.can_name}")

    def _update_positions(self):
        """更新关节位置"""
        try:
            # 获取关节角度信息 (返回0.001度单位)
            joint_msgs = self.interface.GetArmJointMsgs()
            if joint_msgs is not None:
                # 转换为弧度并更新前6个关节
                joint_values = [
                    joint_msgs.joint_state.joint_1,
                    joint_msgs.joint_state.joint_2,
                    joint_msgs.joint_state.joint_3,
                    joint_msgs.joint_state.joint_4,
                    joint_msgs.joint_state.joint_5,
                    joint_msgs.joint_state.joint_6,
                ]

                for i, joint_value in enumerate(joint_values):
                    # 从0.001度转换为弧度
                    angle_deg = joint_value / 1000.0
                    angle_rad = np.deg2rad(angle_deg)
                    self._current_positions[self.motors[i]] = angle_rad

            # 获取夹爪信息
            gripper_msgs = self.interface.GetArmGripperMsgs()
            if gripper_msgs is not None:
                # 夹爪位置单位为0.001mm，转换为米
                gripper_pos_m = gripper_msgs.grippers_angle / 1000000.0  # 0.001mm -> m
                gripper_effort = gripper_msgs.grippers_effort  # 夹爪力矩 (0.1N·m单位)
                gripper_status_code = gripper_msgs.status_code  # 夹爪状态码
                self._current_positions["gripper"] = gripper_pos_m

        except Exception as e:
            logging.error(f"Failed to update positions: {format_exc()}")

    def sync_read(self, parameter: str) -> Dict[str, float]:
        """
        同步读取参数

        Args:
            parameter: 参数名称 ("Present_Position")

        Returns:
            电机名称到值的字典
            {
                "joint_1": ...,
                "joint_2": ...,
                ...
                "gripper": ...
            }
        """
        try:
            if parameter == "Present_Position":
                self._update_positions()
                return self._current_positions.copy()

            else:
                logging.warning(f"Unsupported parameter: {parameter}")
                return {}

        except Exception as e:
            logging.error(f"Failed to sync_read {parameter}: {format_exc()}")
            return {}

    def sync_write(self, parameter: str, values: Dict[str, float] | float):
        """
        同步写入参数

        Args:
            parameter: 参数名称 ("Goal_Position", "Torque_Enable")
            values: 值字典或单个值
            e.g.
            parameter="Goal_Position"
            values = {
                "joint_1": 0.5, # in rad
                "joint_2": 0.5,
                ...
                "gripper": 0.1 # in meters, optional
            }

        """
        try:
            if parameter == "Goal_Position":
                if isinstance(values, dict):
                    # 分离关节角度和夹爪
                    joint_angles = []
                    has_gripper_cmd = False
                    gripper_pos = 0

                    # 提取关节角度（弧度转换为0.001度）
                    for motor in self.motors[:-1]:  # 除了夹爪
                        if motor in values:
                            angle_rad = np.clip(values[motor],
                                              self.joint_limits[motor][0],
                                              self.joint_limits[motor][1])
                            # 弧度转换为0.001度
                            angle_millideg = int(np.rad2deg(angle_rad) * 1000)
                            joint_angles.append(angle_millideg)
                        else:
                            # 使用当前位置
                            current_rad = self._current_positions[motor]
                            angle_millideg = int(np.rad2deg(current_rad) * 1000)
                            joint_angles.append(angle_millideg)

                    # 发送关节指令 (V2版本方法)
                    self.interface.JointCtrl(
                        joint_1=joint_angles[0],
                        joint_2=joint_angles[1],
                        joint_3=joint_angles[2],
                        joint_4=joint_angles[3],
                        joint_5=joint_angles[4],
                        joint_6=joint_angles[5]
                    )

                    # 处理夹爪
                    if "gripper" in values:
                        gripper_pos_m = np.clip(values["gripper"],
                                              self.joint_limits["gripper"][0],
                                              self.joint_limits["gripper"][1])
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
                    self.interface.EnableArm(
                        motor_num=7,       # 7表示所有电机
                        enable_flag=0x02   # 0x02表示使能
                    )
                else:
                    self.interface.DisableArm(
                        motor_num=7,       # 7表示所有电机
                        enable_flag=0x01   # 0x01表示失能
                    )

            else:
                logging.warning(f"Unsupported write parameter: {parameter}")

        except Exception as e:
            logging.error(f"Failed to sync_write {parameter}: {format_exc()}")

    def write(self, parameter: str, motor: str, value: float):
        """
        写入单个电机参数

        Args:
            parameter: 参数名称
            motor: 电机名称
            value: 值
        """
        self.sync_write(parameter, {motor: value})
