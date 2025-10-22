#!/usr/bin/env python

# Copyright 2024 ...
import logging
import os
import sys
import time
from queue import Queue
from typing import Any

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .configuration_keyboard import KeyboardEndEffectorTeleopConfig, KeyboardTeleopConfig

PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")
    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info(f"Could not import pynput: {e}")


class KeyboardTeleop(Teleoperator):
    """
    Teleop class to use keyboard inputs for control.
    """

    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self.logs = {}

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        pass

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "Keyboard is already connected. Do not run `robot.connect()` twice."
            )

        if PYNPUT_AVAILABLE:
            logging.info("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
        else:
            logging.info("pynput not available - skipping local keyboard listener.")
            self.listener = None

    def calibrate(self) -> None:
        pass

    def _on_press(self, key):
        if hasattr(key, "char"):
            self.event_queue.put((key.char, True))
        else:
            self.event_queue.put((key, True))

    def _on_release(self, key):
        if hasattr(key, "char"):
            self.event_queue.put((key.char, False))
        else:
            self.event_queue.put((key, False))
        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_char] = is_pressed

    def configure(self):
        pass

    def get_action(self) -> dict[str, Any]:
        before_read_t = time.perf_counter()

        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()

        # Generate action based on current key states
        action = {key for key, val in self.current_pressed.items() if val}
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return dict.fromkeys(action, None)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `robot.connect()` before `disconnect()`."
            )
        if self.listener is not None:
            self.listener.stop()


class KeyboardEndEffectorTeleop(KeyboardTeleop):
    """
    Teleop class to use keyboard inputs for end effector control.
    Designed to be used with PiperFollowerEndEffector robot (6-DoF per arm).
    """

    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()

    # === 升级：输出 Piper EE 期望的 14 个字段（双臂 6-DoF + 夹爪） ===
    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (14,),
            "names": {
                # 左臂
                "left_delta_x": 0, "left_delta_y": 1, "left_delta_z": 2,
                "left_delta_rx": 3, "left_delta_ry": 4, "left_delta_rz": 5,
                "left_gripper": 6,
                # 右臂
                "right_delta_x": 7, "right_delta_y": 8, "right_delta_z": 9,
                "right_delta_rx": 10, "right_delta_ry": 11, "right_delta_rz": 12,
                "right_gripper": 13,
            },
        }

    def _on_press(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, True))

    def _on_release(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, False))
        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _is_pressed(self, k: Any) -> bool:
        return self.current_pressed.get(k, False) is True

    # === 键位说明 ===
    # 左臂（平移）：←/→: ±x，↑/↓: ∓y，Left-Shift: z-，Right-Shift: z+
    # 左臂（姿态）：R/Y: ±roll，T/G: ±pitch，F/H: ±yaw
    # 左臂（夹爪）：Ctrl-L 收拢，Ctrl-R 张开；默认 1.0 保持
    #
    # 右臂（平移）：A/D: ±x，W/S: ∓y，Z/X: ∓/+ z
    # 右臂（姿态）：U/O: ±roll，I/K: ±pitch，J/L: ±yaw
    # 右臂（夹爪）：N 收拢，M 张开；默认 1.0 保持
    #
    # 说明：实际步长由机器人端 config 的 end_effector_step_sizes / end_effector_rot_step_sizes 决定。

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()

        # 默认：不动（Δ=0），夹爪保持（=1.0）
        l_dx = l_dy = l_dz = 0.0
        l_rx = l_ry = l_rz = 0.0
        l_gr = 1.0

        r_dx = r_dy = r_dz = 0.0
        r_rx = r_ry = r_rz = 0.0
        r_gr = 1.0

        # ---- 左臂 平移（箭头 + Shift 键）----
        if self._is_pressed(keyboard.Key.left):
            l_dx = +1.0
        if self._is_pressed(keyboard.Key.right):
            l_dx = -1.0
        if self._is_pressed(keyboard.Key.up):
            l_dy = -1.0
        if self._is_pressed(keyboard.Key.down):
            l_dy = +1.0
        if self._is_pressed(keyboard.Key.shift):
            l_dz = -1.0
        if self._is_pressed(keyboard.Key.shift_r):
            l_dz = +1.0

        # ---- 左臂 姿态（R/Y: roll，T/G: pitch，F/H: yaw）----
        for k in ("r", "R"):
            if self._is_pressed(k):
                l_rx = -1.0
        for k in ("y", "Y"):
            if self._is_pressed(k):
                l_rx = +1.0
        for k in ("t", "T"):
            if self._is_pressed(k):
                l_ry = +1.0
        for k in ("g", "G"):
            if self._is_pressed(k):
                l_ry = -1.0
        for k in ("f", "F"):
            if self._is_pressed(k):
                l_rz = -1.0
        for k in ("h", "H"):
            if self._is_pressed(k):
                l_rz = +1.0

        # ---- 左臂 夹爪 ----
        if self._is_pressed(keyboard.Key.ctrl_l):
            l_gr = 0.0  # 收拢
        if self._is_pressed(keyboard.Key.ctrl_r):
            l_gr = 2.0  # 张开

        # ---- 右臂 平移（WASD + ZX）----
        for k in ("a", "A"):
            if self._is_pressed(k):
                r_dx = +1.0
        for k in ("d", "D"):
            if self._is_pressed(k):
                r_dx = -1.0
        for k in ("w", "W"):
            if self._is_pressed(k):
                r_dy = -1.0
        for k in ("s", "S"):
            if self._is_pressed(k):
                r_dy = +1.0
        for k in ("z", "Z"):
            if self._is_pressed(k):
                r_dz = -1.0
        for k in ("x", "X"):
            if self._is_pressed(k):
                r_dz = +1.0

        # ---- 右臂 姿态（U/O: roll，I/K: pitch，J/L: yaw）----
        for k in ("u", "U"):
            if self._is_pressed(k):
                r_rx = -1.0
        for k in ("o", "O"):
            if self._is_pressed(k):
                r_rx = +1.0
        for k in ("i", "I"):
            if self._is_pressed(k):
                r_ry = +1.0
        for k in ("k", "K"):
            if self._is_pressed(k):
                r_ry = -1.0
        for k in ("j", "J"):
            if self._is_pressed(k):
                r_rz = -1.0
        for k in ("l", "L"):
            if self._is_pressed(k):
                r_rz = +1.0

        # ---- 右臂 夹爪 ----
        for k in ("n", "N"):
            if self._is_pressed(k):
                r_gr = 0.0
        for k in ("m", "M"):
            if self._is_pressed(k):
                r_gr = 2.0

        # 记录其它键（可用于打标签/事件）
        for key, pressed in list(self.current_pressed.items()):
            if pressed:
                known = {
                    keyboard.Key.left, keyboard.Key.right, keyboard.Key.up, keyboard.Key.down,
                    keyboard.Key.shift, keyboard.Key.shift_r, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r,
                    "r","R","y","Y","t","T","g","G","f","F","h","H",
                    "a","A","d","D","w","W","s","S","z","Z","x","X",
                    "u","U","o","O","i","I","k","K","j","J","l","L",
                    "n","N","m","M"
                }
                if key not in known:
                    self.misc_keys_queue.put(key)

        # 清空已处理的键表（避免持续粘连；需要长按时，pynput 会继续上报）
        self.current_pressed.clear()

        # 输出 Piper EE 期望的动作字典（与机器人端 14 维一一对应）
        return {
            # 左臂：平移 + 姿态 + 夹爪
            "left_delta_x": l_dx, "left_delta_y": l_dy, "left_delta_z": l_dz,
            "left_delta_rx": l_rx, "left_delta_ry": l_ry, "left_delta_rz": l_rz,
            "left_gripper": l_gr,
            # 右臂：平移 + 姿态 + 夹爪
            "right_delta_x": r_dx, "right_delta_y": r_dy, "right_delta_z": r_dz,
            "right_delta_rx": r_rx, "right_delta_ry": r_ry, "right_delta_rz": r_rz,
            "right_gripper": r_gr,
        }
