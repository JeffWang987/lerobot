#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights...
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
    Designed to be used with PiperFollowerEndEffector robot.
    """

    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()

    # === 修改开始：action_features 输出 Piper EE 期望的 8 个字段 ===
    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (8,),
            "names": {
                "left_delta_x": 0, "left_delta_y": 1, "left_delta_z": 2, "left_gripper": 3,
                "right_delta_x": 4, "right_delta_y": 5, "right_delta_z": 6, "right_gripper": 7,
            },
        }
    # === 修改结束 ===

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

    # === 修改开始：将按键映射到左右臂的 EE 增量与夹爪命令 ===
    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()

        # 默认：不动（Δ=0），夹爪保持（=1.0）
        l_dx = l_dy = l_dz = 0.0
        r_dx = r_dy = r_dz = 0.0
        l_gr = 1.0
        r_gr = 1.0

        for key, pressed in self.current_pressed.items():
            if not pressed:
                continue

            # 左臂 — 方向键 & Shift/Ctrl
            if key == keyboard.Key.left:
                l_dx = +1.0
            elif key == keyboard.Key.right:
                l_dx = -1.0
            elif key == keyboard.Key.up:
                l_dy = -1.0
            elif key == keyboard.Key.down:
                l_dy = +1.0
            elif key == keyboard.Key.shift:      # 向下
                l_dz = -1.0
            elif key == keyboard.Key.shift_r:    # 向上
                l_dz = +1.0
            elif key == keyboard.Key.ctrl_l:     # 夹爪收拢
                l_gr = 0.0
            elif key == keyboard.Key.ctrl_r:     # 夹爪张开
                l_gr = 2.0

            # 右臂 — WASD / ZX / N,M
            elif key in ("a", "A"):
                r_dx = +1.0
            elif key in ("d", "D"):
                r_dx = -1.0
            elif key in ("w", "W"):
                r_dy = -1.0
            elif key in ("s", "S"):
                r_dy = +1.0
            elif key in ("z", "Z"):
                r_dz = -1.0
            elif key in ("x", "X"):
                r_dz = +1.0
            elif key in ("n", "N"):              # 夹爪收拢
                r_gr = 0.0
            elif key in ("m", "M"):              # 夹爪张开
                r_gr = 2.0
            else:
                # 记录其它键（可用于打标签/事件）
                self.misc_keys_queue.put(key)

        # 清空已处理的键表，避免“粘连”
        self.current_pressed.clear()

        # 输出 Piper EE 期望的动作字典
        return {
            "left_delta_x": l_dx, "left_delta_y": l_dy, "left_delta_z": l_dz, "left_gripper": l_gr,
            "right_delta_x": r_dx, "right_delta_y": r_dy, "right_delta_z": r_dz, "right_gripper": r_gr,
        }
    # === 修改结束 ===
