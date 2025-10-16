#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logging.getLogger("can.interfaces.socketcan").setLevel(logging.ERROR)

import math
from dataclasses import dataclass

import draccus
from lerobot.errors import DeviceNotConnectedError

# 关键：导入以触发“注册”（不要删除）
from lerobot.robots import (  # noqa: F401
    piper_follower,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.teleoperators import (  # noqa: F401
    keyboard,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.utils.robot_utils import busy_wait


@dataclass
class KeyboardEEDemoConfig:
    # 必填：从命令行传入类型
    teleop: TeleoperatorConfig   # --teleop.type=keyboard_ee
    robot: RobotConfig           # --robot.type=piper_follower_end_effector
    rate_hz: float = 50.0        # 发送频率
    print_every: int = 20        # 每多少帧打印一次动作


@draccus.wrap()
def main(cfg: KeyboardEEDemoConfig):
    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    teleop.connect()
    robot.connect()

    print("\n[INFO] Keyboard EE teleop started.")
    print("      左臂：方向键←→ x，↑↓ y，Shift/-z，右Shift/+z，左Ctrl=闭合(0)，右Ctrl=张开(2)")
    print("      右臂：A/D x，W/S y，Z/-z，X/+z，N=闭合(0)，M=张开(2)")
    print("      按 ESC 退出。\n")

    dt = 1.0 / cfg.rate_hz
    step = 0
    try:
        while True:
            # try:
            action = teleop.get_action()   # 会在 ESC 后抛 DeviceNotConnectedError
            # except DeviceNotConnectedError:
            #     print("[INFO] Teleop disconnected (ESC). Exiting loop.")
            #     break

            # 动作直接下发给 EE 机器人
            robot.send_action(action)

            if step % cfg.print_every == 0:
                # 简要打印，便于确认按键是否生效
                print(f"[tick={step}] action={action}")
            step += 1

            busy_wait(dt)
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt, stopping.")
    finally:
        try:
            teleop.disconnect()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        print("[INFO] Clean exit.\n")


if __name__ == "__main__":
    main()

"""
用法示例（推荐 -m 方式）：
python -m lerobot.scripts.test_piper_keyboard_ee \
  --teleop.type=keyboard_ee \
  --robot.type=piper_follower_end_effector \
  --rate_hz=50
"""
