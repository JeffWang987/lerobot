#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logging.getLogger("can.interfaces.socketcan").setLevel(logging.ERROR)

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np
import draccus
from lerobot.utils.errors import DeviceNotConnectedError

# 关键：导入以触发“注册”（不要删除）
# - 你的 piper_single_follower 是在 robots/piper_follower 包里注册的
#   所以导入这个包能同时触发 piper_follower、piper_single_follower 的注册
from lerobot.robots import (  # noqa: F401
    piper_follower,
    RobotConfig,
    make_robot_from_config,
)

# 相机注册（按需）
from lerobot.cameras import (  # noqa: F401
    opencv,
    realsense,
    dabai,
)

from lerobot.utils.robot_utils import busy_wait


@dataclass
class PiperSingleObservationTestConfig:
    # 必填：从命令行传入类型（注意这里要用 piper_single_follower）
    robot: RobotConfig   # --robot.type=piper_single_follower
    rate_hz: float = 10.0
    print_every: int = 10
    duration_s: float = 10.0
    save_images: bool = True
    image_output_dir: str = "outputs/camera_images"


BANNER = r"""
[INFO] Piper Single-Arm Observation Test started.

将周期性读取并打印单臂机器人的观测数据（关节位置、夹爪/相机图像等）。
按 Ctrl+C 退出测试。
"""


@draccus.wrap()
def main(cfg: PiperSingleObservationTestConfig):
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    print(BANNER)

    out_dir = Path(cfg.image_output_dir)
    if cfg.save_images:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 图像将保存到: {out_dir.absolute()}")

    dt = 1.0 / max(1.0, float(cfg.rate_hz))
    step = 0
    t0 = time.time()

    try:
        while time.time() - t0 < cfg.duration_s:
            try:
                obs = robot.get_observation()
            except DeviceNotConnectedError:
                print("[ERROR] Robot disconnected. Exiting loop.")
                break

            # 保存相机图像（如果开启）
            if cfg.save_images:
                for key, val in obs.items():
                    if isinstance(val, np.ndarray) and val.ndim == 3 and val.shape[-1] in (1, 3, 4):
                        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                        img_path = out_dir / f"{ts}_{key}_{step:05d}.png"
                        arr = val
                        # 常见：RGB → BGR
                        if arr.shape[-1] == 3:
                            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(img_path), arr)

            # 打印
            if cfg.print_every > 0 and (step % cfg.print_every == 0):
                print(f"[tick={step}, t={time.time()-t0:.2f}s]")
                # 只打印关节键（*.pos），便于快速核对键命名
                joint_items = [(k, v) for k, v in obs.items() if isinstance(k, str) and k.endswith(".pos")]
                joint_items = sorted(joint_items, key=lambda kv: kv[0])
                if joint_items:
                    print("  关节观测（.pos）：")
                    for k, v in joint_items:
                        print(f"    {k}: {v}")
                else:
                    print("  未发现 *.pos 关节观测键（检查 robot.get_observation() 返回格式）")

                # 也打印非关节的键（比如相机）
                other_keys = [k for k in obs.keys() if not (isinstance(k, str) and k.endswith(".pos"))]
                if other_keys:
                    print(f"  其它观测键（{len(other_keys)} 个）：{other_keys}")

            step += 1
            busy_wait(dt)

        print(f"\n[INFO] 测试完成，共读取 {step} 次观测。")

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt, stopping.")
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass
        print("[INFO] Clean exit.\n")


if __name__ == "__main__":
    main()

"""
python -m lerobot.scripts.test_piper_single_observation \
  --robot.type=piper_single_follower \
  --rate_hz=10 \
  --duration_s=8 \
  --print_every=2 \
  --save_images=true \
  --image_output_dir=outputs/single_obs

"""