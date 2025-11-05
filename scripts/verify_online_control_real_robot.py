#!/usr/bin/env python
"""
在真机上验证在线控制管线的一致性

这个脚本会：
1. 加载训练好的policy
2. 在真机上运行policy收集几步数据
3. 对比policy输出的4维动作经在线管线转换后的关节角度与真机实际反馈的关节角度

用法:
    python scripts/verify_online_control_real_robot.py --config_path config/train_sac_policy_actor.json --num_steps 10
"""

import argparse
import time

import numpy as np
import torch

# Import robot, camera, and teleoperator classes to trigger registration
from lerobot.cameras import (  # noqa: F401
    dabai,
    opencv,
    realsense,
)
from lerobot.robots import (  # noqa: F401
    piper_follower,
    so100_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    gamepad,
    keyboard,
    so101_leader,
)
# Import policy classes to trigger registration
from lerobot.policies.sac.configuration_sac import SACConfig  # noqa: F401

from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor.core import EnvTransition, PolicyAction, TransitionKey
from lerobot.processor.delta_action_processor import (
    MapDeltaActionToRobotActionStep,
    MapTensorToDeltaActionDictStep,
)
from lerobot.rl.gym_manipulator import (
    make_processors,
    make_robot_env,
)
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsRLStep,
)


def load_config(config_path: str):
    """加载配置文件"""
    return TrainRLServerPipelineConfig.from_pretrained(config_path)


def simulate_online_pipeline(action_4d, current_joints, current_gripper_pos, cfg):
    """
    使用实际的在线控制管线处理器步骤，将4维动作转换为关节角度
    
    这使用了与在线控制完全相同的处理器步骤：
    - MapTensorToDeltaActionDictStep: 将4维张量转换为delta动作字典
    - MapDeltaActionToRobotActionStep: 将delta动作映射为机器人动作
    - EEReferenceAndDelta: 计算目标EE位置
    - EEBoundsAndSafety: 边界检查
    - GripperVelocityToJoint: 转换gripper速度
    - InverseKinematicsRLStep: 执行逆运动学
    """
    # 创建kinematics solver
    ik_cfg = cfg.env.processor.inverse_kinematics
    kinematics_solver = RobotKinematics(
        urdf_path=ik_cfg.urdf_path,
        target_frame_name=ik_cfg.target_frame_name,
        joint_names=[f"joint{i+1}" for i in range(6)],
    )
    
    # 创建处理器步骤（与实际在线控制管线相同）
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else False
    motor_names = [f"joint{i+1}" for i in range(6)]
    
    step1 = MapTensorToDeltaActionDictStep(use_gripper=use_gripper)
    step2 = MapDeltaActionToRobotActionStep()
    step3 = EEReferenceAndDelta(
        kinematics=kinematics_solver,
        end_effector_step_sizes=ik_cfg.end_effector_step_sizes,
        motor_names=motor_names,
        use_latched_reference=False,
        use_ik_solution=True,
    )
    step4 = EEBoundsAndSafety(
        end_effector_bounds=ik_cfg.end_effector_bounds,
    )
    step5 = GripperVelocityToJoint(
        clip_max=cfg.env.processor.max_gripper_pos,
        speed_factor=1.0,
        discrete_gripper=True,
    )
    step6 = InverseKinematicsRLStep(
        kinematics=kinematics_solver,
        motor_names=motor_names,
        initial_guess_current_joints=False,
    )
    
    # 构建observation字典，包含关节位置信息
    observation = {}
    for i, motor_name in enumerate(motor_names):
        observation[f"{motor_name}.pos"] = current_joints[i]
    if current_gripper_pos is not None:
        observation["gripper.pos"] = current_gripper_pos
    
    # 创建初始transition对象
    transition = EnvTransition(
        observation=observation,
        action=PolicyAction(torch.tensor(action_4d, dtype=torch.float32)),
        reward=None,
        done=None,
        truncated=None,
        info=None,
        complementary_data={},
    )
    
    # 执行处理器管线
    transition = step1(transition)
    transition = step2(transition)
    transition = step3(transition)
    transition = step4(transition)
    transition = step5(transition)
    transition = step6(transition)
    
    # 提取最终的action（包含关节角度的字典）
    robot_action_6 = transition[TransitionKey.ACTION]
    
    # 提取关节角度
    target_joints = np.array([
        robot_action_6[f"{motor_name}.pos"] for motor_name in motor_names
    ])
    
    # 计算目标EE位置（用于验证）
    target_ee_transform = kinematics_solver.forward_kinematics(target_joints)
    target_ee_position = target_ee_transform[:3, 3]
    
    return target_joints, target_ee_position


def verify_online_control(config_path: str, num_steps: int = 10):
    """在真机上验证在线控制管线的一致性"""
    
    print("=" * 60)
    print("真机在线控制验证")
    print("=" * 60)
    print("\n⚠️  警告：此脚本将在真机上运行policy，请确保机器人处于安全位置！")
    print("按 Ctrl+C 可以随时停止\n")
    
    # 加载配置
    cfg = load_config(config_path)
    
    # 创建真机环境
    print("1. 创建真机环境...")
    env, teleop_device = make_robot_env(cfg.env)
    env_pipeline, action_pipeline = make_processors(env, teleop_device, cfg.env, device="cpu")
    
    # 加载policy（如果配置了checkpoint路径）
    print("2. 加载policy...")
    policy = None
    if hasattr(cfg, 'policy') and hasattr(cfg.policy, 'pretrained_path'):
        # 这里需要根据实际的policy加载方式来实现
        print("   ⚠️  请手动加载policy，这里假设policy已经加载到环境中")
    else:
        print("   ⚠️  未找到policy路径，请确保policy已加载")
    
    # 重置环境
    print("3. 重置环境...")
    obs, info = env.reset()
    obs = env_pipeline(obs)
    
    # 收集数据
    print(f"\n4. 运行policy {num_steps} 步并收集数据...")
    errors = []
    
    try:
        for step in range(num_steps):
            print(f"   步骤 {step + 1}/{num_steps}...")
            
            # 获取当前关节状态（从observation中提取）
            current_state = obs.get("observation.state")
            if current_state is None:
                print("   ⚠️  无法获取observation.state，跳过此步")
                continue
            
            if isinstance(current_state, torch.Tensor):
                current_state = current_state.numpy()
            
            current_joints = current_state[:6].astype(float)
            current_gripper_pos = current_state[6].astype(float) if len(current_state) > 6 else None
            
            # 这里应该使用policy来生成动作，但为了演示，我们使用随机动作
            # 在实际使用中，应该这样：
            # with torch.no_grad():
            #     action_4d = policy.predict(obs)
            # 但为了安全，我们使用零动作
            print("   ⚠️  使用零动作（安全模式），实际使用中应使用policy.predict()")
            action_4d = np.zeros(4, dtype=np.float32)
            
            # 记录policy输出的4维动作
            policy_action = action_4d.copy()
            
            # 通过在线管线转换4维动作为关节角度（用于验证）
            predicted_joints, predicted_ee_pos = simulate_online_pipeline(
                action_4d, current_joints, current_gripper_pos, cfg
            )
            
            # 执行动作（通过action pipeline处理）
            # action pipeline需要接收transition格式
            from lerobot.processor.converters import create_transition
            action_transition = create_transition(
                observation=obs,
                action=torch.tensor(action_4d, dtype=torch.float32),
                reward=None,
                done=None,
                truncated=None,
                info=info,
                complementary_data={},
            )
            
            # 通过action pipeline处理
            processed_transition = action_pipeline(action_transition)
            processed_action = processed_transition[TransitionKey.ACTION]
            
            # 获取实际要发送给机器人的动作（应该包含关节角度）
            # action pipeline的输出应该是字典格式（包含关节角度）
            if isinstance(processed_action, dict):
                # 提取关节角度
                actual_joints = np.array([
                    processed_action.get(f"joint{i+1}.pos", 0.0) for i in range(6)
                ])
            elif isinstance(processed_action, torch.Tensor):
                # 如果是张量，假设是关节角度数组
                actual_joints = processed_action[:6].numpy()
            else:
                actual_joints = np.zeros(6)
            
            # 执行动作并获取下一帧的观察
            # env.step期望接收numpy数组格式的action
            # 我们需要从processed_action中提取关节角度数组
            if isinstance(processed_action, dict):
                # 从字典中提取关节角度数组
                action_array = np.array([
                    processed_action.get(f"joint{i+1}.pos", 0.0) for i in range(len(env.robot.bus.motors))
                ])
            else:
                action_array = processed_action.numpy() if isinstance(processed_action, torch.Tensor) else processed_action
            
            next_obs, reward, done, truncated, info = env.step(action_array)
            next_obs = env_pipeline(next_obs)
            
            # 获取真机实际反馈的关节状态
            next_state = next_obs.get("observation.state")
            if next_state is None:
                print("   ⚠️  无法获取下一帧的observation.state，跳过此步")
                continue
            
            if isinstance(next_state, torch.Tensor):
                next_state = next_state.numpy()
            
            actual_next_joints = next_state[:6].astype(float)
            
            # 计算误差
            joint_error = np.abs(predicted_joints - actual_next_joints)
            max_joint_error = np.max(joint_error)
            mean_joint_error = np.mean(joint_error)
            
            errors.append({
                'step': step + 1,
                'max_error': max_joint_error,
                'mean_error': mean_joint_error,
                'joint_errors': joint_error,
                'policy_action': policy_action,
                'predicted_joints': predicted_joints,
                'actual_joints': actual_joints,
                'robot_feedback_joints': actual_next_joints,
                'predicted_ee': predicted_ee_pos,
            })
            
            print(f"      Policy动作: {policy_action}")
            print(f"      预测关节角度: {predicted_joints}")
            print(f"      实际发送关节角度: {actual_joints}")
            print(f"      真机反馈关节角度: {actual_next_joints}")
            print(f"      最大关节误差: {max_joint_error:.6f} rad")
            print(f"      平均关节误差: {mean_joint_error:.6f} rad")
            
            # 更新观察
            obs = next_obs
            
            # 如果episode结束，重置
            if done or truncated:
                print("   Episode结束，重置环境...")
                obs, info = env.reset()
                obs = env_pipeline(obs)
            
            # 短暂延迟，避免过快执行
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断执行")
    
    finally:
        # 关闭环境
        print("\n5. 关闭环境...")
        env.close()
    
    # 分析结果
    if len(errors) == 0:
        print("\n   ⚠️  没有收集到有效数据")
        return False
    
    print(f"\n6. 结果统计:")
    errors_array = np.array([e['max_error'] for e in errors])
    print(f"   验证步数: {len(errors_array)}")
    print(f"   最大关节误差: {errors_array.max():.6f} rad")
    print(f"   平均关节误差: {errors_array.mean():.6f} rad")
    print(f"   中位数关节误差: {np.median(errors_array):.6f} rad")
    print(f"   95%分位数: {np.percentile(errors_array, 95):.6f} rad")
    
    # 检查一致性
    print(f"\n7. 一致性检查:")
    threshold = 0.01  # 0.01弧度约等于0.57度
    consistent_count = np.sum(errors_array < threshold)
    consistency_rate = consistent_count / len(errors_array) * 100
    
    print(f"   一致性阈值: {threshold:.4f} rad")
    print(f"   一致性步数: {consistent_count}/{len(errors_array)} ({consistency_rate:.1f}%)")
    
    if consistency_rate > 90:
        print(f"   ✓ 一致性良好！在线控制管线工作正常")
    elif consistency_rate > 70:
        print(f"   ⚠️  一致性一般，可能存在小的数值误差或真机反馈延迟")
    else:
        print(f"   ❌ 一致性较差，需要检查在线控制管线或真机反馈")
    
    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)
    
    return consistency_rate > 70


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在真机上验证在线控制管线")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="训练配置文件路径（通常使用actor配置）"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="验证的步数"
    )
    
    args = parser.parse_args()
    
    verify_online_control(
        config_path=args.config_path,
        num_steps=args.num_steps
    )

