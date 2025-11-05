#!/usr/bin/env python
"""
对比离线转换后的动作和在线控制生成的动作，验证一致性

用法:
    python scripts/compare_offline_online_actions.py --config_path config/train_sac_policy_learner.json
"""

import argparse

import numpy as np
import torch

# Import robot, camera, and teleoperator classes to trigger registration
from lerobot.cameras import (  # noqa: F401
    opencv,
    realsense,
    dabai,
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

from lerobot.datasets.factory import make_dataset
from lerobot.datasets.transforms import JointToEEDeltaConfig, JointToEEDeltaDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.constants import ACTION

# Import actual processor steps for online control pipeline
from lerobot.processor.delta_action_processor import (
    MapDeltaActionToRobotActionStep,
    MapTensorToDeltaActionDictStep,
)
from lerobot.processor.core import PolicyAction, RobotAction, TransitionKey
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsRLStep,
)


def load_config(config_path: str):
    """加载配置文件"""
    from lerobot.configs.train import TrainRLServerPipelineConfig
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
    # 创建kinematics solver（直接使用RobotKinematics，与gym_manipulator.py中的方式相同）
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
    
    # 创建模拟的transition对象
    # 注意：这些处理器步骤需要transition对象来访问observation
    # 我们需要创建一个简化的transition来模拟在线控制
    from lerobot.processor.core import EnvTransition
    
    # 构建observation字典，包含关节位置信息
    # 注意：GripperVelocityToJoint需要gripper位置，所以observation中也需要包含gripper
    observation = {}
    for i, motor_name in enumerate(motor_names):
        observation[f"{motor_name}.pos"] = current_joints[i]
    # 添加gripper位置（GripperVelocityToJoint期望gripper在q_raw的最后一个位置）
    if current_gripper_pos is not None:
        observation["gripper.pos"] = current_gripper_pos
    
    # 创建初始transition对象
    # 注意：我们需要使用完整的transition对象，因为处理器步骤需要访问observation
    transition = EnvTransition(
        observation=observation,
        action=PolicyAction(torch.tensor(action_4d, dtype=torch.float32)),  # 初始action是4维张量
        reward=None,
        done=None,
        truncated=None,
        info=None,
        complementary_data={},
    )
    
    # 执行处理器管线（使用__call__方法，每个步骤都会更新transition）
    # 1. 将4维张量转换为delta动作字典
    transition = step1(transition)
    
    # 2. 将delta动作映射为机器人动作
    transition = step2(transition)
    
    # 3. 计算目标EE位置
    transition = step3(transition)
    
    # 4. 边界检查
    transition = step4(transition)
    
    # 5. 转换gripper速度
    transition = step5(transition)
    
    # 6. 执行逆运动学
    transition = step6(transition)
    
    # 提取最终的action（现在应该是包含关节角度的字典）
    robot_action_6 = transition[TransitionKey.ACTION]
    
    # 提取关节角度
    target_joints = np.array([
        robot_action_6[f"{motor_name}.pos"] for motor_name in motor_names
    ])
    
    # 计算目标EE位置（用于验证）
    # 从step3（EEReferenceAndDelta）中获取desired_ee_pose
    # 注意：这需要在step3之后，但在step6之前访问
    # 作为替代，我们可以从complementary_data中获取，或者重新计算
    # 为了简化，我们使用forward kinematics从目标关节角度计算EE位置
    target_ee_transform = kinematics_solver.forward_kinematics(target_joints)
    target_ee_position = target_ee_transform[:3, 3]
    
    return target_joints, target_ee_position


def compare_actions(config_path: str, num_samples: int = 50):
    """对比离线转换和在线模拟的结果"""
    
    print("=" * 60)
    print("对比离线转换后的动作与在线控制管线")
    print("=" * 60)
    
    # 加载配置
    cfg = load_config(config_path)
    
    # 加载并转换数据集
    print("\n1. 加载并转换数据集...")
    original_dataset = make_dataset(cfg)
    
    ik_cfg = cfg.env.processor.inverse_kinematics
    joint_indices = list(range(0, 6))
    gripper_index = 6
    step_sizes = {
        "x": ik_cfg.end_effector_step_sizes["x"],
        "y": ik_cfg.end_effector_step_sizes["y"],
        "z": ik_cfg.end_effector_step_sizes["z"],
    }
    
    jt_cfg = JointToEEDeltaConfig(
        urdf_path=ik_cfg.urdf_path,
        target_frame_name=ik_cfg.target_frame_name,
        joint_indices=joint_indices,
        gripper_index=gripper_index,
        step_sizes=step_sizes,
        gripper_speed_factor=20.0,
    )
    
    converted_dataset = JointToEEDeltaDataset(original_dataset, jt_cfg)
    
    # 对比验证
    print("\n2. 对比验证（检查转换的可逆性）...")
    errors = []
    
    for i in range(min(num_samples, len(converted_dataset) - 1)):
        sample = converted_dataset[i]
        next_sample = converted_dataset[i + 1]
        
        # 跳过episode边界
        if sample["episode_index"] != next_sample["episode_index"]:
            continue
        
        # 获取转换后的4维动作
        action_4d = sample[ACTION].numpy()
        
        # 获取当前关节状态
        current_state = sample["observation.state"]
        if isinstance(current_state, torch.Tensor):
            current_state = current_state.numpy()
        current_joints = current_state[:6].astype(float)
        current_gripper_pos = current_state[6].astype(float)  # gripper在第7个位置
        
        # 获取下一帧的实际关节状态（这是"真实"的目标）
        next_state = next_sample["observation.state"]
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.numpy()
        actual_next_joints = next_state[:6].astype(float)
        
        # 模拟在线管线：4维动作 -> 关节角度（使用实际的在线控制处理器步骤）
        predicted_joints, predicted_ee_pos = simulate_online_pipeline(
            action_4d, current_joints, current_gripper_pos, cfg
        )
        
        # 计算误差
        joint_error = np.abs(predicted_joints - actual_next_joints)
        max_joint_error = np.max(joint_error)
        mean_joint_error = np.mean(joint_error)
        
        errors.append({
            'max_error': max_joint_error,
            'mean_error': mean_joint_error,
            'joint_errors': joint_error,
            'action_4d': action_4d,
            'predicted_ee': predicted_ee_pos,
        })
    
    if len(errors) == 0:
        print("   ⚠️  没有找到有效的样本进行对比（可能都是episode边界）")
        return False
    
    # 保存原始错误列表用于显示示例
    error_details = errors.copy()
    errors_array = np.array([e['max_error'] for e in errors])
    
    print(f"\n3. 结果统计:")
    print(f"   验证样本数: {len(errors_array)}")
    print(f"   最大关节误差: {errors_array.max():.6f} rad")
    print(f"   平均关节误差: {errors_array.mean():.6f} rad")
    print(f"   中位数关节误差: {np.median(errors_array):.6f} rad")
    print(f"   95%分位数: {np.percentile(errors_array, 95):.6f} rad")
    
    # 检查一致性
    print(f"\n4. 一致性检查:")
    threshold = 0.01  # 1cm的误差阈值（对于关节角度，这大约是0.01弧度）
    consistent_count = np.sum(errors_array < threshold)
    consistency_rate = consistent_count / len(errors_array) * 100
    
    print(f"   一致性阈值: {threshold:.4f} rad")
    print(f"   一致性样本数: {consistent_count}/{len(errors_array)} ({consistency_rate:.1f}%)")
    
    if consistency_rate > 90:
        print(f"   ✓ 一致性良好！转换后的动作与在线管线一致")
    elif consistency_rate > 70:
        print(f"   ⚠️  一致性一般，可能存在小的数值误差")
    else:
        print(f"   ❌ 一致性较差，需要检查转换逻辑")
    
    # 显示一些示例
    print(f"\n5. 示例对比（前5个样本）:")
    for i, e in enumerate(error_details[:5]):
        print(f"   样本 {i+1}:")
        print(f"     动作: {e['action_4d']}")
        print(f"     最大关节误差: {e['max_error']:.6f} rad")
        print(f"     平均关节误差: {e['mean_error']:.6f} rad")
    
    print("\n" + "=" * 60)
    print("对比完成！")
    print("=" * 60)
    
    return consistency_rate > 70


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比离线转换和在线控制管线")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="训练配置文件路径"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="验证的样本数量"
    )
    
    args = parser.parse_args()
    
    compare_actions(
        config_path=args.config_path,
        num_samples=args.num_samples
    )
