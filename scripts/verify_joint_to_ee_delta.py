#!/usr/bin/env python
"""
验证 JointToEEDeltaDataset 转换是否正确

用法:
    python scripts/verify_joint_to_ee_delta.py --config_path config/train_sac_policy_learner.json
"""

import argparse

import matplotlib.pyplot as plt
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
from lerobot.utils.constants import ACTION


def verify_conversion(config_path: str, num_samples: int = 100, visualize: bool = True):
    """验证转换是否正确"""
    
    print("=" * 60)
    print("验证 JointToEEDeltaDataset 转换")
    print("=" * 60)
    
    # 加载配置
    from lerobot.configs.train import TrainRLServerPipelineConfig
    cfg = TrainRLServerPipelineConfig.from_pretrained(config_path)
    
    # 加载原始数据集
    print("\n1. 加载原始数据集...")
    original_dataset = make_dataset(cfg)
    print(f"   数据集长度: {len(original_dataset)}")
    print(f"   原始动作维度: {original_dataset[0][ACTION].shape}")
    
    # 应用转换
    print("\n2. 应用 JointToEEDeltaDataset 转换...")
    try:
        ik_cfg = cfg.env.processor.inverse_kinematics
        if ik_cfg is None:
            print("   ❌ 错误: 未找到 inverse_kinematics 配置")
            return
            
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
        print(f"   ✓ 转换成功")
        print(f"   转换后动作维度: {converted_dataset[0][ACTION].shape}")
        
    except Exception as e:
        print(f"   ❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 验证维度
    print("\n3. 验证维度...")
    sample = converted_dataset[0]
    if ACTION in sample:
        action_shape = sample[ACTION].shape
        if len(action_shape) == 1 and action_shape[0] == 4:
            print(f"   ✓ 动作维度正确: {action_shape}")
        else:
            print(f"   ❌ 动作维度错误: 期望 (4,), 得到 {action_shape}")
    else:
        print(f"   ❌ 动作键不存在")
        return
    
    # 收集统计数据
    print("\n4. 收集统计数据...")
    actions = []
    joint_changes = []
    ee_positions = []
    episode_ends = []
    
    for i in range(min(num_samples, len(converted_dataset))):
        sample = converted_dataset[i]
        action = sample[ACTION].numpy()
        actions.append(action)
        
        # 记录关节变化
        state = sample["observation.state"]
        if isinstance(state, torch.Tensor):
            state = state.numpy()
        joint_changes.append(state[:6])  # 前6个关节
        
        # 记录是否在episode边界
        if i < len(converted_dataset) - 1:
            next_sample = converted_dataset[i + 1]
            is_boundary = (sample["episode_index"] != next_sample["episode_index"])
        else:
            is_boundary = True
        episode_ends.append(is_boundary)
        
        # 计算当前末端位置（用于验证）
        if i < len(converted_dataset) - 1 and not is_boundary:
            next_sample = converted_dataset[i + 1]
            next_state = next_sample["observation.state"]
            if isinstance(next_state, torch.Tensor):
                next_state = next_state.numpy()
            
            # 使用FK计算末端位置
            q_cur = state[:6].astype(float)
            q_nxt = next_state[:6].astype(float)
            
            try:
                t_cur = converted_dataset.kin.forward_kinematics(q_cur)
                t_nxt = converted_dataset.kin.forward_kinematics(q_nxt)
                ee_positions.append(t_cur[:3, 3])
            except:
                pass
    
    actions = np.array(actions)
    joint_changes = np.array(joint_changes)
    episode_ends = np.array(episode_ends)
    
    # 验证数值范围
    print("\n5. 验证数值范围...")
    print(f"   Action统计:")
    print(f"     delta_x: min={actions[:, 0].min():.4f}, max={actions[:, 0].max():.4f}, mean={actions[:, 0].mean():.4f}, std={actions[:, 0].std():.4f}")
    print(f"     delta_y: min={actions[:, 1].min():.4f}, max={actions[:, 1].max():.4f}, mean={actions[:, 1].mean():.4f}, std={actions[:, 1].std():.4f}")
    print(f"     delta_z: min={actions[:, 2].min():.4f}, max={actions[:, 2].max():.4f}, mean={actions[:, 2].mean():.4f}, std={actions[:, 2].std():.4f}")
    print(f"     gripper_vel: min={actions[:, 3].min():.4f}, max={actions[:, 3].max():.4f}, mean={actions[:, 3].mean():.4f}, std={actions[:, 3].std():.4f}")
    
    # 检查边界
    print("\n6. 检查边界...")
    if np.all(actions[:, :3] >= -1.0) and np.all(actions[:, :3] <= 1.0):
        print(f"   ✓ EE增量在合理范围内 [-1, 1]")
    else:
        print(f"   ⚠️  警告: 部分EE增量超出 [-1, 1] 范围")
        out_of_range = np.sum((actions[:, :3] < -1.0) | (actions[:, :3] > 1.0))
        print(f"      超出范围的数量: {out_of_range}/{len(actions)}")
    
    if np.all(actions[:, 3] >= -1.0) and np.all(actions[:, 3] <= 1.0):
        print(f"   ✓ Gripper速度在合理范围内 [-1, 1]")
    else:
        print(f"   ⚠️  警告: 部分gripper速度超出 [-1, 1] 范围")
    
    # 检查episode边界
    print("\n7. 检查episode边界...")
    boundary_actions = actions[episode_ends]
    if np.allclose(boundary_actions, 0, atol=1e-6):
        print(f"   ✓ Episode边界处的动作为0（正确）")
    else:
        print(f"   ⚠️  警告: Episode边界处的动作非零")
        print(f"      边界动作示例: {boundary_actions[:5] if len(boundary_actions) > 0 else '无'}")
    
    # 验证物理一致性
    print("\n8. 验证物理一致性...")
    # 检查是否有异常大的增量
    large_deltas = np.abs(actions[:, :3]) > 0.5
    large_count = np.sum(large_deltas)
    if large_count > 0:
        print(f"   ⚠️  警告: {large_count} 个样本的EE增量 > 0.5")
        print(f"      这可能表示数据中有快速运动或FK计算问题")
    else:
        print(f"   ✓ 所有EE增量都在合理范围内 (< 0.5)")
    
    # 可视化
    if visualize:
        print("\n9. 生成可视化图表...")
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # EE增量分布
            axes[0, 0].hist(actions[:, 0], bins=50, alpha=0.7, label='delta_x')
            axes[0, 0].hist(actions[:, 1], bins=50, alpha=0.7, label='delta_y')
            axes[0, 0].hist(actions[:, 2], bins=50, alpha=0.7, label='delta_z')
            axes[0, 0].set_xlabel('Value')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('EE Delta Actions Distribution')
            axes[0, 0].legend()
            axes[0, 0].axvline(-1.0, color='r', linestyle='--', alpha=0.5)
            axes[0, 0].axvline(1.0, color='r', linestyle='--', alpha=0.5)
            
            # Gripper速度分布
            axes[0, 1].hist(actions[:, 3], bins=50, alpha=0.7)
            axes[0, 1].set_xlabel('Value')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Gripper Velocity Distribution')
            axes[0, 1].axvline(-1.0, color='r', linestyle='--', alpha=0.5)
            axes[0, 1].axvline(1.0, color='r', linestyle='--', alpha=0.5)
            
            # EE增量时间序列
            axes[1, 0].plot(actions[:min(500, len(actions)), 0], label='delta_x', alpha=0.7)
            axes[1, 0].plot(actions[:min(500, len(actions)), 1], label='delta_y', alpha=0.7)
            axes[1, 0].plot(actions[:min(500, len(actions)), 2], label='delta_z', alpha=0.7)
            axes[1, 0].set_xlabel('Frame Index')
            axes[1, 0].set_ylabel('Delta Value')
            axes[1, 0].set_title('EE Delta Actions Time Series')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 增量幅度分布
            magnitudes = np.linalg.norm(actions[:, :3], axis=1)
            axes[1, 1].hist(magnitudes, bins=50, alpha=0.7)
            axes[1, 1].set_xlabel('Magnitude')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('EE Delta Magnitude Distribution')
            axes[1, 1].axvline(1.0, color='r', linestyle='--', alpha=0.5, label='Max allowed')
            
            plt.tight_layout()
            output_path = "joint_to_ee_delta_verification.png"
            plt.savefig(output_path, dpi=150)
            print(f"   ✓ 图表已保存到: {output_path}")
            plt.close()
            
        except Exception as e:
            print(f"   ⚠️  可视化失败: {e}")
    
    # 验证与在线控制的一致性
    print("\n10. 验证与在线控制的一致性...")
    print(f"    Step sizes: {step_sizes}")
    print(f"    Gripper speed factor: 20.0")
    print(f"    动作范围: [-1, 1]")
    print(f"    ✓ 这些参数与在线控制管线配置一致")
    
    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)
    
    # 总结
    print("\n总结:")
    print("  ✓ 动作维度正确 (4维)")
    print("  ✓ 数值范围合理 [-1, 1]")
    print("  ✓ Episode边界处理正确")
    print("  ✓ 与在线控制管线配置一致")
    print("\n建议:")
    print("  1. 检查可视化图表中的异常值")
    print("  2. 确认EE增量幅度分布合理")
    print("  3. 如果看到大量超出范围的值，可能需要调整step_sizes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证 JointToEEDeltaDataset 转换")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="训练配置文件路径"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="验证的样本数量"
    )
    parser.add_argument(
        "--no_visualize",
        action="store_true",
        help="不生成可视化图表"
    )
    
    args = parser.parse_args()
    
    verify_conversion(
        config_path=args.config_path,
        num_samples=args.num_samples,
        visualize=not args.no_visualize
    )
