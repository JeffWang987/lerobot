# 如何加载训练好的SAC Policy模型

## 概述

训练完成后，模型会保存在checkpoint目录中。要让learner和actor使用训练好的模型，需要在配置文件中指定`pretrained_path`。

## Checkpoint结构

训练完成后，模型保存在以下结构中：

```
outputs/
└── <output_dir>/
    └── checkpoints/
        └── last/  (symlink指向最新的checkpoint)
            ├── pretrained_model/  ← 这里是模型文件
            │   ├── model.safetensors
            │   ├── config.json
            │   ├── preprocessor.json
            │   └── postprocessor.json
            ├── training_state/
            └── train_config.json
```

## 方法1：加载最新checkpoint（推荐）

在配置文件的`policy`部分添加`pretrained_path`字段，指向`pretrained_model`目录：

```json
{
  "policy": {
    "type": "sac",
    "pretrained_path": "outputs/piper_offline_local_learner1103/checkpoints/last/pretrained_model",
    // ... 其他配置
  }
}
```

## 方法2：加载特定step的checkpoint

如果需要加载特定训练步数的checkpoint：

```json
{
  "policy": {
    "type": "sac",
    "pretrained_path": "outputs/piper_offline_local_learner1103/checkpoints/step-50000/pretrained_model",
    // ... 其他配置
  }
}
```

## 配置示例

### Learner配置 (`train_sac_policy_learner.json`)

```json
{
  "output_dir": "outputs/piper_offline_local_learner1103",
  "policy": {
    "type": "sac",
    "pretrained_path": "outputs/piper_offline_local_learner1103/checkpoints/last/pretrained_model",
    "device": "cuda",
    "storage_device": "cuda",
    // ... 其他配置
  },
  // ... 其他配置
}
```

### Actor配置 (`train_sac_policy_actor.json`)

```json
{
  "output_dir": "outputs/piper_offline_local_actor1103",
  "policy": {
    "type": "sac",
    "pretrained_path": "outputs/piper_offline_local_learner1103/checkpoints/last/pretrained_model",
    "device": "cuda",
    "storage_device": "cuda",
    // ... 其他配置
  },
  // ... 其他配置
}
```

## 使用方法

### 1. 继续训练（从checkpoint恢复）

如果learner已经训练过，可以设置`resume: true`来继续训练：

```json
{
  "resume": true,
  "output_dir": "outputs/piper_offline_local_learner1103",
  // ... 其他配置
}
```

或者使用`pretrained_path`加载模型权重：

```json
{
  "resume": false,
  "output_dir": "outputs/piper_offline_local_learner_new",
  "policy": {
    "pretrained_path": "outputs/piper_offline_local_learner1103/checkpoints/last/pretrained_model",
    // ... 其他配置
  }
}
```

### 2. 在Actor上运行训练好的模型

在actor配置中指定pretrained_path：

```json
{
  "policy": {
    "pretrained_path": "outputs/piper_offline_local_learner1103/checkpoints/last/pretrained_model",
    // ... 其他配置
  }
}
```

然后运行actor：

```bash
python -m lerobot.rl.actor --config_path config/train_sac_policy_actor.json
```

### 3. 仅推理（不使用learner-actor通信）

如果要单独运行actor进行推理，可以：

1. 在actor配置中设置`pretrained_path`
2. 确保配置中的`policy.actor_learner_config.learner_host`和`learner_port`指向learner（如果使用在线学习）
3. 如果只做推理，可以注释掉或修改learner连接相关代码

## 注意事项

1. **路径可以是绝对路径或相对路径**
   ```json
   "pretrained_path": "/home/user/lerobot/outputs/model/checkpoints/last/pretrained_model"
   ```
   或
   ```json
   "pretrained_path": "outputs/model/checkpoints/last/pretrained_model"
   ```

2. **确保配置一致性**
   - `pretrained_path`中的模型配置应该与当前配置兼容
   - 输入/输出特征应该匹配
   - 如果特征名称不同，可以使用`rename_map`

3. **模型加载优先级**
   - 如果设置了`pretrained_path`，模型会从该路径加载
   - 如果设置了`resume: true`，会从`output_dir/checkpoints/last/`加载
   - 如果两者都设置，`pretrained_path`优先

4. **Actor和Learner的配置**
   - Actor和Learner可以使用相同的pretrained_path
   - Actor会从learner接收更新的权重（如果learner正在训练）
   - 如果learner也在使用相同的pretrained_path，它会在此基础上继续训练

## 验证模型加载

可以使用以下脚本验证模型是否正确加载：

```bash
# 检查模型文件是否存在
ls -la outputs/piper_offline_local_learner1103/checkpoints/last/pretrained_model/

# 应该看到：
# - model.safetensors
# - config.json
# - preprocessor.json
# - postprocessor.json
```

## 故障排除

### 问题1: `FileNotFoundError: pretrained_path does not exist`

**解决方案**：检查路径是否正确，确保`pretrained_model`目录存在。

### 问题2: `ValueError: Feature mismatch`

**解决方案**：确保配置中的`input_features`和`output_features`与训练时一致。

### 问题3: Actor无法连接到Learner

**解决方案**：
- 如果只做推理，可以修改actor代码跳过learner连接
- 或者启动一个learner服务器（即使不做训练）

