# HIL-SERL 中 SAC Policy 原理与算法实现详解

## 目录
1. [SAC 算法原理概述](#sac-算法原理概述)
2. [网络架构](#网络架构)
3. [损失函数与优化目标](#损失函数与优化目标)
4. [训练流程](#训练流程)
5. [关键实现细节](#关键实现细节)
6. [配置参数说明](#配置参数说明)

---

## SAC 算法原理概述

### 1.1 什么是 SAC（Soft Actor-Critic）

SAC（Soft Actor-Critic）是一种基于最大熵强化学习的离线策略（off-policy）算法。相比传统的 RL 算法，SAC 在最大化期望累积奖励的同时，还最大化策略的熵（entropy），从而鼓励探索并提高策略的鲁棒性。

### 1.2 核心思想

SAC 的目标函数为：

```
J(π) = E[∑(r_t + α * H(π(·|s_t)))]
```

其中：
- `r_t` 是时刻 t 的奖励
- `α` 是温度参数（temperature），控制探索与利用的平衡
- `H(π(·|s_t))` 是策略在状态 `s_t` 下的熵
- 最大化熵意味着策略更加均匀，鼓励探索

### 1.3 算法组件

SAC 包含以下主要组件：

1. **Actor（策略网络）**：输出动作的均值和标准差，使用重参数化技巧采样动作
2. **Critic（Q 网络）**：估计状态-动作对的价值函数，使用双 Q 网络（double Q-learning）减少过估计
3. **Target Networks**：使用软更新（soft update）保持目标网络的稳定性
4. **Temperature Parameter**：自动调整温度参数 `α`，平衡探索与利用

---

## 网络架构

### 2.1 整体架构

```
输入观测 (图像/状态)
    ↓
[观察编码器] (SACObservationEncoder)
    ↓
    ├──→ [Actor 网络] ──→ 动作分布 (均值 + 标准差)
    │
    └──→ [Critic 网络] ──→ Q 值估计
```

### 2.2 观察编码器 (SACObservationEncoder)

观察编码器负责处理多模态输入（图像和状态向量）：

**图像编码**：
- 使用 CNN 或预训练的视觉编码器提取图像特征
- 应用空间嵌入（SpatialLearnedEmbeddings）进行空间池化
- 通过后处理层（Post-Encoder）生成固定维度的特征向量

**状态编码**：
- 使用线性层 + LayerNorm + Tanh 编码状态向量

**特征融合**：
- 将图像特征和状态特征拼接成统一的观察向量

**关键实现**（`modeling_sac.py:467-618`）：
```python
class SACObservationEncoder(nn.Module):
    def forward(self, obs, cache=None, detach=False):
        parts = []
        if self.has_images:
            # 使用缓存的图像特征（如果共享编码器）
            if cache is None:
                cache = self.get_cached_image_features(obs)
            parts.append(self._encode_images(cache, detach))
        if self.has_env:
            parts.append(self.env_encoder(obs[OBS_ENV_STATE]))
        if self.has_state:
            parts.append(self.state_encoder(obs[OBS_STATE]))
        return torch.cat(parts, dim=-1)
```

### 2.3 Actor 网络（策略网络）

Actor 网络输出动作的分布参数：

**网络结构**：
```
观察编码 → MLP → [均值层, 标准差层] → Tanh 变换 → 动作
```

**动作采样**：
- 使用重参数化技巧（reparameterization trick）：`a = μ + σ * ε`，其中 `ε ~ N(0,1)`
- 应用 Tanh 变换将动作限制在 [-1, 1] 范围内
- 使用 `TanhMultivariateNormalDiag` 分布处理变换后的对数概率

**关键实现**（`modeling_sac.py:798-873`）：
```python
class Policy(nn.Module):
    def forward(self, observations, observation_features=None):
        obs_enc = self.encoder(observations, cache=observation_features, 
                              detach=self.encoder_is_shared)
        outputs = self.network(obs_enc)
        means = self.mean_layer(outputs)
        
        # 计算标准差
        log_std = self.std_layer(outputs)
        std = torch.exp(log_std)
        std = torch.clamp(std, self.std_min, self.std_max)
        
        # 构建变换分布（Tanh + 可选的 Rescale）
        dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)
        
        # 重参数化采样
        actions = dist.rsample()
        log_probs = dist.log_prob(actions)
        
        return actions, log_probs, means
```

### 2.4 Critic 网络（Q 网络）

Critic 网络估计状态-动作对的 Q 值：

**网络结构**：
```
[观察编码, 动作] → MLP → Q 值输出
```

**双 Q 网络（Double Q-Learning）**：
- 使用多个 Q 网络（通常为 2 个）组成集成（ensemble）
- 取最小值以减少过估计：`Q_target = min(Q1, Q2)`

**关键实现**（`modeling_sac.py:709-753`）：
```python
class CriticEnsemble(nn.Module):
    def forward(self, observations, actions, observation_features=None):
        obs_enc = self.encoder(observations, cache=observation_features)
        inputs = torch.cat([obs_enc, actions], dim=-1)
        
        # 每个 critic 独立计算 Q 值
        q_values = []
        for critic in self.critics:
            q_values.append(critic(inputs))
        
        # 返回形状为 [num_critics, batch_size] 的张量
        return torch.stack([q.squeeze(-1) for q in q_values], dim=0)
```

---

## 损失函数与优化目标

### 3.1 Critic 损失（Q 网络损失）

Critic 使用 TD（Temporal Difference）学习更新 Q 值：

**目标值计算**：
```python
# 1. 从下一个状态采样动作
next_actions, next_log_probs, _ = actor(next_observations)

# 2. 计算目标 Q 值（使用目标网络）
q_targets = critic_target(next_observations, next_actions)

# 3. 取最小值（双 Q 网络）
min_q = q_targets.min(dim=0)[0]

# 4. 添加熵项（如果启用）
if use_backup_entropy:
    min_q = min_q - (temperature * next_log_probs)

# 5. 计算 TD 目标
td_target = rewards + (1 - done) * discount * min_q
```

**损失函数**：
```python
# 计算所有 critic 的 MSE 损失
td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
critics_loss = F.mse_loss(q_preds, td_target_duplicate, reduction="none").mean(dim=1).sum()
```

**关键实现**（`modeling_sac.py:245-304`）：
```python
def compute_loss_critic(self, observations, actions, rewards, 
                       next_observations, done, ...):
    with torch.no_grad():
        # 使用目标网络计算目标 Q 值
        next_action_preds, next_log_probs, _ = self.actor(next_observations, ...)
        q_targets = self.critic_forward(..., use_target=True)
        
        # 子采样 critic（如果使用高 UTD 比例）
        if self.config.num_subsample_critics is not None:
            indices = torch.randperm(self.config.num_critics)
            indices = indices[:self.config.num_subsample_critics]
            q_targets = q_targets[indices]
        
        min_q, _ = q_targets.min(dim=0)
        if self.config.use_backup_entropy:
            min_q = min_q - (self.temperature * next_log_probs)
        
        td_target = rewards + (1 - done) * self.config.discount * min_q
    
    # 计算预测 Q 值
    q_preds = self.critic_forward(..., use_target=False)
    
    # 计算损失
    td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
    critics_loss = F.mse_loss(q_preds, td_target_duplicate, 
                              reduction="none").mean(dim=1).sum()
    return critics_loss
```

### 3.2 Actor 损失（策略损失）

Actor 的目标是最大化 Q 值同时保持策略的熵：

**损失函数**：
```python
# 1. 从当前策略采样动作
actions_pi, log_probs, _ = actor(observations)

# 2. 计算 Q 值
q_preds = critic(observations, actions_pi)
min_q_preds = q_preds.min(dim=0)[0]  # 取最小值

# 3. Actor 损失 = 温度 * 对数概率 - Q 值
#    最大化 Q 值，同时通过熵项鼓励探索
actor_loss = ((temperature * log_probs) - min_q_preds).mean()
```

**关键实现**（`modeling_sac.py:373-389`）：
```python
def compute_loss_actor(self, observations, observation_features=None):
    actions_pi, log_probs, _ = self.actor(observations, observation_features)
    
    q_preds = self.critic_forward(observations, actions_pi, 
                                  use_target=False, ...)
    min_q_preds = q_preds.min(dim=0)[0]
    
    # 损失 = 温度 * log_prob - Q
    # 最大化 Q（最小化 -Q），同时通过熵项鼓励探索
    actor_loss = ((self.temperature * log_probs) - min_q_preds).mean()
    return actor_loss
```

### 3.3 Temperature 损失（温度参数自动调整）

Temperature 参数控制探索与利用的平衡。SAC 通过自动调整温度来保持目标熵：

**目标熵**：
- 如果未指定，默认值为：`target_entropy = -action_dim / 2`
- 例如，4 维动作空间：`target_entropy = -2`

**损失函数**：
```python
# 计算当前策略的熵
_, log_probs, _ = actor(observations)

# 温度损失 = -log_alpha * (log_prob + target_entropy)
# 最小化此损失会调整温度，使策略熵接近目标熵
temperature_loss = (-log_alpha.exp() * (log_probs + target_entropy)).mean()
```

**关键实现**（`modeling_sac.py:365-371`）：
```python
def compute_loss_temperature(self, observations, observation_features=None):
    with torch.no_grad():
        _, log_probs, _ = self.actor(observations, observation_features)
    
    # log_alpha 是可学习参数
    temperature_loss = (-self.log_alpha.exp() * 
                       (log_probs + self.target_entropy)).mean()
    return temperature_loss
```

---

## 训练流程

### 4.1 训练循环结构

训练采用 UTD（Update-To-Data）比例，即每个环境步进行多次梯度更新：

**训练步骤**（`learner.py:469-573`）：
```python
for optimization_step in range(max_steps):
    # 1. 多次 Critic 更新（UTD 比例）
    for _ in range(utd_ratio - 1):
        batch = sample_from_buffer()
        loss_critic = policy.compute_loss_critic(...)
        loss_critic.backward()
        optimizer_critic.step()
        policy.update_target_networks()
    
    # 2. 最后一次 Critic 更新
    batch = sample_from_buffer()
    loss_critic = policy.compute_loss_critic(...)
    loss_critic.backward()
    optimizer_critic.step()
    
    # 3. Actor 和 Temperature 更新（按频率）
    if optimization_step % policy_update_freq == 0:
        loss_actor = policy.compute_loss_actor(...)
        loss_actor.backward()
        optimizer_actor.step()
        
        loss_temperature = policy.compute_loss_temperature(...)
        loss_temperature.backward()
        optimizer_temperature.step()
        policy.update_temperature()  # 更新温度值
    
    # 4. 更新目标网络
    policy.update_target_networks()
```

### 4.2 目标网络更新

使用软更新（soft update）保持目标网络的稳定性：

**更新公式**：
```python
target_param = τ * param + (1 - τ) * target_param
```

其中 `τ` 是更新权重（通常为 0.005）。

**关键实现**（`modeling_sac.py:220-240`）：
```python
def update_target_networks(self):
    for target_param, param in zip(
        self.critic_target.parameters(),
        self.critic_ensemble.parameters(),
        strict=True,
    ):
        target_param.data.copy_(
            param.data * self.config.critic_target_update_weight
            + target_param.data * (1.0 - self.config.critic_target_update_weight)
        )
```

### 4.3 离线训练 vs 在线训练

**离线训练**（`learner_offline.py`）：
- 仅使用离线数据集进行训练
- 不需要与环境交互
- 适用于已有大量演示数据的情况

**在线训练**（`learner.py`）：
- Actor 收集数据并发送到 Learner
- Learner 从在线缓冲区和离线缓冲区混合采样
- 适用于需要在线学习新技能的场景

---

## 关键实现细节

### 5.1 共享编码器机制

当 `shared_encoder=True` 时，Actor 和 Critic 共享同一个观察编码器：

**优势**：
- 减少参数量
- 加快训练速度（只需编码一次）

**实现细节**：
```python
# Actor 前向传播时，如果共享编码器，需要 detach 编码器输出
# 避免通过策略损失更新编码器
obs_enc = self.encoder(observations, cache=observation_features, 
                      detach=self.encoder_is_shared)
```

### 5.2 图像特征缓存

当使用预训练且冻结的视觉编码器时，可以缓存图像特征：

**缓存机制**（`modeling_sac.py:563-589`）：
```python
def get_cached_image_features(self, obs):
    """提取并缓存图像特征，避免重复计算"""
    batched = torch.cat([obs[k] for k in self.image_keys], dim=0)
    out = self.image_encoder(batched)
    chunks = torch.chunk(out, len(self.image_keys), dim=0)
    return dict(zip(self.image_keys, chunks, strict=False))
```

**性能提升**：
- 视觉编码器通常是计算瓶颈
- 缓存特征可获得 2-4x 的训练加速

### 5.3 Tanh 变换与动作限制

动作通过 Tanh 变换限制在 [-1, 1] 范围内：

**变换过程**：
1. 从正态分布采样：`a ~ N(μ, σ)`
2. 应用 Tanh 变换：`a_tanh = tanh(a)`
3. 可选：重新缩放到 [low, high] 范围

**对数概率计算**：
- 需要考虑 Tanh 变换的雅可比行列式
- `log_prob_total = log_prob_base + log_det_jacobian`

**关键实现**（`modeling_sac.py:1030-1063`）：
```python
class TanhMultivariateNormalDiag(TransformedDistribution):
    def __init__(self, loc, scale_diag, low=None, high=None):
        base_dist = MultivariateNormal(loc, torch.diag_embed(scale_diag))
        transforms = [TanhTransform(cache_size=1)]
        
        if low is not None and high is not None:
            transforms.insert(0, RescaleFromTanh(low, high))
        
        super().__init__(base_dist, transforms)
```

### 5.4 离散动作支持

对于离散动作（如夹爪控制），使用单独的离散 Critic：

**实现**（`modeling_sac.py:306-363`）：
```python
def compute_loss_discrete_critic(self, ...):
    # 使用 DQN 风格的目标计算
    # 1. 使用在线网络选择最佳动作
    next_discrete_qs = self.discrete_critic_forward(..., use_target=False)
    best_next_action = torch.argmax(next_discrete_qs, dim=-1)
    
    # 2. 使用目标网络评估 Q 值
    target_next_qs = self.discrete_critic_forward(..., use_target=True)
    target_next_q = torch.gather(target_next_qs, dim=1, index=best_next_action)
    
    # 3. 计算 TD 目标
    target_q = rewards + (1 - done) * discount * target_next_q
    
    # 4. 计算损失
    predicted_q = self.discrete_critic_forward(...)
    loss = F.mse_loss(predicted_q, target_q)
```

### 5.5 UTD（Update-To-Data）比例

UTD 比例允许每个环境步进行多次梯度更新：

**优势**：
- 提高数据利用效率
- 加快收敛速度

**实现**：
```python
# 每个环境步，进行 utd_ratio 次 Critic 更新
for _ in range(utd_ratio - 1):
    batch = sample_from_buffer()
    update_critic()
    
# 最后一次更新
batch = sample_from_buffer()
update_critic()
update_actor()  # 仅在特定频率更新
```

### 5.6 Critic 子采样

当使用高 UTD 比例时，可以子采样 Critic 以减少过拟合：

**实现**（`modeling_sac.py:268-271`）：
```python
if self.config.num_subsample_critics is not None:
    indices = torch.randperm(self.config.num_critics)
    indices = indices[:self.config.num_subsample_critics]
    q_targets = q_targets[indices]
```

---

## 配置参数说明

### 6.1 核心算法参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `discount` | 0.99 | 折扣因子，控制未来奖励的重要性 |
| `temperature_init` | 1.0 | 初始温度参数 |
| `num_critics` | 2 | Critic 网络数量（双 Q 网络） |
| `critic_target_update_weight` | 0.005 | 目标网络软更新权重（τ） |
| `utd_ratio` | 1 | 每个环境步的更新次数 |
| `target_entropy` | None | 目标熵（None 时自动计算为 -action_dim/2） |
| `use_backup_entropy` | True | 是否在目标 Q 值计算中使用熵项 |

### 6.2 网络架构参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `latent_dim` | 256 | 编码器输出维度 |
| `state_encoder_hidden_dim` | 256 | 状态编码器隐藏层维度 |
| `image_encoder_hidden_dim` | 32 | 图像编码器隐藏层维度 |
| `shared_encoder` | True | 是否共享 Actor 和 Critic 的编码器 |
| `freeze_vision_encoder` | True | 是否冻结视觉编码器 |
| `vision_encoder_name` | None | 预训练视觉编码器名称 |

### 6.3 优化器参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `actor_lr` | 3e-4 | Actor 学习率 |
| `critic_lr` | 3e-4 | Critic 学习率 |
| `temperature_lr` | 3e-4 | 温度参数学习率 |
| `grad_clip_norm` | 40.0 | 梯度裁剪范数 |

### 6.4 策略参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_tanh_squash` | True | 是否使用 Tanh 变换限制动作 |
| `std_min` | 1e-5 | 标准差最小值 |
| `std_max` | 10.0 | 标准差最大值 |
| `init_final` | 0.05 | 最终层初始化范围 |

### 6.5 训练流程参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `policy_update_freq` | 1 | Actor 更新频率（每 N 步更新一次） |
| `online_step_before_learning` | 100 | 开始学习前需要收集的环境步数 |
| `online_buffer_capacity` | 100000 | 在线缓冲区容量 |
| `offline_buffer_capacity` | 100000 | 离线缓冲区容量 |

---

## 总结

HIL-SERL 中的 SAC 实现具有以下特点：

1. **完整的 SAC 算法实现**：包括 Actor、Critic、Temperature 的自动调整
2. **多模态输入支持**：支持图像和状态向量的混合输入
3. **高效的训练机制**：共享编码器、特征缓存、UTD 比例等优化
4. **灵活的配置**：支持离线训练、在线训练、混合训练等多种模式
5. **离散动作支持**：可处理连续动作和离散动作的混合动作空间

该实现遵循了 SAC 算法的核心原理，同时针对机器人操作任务进行了优化，是一个高效且实用的离线策略强化学习实现。

