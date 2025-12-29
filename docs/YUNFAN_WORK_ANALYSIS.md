# 云帆 ManiSkill + Pi0 工作详细分析

> 调研时间: 2024-12-28
> 目的: 深入理解云帆的实现方案，为我们的 Pi0.5 训练提供参考

---

## 1. 云帆做了什么

### 1.1 核心工作

云帆成功将 **Pi0 模型** 在 **ManiSkill3 仿真环境** 中进行了 **SFT 微调**，实现了多个操作任务的自主执行。

**技术栈**:
- **模型**: Pi0 (不是 Pi0.5)
- **框架**: OpenPI (Physical Intelligence 官方代码)
- **仿真**: ManiSkill3 (GPU 并行仿真)
- **数据格式**: LeRobot

### 1.2 训练的任务

| 任务 | 环境ID | 训练步数 | batch_size | action_horizon |
|------|--------|----------|------------|----------------|
| StackCube | StackCube-v1 | 30,000 | 256 | 20 |
| PickCube | PickCube-v1 | 30,000 | 256 | 10 |
| PushCube | PushCube-v1 | 20,000 | 16 | 10 |
| PullCube | PullCube-v1 | 20,000 | 16 | 10 |
| PlaceSphere | PlaceSphere-v1 | 30,000 | 256 | 10 |
| PegInsertionSide | PegInsertionSide-v1 | - | 16 | 10 |
| PullCubeTool | PullCubeTool-v1 | 20,000 | 16 | 10 |

### 1.3 评估结果 (40 episodes)

| 任务 | 成功率 | 评估 |
|------|--------|------|
| PullCube | 87.50% (35/40) | ⭐ 最佳 |
| PushCube | 70.00% (28/40) | ✅ 好 |
| StackCube | 60.00% (24/40) | ✅ 中等 |
| PlaceSphere | 27.50% (11/40) | ⚠️ 较低 |
| PullCubeTool | 7.50% (3/40) | ❌ 困难 |
| PickCube | 2.50% (1/40) | ❌ 很低 |
| PegInsertionSide | 0% (0/20) | ❌ 失败 |

---

## 2. 代码架构分析

### 2.1 目录结构

```
/share/project/yunfan/RL/caurft/openpi/
├── checkpoints/
│   ├── pi0_base/                         # Pi0 预训练权重
│   │   └── openpi/openpi-assets/checkpoints/pi0_base/params/
│   ├── pi0_maniskill_stackcube/          # 训练后的 checkpoint
│   │   └── stackcube/                    # 实验目录
│   │       ├── 500/                      # step 500
│   │       ├── 1000/
│   │       ├── ...
│   │       └── 29999/                    # 最终 checkpoint
│   └── pi0_maniskill_xxx/                # 其他任务
├── src/openpi/training/config.py         # 训练配置
├── scripts/
│   ├── eval_maniskill.py                 # 评估脚本
│   └── run_maniskill_eval.sh             # 评估启动脚本
└── video_xxx/                            # 评估视频
```

### 2.2 数据目录

```
/share/project/yunfan/RL/maniskill_lerobot/
├── maniskill_stackcube/                  # LeRobot 格式数据
├── maniskill_pickcube/
├── maniskill_pushcube/
├── maniskill_pullcube/
├── maniskill_placesphere/
├── maniskill_peginsertionside/
└── maniskill_pullcubetool/
```

---

## 3. 关键代码分析

### 3.1 训练配置 (config.py)

云帆定义了两种 ManiSkill 数据配置:

```python
# 有腕部相机的任务 (如 StackCube)
class LeRobotManiskillDataConfig(DataConfigFactory):
    """使用 LiberoInputs 处理，包含腕部相机"""
    pass

# 无腕部相机的任务 (如 PickCube, PushCube)
class LeRobotManiskillNoWristDataConfig(DataConfigFactory):
    """使用 LiberoInputs_no_wrist 处理，无腕部相机"""
    pass
```

**训练配置示例 (StackCube)**:

```python
TrainConfig(
    name="pi0_maniskill_stackcube",
    model=pi0_config.Pi0Config(action_horizon=20),  # Pi0，非 Pi0.5
    data=LeRobotManiskillDataConfig(
        repo_id="/share/project/yunfan/RL/maniskill_lerobot/maniskill_stackcube",
        base_config=DataConfig(prompt_from_task=True),  # 从数据集读取 prompt
        extra_delta_transform=False,
    ),
    batch_size=256,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10_000,
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/share/project/yunfan/.../pi0_base/params"
    ),
    num_train_steps=30_000,
    save_interval=500,
    keep_period=500,
)
```

### 3.2 评估脚本 (eval_maniskill.py)

**关键函数: 观测预处理**

```python
def _extract_policy_inputs_from_obs(raw_obs: dict, prompt: str) -> dict:
    """将 ManiSkill 观测转换为 Pi0 输入格式"""

    # 1. 提取基座相机图像 (去掉 alpha 通道)
    base = raw_obs["sensor_data"]["base_camera"]["Color"][..., :3]

    # 2. 提取腕部相机图像 (如果有)
    wrist = raw_obs["sensor_data"]["hand_camera"]["Color"][..., :3]

    # 3. 提取机器人状态 (取 qpos 前 8 维)
    qpos = raw_obs["agent"]["qpos"]
    state8 = np.asarray(qpos[:8], dtype=np.float32)

    return {
        "observation/image": base,           # (H, W, 3)
        "observation/wrist_image": wrist,    # (H, W, 3) 或零图像
        "observation/state": state8,         # (8,)
        "prompt": prompt,
    }
```

**关键: State 格式**

云帆使用的是 **qpos[:8]**，即关节位置的前 8 维:
- 7 个关节角度 + 1 个夹爪位置
- **不是 EEF pose (末端位姿)**

### 3.3 评估执行流程

```python
def evaluate(...):
    # 1. 创建 ManiSkill 环境
    env = gym.make(
        id=env_id,
        num_envs=1,
        obs_mode="sensor_data",              # 获取相机数据
        control_mode="pd_ee_delta_pose",     # delta pose 控制
        sim_backend="cpu",
        max_episode_steps=500,
        sensor_configs={"width": 256, "height": 256},
    )

    # 2. 评估循环
    for ep in range(num_episodes):
        raw_obs, _ = env.reset()
        while not done:
            # 转换观测
            inputs = _extract_policy_inputs_from_obs(raw_obs, prompt)

            # 模型推理
            outputs = policy.infer(inputs)
            actions_chunk = outputs["actions"]  # (horizon, 7)

            # 执行动作序列
            for a in actions_chunk[:20]:  # 最多执行 20 步
                raw_obs, reward, terminated, truncated, info = env.step(a)
                if terminated or truncated:
                    break
```

---

## 4. 数据预处理流程

### 4.1 数据转换 (ManiSkill → LeRobot)

原始 ManiSkill 演示 → LeRobot 格式:

```
ManiSkill trajectory.h5:
├── obs/
│   ├── sensor_data/base_camera/Color  → observation/image
│   ├── sensor_data/hand_camera/Color  → observation/wrist_image
│   └── agent/qpos                     → observation/state[:8]
├── actions                            → actions (7D)
└── task_name                          → prompt
```

### 4.2 输入/输出格式

**训练时输入**:

| 键 | 形状 | 说明 |
|----|------|------|
| observation/image | (256, 256, 3) | 基座相机 RGB |
| observation/wrist_image | (256, 256, 3) | 腕部相机 RGB (可选) |
| observation/state | (8,) | 关节位置 qpos[:8] |
| prompt | str | 任务描述 |

**训练时输出**:

| 键 | 形状 | 说明 |
|----|------|------|
| actions | (horizon, 7) | 动作序列 |

动作 7 维 = `[dx, dy, dz, drx, dry, drz, gripper]`

---

## 5. 训练细节

### 5.1 训练时间

从云帆的日志分析:

```
StackCube 训练 (30k steps):
- 开始: 2025-12-26 17:34 (step 500)
- 结束: 2025-12-27 13:13 (step 29999)
- 总时间: ~20 小时
- 速度: ~1.42 秒/step
```

### 5.2 Loss 收敛

```
Step 0:     loss=? (初始)
Step 29600: loss=0.0036
Step 29900: loss=0.0035
```

最终 loss 收敛到 ~0.0035。

### 5.3 模型大小

Pi0 checkpoint 大小: ~12 GB (每个 step)

---

## 6. 与我们的方案对比

| 方面 | 云帆 | 我们 |
|------|------|------|
| **模型** | Pi0 | Pi0.5 |
| **模型大小** | ~3GB | ~14GB |
| **FSDP** | 不需要 | 需要 (8卡分片) |
| **batch_size** | 256 | 64→128 |
| **训练时间** | ~20小时 | ~10小时 (预计) |
| **预训练权重** | pi0_base | pi05_base |

### 6.1 Pi0 vs Pi0.5 差异

```python
# Pi0
Pi0Config()  # 3 个子模块

# Pi0.5
Pi0Config(pi05=True)  # 5 个子模块 (多了 time_mlp_in, time_mlp_out)
```

**不兼容性**: Pi0 和 Pi0.5 的 checkpoint 不能互用。

### 6.2 我们需要调整的

1. **使用 Pi0.5 权重**: `checkpoints/pi05_base/params`
2. **启用 FSDP**: `fsdp_devices=8` (解决 OOM)
3. **减小 batch_size**: 256 → 128 或 64

---

## 7. 关键发现

### 7.1 成功因素

1. **SFT 微调是必须的**: Zero-shot 成功率为 0%
2. **充足的训练步数**: 30k steps
3. **合适的数据格式**: LeRobot 标准格式
4. **正确的 prompt**: 从数据集读取任务描述

### 7.2 失败原因分析

为什么某些任务成功率低:

| 任务 | 成功率 | 可能原因 |
|------|--------|----------|
| PullCube | 87.5% | 简单任务，拉动动作直接 |
| PickCube | 2.5% | 需要精确抓取，对齐困难 |
| PegInsertion | 0% | 精度要求极高，force feedback 缺失 |

### 7.3 数据量

每个任务约 200 条轨迹，这是云帆使用的数据规模。

---

## 8. 评估命令示例

```bash
# 云帆的评估命令
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDA_VISIBLE_DEVICES=0 \
python scripts/eval_maniskill.py \
    --config-name pi0_maniskill_stackcube \
    --exp-name stackcube \
    --env-id StackCube-v1 \
    --checkpoint-step 29999 \
    --num-episodes 40 \
    --save-video True \
    --prompt "Pick up a red cube and stack it on top of a green cube..."
```

---

## 9. 对我们的启示

### 9.1 可以复用

- ✅ 数据转换流程
- ✅ 评估脚本结构
- ✅ 训练超参数 (lr, warmup, etc.)
- ✅ 任务 prompt

### 9.2 需要修改

- ❌ Pi0 → Pi0.5 配置
- ❌ 添加 FSDP 支持
- ❌ 调整 batch_size

### 9.3 预期结果

- 云帆 Pi0 最佳: PullCube 87.5%
- 我们 Pi0.5 应该类似或更好 (更大的模型)

---

## 10. 总结

云帆的工作证明了:

1. **VLA 在仿真中可行**: 通过 SFT 可以适应 ManiSkill 环境
2. **数据量要求适中**: 200 条轨迹/任务足够
3. **训练时间合理**: 单任务 ~20 小时
4. **成功率有上限**: 最高 87.5%，复杂任务仍然困难

这为我们的 Pi0.5 训练提供了清晰的 baseline 和参考。
