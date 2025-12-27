# 云帆 ManiSkill 训练方案分析

> 分析云帆的 Pi0 + ManiSkill 训练方案，为 Pi0.5 迁移提供参考

## 1. 云帆的仓库结构

```
/share/project/yunfan/RL/caurft/openpi/
├── checkpoints/
│   ├── pi0_base/                    # Pi0 预训练权重 (从 GCS 下载)
│   │   └── openpi/openpi-assets/checkpoints/pi0_base/params/
│   ├── pi0_maniskill_stackcube/     # StackCube 训练后的 checkpoint
│   │   ├── stackcube/               # 实验 1
│   │   ├── stackcube_42/            # 实验 2 (seed=42?)
│   │   ├── stackcube_full/          # 实验 3
│   │   ├── stackcube_new/           # 实验 4
│   │   └── stackcube_new_stats/     # 实验 5
│   ├── pi0_maniskill_pickcube/
│   ├── pi0_maniskill_pushcube/
│   ├── pi0_maniskill_pullcube/
│   ├── pi0_maniskill_placesphere/
│   └── ...
├── src/openpi/training/config.py    # 训练配置
└── scripts/
    ├── train.py                     # 训练脚本
    └── eval_maniskill.py            # 评估脚本
```

## 2. 云帆的训练配置

### 2.1 StackCube 配置示例

```python
TrainConfig(
    name="pi0_maniskill_stackcube",
    model=pi0_config.Pi0Config(action_horizon=20),  # Pi0，不是 Pi0.5
    data=LeRobotManiskillDataConfig(
        repo_id="/share/project/yunfan/RL/maniskill_lerobot/maniskill_stackcube",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    batch_size=256,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10_000,
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/share/project/yunfan/RL/caurft/openpi/checkpoints/pi0_base/openpi/openpi-assets/checkpoints/pi0_base/params"
    ),
    num_train_steps=30_000,
    save_interval=500,
    keep_period=500,
)
```

### 2.2 关键配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `model` | `Pi0Config(action_horizon=20)` | **Pi0 模型**，不是 Pi0.5 |
| `batch_size` | 256 | 批大小 |
| `num_train_steps` | 30,000 | 总训练步数 |
| `save_interval` | 500 | 每 500 步保存 |
| `peak_lr` | 5e-5 | 学习率峰值 |
| `warmup_steps` | 10,000 | 预热步数 |

### 2.3 云帆训练的任务列表

| 任务 | config_name | action_horizon | batch_size |
|------|-------------|----------------|------------|
| StackCube | pi0_maniskill_stackcube | 20 | 256 |
| PickCube | pi0_maniskill_pickcube | 10 | 256 |
| PushCube | pi0_maniskill_pushcube | 10 | 16 |
| PullCube | pi0_maniskill_pullcube | 10 | 16 |
| PlaceSphere | pi0_maniskill_placesphere | 10 | 256 |
| PegInsertionSide | pi0_maniskill_peginsertionside | 10 | 16 |

## 3. 训练命令

```bash
cd /share/project/yunfan/RL/caurft/openpi

# 训练命令格式
uv run python scripts/train.py <config_name> --exp-name=<experiment_name>

# 示例：训练 StackCube
uv run python scripts/train.py pi0_maniskill_stackcube --exp-name=stackcube

# Checkpoint 保存位置
# checkpoints/<config_name>/<exp_name>/<step>/
```

## 4. 训练时间估算

根据云帆的 checkpoint 时间戳分析：

```
stackcube 实验:
- 500 步: 2025-12-27 01:34
- 9500 步: 2025-12-27 05:08
- 9000 步耗时约 3.5 小时
- 速率: ~43 分钟 / 1000 步
```

**预估 30,000 步总时间: ~20-22 小时**

## 5. Pi0 vs Pi0.5 关键差异

### 5.1 模型结构差异

| 特性 | Pi0 | Pi0.5 |
|------|-----|-------|
| 子模块数量 | 3 | 5 (多了 time_mlp_in, time_mlp_out) |
| 配置参数 | `Pi0Config()` | `Pi0Config(pi05=True)` |
| 预训练权重 | `pi0_base` | `pi05_base` |
| 模型大小 | ~3GB | ~14GB |

### 5.2 不兼容性验证

尝试用 Pi0 checkpoint 加载到 Pi0.5 配置会报错：
```
ValueError: PyTrees have different structure:
   - expected 5 children, got 3 children
   - symmetric difference: {'time_mlp_in', 'time_mlp_out'}
```

**结论：Pi0 和 Pi0.5 的 checkpoint 不能互用**

## 6. 对我们的启示

### 6.1 我们需要做的

1. **使用 Pi0.5 预训练权重**: `checkpoints/pi05_base_hf`
2. **修改配置为 Pi0.5**: `Pi0Config(pi05=True, action_horizon=20, discrete_state_input=False)`
3. **重新训练**: 不能复用云帆的 checkpoint，需要从 pi05_base 开始 SFT

### 6.2 可以复用的

- ✅ 数据转换脚本 (HDF5 → LeRobot)
- ✅ 评估脚本 (eval_maniskill.py)
- ✅ 训练配置模板 (lr, batch_size, steps 等)
- ✅ 数据源 (ManiSkill demo 数据)

### 6.3 不能复用的

- ❌ Pi0 预训练权重
- ❌ Pi0 训练后的 checkpoint
- ❌ 需要重新训练 Pi0.5

## 7. 评估结果对比

| 模型 | 配置 | 成功率 | 备注 |
|------|------|--------|------|
| Pi0 base | zero-shot | N/A | 结构不兼容，无法加载 |
| Pi0 SFT (云帆) | 19k steps | 0/10 = 0% | checkpoint 可能有问题 |
| Pi0.5 base | zero-shot | 0/10 = 0% | 预期内，未见过 ManiSkill |
| Pi0.5 SFT | 待训练 | - | 下一步 |

## 8. 下一步计划

1. 启动 Pi0.5 SFT 训练:
   ```bash
   cd /share/project/guoyichen/openpi
   CUDA_VISIBLE_DEVICES=0,1 uv run python scripts/train.py pi05_maniskill_stackcube --exp-name=stackcube_sft_v1
   ```

2. 训练完成后评估

3. 与云帆的 Pi0 结果对比
