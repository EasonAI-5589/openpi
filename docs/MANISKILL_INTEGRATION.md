# Pi0.5 + ManiSkill 集成文档

> 记录将 Pi0.5 集成到 ManiSkill3 仿真环境的完整过程

## 概述

**目标**：让 Pi0.5 模型能在 ManiSkill3 仿真环境中运行并评估

**参考**：云帆的 caurft 仓库 (`/share/project/yunfan/RL/caurft/openpi/`)，他已实现 Pi0 + ManiSkill

**关键区别**：
- 云帆用 **Pi0**，我们用 **Pi0.5**
- 数据转换格式相同，模型配置不同

---

## 文件修改清单

### 1. 新增文件

| 文件路径 | 来源 | 说明 |
|----------|------|------|
| `examples/maniskill/convert_maniskill_data_to_lerobot.py` | 云帆 caurft | ManiSkill HDF5 → LeRobot 格式转换 |
| `scripts/eval_maniskill.py` | 云帆 caurft | ManiSkill 环境评估脚本 |
| `src/openpi/policies/libero_policy_no_wrist.py` | 云帆 caurft | 无 wrist camera 的策略类 |

### 2. 修改文件

#### `src/openpi/training/config.py`

**修改内容**：
1. 添加 import：`import openpi.policies.libero_policy_no_wrist as libero_policy_no_wrist`
2. 添加两个 DataConfig 类：
   - `LeRobotManiskillDataConfig` - 有 wrist camera
   - `LeRobotManiskillNoWristDataConfig` - 无 wrist camera
3. 添加三个训练配置：
   - `pi05_maniskill_stackcube`
   - `pi05_maniskill_pickcube`
   - `pi05_maniskill_pushcube`

**配置示例**：
```python
TrainConfig(
    name="pi05_maniskill_stackcube",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=20, discrete_state_input=False),
    data=LeRobotManiskillDataConfig(
        repo_id="/share/project/guoyichen/maniskill_lerobot/maniskill_stackcube",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    batch_size=256,
    weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/pi05_base_hf"),
    num_train_steps=30_000,
    save_interval=500,
    keep_period=500,
)
```

#### `scripts/eval_maniskill.py`

**修改内容**：
- `_find_checkpoint_dir()` 函数：支持 HuggingFace checkpoint 格式 (`model.safetensors`)，不仅限于 OpenPI 原生格式 (`params/`)

---

## 数据转换

### 原始数据位置
```
/share/project/zooy/mani_data/StackCube-v1/trajectory.h5
```

### 转换命令
```bash
cd /share/project/guoyichen/openpi
uv run python examples/maniskill/convert_maniskill_data_to_lerobot.py \
  --h5-file="/share/project/zooy/mani_data/StackCube-v1/trajectory.h5" \
  --task-name=stackcube
```

### 输出
```
/share/project/guoyichen/maniskill_lerobot/maniskill_stackcube/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── ...
│       └── episode_000199.parquet
├── meta/
│   └── info.json
└── norm_stats.json
```

**统计**：
- 200 条轨迹
- 3.5GB 总大小
- 包含 norm_stats 归一化统计

---

## 评估命令

### Zero-shot 评估 (不训练直接测试)
```bash
cd /share/project/guoyichen/openpi
PYTHONPATH=src:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 \
/share/project/guoyichen/miniconda3/envs/openpi/bin/python scripts/eval_maniskill.py \
  --config-name pi05_maniskill_stackcube \
  --checkpoint-dir checkpoints/pi05_base_hf \
  --env-id StackCube-v1 \
  --num-episodes 10 \
  --prompt "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling" \
  --save-video False
```

### SFT 训练后评估
```bash
cd /share/project/guoyichen/openpi
PYTHONPATH=src:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 \
/share/project/guoyichen/miniconda3/envs/openpi/bin/python scripts/eval_maniskill.py \
  --config-name pi05_maniskill_stackcube \
  --checkpoint-dir checkpoints/pi05_maniskill_stackcube/<exp_name> \
  --checkpoint-step <step> \
  --env-id StackCube-v1 \
  --num-episodes 50 \
  --save-video True
```

---

## 训练命令

```bash
cd /share/project/guoyichen/openpi
CUDA_VISIBLE_DEVICES=0,1 \
/share/project/guoyichen/miniconda3/envs/openpi/bin/python -m openpi.training.train \
  pi05_maniskill_stackcube \
  --exp-name=stackcube_sft_v1
```

---

## 评估结果

| 模型 | 配置 | 成功率 | 备注 |
|------|------|--------|------|
| Pi0.5 base | zero-shot | 0/10 = 0% | 预期内，未见过 ManiSkill 数据 |
| Pi0.5 SFT | 待训练 | - | 待评估 |
| Pi0 SFT (云帆) | 19k steps | 0/10 = 0% | 参考，可能 checkpoint 问题 |

---

## 与云帆进度对比

| 项目 | 云帆 (Pi0) | 我们 (Pi0.5) |
|------|-----------|--------------|
| 数据转换 | 10 个任务 | 1 个 (StackCube) |
| SFT 训练 | ✅ 已完成多个 | ❌ 未开始 |
| 模型 | Pi0 | Pi0.5 |

---

## 环境依赖

- Python 3.11+ (conda env: `/share/project/guoyichen/miniconda3/envs/openpi`)
- ManiSkill3: `mani_skill==3.0.0b22`
- OpenPI 依赖 (见 `pyproject.toml`)

---

## 待办

- [ ] 运行 Pi0.5 SFT 训练
- [ ] 评估训练后效果
- [ ] 转换更多任务数据 (PickCube, PushCube, etc.)
- [ ] 对比 Pi0 vs Pi0.5 效果差异
