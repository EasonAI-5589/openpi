# 待讨论问题

> 与洪洋、云帆讨论

---

## 0. 🔥 为什么 OpenPI 官方不支持 ManiSkill？

### 官方支持的 Benchmark

| Benchmark | 数据格式 | 机器人 | 状态维度 | 动作维度 |
|-----------|----------|--------|----------|----------|
| **LIBERO** | RLDS (TensorFlow) | Panda | 8D | 7D |
| **DROID** | LeRobot | DROID平台 | 14D | 7D |
| **ALOHA** | LeRobot | Trossen双臂 | 14D | 14D |

**ManiSkill 不在官方支持列表中**。

### ManiSkill 不被支持的原因分析

1. **数据格式不同**
   - 官方使用: RLDS / LeRobot
   - ManiSkill 使用: **HDF5 (.h5)**
   - 需要额外的格式转换步骤

2. **没有现成的 Transforms**
   - 官方提供: `LiberoInputs`, `DroidInputs`, `AlohaInputs`
   - ManiSkill: **没有对应的 `ManiskillInputs`**

3. **State 表示差异**
   - LIBERO: `state` = 8D (qpos 前 8 维)
   - DROID: `state` = 14D (关节位置)
   - ManiSkill: `qpos` = 9D (需要取前 8 维)

4. **训练配置缺失**
   - 官方提供: `pi05_libero`, `pi05_droid`, `pi0_aloha_*`
   - ManiSkill: **需要自己创建 config**

### 云帆的解决方案（需确认）

云帆通过以下步骤让 Pi0 支持 ManiSkill：

```
1. 数据转换: ManiSkill HDF5 → LeRobot 格式
   文件: examples/maniskill/convert_maniskill_data_to_lerobot.py

2. 复用 LIBERO Transforms:
   - LiberoInputs (state=8D, action=7D)
   - LiberoOutputs

3. 创建训练配置:
   - LeRobotManiskillDataConfig (模仿 LeRobotLiberoDataConfig)
   - pi0_maniskill_* 训练配置
```

### 待确认问题

- [ ] 云帆的数据转换脚本是否可以直接用？
- [ ] 需要修改哪些配置文件？
- [ ] 是否需要重新计算 norm_stats？
- [ ] SFT 训练需要多少数据量？多长时间？

---

## 1. ManiSkill3 Zero-shot 0% 成功率

### 问题描述
使用 Pi0.5 base 模型在 ManiSkill3 上 zero-shot 评估，6 个任务全部 0% 成功率。

### 诊断结果
1. **Gripper 始终打开** - 模型输出 gripper = -1，无法抓取
2. **State 格式不匹配** - Pi0.5 期望 EEF pose (7D)，我们给的是 qpos (9D)
3. **无 ManiSkill 微调数据** - Base 模型没见过 ManiSkill 格式

### 待确认
- [ ] 云帆的 Pi0 使用的是微调后的模型？（CalQL + RL）
- [ ] 是否有现成的 ManiSkill 专家轨迹数据可用？
- [ ] State 格式应该用 `qpos[:8]` 还是 `tcp_pose + gripper`？

---

## 2. 🔥 向云帆确认：数据和训练流程

### 云帆的数据位置（已找到）

```
/share/project/zooy/mani_data/
├── StackCube-v1/motionplanning/
│   └── StackCube_New.sensor_data.pd_ee_delta_pose.physx_cpu.h5  (136GB, 200条轨迹)
├── PickCube-v1/
├── PushCube-v1/
├── PlaceSphere-v1/
├── PegInsertionSide-v1/
├── PlugCharger-v1/
├── PullCube-v1/
└── PullCubeTool-v1/
```

### 待确认问题

**数据相关**:
- [ ] 用的是哪个 `.h5` 文件？`StackCube_New.h5` 还是 `StackCube_New.sensor_data.pd_ee_delta_pose.physx_cpu.h5`？
- [ ] 数据转换命令是什么？
  ```bash
  python examples/maniskill/convert_maniskill_data_to_lerobot.py --h5_file ???
  ```
- [ ] 转换后的 LeRobot 数据集在哪里？`/share/project/yunfan/RL/maniskill_lerobot/`？

**训练相关**:
- [ ] 用的是哪个 config？`pi0_maniskill_stackcube`？
- [ ] 训练了多少步？用了几张卡？
- [ ] Checkpoint 保存在哪里？

**评估相关**:
- [ ] 评估脚本是 `scripts/eval_maniskill.py`？
- [ ] 评估命令示例？

### LIBERO vs ManiSkill 格式对比（已确认一致）

| 字段 | LIBERO | ManiSkill (云帆适配) | 一致性 |
|------|--------|---------------------|--------|
| image | 256×256×3 RGB | 256×256×3 RGB | ✅ |
| wrist_image | 256×256×3 RGB | zeros (无手腕相机) | ⚠️ |
| state | 8D (observation/state) | 8D (qpos[:8]) | ✅ |
| actions | 7D | 7D | ✅ |
| task | 从数据集读取 | 硬编码字符串 | ⚠️ |

### 复现步骤（待确认）

```bash
# 1. 数据转换
python examples/maniskill/convert_maniskill_data_to_lerobot.py \
    --h5_file /share/project/zooy/mani_data/StackCube-v1/motionplanning/???.h5

# 2. 计算归一化统计
uv run scripts/compute_norm_stats.py --config-name pi0_maniskill_stackcube

# 3. 训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_maniskill_stackcube

# 4. 评估
bash scripts/run_maniskill_eval.sh --config-name pi0_maniskill_stackcube ...
```

---

## 3. Pi0 → Pi0.5 迁移

### 待确认
- [ ] 云帆修改过的 config 文件在哪里？
- [ ] 造好的数据集路径？
- [ ] Pi0.5 base 权重是否已下载到集群？（我已下载到 `checkpoints/pi05_base_hf/`）

### 架构差异确认
| 配置项 | Pi0 | Pi0.5 |
|--------|-----|-------|
| `pi05` | False | True |
| `max_token_len` | 48 | 200 |
| Action Expert | RMSNorm | adaRMSNorm |

---

## 4. 架构更改任务

### 待确认
- [ ] 当前 loss 组成：diffusion loss + a loss + q loss，具体是什么？
- [ ] 换成 Flow Matching 头后，a loss 和 q loss 怎么调整？
- [ ] 有没有泰玲的代码可以参考？

---

## 5. 集群资源

### 待确认
- [ ] 有卡的集群账号？（洪洋说 12/26 上午提供）
- [ ] 训练任务需要多少显存？多少卡？

---

## 6. LIBERO 评估结果

### 已完成
- libero_spatial: **96%** (48/50)，官方 98.8%

### 待确认
- [ ] 是否需要跑完整的 libero_90？
- [ ] 每个任务需要多少 trials？（目前用的 5，官方用 50）
