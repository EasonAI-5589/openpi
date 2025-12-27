# 郭奕辰 - OpenPi 实习任务清单

> 来源：2025年12月25日会议纪要（程洪洋、娄云帆、张书逸）
> 开始时间：2025年12月26日

---

## 🔧 Pi0 → Pi0.5 迁移方案

### 核心差异

| 配置项 | Pi0 | Pi0.5 | 说明 |
|--------|-----|-------|------|
| `pi05` | `False` | `True` | 主开关 |
| `max_token_len` | 48 | 200 | Token 序列长度 |
| `discrete_state_input` | `False` | `True` | 状态输入方式 |
| Checkpoint | `pi0_base` | `pi05_base` | 预训练权重路径 |
| Action Expert | 标准 RMSNorm | adaRMSNorm | 注入 flow matching timestep |

### 迁移步骤

#### Step 1: 修改 Config 文件
找到 `src/openpi/training/config.py`，复制一份 Pi0 的 config（如 `pi0_libero`），改为：

```python
@register_config
def pi05_custom() -> TrainConfig:
    return TrainConfig(
        name="pi05_custom",
        model=pi0_config.Pi0Config(
            pi05=True,  # 关键：启用 Pi0.5 模式
            # max_token_len 和 discrete_state_input 会自动设置
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"  # Pi0.5 权重
        ),
        # ... 其他配置
    )
```

#### Step 2: 数据预处理（重要）
Pi0.5 使用 **分位数归一化**，需要运行：
```bash
python scripts/compute_norm_stats.py --config pi05_custom
```

#### Step 3: 训练命令
```bash
python scripts/train.py --config pi05_custom
```

#### Step 4: 推理验证
```bash
python scripts/serve_policy.py --config pi05_custom
```

### 关键文件位置

```
src/openpi/
├── models/
│   ├── pi0.py           # Pi0/Pi0.5 模型实现
│   ├── pi0_config.py    # Pi0Config 定义（pi05 参数在这里）
│   └── gemma.py         # Action Expert 实现
├── training/
│   └── config.py        # 所有训练配置（在这里创建新 config）
└── transforms/
    └── transforms.py    # 数据预处理（归一化方式）
```

---

## 📋 当前任务

### 1. 基础配置修改
- [x] 查找 Pi 0.5 的 config 文件 → `src/openpi/training/config.py`
- [ ] 基于娄云帆共享的 config 进行修改
- [x] 对比官方仓库的改动处 → 见上方迁移方案
- [ ] 将数据集替换成娄云帆造好的数据集
- [ ] 下载 Pi 0.5 的 base（等娄云帆完成下载后获取）

### 2. 仿真任务（上手任务）
- [ ] 完成 6 个在 **ManiSkill** 上的仿真任务
- [ ] 将 Pi 0 接 ManiSkill 的仿真 → 换成 Pi 0.5 接 ManiSkill 的仿真
- [ ] 完成 Pi 0.5 在 ManiSkill Benchmark 的跑分

> **注**：ManySkill/ManyScore 实际上是 **ManiSkill**（SAPIEN Manipulation Skill Framework）
> - GitHub: https://github.com/haosulab/ManiSkill
> - GPU 并行机器人操作仿真器和基准测试平台

#### ManiSkill 集成调研（2025-12-27）

**结论**：openpi 官方**不直接支持 ManiSkill**，只支持 LIBERO、ALOHA Sim、DROID。

**✅ 已完成 Pi0.5 + ManiSkill3 集成**（2025-12-27）

**集成架构**：
```
ManiSkill3 环境 (PickCube-v1, StackCube-v1, etc.)
    ↓ (obs: RGBD + qpos/qvel)
ManiSkillInputs Transform (src/openpi/policies/maniskill_policy.py)
    ↓ (转换为 Pi0.5 输入格式: image dict + state)
Pi0.5 Model Inference
    ↓ (50-step action chunks, 32-dim)
ManiSkillOutputs Transform
    ↓ (转换为 7D actions: dx, dy, dz, dax, day, daz, gripper)
ManiSkill3 env.step()
```

**新增文件**：
- `src/openpi/policies/maniskill_policy.py` - ManiSkill transforms
- `src/openpi/maniskill/pi05_maniskill_adapter.py` - 适配器（备用）
- `src/openpi/maniskill/pi05_maniskill_evaluator.py` - 评估循环
- `scripts/test_maniskill_integration.py` - 测试脚本
- `src/openpi/training/config.py` - 新增 `pi05_maniskill` 和 `pi05_maniskill_droid` 配置

**ManiSkill3 全部任务结构图**：

```
ManiSkill3 (34 environments)
│
├── 🧊 Cube Manipulation (基础方块操作)
│   ├── [✓] PickCube-v1          - 云帆 Pi0:  1/40 =  2.50%
│   ├── [✓] StackCube-v1         - 云帆 Pi0: 24/40 = 60.00%
│   ├── [✓] PushCube-v1          - 云帆 Pi0: 28/40 = 70.00%
│   ├── [✓] PullCube-v1          - 云帆 Pi0: 35/40 = 87.50%
│   ├── [✓] PullCubeTool-v1      - 云帆 Pi0:  3/40 =  7.50%
│   ├── [ ] PokeCube-v1
│   ├── [ ] TwoRobotPickCube-v1
│   └── [ ] TwoRobotStackCube-v1
│
├── 🔴 Sphere/Object Placement
│   ├── [✓] PlaceSphere-v1       - 云帆 Pi0: 11/40 = 27.50%
│   └── [ ] UnitreeG1PlaceAppleInBowl-v1
│
├── 🔧 Precision Tasks (精密操作)
│   ├── [ ] PegInsertionSide-v1
│   ├── [ ] PlugCharger-v1
│   └── [ ] LiftPegUpright-v1
│
├── 🚿 Articulated Objects (关节物体)
│   ├── [ ] TurnFaucet-v1
│   ├── [ ] OpenCabinetDoor-v1
│   └── [ ] OpenCabinetDrawer-v1
│
├── 🤏 Grasping Tasks (抓取任务)
│   ├── [ ] PickSingleYCB-v1
│   └── [ ] PickClutterYCB-v1
│
├── 🎮 Push Tasks
│   └── [ ] PushT-v1
│
├── 🤖 Dexterous Hand (灵巧手)
│   ├── [ ] RotateSingleObjectInHandLevel0-v1
│   ├── [ ] RotateSingleObjectInHandLevel1-v1
│   ├── [ ] RotateSingleObjectInHandLevel2-v1
│   └── [ ] RotateSingleObjectInHandLevel3-v1
│
├── 🔄 Valve Rotation
│   ├── [ ] RotateValveLevel0-v1
│   ├── [ ] RotateValveLevel1-v1
│   ├── [ ] RotateValveLevel2-v1
│   ├── [ ] RotateValveLevel3-v1
│   └── [ ] RotateValveLevel4-v1
│
├── 🖐️ TriFinger (三指机器人)
│   ├── [ ] TriFingerRotateCubeLevel0-v1
│   ├── [ ] TriFingerRotateCubeLevel1-v1
│   ├── [ ] TriFingerRotateCubeLevel2-v1
│   ├── [ ] TriFingerRotateCubeLevel3-v1
│   └── [ ] TriFingerRotateCubeLevel4-v1
│
└── 🎨 Scene Tasks
    └── [ ] StackGreenCubeOnYellowCubeBakedTexInScene-v1

Legend: [✓] = 云帆 Pi0 已评估  [ ] = 未评估
```

**云帆 Pi0 评估的 6 个任务**：
- PlaceSphere-v1: 11/40 = 27.50%
- PickCube-v1: 1/40 = 2.50%
- StackCube-v1: 24/40 = 60.00%
- PushCube-v1: 28/40 = 70.00%
- PullCube-v1: 35/40 = 87.50%
- PullCubeTool-v1: 3/40 = 7.50%

**评估结果（40 episodes/task）**：

| Task | Pi0.5 (ours) | Pi0 (云帆) |
|------|--------------|------------|
| PlaceSphere-v1 | 0/40 = 0.00% | 11/40 = 27.50% |
| PickCube-v1 | 0/40 = 0.00% | 1/40 = 2.50% |
| StackCube-v1 | 0/40 = 0.00% | 24/40 = 60.00% |
| PushCube-v1 | 0/40 = 0.00% | 28/40 = 70.00% |
| PullCube-v1 | 0/40 = 0.00% | 35/40 = 87.50% |
| PullCubeTool-v1 | 0/40 = 0.00% | 3/40 = 7.50% |
| PegInsertionSide-v1 | 0/40 = 0.00% | N/A |
| PlugCharger-v1 | 0/40 = 0.00% | N/A |
| TurnFaucet-v1 | 0/40 = 0.00% | N/A |

**注意**：base 模型未针对 ManiSkill 任务微调，低成功率是预期的。需要在 ManiSkill 任务示范上微调以获得更好效果。

---

### 🔬 0% 成功率诊断分析（2025-12-27）

**问题**：Pi0.5 在 ManiSkill3 上的成功率始终为 0%，而云帆的 Pi0 能达到 60-87.5%。

#### 诊断脚本
- `scripts/debug_action_space.py` - 分析 action 范围和 magnitude
- `scripts/debug_observation.py` - 分析 observation 对齐问题

#### 诊断结论

**1. Gripper 始终打开（核心问题！）**
```
Gripper values: min=-1.0000, max=-1.0000
Unique gripper values: [-1.]
```
- Pi0.5 输出的 gripper 始终是 -1（打开状态）
- PickCube 等任务需要 gripper=1（闭合）才能抓取
- **这是 0% 成功率的直接原因！**

**2. State 不对齐**
```
Pi0.5 期望 (DROID 格式):          ManiSkill 提供:
- EEF pose (7D: xyz + quat)       - Joint positions (9D: qpos)
- Gripper state                    - Joint velocities (9D: qvel)
- 3 个相机视角                     - 1-2 个相机视角
```
- Pi0.5 训练数据使用 **EEF pose**（末端执行器位姿）
- 我们给的是 **Joint positions**（关节角度）
- 模型完全不理解输入的含义！

**3. 好消息：ManiSkill 提供了 EEF pose**
```python
obs["extra"]["tcp_pose"]  # tensor shape=[1, 7] (xyz + quaternion)
```
但我们没有使用它！

**4. Action 范围看起来正常**
- Mean action magnitude = 0.303（在 [-1, 1] 范围内合理）
- 不是 action scale 的问题

#### 解决方案

**方案 A：修改 State 输入（推荐）**
```python
# 在 ManiSkillInputs transform 中：
# 现在：使用 qpos + qvel (18D)
state = np.concatenate([qpos, qvel])

# 改为：使用 tcp_pose + gripper (8D)
tcp_pose = obs["extra"]["tcp_pose"]  # 7D
gripper_state = obs["agent"]["qpos"][:, 7:9].mean()  # 1D
state = np.concatenate([tcp_pose, [gripper_state]])
```

**方案 B：在 ManiSkill 数据上微调**
- 收集 ManiSkill 上的专家轨迹
- 使用 ManiSkill 的 observation 格式微调 Pi0.5
- 这是云帆 Pi0 成功的原因！

**方案 C：使用不同的 control mode**
```python
# 现在：pd_ee_delta_pose (7D delta)
# 可尝试：pd_joint_delta_pos (关节空间 delta)
```

#### 下一步行动

1. [ ] **修改 ManiSkillInputs transform** - 使用 tcp_pose 替代 qpos
2. [ ] **测试修复后的成功率**
3. [x] **分析云帆的代码** - 找到他们的实现细节

---

### 🔍 云帆代码分析结果（2025-12-27）

**代码位置**：`/share/project/yunfan/RL/caurft/`

#### 关键发现 1：云帆使用微调后的模型！

云帆的代码是一个**完整的 RL 微调框架**（CalQL + Pi0），不是直接用 base 模型评估！

```
/share/project/yunfan/RL/caurft/
├── openpi/                     # 修改过的 openpi（加了 ManiSkill 支持）
├── example/
│   └── train_main_sim.py      # 主训练脚本（79KB！非常复杂）
└── jaxrl_m/
    └── envs/maniskill.py      # ManiSkill Wrapper
```

**训练流程**：
1. 离线预训练（CalQL on demo data）
2. 在线微调（RL + demo 混合）
3. 评估

#### 关键发现 2：State 格式

云帆的代码**确实使用了 qpos 前 8 维**，但他们是在**微调数据**中统一了这个格式：

```python
# 云帆的 convert_maniskill_data_to_lerobot.py
"state": qpos[t, :8],  # Take first 8 dimensions
"actions": actions[t],  # shape: (7,)
```

评估脚本也使用同样的格式：
```python
# 云帆的 eval_maniskill.py
state8 = np.asarray(qpos[:8], dtype=np.float32)
```

#### 关键发现 3：LiberoInputs Transform

云帆的配置使用 `LeRobotManiskillDataConfig`，它复用了 `LiberoInputs`：
```python
data_transforms = _transforms.Group(
    inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
    outputs=[libero_policy.LiberoOutputs()],
)
```

LiberoInputs 期望的输入格式：
- `observation/state`: 8D float32（前 8 维 qpos）
- `observation/image`: base camera RGB
- `observation/wrist_image`: wrist camera RGB（如果没有就用 zeros）

#### 关键发现 4：评估脚本

云帆有专门的评估脚本：
```bash
python /share/project/yunfan/RL/caurft/openpi/scripts/eval_maniskill.py \
    --config-name pi0_maniskill \
    --checkpoint-dir /path/to/trained/checkpoint \
    --env-id StackCube-v1 \
    --num-episodes 40
```

#### 为什么云帆成功率高？

| 因素 | 我们（Pi0.5 base） | 云帆（Pi0 微调） |
|------|-------------------|-----------------|
| 模型 | 未微调的 base 模型 | 在 ManiSkill 示范上微调 |
| 训练 | 无 | CalQL + RL 在线微调 |
| 数据 | 无 ManiSkill 数据 | 使用 ManiSkill 专家轨迹 |
| State | 我们用 18D (qpos+qvel) | 他们用 8D (qpos[:8]) |

**结论**：云帆的高成功率来自于**在 ManiSkill 数据上微调**，而不是 zero-shot！

#### 下一步行动（更新）

1. [ ] **统一 State 格式为 8D** - 与云帆一致
2. [ ] **收集 ManiSkill 专家数据** - 用于微调
3. [ ] **参考云帆的训练流程** - CalQL + Pi0 微调
4. [ ] **对比 zero-shot vs fine-tuned** - 理解差距来源

**可参考的集成方案**：

| 项目 | 说明 | 链接 |
|------|------|------|
| **云帆的 caurft** | CalQL + Pi0 + ManiSkill 微调 | `/share/project/yunfan/RL/caurft/` |
| **open-pi-zero** | Pi0 重实现，支持 SimplerEnv + ManiSkill2 | https://github.com/allenzren/open-pi-zero |
| **SimplerEnv** | Real2Sim 评估框架，包含 ManiSkill2_real2sim | https://github.com/DelinQu/SimplerEnv-OpenVLA |
| **VLABench** | VLA 评估基准，支持 Pi0/Pi0.5 | https://github.com/OpenMOSS/VLABench |

### 3. 架构更改任务
- [ ] 将当前架构的动作头换成简单的 Flow Matching 头（参照 Pi 0 的设计）
- [ ] 微调参数：当前 loss 是 diffusion loss + a + q loss
- [ ] 微调 q 使其适配 Flow Matching loss

### 4. 学习准备
- [ ] 阅读 CRFT 论文（程洪洋推荐）
- [ ] 熟悉泰玲的代码（基于 CRFT 改进）

---

## ⏳ 娄云帆需要提供的前置材料

> **发给云帆确认进度用**

### 必须材料（阻塞我开始工作）

| # | 材料 | 说明 | 我拿到后做什么 | 状态 |
|---|------|------|----------------|------|
| 1 | **你修改过的 config 文件** | 基于 Pi 0 改的那份 | 对比官方改动，理解你们的定制化配置 | ⏳ |
| 2 | **造好的数据集** | 替换官方数据集用 | 在 config 里指向这个数据集路径 | ⏳ |
| 3 | **Pi 0.5 base 权重** | 你说 12/25 下载完成 | 放到集群上，config 指向这个路径 | ⏳ |
| 4 | **ManiSkill 仿真代码** | 6 个仿真任务的代码 | 跑通后改成 Pi 0.5 | ⏳ |
| 5 | **Pi 0 接 ManiSkill 仿真代码** | 现有的 Pi 0 版本 | 改成 Pi 0.5 版本 | ⏳ |

### 建议材料（加速我上手）

| # | 材料 | 说明 | 状态 |
|---|------|------|------|
| 6 | **详细任务说明文档** | 包括修改位置、参数、预期结果 | ⏳ |
| 7 | **Pi 0 的 Flow Matching 设计参考** | 架构任务要用 | ⏳ |

### 依赖关系

```
云帆提供 config + 数据集 + 权重
         ↓
    我修改 config，跑通训练
         ↓
云帆提供 ManiSkill 仿真代码
         ↓
    我跑通 6 个仿真任务
         ↓
云帆提供 Pi0 接 ManiSkill 代码
         ↓
    我改成 Pi0.5 版本并跑分
```

---

## ⏳ 程洪洋需要提供

| 待办 | 状态 | 备注 |
|------|------|------|
| 有卡的集群账号 | ⏳ | 12/26 上午 |

---

## 🎯 后续规划

- 如果前两项任务效果好 → 会有进一步迁移任务
- **Robobrain 项目**: Robot 这边的 VLM 通用基座项目
  - 目标：冲击顶尖期刊
  - 内心目标：Science

---

## 📝 进度记录

### 2025-12-26
- [x] Fork openpi 仓库到个人账户
- [x] 创建 TODO.md 任务清单
- [x] 调研 Pi0 → Pi0.5 迁移方案
- [x] 确认 ManiSkill 基准测试平台

### 2025-12-27
- [x] 配置商庄服务器环境（Conda + 代理）
- [x] 部署 openpi 环境（PyTorch 2.7.1 + JAX 0.5.3）
- [x] 调研 ManiSkill 集成方案 → 官方不支持，需要 SimplerEnv 适配层
- [x] **配置 HuggingFace 镜像**（hf-mirror.com）加速下载
- [x] **下载 pi05_base 模型** → `checkpoints/pi05_base_hf/` (14GB)
- [x] **下载 pi05_droid 模型** → `checkpoints/pi05_droid_hf/` (6.8GB)
- [x] **pi05_base PyTorch 推理测试成功** ✅
- [x] **下载 pi05_libero 模型** → `checkpoints/pi05_libero/` (12GB, GCS)
- [x] **LIBERO Benchmark 评估完成** ✅

#### LIBERO Benchmark 评估结果（2025-12-27）

**Task Suite**: `libero_spatial` (10 tasks × 5 trials = 50 episodes)

| 指标 | 结果 |
|------|------|
| **总成功率** | **96%** (48/50) |
| **官方报告** | 98.8% |
| **评估时间** | 5分26秒 |

**各任务成功率**：
| Task | 成功率 |
|------|--------|
| Task 1-8 | 100% (5/5) |
| Task 9 | 80% (4/5) |
| Task 10 | 80% (4/5) |

**运行环境**：
- Policy Server: `uv run scripts/serve_policy.py --env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir checkpoints/pi05_libero`
- Client: `python examples/libero/main.py --args.task-suite-name libero_spatial --args.num-trials-per-task 5`
- Conda 环境: `libero` (Python 3.8)

**对比 ManiSkill3**：
| Benchmark | 成功率 | 备注 |
|-----------|--------|------|
| LIBERO (libero_spatial) | **96%** | 使用 fine-tuned checkpoint |
| ManiSkill3 (6 tasks) | 0% | Zero-shot，无 ManiSkill 微调数据 |

**结论**：Pi0.5 在有微调数据的 LIBERO 上表现优异，接近官方报告水平。

#### 推理测试结果（2025-12-27）

**pi05_base 模型推理成功**：

| 指标 | 结果 |
|------|------|
| 动作形状 | `[1, 50, 32]` (batch=1, chunk=50步, dim=32维) |
| 动作范围 | `[-0.26, 0.56]` |
| 显存使用 | **14.49 GB** |
| 峰值显存 | **14.83 GB** |
| 首次推理耗时 | ~7分钟（Triton 自动调优内核） |

**注意事项**：
1. 首次推理较慢是正常的 - PyTorch/Triton 在进行 AUTOTUNE
2. 后续推理会快很多（内核已缓存）
3. 需要安装 `transformers_replace`：
   ```bash
   TRANSFORMERS_PATH=$(python -c "import transformers; print(transformers.__path__[0])")
   cp -r ./src/openpi/models_pytorch/transformers_replace/* "$TRANSFORMERS_PATH/"
   ```
4. Weight tying：`embed_tokens = lm_head.weight`（HF 模型需要手动处理）

#### 模型下载说明（2025-12-27）

**问题**：GCS (Google Cloud Storage) 下载速度极慢（~240KB/s）

**解决方案**：使用 HuggingFace 镜像（hf-mirror.com）

```bash
# 配置 HuggingFace 镜像
export HF_ENDPOINT="https://hf-mirror.com"

# 下载模型（使用 PyTorch 格式）
from huggingface_hub import snapshot_download
snapshot_download('lerobot/pi05_base', local_dir='checkpoints/pi05_base_hf')
snapshot_download('s3y/pi05_droid_pytorch', local_dir='checkpoints/pi05_droid_hf')
```

**下载速度对比**：
| 来源 | 速度 | 21GB 下载时间 |
|------|------|---------------|
| GCS | ~240KB/s | ~25小时 |
| HF 镜像 | ~30MB/s | **~12分钟** |

**模型路径**：
```
checkpoints/
├── pi05_base_hf/           # 14GB
│   ├── model.safetensors
│   ├── config.json
│   └── ...
└── pi05_droid_hf/          # 6.8GB
    ├── model.safetensors
    ├── config.json
    └── ...

---

## 🔗 相关资源

- **仓库地址**: https://github.com/EasonAI-5589/openpi
- **官方仓库**: https://github.com/Physical-Intelligence/openpi
- **会议纪要**: https://jwolpxeehx.feishu.cn/docx/NdiNdlHobooUZYxBo95ckA0fn2e
- **ManiSkill**: https://github.com/haosulab/ManiSkill

---

## 💬 交流沟通

- 有问题在群里直接发
- 多与娄云帆和书逸交流

---

## 📚 Preliminary: Diffusion vs Flow Matching

> 架构更改任务的背景知识：理解为什么要把动作头换成 Flow Matching

### 1. 问题背景：机器人动作生成

机器人控制需要生成**连续的动作序列**（如关节角度、末端执行器位置）。传统方法直接回归动作，但存在问题：
- 动作分布是**多模态**的（同一个任务可能有多种完成方式）
- 需要生成**平滑连续**的轨迹
- 要能处理**不确定性**

**生成模型**（Diffusion/Flow Matching）可以解决这些问题：从噪声中"生成"动作。

---

### 2. Diffusion Model（扩散模型）

#### 2.1 核心思想

```
前向过程（加噪）：x_0 → x_1 → x_2 → ... → x_T（纯噪声）
反向过程（去噪）：x_T → x_{T-1} → ... → x_0（干净数据）
```

#### 2.2 数学形式

**前向过程**（固定的马尔可夫链）：
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```
- `β_t` 是噪声调度（noise schedule），控制每步加多少噪声
- 经过 T 步后，`x_T ≈ N(0, I)`（近似标准高斯）

**反向过程**（需要学习）：
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```
- 模型学习预测 `μ_θ`（去噪后的均值）
- 实际中常用 **ε-prediction**：预测噪声 ε，然后计算 μ

#### 2.3 训练目标

```python
# DDPM 风格的训练
ε = random_noise()
x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε   # 一步加噪到 t
ε_pred = model(x_t, t)               # 预测噪声
loss = MSE(ε_pred, ε)                # 重建噪声
```

#### 2.4 采样过程

```python
x_T = random_noise()
for t in range(T, 0, -1):
    ε_pred = model(x_t, t)
    x_{t-1} = denoise_step(x_t, ε_pred, t)  # 复杂的去噪公式
return x_0
```

#### 2.5 问题

| 问题 | 说明 |
|------|------|
| **采样慢** | 需要 T 步（通常 100-1000 步） |
| **噪声调度复杂** | β_t 的设计影响很大，需要调参 |
| **数学复杂** | 涉及 SDE/ODE 理论 |

---

### 3. Flow Matching（流匹配）

#### 3.1 核心思想

不再"逐步去噪"，而是学习一个**向量场**（velocity field），直接描述从噪声到数据的"流动"方向。

```
数据点 x_0 和噪声点 x_1 之间存在一条"流动路径"
模型学习：在路径上任意一点 x_t，往哪个方向走？
```

#### 3.2 数学形式

**线性插值路径**（最简单的形式）：
```
x_t = t * x_1 + (1 - t) * x_0
    = t * noise + (1 - t) * data
```
- `t=0`：纯数据
- `t=1`：纯噪声
- `t∈(0,1)`：中间状态

**目标向量场**（真实的"流动方向"）：
```
u_t = dx_t/dt = x_1 - x_0 = noise - data
```

**模型预测**：
```
v_θ(x_t, t) ≈ u_t
```

#### 3.3 训练目标

```python
# Flow Matching 训练（极其简单！）
noise = random_noise()
t = random_uniform(0, 1)              # 随机采样时间步
x_t = t * noise + (1 - t) * data      # 线性插值
u_t = noise - data                    # 目标向量
v_t = model(x_t, t)                   # 模型预测
loss = MSE(v_t, u_t)                  # 就是这么简单
```

#### 3.4 采样过程（ODE 积分）

```python
x = noise  # 从 t=1 开始
dt = -1.0 / num_steps

for step in range(num_steps):
    t = 1.0 - step * abs(dt)
    v = model(x, t)      # 预测向量场
    x = x + dt * v       # 欧拉积分

return x  # t=0 时得到干净数据
```

---

### 4. 对比：为什么 Flow Matching 更好？

| 维度 | Diffusion | Flow Matching |
|------|-----------|---------------|
| **训练目标** | 预测噪声 ε | 预测速度 v |
| **数学复杂度** | 高（SDE/马尔可夫链） | 低（线性插值） |
| **噪声调度** | 需要设计 β_t | 不需要 |
| **采样步数** | 通常 20-100 步 | 可以 5-20 步 |
| **采样速度** | 慢 | **快** |
| **代码实现** | 复杂 | **简单** |
| **理论基础** | 去噪得分匹配 | 连续归一化流 |

---

### 5. Pi0 中的 Flow Matching 实现

#### 5.1 训练代码（`pi0.py`）

```python
def compute_loss(self, rng, observation, actions, train=False):
    # 1. 采样噪声
    noise = jax.random.normal(noise_rng, actions.shape)

    # 2. 采样时间步（Beta 分布，偏向低 t）
    time = jax.random.beta(time_rng, 1.5, 1) * 0.999 + 0.001

    # 3. 线性插值得到 x_t
    x_t = time * noise + (1 - time) * actions

    # 4. 目标向量
    u_t = noise - actions

    # 5. 模型预测
    v_t = self.forward(observation, x_t, time)

    # 6. MSE Loss
    return jnp.mean(jnp.square(v_t - u_t), axis=-1)
```

#### 5.2 采样代码（`pi0.py`）

```python
def sample_actions(self, rng, observation, num_steps=10):
    dt = -1.0 / num_steps
    x = random_noise()  # 从噪声开始
    time = 1.0

    while time >= 0:
        v_t = self.forward(observation, x, time)
        x = x + dt * v_t  # 欧拉积分
        time = time + dt

    return x
```

#### 5.3 关键设计选择

| 设计 | Pi0 的选择 | 原因 |
|------|-----------|------|
| 时间采样 | `Beta(1.5, 1)` | 偏向低 t（更接近噪声的区域更难） |
| 积分步数 | 10 步 | 足够精度，50Hz 实时性 |
| 时间注入 | 拼接到 action token | 简单有效 |

---

### 6. Pi0 vs Pi0.5 的 Flow Matching 区别

| 部分 | Pi0 | Pi0.5 |
|------|-----|-------|
| 时间步注入 | MLP 拼接 | **adaRMSNorm** |
| 状态输入 | 连续 token（suffix） | 离散 token（prefix） |

**adaRMSNorm**（自适应 RMS 归一化）：

```python
# Pi0: 时间信息通过拼接注入
action_time_tokens = concat([action_tokens, time_tokens])
action_time_tokens = MLP(action_time_tokens)

# Pi0.5: 时间信息通过调制归一化参数注入
time_emb = time_MLP(timestep)
# 在 Transformer 的每个 RMSNorm 层：
output = RMSNorm(input) * (1 + scale(time_emb)) + shift(time_emb)
```

adaRMSNorm 的好处：
- 时间信息渗透到网络的每一层
- 类似 DiT（Diffusion Transformer）的设计
- 更好的条件控制

---

### 7. 你的任务：换 Flow Matching 头

根据会议纪要，当前架构可能用的是 Diffusion（或其他变体），需要：

1. **理解当前 loss 组成**
   - `diffusion loss`：应该是 ε-prediction 的 MSE
   - `a loss` 和 `q loss`：可能是辅助任务（需要看代码确认）

2. **替换为 Flow Matching**
   - 核心改动：`ε-prediction` → `v-prediction`
   - 简化训练目标为纯 MSE
   - 可能需要调整辅助 loss 的权重

3. **参考 Pi0 的实现**
   - `src/openpi/models/pi0.py` 的 `compute_loss` 和 `sample_actions`

---

### 8. 参考资料

- **Pi0 论文**: [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
- **Flow Matching 原论文**: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- **Rectified Flow**: [Flow Straight and Fast](https://arxiv.org/abs/2209.03003)
- **openpi 源码**: `src/openpi/models/pi0.py`
