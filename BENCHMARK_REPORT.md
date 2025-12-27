# OpenPI Benchmark 汇报

> 郭奕辰 2025-12-27
> 给云帆和洪洋的汇报材料

---

## ⚠️ 当前环境限制

| 限制 | 影响 | 状态 |
|------|------|------|
| **无 Docker** | 无法使用官方推荐的 Docker 部署 | 手动安装依赖 |
| **Python 3.8 不可用** | LIBERO 需要 Python 3.8 + PyTorch 1.11 | ❌ 需另建环境 |
| **网络慢** | GCS/HuggingFace 下载 checkpoint 慢 | 使用 HF 镜像 |
| **无 GUI** | 部分可视化功能受限 | 使用 EGL 渲染 |

**当前可运行**：
- ✅ **ManiSkill3** - 已完成集成和评估（全部 0%）
- ⚠️ **ALOHA Sim** - 依赖已装，需下载 checkpoint（~14GB，正在下载）
- ❌ **LIBERO** - 需要 Python 3.8 环境

---

## 📊 核心结论

### Pi0.5 Zero-shot 在 ManiSkill3 上的表现

| Task | Pi0.5 (我们 zero-shot) | Pi0 (云帆 微调后) | 差距原因 |
|------|------------------------|-------------------|----------|
| StackCube-v1 | **0%** (0/40) | 60% (24/40) | 云帆有微调 |
| PullCube-v1 | **0%** (0/40) | 87.5% (35/40) | 云帆有微调 |
| PushCube-v1 | **0%** (0/40) | 70% (28/40) | 云帆有微调 |
| PickCube-v1 | **0%** (0/40) | 2.5% (1/40) | 云帆有微调 |
| PlaceSphere-v1 | **0%** (0/40) | 27.5% (11/40) | 云帆有微调 |
| PullCubeTool-v1 | **0%** (0/40) | 7.5% (3/40) | 云帆有微调 |

**关键发现**：云帆的高成功率来自 **CalQL + Pi0 微调**，不是 zero-shot！

---

## 🔧 OpenPI 官方内置 Benchmark

### 可以直接跑的 Benchmark

| Benchmark | 类型 | 官方 Checkpoint | 预期成功率 | 难度 |
|-----------|------|-----------------|-----------|------|
| **LIBERO** | 仿真 | `pi05_libero` | 92-98% | ⭐ 推荐 |
| **ALOHA Sim** | 仿真 | `pi0_aloha_sim` | 高 | ⭐ 简单 |
| **DROID** | 实物 | `pi05_droid` | 需实物机器人 | ⭐⭐⭐ |
| **ALOHA Real** | 实物 | `pi0_aloha_*` | 需实物机器人 | ⭐⭐⭐ |
| **ManiSkill3** | 仿真 | 无（需微调） | 0% (zero-shot) | ⭐⭐ |

---

## 🏆 LIBERO Benchmark（推荐）

**官方成绩** (π₀.₅ @ 30k fine-tuned):

| Task Suite | 成功率 |
|------------|--------|
| libero_spatial | **98.8%** |
| libero_object | **98.2%** |
| libero_goal | **98.0%** |
| libero_10 | **92.4%** |
| **平均** | **96.85%** |

### 运行命令

```bash
# 方式 1: Docker（推荐）
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# 方式 2: 手动运行
# 终端 1 - 启动 policy server
uv run scripts/serve_policy.py --env LIBERO

# 终端 2 - 运行评估
python examples/libero/main.py --task-suite-name libero_spatial
```

### 文件位置
- 评估脚本: `examples/libero/main.py`
- 配置: `src/openpi/training/config.py` → `pi05_libero`

---

## 🤖 ALOHA Simulator Benchmark

**支持任务**: `gym_aloha/AlohaTransferCube-v0`

### 运行命令

```bash
# 终端 1
uv run scripts/serve_policy.py --env ALOHA_SIM

# 终端 2
MUJOCO_GL=egl python examples/aloha_sim/main.py
```

### 文件位置
- 评估脚本: `examples/aloha_sim/main.py`
- 配置: `pi0_aloha_sim`

---

## 🎮 ManiSkill3 Benchmark（我们集成的）

**当前状态**: Zero-shot 成功率 0%，需要微调

### 已支持的任务 (9个)

```
PickCube-v1, StackCube-v1, PushCube-v1, PullCube-v1, PullCubeTool-v1
PlaceSphere-v1, PegInsertionSide-v1, PlugCharger-v1, TurnFaucet-v1
```

### 运行命令

```bash
# 单任务评估
python scripts/test_maniskill_integration.py --run-eval

# 多任务评估
python scripts/run_all_maniskill_tasks.py

# 多种子评估 (40 episodes)
python scripts/run_multiseed_eval.py
```

### 文件位置
- 评估脚本: `scripts/run_all_maniskill_tasks.py`
- Policy: `src/openpi/policies/maniskill_policy.py`
- Evaluator: `src/openpi/maniskill/pi05_maniskill_evaluator.py`

---

## 📁 所有预训练 Checkpoint

```
# Base 模型
gs://openpi-assets/checkpoints/pi0_base
gs://openpi-assets/checkpoints/pi0_fast_base
gs://openpi-assets/checkpoints/pi05_base

# Fine-tuned 模型
gs://openpi-assets/checkpoints/pi0_droid
gs://openpi-assets/checkpoints/pi0_fast_droid
gs://openpi-assets/checkpoints/pi05_droid       # 最强通用 Franka 策略
gs://openpi-assets/checkpoints/pi05_libero      # LIBERO 专用
gs://openpi-assets/checkpoints/pi0_aloha_sim
gs://openpi-assets/checkpoints/pi0_aloha_towel
gs://openpi-assets/checkpoints/pi0_aloha_tupperware
gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap
```

**本地已下载**:
- `checkpoints/pi05_base_hf/` (14GB)
- `checkpoints/pi05_droid_hf/` (6.8GB)

---

## 🔬 分析云帆代码的发现

**代码位置**: `/share/project/yunfan/RL/caurft/`

### 云帆的训练流程
1. 离线预训练（CalQL on demo data）
2. 在线微调（RL + demo 混合）
3. 评估

### 关键差异

| 因素 | 我们 (Pi0.5 base) | 云帆 (Pi0 微调) |
|------|-------------------|-----------------|
| **模型** | 未微调的 base | 在 ManiSkill 上微调 |
| **训练** | 无 | CalQL + RL 在线微调 |
| **数据** | 无 | ManiSkill 专家轨迹 |
| **State** | 18D (qpos+qvel) | 8D (qpos[:8]) |

---

## 📋 下一步计划

### 可以立即做的

1. **跑 LIBERO Benchmark** - 官方有 fine-tuned checkpoint，预期 96%+ 成功率
2. **跑 ALOHA Sim** - 简单，有官方 checkpoint

### 需要准备的

3. **在 ManiSkill 数据上微调 Pi0.5**
   - 参考云帆的 CalQL + Pi0 框架
   - 收集 ManiSkill 专家轨迹

---

## 🔗 相关资源

- **OpenPI 官方**: https://github.com/Physical-Intelligence/openpi
- **我的 Fork**: https://github.com/EasonAI-5589/openpi
- **云帆代码**: `/share/project/yunfan/RL/caurft/`
- **ManiSkill**: https://github.com/haosulab/ManiSkill

---

## 📝 代码已提交

Commit: https://github.com/EasonAI-5589/openpi/commit/917c9cc

包含:
- ManiSkill3 集成代码
- 9 个任务的评估结果
- 诊断分析脚本
- 云帆代码分析
