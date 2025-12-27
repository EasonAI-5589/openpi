# OpenPI 项目背景介绍

> Pi0/Pi0.5 模型、Benchmark 介绍、整体流程概览

---

## 1. Pi0 / Pi0.5 是什么？

### 1.1 概述

**Pi0** (π₀) 是 Physical Intelligence 开发的 **Vision-Language-Action (VLA)** 模型，用于机器人操控任务。

```
输入: 图像 + 语言指令 + 机器人状态
  ↓
Pi0 模型
  ↓
输出: 动作序列 (action chunks)
```

### 1.2 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                        Pi0 架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Images    │    │  Language   │    │   State     │     │
│  │  (cameras)  │    │ (instruction)│   │  (qpos/EEF) │     │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘     │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Vision-Language Model (VLM)             │   │
│  │                    PaliGemma 3B                      │   │
│  └─────────────────────────┬───────────────────────────┘   │
│                            │                               │
│                            ▼                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Action Expert (Flow Matching)           │   │
│  │           生成 50-step action chunks                 │   │
│  └─────────────────────────┬───────────────────────────┘   │
│                            │                               │
│                            ▼                               │
│                    Actions [50, 7-32]                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Pi0 vs Pi0.5 区别

| 特性 | Pi0 | Pi0.5 |
|------|-----|-------|
| 时间步注入 | MLP 拼接 | **adaRMSNorm** |
| 状态输入 | 连续 token (suffix) | **离散 token (prefix)** |
| max_token_len | 48 | **200** |
| 训练数据 | 单数据集 | **多数据集混合** |

**adaRMSNorm**：时间信息通过调制归一化参数注入，渗透到网络每一层。

---

## 2. Flow Matching 动作生成

### 2.1 为什么用 Flow Matching？

机器人动作是**多模态分布**（同一任务可能有多种完成方式），传统回归方法无法处理。

**Flow Matching** 学习从噪声到动作的"流动"路径：

```
噪声 ────────────────────────────────> 动作
     t=1                          t=0
         学习向量场 v(x, t)
```

### 2.2 训练目标

```python
# 极其简单的训练目标
noise = random_noise()
t = random_uniform(0, 1)
x_t = t * noise + (1 - t) * actions  # 线性插值
u_t = noise - actions                 # 目标向量
v_t = model(x_t, t)                   # 模型预测
loss = MSE(v_t, u_t)                  # 就是这么简单
```

### 2.3 采样过程

```python
x = noise  # 从 t=1 开始
for step in range(10):  # 10 步积分
    t = 1.0 - step * 0.1
    v = model(x, t)
    x = x + (-0.1) * v  # 欧拉积分
return x  # t=0 时得到动作
```

---

## 3. Benchmark 介绍

### 3.1 LIBERO

**LIBERO** 是基于 robosuite 的操控任务基准测试。

```
LIBERO Task Suites
├── libero_spatial (10 tasks)  - 空间关系理解
├── libero_object (10 tasks)   - 物体识别
├── libero_goal (10 tasks)     - 目标导向
├── libero_10 (10 tasks)       - 混合任务
└── libero_90 (90 tasks)       - 完整评估集
```

**典型任务**：`pick up the black bowl and place it on the plate`

**评估指标**：成功率 (Success Rate)

**官方结果**：

| Model | libero_spatial | libero_object | libero_goal | libero_10 |
|-------|----------------|---------------|-------------|-----------|
| Pi0.5 | 98.8% | 96.8% | 97.4% | 96.2% |

### 3.2 ManiSkill3

**ManiSkill3** 是基于 SAPIEN 的 GPU 并行机器人仿真器。

```
ManiSkill3 (34 environments)
├── Cube Manipulation      - PickCube, StackCube, PushCube...
├── Precision Tasks        - PegInsertion, PlugCharger...
├── Articulated Objects    - TurnFaucet, OpenCabinet...
├── Dexterous Hand         - RotateObject...
└── ...
```

**特点**：
- GPU 并行仿真（1000+ envs 同时运行）
- 支持多种机器人（Panda, UR5, 灵巧手...）
- Real2Sim 评估

### 3.3 其他 Benchmark

| Benchmark | 说明 | 链接 |
|-----------|------|------|
| ALOHA Sim | 双臂协作 | openpi 官方支持 |
| DROID | 真实机器人数据集 | openpi 官方支持 |
| SimplerEnv | Real2Sim 评估框架 | 包含 ManiSkill2 |
| VLABench | VLA 模型评估 | 支持 Pi0/Pi0.5 |

---

## 4. 评估流程

### 4.1 Client-Server 架构

```
┌─────────────────┐         HTTP/WebSocket        ┌─────────────────┐
│                 │ ◄──────────────────────────► │                 │
│  Policy Server  │         port 8000            │  Env Client     │
│  (Pi0.5 模型)   │                              │  (LIBERO/MS3)   │
│                 │                              │                 │
└─────────────────┘                              └─────────────────┘
     GPU 机器                                       CPU/GPU 机器
```

### 4.2 评估步骤

```bash
# 1. 启动 Policy Server
uv run scripts/serve_policy.py --env LIBERO \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir checkpoints/pi05_libero

# 2. 运行评估
python examples/libero/main.py \
    --args.task-suite-name libero_spatial \
    --args.num-trials-per-task 5
```

### 4.3 数据流

```
1. Env 重置 → 获取初始 obs
2. obs → Policy Server → 返回 action chunk [50, 7]
3. 执行 action chunk（每步取第一个动作）
4. 检查 success/fail
5. 重复直到成功或超时
```

---

## 5. 模型 Checkpoint

### 5.1 可用 Checkpoint

| Checkpoint | 大小 | 说明 | 路径 |
|------------|------|------|------|
| pi05_base | 14GB | Base 预训练模型 | `checkpoints/pi05_base_hf/` |
| pi05_droid | 6.8GB | DROID 数据微调 | `checkpoints/pi05_droid_hf/` |
| pi05_libero | 12GB | LIBERO 数据微调 | `checkpoints/pi05_libero/` |
| pi05_aloha_sim | - | ALOHA 仿真微调 | GCS |
| pi05_aloha_towel | - | ALOHA 毛巾任务 | GCS |

### 5.2 下载方式

```bash
# 方式 1: GCS (推荐)
gsutil -m cp -r "gs://openpi-assets/checkpoints/pi05_libero/*" checkpoints/pi05_libero/

# 方式 2: HuggingFace
export HF_ENDPOINT="https://hf-mirror.com"
python -c "
from huggingface_hub import snapshot_download
snapshot_download('lerobot/pi05_base', local_dir='checkpoints/pi05_base_hf')
"
```

---

## 6. 训练流程（参考）

### 6.1 数据准备

```bash
# 1. 转换数据到 LeRobot 格式
python scripts/convert_xxx_to_lerobot.py

# 2. 计算归一化统计
python scripts/compute_norm_stats.py --config pi05_custom
```

### 6.2 训练

```bash
# 配置在 src/openpi/training/config.py
python scripts/train.py --config pi05_custom
```

### 6.3 评估

```bash
# 启动 server
uv run scripts/serve_policy.py ...

# 运行评估
python examples/libero/main.py ...
```

---

## 7. 关键文件位置

```
openpi/
├── src/openpi/
│   ├── models/
│   │   ├── pi0.py           # Pi0/Pi0.5 模型实现
│   │   ├── pi0_config.py    # 模型配置
│   │   └── gemma.py         # Action Expert
│   ├── training/
│   │   └── config.py        # 训练配置
│   └── policies/
│       ├── libero_policy.py # LIBERO transforms
│       └── droid_policy.py  # DROID transforms
├── scripts/
│   ├── serve_policy.py      # Policy Server
│   ├── train.py             # 训练脚本
│   └── compute_norm_stats.py
├── examples/
│   └── libero/
│       └── main.py          # LIBERO 评估客户端
└── checkpoints/             # 模型权重
```

---

## 8. 参考资料

- **Pi0 论文**: [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
- **OpenPI 仓库**: https://github.com/Physical-Intelligence/openpi
- **LIBERO 论文**: [arXiv:2310.08587](https://arxiv.org/abs/2310.08587)
- **ManiSkill3**: https://github.com/haosulab/ManiSkill
- **Flow Matching**: [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
