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
- [ ] 将 Pi 0 接 MySQL 的仿真 → 换成 Pi 0.5 接 MySQL 的仿真
- [ ] 完成 Pi 0.5 在 ManiSkill Benchmark 的跑分

> **注**：ManySkill/ManyScore 实际上是 **ManiSkill**（SAPIEN Manipulation Skill Framework）
> - GitHub: https://github.com/haosulab/ManiSkill
> - GPU 并行机器人操作仿真器和基准测试平台

### 3. 架构更改任务
- [ ] 将当前架构的动作头换成简单的 Flow Matching 头（参照 Pi-Lin 的设计）
- [ ] 微调参数：当前 loss 是 diffusion loss + a + q loss
- [ ] 微调 q 使其适配 Flow Matching loss

### 4. 学习准备
- [ ] 阅读 CRFT 论文（程洪洋推荐）
- [ ] 熟悉泰玲的代码（基于 CRFT 改进）

---

## ⏳ 等待他人提供

| 待办 | 负责人 | 状态 | 备注 |
|------|--------|------|------|
| 下载 Pi 0.5 的 base 数据集 | 娄云帆 | ⏳ | 12/25 应完成 |
| 发送 ManiSkill 仿真代码 | 娄云帆 | ⏳ | |
| 发送 Pi0 接 MySQL 仿真代码 | 娄云帆 | ⏳ | |
| 整理详细任务说明文档 | 娄云帆 | ⏳ | |
| 提供有卡的集群账号 | 程洪洋 | ⏳ | 12/26 上午 |

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
