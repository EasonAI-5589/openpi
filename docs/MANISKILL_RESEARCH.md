# ManiSkill 深度调研

> 调研时间: 2024-12-28
> 目的: 深入了解ManiSkill平台，为Pi0.5训练和评估提供指导

---

## 1. ManiSkill 演进历程

| 版本 | 发布时间 | 会议 | 核心特点 |
|------|----------|------|----------|
| ManiSkill | 2021 | NeurIPS 2021 | 首个大规模操作技能benchmark |
| ManiSkill2 | 2023 | ICLR 2023 | 20个任务族，4M+演示帧 |
| ManiSkill3 | 2024 | RSS 2025 | GPU并行，30,000+ FPS |

### ManiSkill3 核心优势

- **速度**: 比其他平台快10-1000倍，30,000+ FPS
- **内存**: 比Isaac Lab省2-3倍GPU显存（4.4GB vs 14.1GB）
- **开源**: 基于SAPIEN，完全开源（vs Isaac Lab闭源）
- **异构**: 唯一支持GPU异构仿真的平台

---

## 2. 任务列表与详情

### 2.1 我们正在用的任务

| 任务 | 环境ID | 难度 | 特点 |
|------|--------|------|------|
| **PickCube** | PickCube-v1 | 简单 | 抓取方块移动到目标位置 |
| **StackCube** | StackCube-v1 | 中等 | 将一个方块堆到另一个上 |
| **PushCube** | PushCube-v1 | 简单 | 推动方块到目标位置 |
| **PullCube** | PullCube-v1 | 简单 | 拉动方块 |
| **PlaceSphere** | PlaceSphere-v1 | 中等 | 放置球体 |
| **PegInsertionSide** | PegInsertionSide-v1 | 困难 | 侧向插入圆柱体到孔中 |

### 2.2 任务配置参数

```python
# 创建环境示例
env = gym.make(
    "PickCube-v1",
    num_envs=128,              # 并行环境数
    obs_mode="rgbd",           # state/rgbd/pointcloud
    control_mode="pd_ee_delta_pose",  # 末端执行器控制
    render_mode="human"        # 渲染模式
)
```

### 2.3 支持的机器人

| 机器人 | 类型 | 任务支持 |
|--------|------|----------|
| Panda | 固定臂+夹爪 | 所有tabletop任务 |
| Panda_wristcam | Panda+腕部相机 | 视觉任务 |
| Fetch | 移动底盘+臂 | 移动操作 |
| XArm6 | 6自由度臂 | 工业任务 |

---

## 3. 观测空间与动作空间

### 3.1 观测模式

| 模式 | 内容 | 适用场景 |
|------|------|----------|
| **state** | 完整物理状态（物体位姿等） | RL快速验证 |
| **rgbd** | RGB-D图像 | 视觉策略训练 |
| **pointcloud** | 点云 | 泛化能力研究 |
| **state_dict** | 层次化状态字典 | 调试 |

### 3.2 观测组成

```
observation = {
    "agent": {
        "qpos": 关节角度,
        "qvel": 关节速度,
        "controller": 控制器状态
    },
    "extra": {
        "tcp_pose": 末端位姿,
        "goal_pos": 目标位置（如有）
    },
    "sensor_data": {
        "base_camera": {"rgb": ..., "depth": ...},
        "hand_camera": {"rgb": ..., "depth": ...}
    }
}
```

### 3.3 控制模式

| 控制模式 | 维度 | 说明 |
|----------|------|------|
| pd_joint_delta_pos | 7+1 | 关节位置增量 |
| pd_ee_delta_pose | 6+1 | 末端位姿增量（我们用的） |
| pd_ee_delta_pos | 3+1 | 末端位置增量 |
| pd_joint_pos | 7+1 | 关节绝对位置 |
| pd_ee_pose | 6+1 | 末端绝对位姿 |

**注**：+1表示夹爪控制

---

## 4. 奖励函数设计

### 4.1 稀疏 vs 稠密

| 类型 | 定义 | 难度 |
|------|------|------|
| **sparse** | 成功+1，否则0 | 学习困难 |
| **dense** | 精心设计的中间奖励 | 需要调参 |

### 4.2 Dense Reward 设计原则

ManiSkill团队花费**超过1个月**设计复杂任务的dense reward。

典型组成：
1. **距离奖励**: 末端到目标距离
2. **抓取奖励**: 是否成功抓取
3. **放置奖励**: 物体到目标距离
4. **平滑奖励**: 动作变化惩罚

每个奖励项涉及~4个超参数：归一化函数、裁剪上下界、缩放系数。

### 4.3 自动奖励设计

| 方法 | 说明 |
|------|------|
| **DrS** | 从演示学习可复用dense reward |
| **Text2Reward** | LLM生成奖励代码 |
| **Auto MC-Reward** | 迭代优化奖励函数 |

---

## 5. 演示数据收集

### 5.1 数据来源

| 来源 | 适用场景 | 数据量 |
|------|----------|--------|
| **Motion Planning** | 简单任务自动生成 | 无限 |
| **RL Policy** | 有dense reward的任务 | 无限 |
| **Teleoperation** | 复杂任务人工收集 | ~10演示起 |
| **RFCL/RLPD** | 从少量演示扩展 | 放大10-100x |

### 5.2 遥操作方式

| 方式 | 设备 | 易用性 |
|------|------|--------|
| Click+Drag | 鼠标 | 简单 |
| SpaceMouse | 3D鼠标 | 中等 |
| VR | VR头显 | 直观 |

### 5.3 数据格式

演示数据组织结构：
```
demos/
├── PickCube-v1/
│   ├── motionplanning/
│   │   ├── trajectory.h5
│   │   └── ...
│   ├── rl/
│   └── teleop/
└── StackCube-v1/
    └── ...
```

HuggingFace数据集: [haosulab/ManiSkill_Demonstrations](https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations)

---

## 6. RL 基线性能

### 6.1 PPO 训练时间 (RTX 4090)

| 任务 | State-based | Image-based |
|------|-------------|-------------|
| PushCube-v1 | <1分钟 | 1-5分钟 |
| PickCube-v1 | 2-5分钟 | 15-45分钟 |
| PegInsertionSide-v1 | ~1小时 | - |

### 6.2 PPO vs SAC

| 任务类型 | 推荐算法 | 原因 |
|----------|----------|------|
| 简单任务 | PPO | 训练更快 |
| 困难任务 | SAC | 性能更好 |
| Pick/Place | SAC | 更好的多样性采样 |
| Open/Close | PPO | 足够且更快 |

### 6.3 标准Benchmark

- **小型**: 8个任务（快速验证）
- **大型**: 50个任务（全面评估）
- 所有任务都有归一化的dense reward

---

## 7. 模仿学习基线

### 7.1 支持的算法

| 算法 | 类型 | 特点 |
|------|------|------|
| **Behavior Cloning** | 监督学习 | 简单快速 |
| **Diffusion Policy** | 扩散模型 | 多模态动作 |
| **ACT** | Transformer | 精细操作 |

### 7.2 与VLA集成

ManiSkill支持的VLA模型：
- Octo
- RDT-1B
- RT-x系列
- OpenVLA

### 7.3 RLinf-VLA 结果

在25个ManiSkill任务上：**97.66%** 成功率

---

## 8. 域随机化

### 8.1 支持的随机化类型

| 类型 | API | 用途 |
|------|-----|------|
| **光照** | `_load_lighting()` | 视觉鲁棒性 |
| **相机** | CameraConfig.pose | 视角变化 |
| **纹理** | Actor.set_texture() | 外观变化 |
| **物体** | 异构仿真 | 形状泛化 |

### 8.2 相机随机化示例

```python
# 每个并行环境不同的相机位姿
camera_config = CameraConfig(
    pose=batched_poses,  # [N, 7] 位姿
    fov=batched_fovs,    # [N] 视场角
)
```

### 8.3 异构仿真

ManiSkill3独有功能：
- 每个并行环境可以有不同物体
- 不同数量的物体
- 不同自由度的机构

示例任务：
- Open Cabinet Drawer：每个环境不同柜子
- Pick Clutter YCB：每个环境不同数量YCB物体

---

## 9. Sim2Real 迁移

### 9.1 成功案例

| 任务 | 真实成功率 | 方法 |
|------|------------|------|
| Cube Picking | **95%** | PPO + PointNet |
| Peg Insertion (触觉) | **95.08%** | 软体触觉模拟 |

### 9.2 数字孪生

ManiSkill3支持的Sim2Real工具：
- SIMPLER评估环境
- 真实机器人数字孪生
- 触觉传感器模拟

### 9.3 ManiSkill-ViTac挑战赛

- 2024年首届挑战赛：18支队伍
- 2025年挑战赛：3个赛道
- 最终在真实机器人上评估

---

## 10. 评估协议

### 10.1 成功率指标

| 指标 | 定义 | 用途 |
|------|------|------|
| **success_once** | 任何时刻成功 | 子任务评估 |
| **success_at_end** | 结束时成功 | 长期稳定性 |
| **success** | 连续10步成功 | 严格标准 |

### 10.2 评估设置

ManiSkill-HAB标准：
- **1000 episodes** 每次评估
- **200 timesteps** 每个episode
- **3 seeds** 训练重复

### 10.3 长任务评估

Progressive Completion Rate：
- 必须按顺序完成所有子任务
- 最后子任务成功率 = 整体任务成功率

---

## 11. 与我们项目的关系

### 11.1 数据格式转换

我们的数据流：
```
ManiSkill演示 → LeRobot格式 → OpenPI训练 → Pi0.5模型
```

关键点：
- 200条轨迹/任务
- LeRobot v2.1格式
- 包含RGB图像（256x256）

### 11.2 评估脚本

```bash
# 评估训练好的模型
python scripts/eval_maniskill.py \
    --config-name pi05_maniskill_pickcube \
    --exp-name pickcube_sft_v1 \
    --env-id PickCube-v1 \
    --num-episodes 50
```

### 11.3 预期结果对比

| 方法 | 模型 | 任务 | 成功率 |
|------|------|------|--------|
| 云帆 | Pi0 | StackCube | ~70% |
| 我们 | Pi0.5 | PickCube | TBD |
| RLinf-VLA | OpenVLA | 25任务avg | 97.66% |

---

## 12. 最佳实践建议

### 12.1 训练

1. **先用state模式验证**：确认任务可学习
2. **再加视觉输入**：rgbd或pointcloud
3. **域随机化**：提升泛化能力
4. **充足演示**：200条是基线

### 12.2 评估

1. **多次评估**：至少50-100 episodes
2. **不同初始化**：测试泛化
3. **记录失败模式**：分析改进方向

### 12.3 Sim2Real

1. **匹配控制频率**：仿真和真实保持一致
2. **相机校准**：内参外参对齐
3. **域随机化**：覆盖真实变化范围

---

## 13. 参考资源

### 文档
- [ManiSkill3 官方文档](https://maniskill.readthedocs.io/)
- [ManiSkill GitHub](https://github.com/haosulab/ManiSkill)
- [ManiSkill3 论文](https://arxiv.org/abs/2410.00425)

### 数据集
- [ManiSkill Demonstrations (HuggingFace)](https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations)
- [ManiSkill LeRobot格式示例](https://huggingface.co/datasets/dancher00/maniskill-panda-pickcube)

### 相关论文
- ManiSkill3: GPU Parallelized Robotics Simulation (RSS 2025)
- ManiSkill2: A Unified Benchmark (ICLR 2023)
- ManiSkill-HAB: Home Rearrangement (ICLR 2024)
- DrS: Learning Reusable Dense Rewards (ICLR 2024)

### 工具
- [SimplerEnv](https://github.com/simpler-env/SimplerEnv) - Sim2Real评估
- [RLinf-VLA](https://arxiv.org/abs/2510.06710) - VLA+RL框架
