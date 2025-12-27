# 问云帆的问题清单

> 郭奕辰 2025-12-27
> Pi0.5 + ManiSkill3 集成遇到的问题

---

## 🔥 最关键的问题

**云帆，你的 Pi0 + ManiSkill 集成代码在哪？**

我想直接看你的实现，特别是：
1. **observation 预处理代码** - 在哪个文件？哪个函数？
2. **action 后处理代码** - 怎么把 Pi0 输出转成 ManiSkill action？
3. **完整的 evaluation 脚本** - 你跑 40 episodes 用的脚本

如果在 GitHub 上，给我 repo 链接 + 文件路径就行。
如果在服务器上，告诉我路径我自己去看。

---

## 背景

我已经完成了 Pi0.5 + ManiSkill3 的基础集成，跑了云帆评估的 6 个任务（40 episodes each），但**成功率全部是 0%**。

### 评估结果对比

| Task | Pi0.5 (我跑的) | Pi0 (云帆) | 差距 |
|------|----------------|------------|------|
| PlaceSphere-v1 | 0/40 = 0.00% | 11/40 = 27.50% | -27.5% |
| PickCube-v1 | 0/40 = 0.00% | 1/40 = 2.50% | -2.5% |
| StackCube-v1 | 0/40 = 0.00% | 24/40 = 60.00% | -60% |
| PushCube-v1 | 0/40 = 0.00% | 28/40 = 70.00% | -70% |
| PullCube-v1 | 0/40 = 0.00% | 35/40 = 87.50% | -87.5% |
| PullCubeTool-v1 | 0/40 = 0.00% | 3/40 = 7.50% | -7.5% |

---

## 我诊断出的问题

### 问题 1：Gripper 始终打开

Pi0.5 输出的 gripper action 始终是 -1（打开状态），从不闭合：
```
Gripper values: min=-1.0000, max=-1.0000
Unique gripper values: [-1.]
```

这导致**无法抓取任何物体**。

### 问题 2：State 格式不匹配

| 数据来源 | State 格式 |
|----------|-----------|
| **Pi0.5 期望**（DROID 格式） | EEF pose (7D: xyz + quaternion) + gripper |
| **我给的**（ManiSkill 原始） | Joint positions (9D) + Joint velocities (9D) = 18D |

模型可能不理解 joint angles，因为它是在 EEF pose 上训练的。

### 问题 3：相机数量不同

- Pi0.5 训练：3 个相机（base, left_wrist, right_wrist）
- ManiSkill：1-2 个相机（base_camera, hand_camera）

---

## 想问云帆的问题

### Q1: 你们的 Pi0 在 ManiSkill 上**有没有做微调**？

还是纯 zero-shot？如果做了微调：
- 用的什么数据？ManiSkill 的 demos？
- 微调了多少步？
- 能分享微调后的 checkpoint 吗？

### Q2: 你们的 **observation 格式**是怎么处理的？

我现在给模型的是：
```python
state = np.concatenate([qpos, qvel])  # 18D joint state
```

你们是不是用了 EEF pose？
```python
state = obs["extra"]["tcp_pose"]  # 7D EEF pose
```

### Q3: 你们的 **action transform** 是怎么做的？

我的处理：
```python
# Pi0.5 输出 32D actions
# 取前 7D 作为 ManiSkill action
output[:, 0:3] = raw[:, 0:3]  # position delta
output[:, 3:6] = raw[:, 3:6]  # rotation delta
output[:, 6] = binary(raw[:, 6])  # gripper
```

有没有需要特殊处理的地方？比如 action scaling？

### Q4: 你们用的 **control mode** 是什么？

我用的是 `pd_ee_delta_pose`，你们是不是也是这个？

### Q5: 能不能分享你们的**集成代码**？

特别是：
- observation 预处理部分
- action 后处理部分
- 任何特殊的 adapter/wrapper

### Q6: 你们评估用的是哪个 **checkpoint**？

- `pi0_base`？
- `pi0_droid`？
- 还是 ManiSkill 微调后的版本？

---

## 我的代码位置

已提交到 GitHub，可以参考：

- **Repository**: https://github.com/EasonAI-5589/openpi
- **ManiSkill transforms**: `src/openpi/policies/maniskill_policy.py`
- **Evaluator**: `src/openpi/maniskill/pi05_maniskill_evaluator.py`
- **Config**: `src/openpi/training/config.py` (搜索 `pi05_maniskill`)
- **诊断脚本**: `scripts/debug_action_space.py`, `scripts/debug_observation.py`

---

## 技术细节补充

### 我的环境配置

```
ManiSkill3: GPU backend
Robot: Panda
Control mode: pd_ee_delta_pose
Obs mode: rgbd
Model: pi05_base (HuggingFace checkpoint)
```

### Action 分析结果

```
Action ranges per dimension:
  Dim 0 (dx): [-0.52, -0.02]  ← 一直往负 x 方向动
  Dim 1 (dy): [-0.06, 0.08]
  Dim 2 (dz): [-0.04, 0.01]
  Dim 3 (dax): [-0.53, -0.34]
  Dim 4 (day): [-0.72, -0.03]
  Dim 5 (daz): [0.33, 0.59]
  Dim 6 (gripper): [-1.0, -1.0]  ← 始终打开！

Mean action magnitude: 0.303 (合理范围)
```

动作幅度看起来正常，但方向和 gripper 有问题。

---

## 总结

我怀疑主要问题是：

1. **没有微调** - Pi0.5 base 模型是在真实 DROID 数据上训练的，直接用在 ManiSkill sim 上有 domain gap
2. **State 格式错** - 应该用 EEF pose 而不是 joint angles
3. **可能需要你们的 adapter 代码** - 看看你们是怎么处理 obs/action 的

期待你的回复！

---

*生成时间: 2025-12-27*
