# 待讨论问题

> 与洪洋、云帆讨论

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

## 2. Pi0 → Pi0.5 迁移

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

## 3. 架构更改任务

### 待确认
- [ ] 当前 loss 组成：diffusion loss + a loss + q loss，具体是什么？
- [ ] 换成 Flow Matching 头后，a loss 和 q loss 怎么调整？
- [ ] 有没有泰玲的代码可以参考？

---

## 4. 集群资源

### 待确认
- [ ] 有卡的集群账号？（洪洋说 12/26 上午提供）
- [ ] 训练任务需要多少显存？多少卡？

---

## 5. LIBERO 评估结果

### 已完成
- libero_spatial: **96%** (48/50)，官方 98.8%

### 待确认
- [ ] 是否需要跑完整的 libero_90？
- [ ] 每个任务需要多少 trials？（目前用的 5，官方用 50）
