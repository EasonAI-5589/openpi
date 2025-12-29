# OpenPi 实习任务清单

> 负责人：郭奕辰
> 来源：2025年12月25日会议纪要（程洪洋、娄云帆、张书逸）
> 开始时间：2025年12月26日

---

## 当前任务状态

### 1. 基础配置修改
- [x] 查找 Pi 0.5 的 config 文件 → `src/openpi/training/config.py`
- [x] 基于娄云帆共享的 config 进行修改 ✅
- [x] 对比官方仓库的改动处
- [x] 将数据集替换成娄云帆造好的数据集 (使用相同的原始数据源)
- [x] 下载 Pi 0.5 的 base 权重

### 2. Benchmark 评估
- [x] **LIBERO Benchmark** → 96% 成功率 ✅
- [x] ManiSkill3 Zero-shot 评估 → 0% (预期内) ✅
- [ ] ManiSkill3 SFT 后评估（待训练）

### 3. 仿真任务
- [x] 完成数据转换流程 (StackCube)
- [x] 将 Pi 0 接 ManiSkill 的仿真 → 换成 Pi 0.5 配置 ✅
- [ ] 完成 Pi 0.5 SFT 训练
- [ ] 完成 Pi 0.5 在 ManiSkill Benchmark 的跑分

### 4. 架构更改任务
- [ ] 将当前架构的动作头换成简单的 Flow Matching 头
- [ ] 微调参数：当前 loss 是 diffusion loss + a + q loss
- [ ] 微调 q 使其适配 Flow Matching loss

### 5. 学习准备
- [ ] 阅读 CRFT 论文（程洪洋推荐）

---

## 待协调材料

### 娄云帆需提供

| # | 材料 | 说明 | 状态 |
|---|------|------|------|
| 1 | 修改过的 config 文件 | 基于 Pi 0 改的 | ⏳ |
| 2 | 造好的数据集 | 替换官方数据集 | ⏳ |
| 3 | ManiSkill 仿真代码 | 6 个仿真任务 | ⏳ |
| 4 | Pi 0 接 ManiSkill 代码 | 改成 Pi 0.5 版本 | ⏳ |

### 程洪洋需提供

| 待办 | 状态 |
|------|------|
| 有卡的集群账号 | ⏳ |

---

## 进度记录

### 2025-12-26
- [x] Fork openpi 仓库到个人账户
- [x] 创建任务清单
- [x] 调研 Pi0 → Pi0.5 迁移方案

### 2025-12-27
- [x] 配置商庄服务器环境
- [x] 部署 openpi 环境
- [x] 下载 pi05_base 模型 (14GB)
- [x] 下载 pi05_droid 模型 (6.8GB)
- [x] 下载 pi05_libero 模型 (12GB)
- [x] **LIBERO Benchmark 评估完成** → 96%

#### ManiSkill 集成工作
- [x] 分析云帆的 caurft 仓库结构和数据转换方案
- [x] 复制 ManiSkill 相关脚本到我们仓库：
  - `examples/maniskill/convert_maniskill_data_to_lerobot.py` - 数据转换
  - `scripts/eval_maniskill.py` - 评估脚本
  - `src/openpi/policies/libero_policy_no_wrist.py` - 无 wrist camera 策略
- [x] 添加 Pi0.5 ManiSkill 训练配置到 `config.py`
- [x] 转换 StackCube 数据 (200条轨迹 → 3.5GB LeRobot 格式)
- [x] 计算 norm_stats 归一化统计
- [x] 修改评估脚本支持 HuggingFace checkpoint 格式
- [x] **Pi0.5 Zero-shot 评估** → 0% 成功率 (预期内，未经训练)
- [x] **转换全部 6 个 ManiSkill 任务数据** ✅
  - StackCube (200 条轨迹)
  - PickCube (200 条轨迹)
  - PushCube (200 条轨迹)
  - PullCube (200 条轨迹)
  - PlaceSphere (188 条轨迹)
  - PegInsertionSide (200 条轨迹)
- [x] **添加全部 6 个任务的 Pi0.5 训练配置** ✅
- [x] **PickCube Pi0.5 SFT 训练完成** ✅ (2025-12-28)
  - 30000 步训练完成
  - Loss: 0.0833 → 0.0045 (下降 95%)
  - Checkpoint: `checkpoints/pi05_maniskill_pickcube/pickcube_sft_v1/29999`
- [x] **PickCube 训练后评估** ✅ (2025-12-29)
  - 结果：0% (0/50) - 训练 loss 降但评估失败
- [ ] 剩余 5 个任务 SFT 训练

### 2025-12-29
- [x] **问题排查：训练 loss 降但评估 0%**
  - 排查 wrist_image：训练数据也是全零，与评估一致 ✅
  - 排查 Pi0 vs Pi0.5：云帆 Pi0 PickCube 也是 0% ✅
  - 结论：**PickCube 任务本身较难**
- [x] **使用云帆 checkpoint 验证评估流程**
  - StackCube (云帆 Pi0): **15% (3/20)** ✅ 验证通过
  - PickCube (云帆 Pi0): **0% (0/20)** - 与云帆视频记录一致
  - PickCube_50 (云帆 Pi0): **0% (0/20)**
- [x] **更新文档**
  - BENCHMARK_REPORT.md - 添加验证结果
  - ISSUES_2024-12-28.md - 修正 18.5% 成功率的错误
  - TODO.md - 更新进度
- [x] **问云帆**：18.5% PickCube 成功率是怎么跑出来的？ ✅ 已回复
  - 云帆：**"你需要先训练一个 pi0.5 然后测试一下"**
  - 关键发现：StackCube 用 action_horizon=20 + 带 wrist，PickCube 用 10 + NoWrist
- [x] **修改所有 Pi0.5 ManiSkill 配置** ✅
  - 全部改为 action_horizon=20 + LeRobotManiskillDataConfig（带 wrist）
- [ ] **Pi0.5 StackCube 训练** 🔄 进行中
  - 启动时间：20:10
  - 当前进度：117/30000 steps
  - 预计完成：~43 小时（约 2 天）
  - 日志：`nohup_stackcube.out`
  - Checkpoint：`checkpoints/pi05_maniskill_stackcube/stackcube_sft_v1/`
- [ ] 训练完成后评估验证（目标：>10%，对比云帆 Pi0 的 15%）

---

## 后续规划

- 如果前两项任务效果好 → 会有进一步迁移任务
- **Robobrain 项目**: Robot 这边的 VLM 通用基座项目
  - 目标：冲击顶尖期刊

---

## 相关资源

- **仓库地址**: https://github.com/EasonAI-5589/openpi
- **官方仓库**: https://github.com/Physical-Intelligence/openpi
- **会议纪要**: https://jwolpxeehx.feishu.cn/docx/NdiNdlHobooUZYxBo95ckA0fn2e
