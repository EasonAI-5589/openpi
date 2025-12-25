# 郭奕辰 - OpenPi 实习任务清单

> 来源：2025年12月25日会议纪要（程洪洋、娄云帆、张书逸）
> 开始时间：2025年12月26日

---

## 📋 当前任务

### 1. 基础配置修改
- [ ] 查找 Pi 0.5 的 config 文件
- [ ] 基于娄云帆共享的 config 进行修改
- [ ] 对比官方仓库的改动处
- [ ] 将数据集替换成娄云帆造好的数据集
- [ ] 下载 Pi 0.5 的 base（等娄云帆完成下载后获取）

### 2. 仿真任务（上手任务）
- [ ] 完成 6 个在 ManySkill 上的仿真任务
- [ ] 将 Pi 0 接 MySQL 的仿真 → 换成 Pi 0.5 接 MySQL 的仿真
- [ ] 完成 Pi 0.5 在 ManyScore 的跑分（把 Pi 0 改成 Pi 0.5）

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
| 发送 ManySkill 仿真代码 | 娄云帆 | ⏳ | |
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

---

## 🔗 相关资源

- **仓库地址**: https://github.com/EasonAI-5589/openpi
- **官方仓库**: https://github.com/Physical-Intelligence/openpi
- **会议纪要**: https://jwolpxeehx.feishu.cn/docx/NdiNdlHobooUZYxBo95ckA0fn2e

---

## 💬 交流沟通

- 有问题在群里直接发
- 多与娄云帆和书逸交流
