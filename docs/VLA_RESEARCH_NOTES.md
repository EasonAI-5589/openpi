# VLA (Vision-Language-Action) 领域调研笔记

> 调研时间: 2024-12-28
> 整理者: Claude

---

## 1. Physical Intelligence 模型演进

### 1.1 π₀ (Pi-Zero) - 2024年10月

**核心创新**：首个使用 Flow Matching（一种扩散变体）而非自回归离散化的VLA模型

| 特性 | 详情 |
|------|------|
| VLM骨干 | PaliGemma (SigLIP + Gemma) |
| 动作生成 | Flow Matching + Action Chunking |
| 控制频率 | 最高50Hz |
| 数据来源 | Open X-Embodiment |

**为什么用Flow Matching?**
- 自回归方法：生成50个动作需要50步解码
- Flow Matching：约10次推理即可生成整个chunk
- 更适合连续动作空间，动作更平滑

### 1.2 π₀.₅ (Pi-Zero-Point-Five) - 2025年4月

**核心创新**：Knowledge Insulation（知识隔离）+ 异构数据Co-training

**Knowledge Insulation 原理**：
1. 阻断从Action Expert到VLM backbone的梯度回传
2. VLM用FAST-tokenized离散动作微调（学习表示）
3. Action Expert单独学习连续动作（不干扰VLM知识）

**优势**：
- 训练步数减少7.5倍
- 保留VLM的预训练知识
- 推理速度更快

**Co-training数据源**：
- 多模态网页数据
- 语言指令
- 子任务命令
- 跨机器人数据
- 约400小时移动操作演示

### 1.3 π*₀.₆ - 2025年11月

**核心创新**：RECAP（通过优势条件策略进行RL）

**RECAP三步法**：
1. **Pre-training**：用离线RL预训练VLA
2. **Coaching**：用专家干预进行修正
3. **Experience**：从自主经验中学习

**关键技术**：
- Advantage Conditioning：训练时学习"高优势"动作
- 推理时只执行高优势动作 → 超越训练数据

**成果**：
- 任务吞吐量翻倍
- 故障率降低50%以上
- 机器人连续运行18小时（5:30AM-11:30PM）

---

## 2. 开源VLA模型对比

| 模型 | 参数量 | 优势 | 局限 |
|------|--------|------|------|
| **RT-2-X** | 55B | 大规模VLM预训练，泛化强 | 闭源，计算需求高 |
| **OpenVLA** | 7B | 开源，比RT-2-X高16.5%成功率 | - |
| **Octo** | 27M-93M | 轻量，支持goal image | 性能较弱 |
| **π₀** | ~3B | Flow Matching，高频控制 | - |
| **π₀.₅** | ~3B+ | 开放世界泛化 | - |

### OpenVLA 亮点
- 在29个任务上比RT-2-X高16.5%（参数少7倍）
- 支持LoRA微调，可在消费级GPU运行
- 4bit量化不影响性能

### Octo 亮点
- 基于扩散策略（Diffusion Policy）
- 同时支持语言和目标图像条件
- 用目标图像时成功率提高25%

---

## 3. 动作表示方法对比

### 3.1 Discrete Token Output
- 代表：RT-2, OpenVLA
- 将动作编码为离散token序列
- 像文本生成一样自回归解码

### 3.2 Diffusion/Flow Matching
- 代表：π₀, Octo
- 输出连续动作轨迹
- 动作更平滑，适合精细操作

### 3.3 Action Chunking
- 一次预测多个时间步的动作
- 减少推理次数，提高效率
- π₀：并行Flow Matching生成chunk

### 3.4 Real-Time Chunking (RTC)
- 边执行当前chunk边生成下一个
- "冻结"确定执行的动作，"修复"其余部分
- 无需重新训练，即插即用

---

## 4. 数据集与Benchmark

### 4.1 Open X-Embodiment
- 22个机器人，21个机构
- 527种技能
- 标准化数据格式

**成果**：
- RT-1-X比单机器人baseline高50%
- RT-2-X在新任务上emergent skill成功率3倍

### 4.2 LIBERO
- 130个任务，4个任务套件
- 测试终身学习能力
- 最新方法达到98.1%平均成功率

### 4.3 DROID
- 真实世界Franka机器人数据
- 6个任务，4个场景（实验室、办公室、家庭）
- Co-training显著提升OOD性能（+17%）

### 4.4 ManiSkill3
- GPU并行仿真，30,000+ FPS
- 比其他平台快10-1000倍
- 支持域随机化（相机、纹理）

**Sim2Real能力**：
- 实现SIMPLER数字孪生
- 评估速度是真实世界的60-100倍
- Peg insertion任务真实成功率95%

---

## 5. 微调技术

### 5.1 Full Fine-tuning (我们目前用的)
- 微调所有参数
- 需要更多显存和计算
- 效果通常最好

### 5.2 LoRA/QLoRA
- 注入低秩矩阵，冻结原参数
- 显著减少可训练参数
- QLoRA：4bit量化进一步节省内存

**OpenPI LoRA配置示例**：
```python
model=pi0_config.Pi0Config(
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora"
),
freeze_filter=model.get_freeze_filter(),
ema_decay=None,  # LoRA时关闭EMA
```

### 5.3 Few-Shot Learning
- 用极少样本适应新任务
- LoRA Recycle：复用预调好的LoRA进行零样本适应

---

## 6. 对我们项目的启示

### 6.1 当前配置评估

| 配置项 | 当前值 | 建议 |
|--------|--------|------|
| 微调方式 | Full SFT | 可考虑LoRA减少资源 |
| batch_size | 64→128 | 已优化 |
| fsdp_devices | 8 | 合适 |
| 训练步数 | 30k | 可能偏多 |

### 6.2 可能的改进方向

1. **加入LoRA**：减少显存，可能加速收敛
2. **Knowledge Insulation**：参考π₀.₅，避免遗忘VLM知识
3. **域随机化**：ManiSkill3支持，可提升泛化
4. **RL微调**：参考RECAP，从经验中学习

### 6.3 评估注意事项

参考LIBERO评估协议：
- 每个任务20次rollout
- 测试不同初始状态
- 报告per-task和平均成功率

---

## 7. 相关链接

### 论文
- [π₀ Paper](https://www.physicalintelligence.company/download/pi0.pdf)
- [π₀.₅ Paper](https://arxiv.org/abs/2504.16054)
- [π*₀.₆ Paper](https://arxiv.org/abs/2511.14759)
- [OpenVLA Paper](https://arxiv.org/abs/2406.09246)
- [Octo Paper](https://arxiv.org/abs/2405.12213)
- [ManiSkill3 Paper](https://arxiv.org/abs/2410.00425)

### 代码
- [OpenPI GitHub](https://github.com/Physical-Intelligence/openpi)
- [OpenVLA GitHub](https://openvla.github.io/)
- [Octo](https://octo-models.github.io/)
- [ManiSkill3](https://github.com/haosulab/ManiSkill)
- [SimplerEnv](https://github.com/simpler-env/SimplerEnv)

### 数据集
- [Open X-Embodiment](https://arxiv.org/abs/2310.08864)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [DROID](https://droid-dataset.github.io/)

---

## 8. Benchmark 详细对比

### 8.1 SIMPLER - 仿真评估真实策略

**核心贡献**：通过1500+组配对的sim-real评估，证明仿真评估与真实世界强相关

| 评估方法 | 说明 |
|----------|------|
| Visual Matching | 将真实图像叠加到仿真背景，调整前景纹理 |
| Variant Aggregation | 创建多种仿真变体（背景、光照、干扰物），取平均 |

**评估指标**：
- **MMRV** (Mean Maximum Rank Violation)：越低越好，衡量策略排序一致性
- **Pearson相关系数**：衡量sim-real性能相关性

**结论**：除1个任务外，所有策略排序都正确

### 8.2 主要Benchmark对比

| Benchmark | 任务数 | 特点 | 最新SOTA |
|-----------|--------|------|----------|
| **LIBERO** | 130 | 终身学习，4个套件 | 98.1% avg |
| **RLBench** | 100 | 多样任务，运动规划演示 | - |
| **MetaWorld** | 50 | 多任务/元RL | - |
| **CALVIN** | - | 语言条件，长horizon | - |
| **ManiSkill3** | 12域 | GPU并行，sim2real | 95% real |

### 8.3 评估最佳实践

参考 "Robot Learning as an Empirical Science" (2024):

**指标类型**：
1. **语义指标**：成功/失败（二值），子目标完成
2. **行为指标**：轨迹平滑度(SPARC)、速度峰值数
3. **阶段指标**：各子阶段成功率

**建议**：
- 简单/困难任务：二值成功率信息量低
- 中等任务：行为和阶段指标更有诊断价值
- 相同成功率的策略可能有不同执行质量

---

## 9. 策略学习方法深度对比

### 9.1 Diffusion Policy vs ACT

| 方面 | Diffusion Policy | ACT |
|------|------------------|-----|
| 架构 | CNN/Transformer + 扩散 | Encoder-Decoder Transformer |
| 动作生成 | 从噪声迭代去噪 | 直接预测 |
| 多模态 | 天然支持 | 需要style variable z |
| 性能 | 平均提升46.9% | 精细双臂任务强 |

**Diffusion Policy优势**：
- 优雅处理多模态动作分布
- 适合高维动作空间
- 训练稳定性好

### 9.2 ALOHA系统

| 版本 | 发布 | 成本 | 特点 |
|------|------|------|------|
| ALOHA | 2023 | ~$20k | 双臂遥操作，ACT策略 |
| Mobile ALOHA | 2024.01 | $32k | 加移动底盘，全身遥操作 |
| ALOHA 2 | 2024.02 | - | 改进夹爪、重力补偿、MuJoCo模型 |

**Mobile ALOHA关键发现**：
- 50个演示 + co-training可提升成功率90%
- 可完成：炒虾、开柜门、叫电梯、洗锅

### 9.3 行为克隆 vs 逆强化学习

| 方法 | 优点 | 缺点 |
|------|------|------|
| **BC** | 简单、快速、数据高效 | 分布偏移、泛化差 |
| **IRL** | 学习奖励函数、更鲁棒 | 计算复杂、需要RL子问题 |

**2024趋势**：BC重新主导操作研究，尤其是结合大模型

---

## 10. 动作Tokenization技术

### 10.1 FAST (Frequency-space Action Sequence Tokenization)

**问题**：简单per-dimension binning在高频精细任务上失败

**解决方案**：
1. 对动作chunk应用DCT（离散余弦变换）
2. 量化DCT系数
3. 用BPE压缩

**效果**：
- 30-60个token/chunk（10倍压缩）
- 训练速度提升5倍
- 可完成折叠衣物、收拾餐桌等精细任务

### 10.2 BEAST (B-spline Encoded Action Sequence)

- 用B样条编码动作序列
- 无需单独训练tokenizer
- 固定长度token，支持并行解码

---

## 11. 数据增强技术

### 11.1 视觉增强

| 方法 | 说明 | 效果 |
|------|------|------|
| **RoVi-Aug** | 用扩散模型换机器人/视角 | 零样本跨机器人迁移 |
| **NeRF-Aug** | NeRF渲染新物体blend进演示 | 泛化到新物体 |
| **Domain Randomization** | 随机化颜色、纹理、光照 | 提升sim2real |

### 11.2 物理增强

- 随机化质量、摩擦力、阻尼
- OpenAI魔方手：大量物理随机化使策略鲁棒

### 11.3 深度图像优势

- 对光照和外观变化不敏感
- sim2real差距更小
- 推荐用于感知任务

---

## 12. Sim2Real 迁移

### 12.1 域差距来源

| 类型 | 问题 | 解决方案 |
|------|------|----------|
| 视觉 | 纹理、光照不真实 | Visual Randomization |
| 物理 | 摩擦、质量不准确 | Physics Randomization |
| 几何 | 碰撞模型简化 | 高精度mesh |

### 12.2 2024最佳实践

- **混合方法**：视觉+物理同时随机化
- **迭代优化**：根据真实失败调整仿真
- **深度图像**：比RGB更易迁移

**成功案例**：
- 果园机器人：50+光照条件训练，92%采摘准确率
- 苹果采摘：比人类减少37%损伤
- NVIDIA Isaac Sim + Replicator：5%→87%精度提升

---

## 13. 人形机器人学习

### 13.1 2024-2025关键论文

| 论文 | 贡献 |
|------|------|
| HugWBC (2025) | 统一精细运动全身控制 |
| ASAP (2025) | 仿真-真实物理对齐 |
| ExBody2 (2024) | 表情人形全身控制 |
| Science Robotics (2024) | 纯学习的真实人形运动 |

### 13.2 行业进展

| 公司 | 机器人 | 价格/特点 |
|------|--------|----------|
| Boston Dynamics | Atlas (电动版) | 2024发布 |
| Unitree | G1 | $16,000起 |
| 1XTech | NEO | 双足 |
| Agility | Digit | GXO物流部署 |
| 优必选 | Walker S1 | 500+订单(BYD等) |

---

## 14. 奖励函数设计

### 14.1 稀疏 vs 稠密奖励

| 类型 | 定义 | 优缺点 |
|------|------|--------|
| **稀疏** | 只在完成目标时给奖励 | 简单但探索困难 |
| **稠密** | 中间步骤持续反馈 | 学习快但设计难 |

### 14.2 LLM自动设计奖励

| 方法 | 会议 | 说明 |
|------|------|------|
| Auto MC-Reward | CVPR 2024 | 迭代优化奖励函数 |
| Text2Reward | ICLR 2024 | 自然语言→代码奖励 |
| HPRS | 2024 | 层次势能奖励塑形 |

**DrS方法**：从稀疏奖励+演示学习可复用稠密奖励

---

## 15. 视觉表示学习

### 15.1 Vision Transformer in Robotics

**挑战**：
- ViT需要大量数据
- 只用最后一层特征可能丢失信息

**VAT (Vision Action Transformer) 2024**：
- 利用ViT全层特征层次
- LIBERO：98.15%成功率（SOTA）
- RoboTwin：40.66%（接近Pi0的46.42%，但参数少一半）

### 15.2 ViT-VS (Visual Servoing)

- 预训练ViT特征用于视觉伺服
- 无需微调，通用性强
- 结合了IBVS通用性和PBVS高收敛率

---

## 16. 术语表

| 术语 | 解释 |
|------|------|
| VLA | Vision-Language-Action，视觉-语言-动作模型 |
| Flow Matching | 一种扩散变体，用于生成连续动作 |
| Action Chunking | 一次预测多个时间步动作 |
| FSDP | Fully Sharded Data Parallelism，全分片数据并行 |
| Knowledge Insulation | 阻断梯度保护VLM知识的技术 |
| RECAP | RL with Experience and Corrections via Advantage-conditioned Policies |
| Sim2Real | 仿真到真实机器人的迁移 |
| Domain Randomization | 在仿真中随机化视觉/物理参数以提升泛化 |
| FAST | Frequency-space Action Sequence Tokenization |
| DCT | Discrete Cosine Transform，离散余弦变换 |
| BPE | Byte Pair Encoding，字节对编码 |
| MMRV | Mean Maximum Rank Violation，评估sim-real一致性 |
| BC | Behavioral Cloning，行为克隆 |
| IRL | Inverse Reinforcement Learning，逆强化学习 |
| ACT | Action Chunking with Transformers |
| IBVS | Image-Based Visual Servoing |
| PBVS | Position-Based Visual Servoing |

---

## 17. 参考资源汇总

### 论文
- [π₀](https://www.physicalintelligence.company/download/pi0.pdf) | [π₀.₅](https://arxiv.org/abs/2504.16054) | [π*₀.₆](https://arxiv.org/abs/2511.14759)
- [OpenVLA](https://arxiv.org/abs/2406.09246) | [Octo](https://arxiv.org/abs/2405.12213)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [ALOHA](https://tonyzhaozh.github.io/aloha/) | [Mobile ALOHA](https://arxiv.org/abs/2401.02117)
- [FAST Tokenizer](https://arxiv.org/abs/2501.09747)
- [SIMPLER](https://simpler-env.github.io/)
- [ManiSkill3](https://arxiv.org/abs/2410.00425)
- [VAT](https://arxiv.org/abs/2512.06013)

### 代码库
- [OpenPI](https://github.com/Physical-Intelligence/openpi)
- [LeRobot](https://github.com/huggingface/lerobot)
- [ManiSkill](https://github.com/haosulab/ManiSkill)
- [SimplerEnv](https://github.com/simpler-env/SimplerEnv)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

### 数据集
- [Open X-Embodiment](https://arxiv.org/abs/2310.08864)
- [DROID](https://droid-dataset.github.io/)
- [BridgeData V2](https://rail-berkeley.github.io/bridgedata/)
