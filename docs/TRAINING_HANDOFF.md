# Pi0.5 ManiSkill SFT 训练交接文档

## 当前状态

**PickCube 训练已完成！**

### 训练进度 (2025-12-29)

| 任务 | 状态 | 进度 | 完成时间 |
|------|------|------|----------|
| PickCube | ✅ 已完成 | 30000/30000 (100%) | 2025-12-29 11:28 |
| StackCube | ⏳ 待训练 | - | - |
| PushCube | ⏳ 待训练 | - | - |
| PullCube | ⏳ 待训练 | - | - |
| PlaceSphere | ⏳ 待训练 | - | - |
| PegInsertionSide | ⏳ 待训练 | - | - |

### 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | Pi0.5 (pi05=True) |
| batch_size | 64 |
| fsdp_devices | 8 (8卡模型分片) |
| 学习率 | 5e-5 (cosine decay) |
| 总步数 | 30,000 |
| 训练速度 | ~1.3秒/step |

### Loss 收敛情况

| Step | Loss | Grad Norm |
|------|------|-----------|
| 0 | 0.0833 | 0.8746 |
| 500 | 0.0275 | 0.3594 |
| 1000 | 0.0199 | 0.2358 |
| 1500 | 0.0173 | 0.1856 |
| 2000 | 0.0160 | 0.1573 |
| 10000 | 0.0078 | 0.0862 |
| 20000 | 0.0051 | 0.0598 |
| 29900 | 0.0045 | 0.0526 |

**最终结果**: Loss 从 0.0833 → 0.0045，下降 95%，训练收敛良好。

### 解决的问题

1. **OOM问题** - 添加 `fsdp_devices=8` 启用模型分片
2. **LeRobot兼容性** - 修复 `torch.stack()` 对 Column 对象的TypeError
3. **权重格式** - 使用 Orbax 格式 (`checkpoints/pi05_base/params`)，非 HuggingFace 格式
4. **权重下载** - 从 GCS 重新下载（原有权重已损坏）

### 监控命令

```bash
# 查看训练日志
tail -f /share/project/guoyichen/openpi/nohup.out

# 查看loss趋势
grep "Step [0-9]*:" /share/project/guoyichen/openpi/nohup.out

# 查看GPU状态
nvidia-smi
```

---

## 任务数据概览

| 任务 | 数据 | 配置 | 轨迹数 |
|------|------|------|--------|
| StackCube | ✅ | `pi05_maniskill_stackcube` | 200 |
| PickCube | ✅ | `pi05_maniskill_pickcube` | 200 |
| PushCube | ✅ | `pi05_maniskill_pushcube` | 200 |
| PullCube | ✅ | `pi05_maniskill_pullcube` | 200 |
| PlaceSphere | ✅ | `pi05_maniskill_placesphere` | 188 |
| PegInsertionSide | ✅ | `pi05_maniskill_peginsertionside` | 200 |

## 训练命令

```bash
cd /share/project/guoyichen/openpi

# 训练前先检查 GPU 状态
nvidia-smi

# 8卡训练（约 1.5-2 小时）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/share/project/guoyichen/miniconda3/envs/openpi/bin/python -m openpi.training.train \
    pi05_maniskill_stackcube \
    --exp-name=stackcube_sft_v1
```

## 评估命令

训练完成后：

```bash
cd /share/project/guoyichen/openpi

/share/project/guoyichen/miniconda3/envs/openpi/bin/python scripts/eval_maniskill.py \
    --config-name pi05_maniskill_stackcube \
    --exp-name stackcube_sft_v1 \
    --env-id StackCube-v1 \
    --num-episodes 50
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `src/openpi/training/config.py` | Pi0.5 ManiSkill 训练配置 |
| `scripts/eval_maniskill.py` | ManiSkill 评估脚本 |
| `docs/YUNFAN_TRAINING_ANALYSIS.md` | 云帆方案分析 + SFT 核心概念 |
| `docs/MANISKILL_INTEGRATION.md` | 完整集成指南 |

## 预期结果

- 云帆 Pi0 在 StackCube 上：~70% 成功率
- 我们 Pi0.5 预期应该相近或更好
