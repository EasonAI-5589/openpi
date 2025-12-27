# OpenPi Benchmark 评估报告

> 负责人：郭奕辰
> 日期：2025-12-27
> 仓库：https://github.com/EasonAI-5589/openpi

---

## 评估结果汇总

| Benchmark | Task Suite | 成功率 | 官方报告 | 备注 |
|-----------|------------|--------|----------|------|
| **LIBERO** | libero_spatial | **96%** (48/50) | 98.8% | Fine-tuned checkpoint |
| ManiSkill3 | 6 tasks | 0% | N/A | Zero-shot，无微调数据 |

---

## 1. LIBERO Benchmark 评估

### 1.1 评估结果

**Task Suite**: `libero_spatial` (10 tasks × 5 trials = 50 episodes)

| 指标 | 结果 |
|------|------|
| **总成功率** | **96%** (48/50) |
| **官方报告** | 98.8% |
| **评估时间** | 5分26秒 |

**各任务成功率**：

| Task ID | 任务名称 | 成功率 |
|---------|----------|--------|
| Task 1 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 100% (5/5) |
| Task 2 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 100% (5/5) |
| Task 3 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 100% (5/5) |
| Task 4 | pick_up_the_black_bowl_on_the_cookie_sheet_and_place_it_on_the_plate | 100% (5/5) |
| Task 5 | pick_up_the_black_bowl_next_to_the_cookie_sheet_and_place_it_on_the_plate | 100% (5/5) |
| Task 6 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 100% (5/5) |
| Task 7 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 100% (5/5) |
| Task 8 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 100% (5/5) |
| Task 9 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 80% (4/5) |
| Task 10 | pick_up_the_black_bowl_next_to_the_stove_and_place_it_on_the_plate | 80% (4/5) |

### 1.2 复现流程

#### Step 1: 环境准备

```bash
# 1. 进入项目目录
cd /share/project/guoyichen/openpi

# 2. 确保 openpi 环境已配置（使用 uv）
# 如果未安装 uv:
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 3. 同步依赖
uv sync
```

#### Step 2: 下载 Checkpoint

**方法 A: 使用 gsutil（推荐，速度更快）**

```bash
# 安装 gsutil
pip install gsutil

# 下载 pi05_libero checkpoint (12GB)
mkdir -p checkpoints/pi05_libero
gsutil -m cp -r "gs://openpi-assets/checkpoints/pi05_libero/*" checkpoints/pi05_libero/
```

**方法 B: 使用 HuggingFace**

```bash
# 如果 gsutil 下载失败，可以尝试 HuggingFace
export HF_ENDPOINT="https://hf-mirror.com"  # 国内用户使用镜像
export HF_TOKEN="your_token_here"

python -c "
from huggingface_hub import snapshot_download
snapshot_download('physical-intelligence/pi05_libero', local_dir='checkpoints/pi05_libero')
"
```

#### Step 3: 配置 LIBERO 环境

```bash
# 1. 创建 LIBERO conda 环境（Python 3.8 + PyTorch 1.11）
conda create -n libero python=3.8 -y
conda activate libero

# 2. 安装 PyTorch（CUDA 11.3）
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# 3. 安装 LIBERO
cd third_party/libero
pip install -e .
cd ../..

# 4. 安装额外依赖
pip install bddl easydict cloudpickle gym tyro tqdm imageio

# 5. 创建 LIBERO 配置文件
mkdir -p ~/.libero
cat > ~/.libero/config.yaml << 'EOF'
libero_root: /share/project/guoyichen/openpi/third_party/libero
assets_root: /share/project/guoyichen/openpi/third_party/libero/libero/libero/assets
EOF
```

#### Step 4: 启动 Policy Server

```bash
# Terminal 1: 启动 Policy Server（使用 openpi 的 uv 环境）
cd /share/project/guoyichen/openpi
uv run scripts/serve_policy.py \
    --env LIBERO \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir checkpoints/pi05_libero

# 等待看到: "Uvicorn running on http://0.0.0.0:8000"
```

#### Step 5: 运行评估

```bash
# Terminal 2: 运行 LIBERO 评估客户端
source /share/project/guoyichen/miniconda3/bin/activate libero

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:/share/project/guoyichen/openpi/third_party/libero
export MUJOCO_GL=egl  # 无头渲染

# 运行评估
cd /share/project/guoyichen/openpi
python examples/libero/main.py \
    --args.task-suite-name libero_spatial \
    --args.num-trials-per-task 5
```

#### Step 6: 查看结果

评估完成后，输出类似：

```
INFO:root:Task 0 success rate: 1.0
INFO:root:Task 1 success rate: 1.0
...
INFO:root:Task 9 success rate: 0.8
INFO:root:Total success rate: 0.96
INFO:root:Total episodes: 50
```

### 1.3 其他 Task Suite

LIBERO 包含多个 task suite，可以替换 `--args.task-suite-name`:

| Task Suite | 任务数 | 说明 |
|------------|--------|------|
| `libero_spatial` | 10 | 空间关系理解 |
| `libero_object` | 10 | 物体识别 |
| `libero_goal` | 10 | 目标导向 |
| `libero_10` | 10 | 混合任务 |
| `libero_90` | 90 | 完整评估集 |

完整评估命令：

```bash
# 增加每个任务的试验次数以获得更准确的统计
python examples/libero/main.py \
    --args.task-suite-name libero_90 \
    --args.num-trials-per-task 50
```

---

## 2. ManiSkill3 评估（参考）

### 2.1 评估结果

使用 Pi0.5 base 模型（未微调）进行 zero-shot 评估：

| Task | Pi0.5 (ours) | Pi0 (云帆微调) |
|------|--------------|----------------|
| PlaceSphere-v1 | 0/40 = 0% | 11/40 = 27.5% |
| PickCube-v1 | 0/40 = 0% | 1/40 = 2.5% |
| StackCube-v1 | 0/40 = 0% | 24/40 = 60% |
| PushCube-v1 | 0/40 = 0% | 28/40 = 70% |
| PullCube-v1 | 0/40 = 0% | 35/40 = 87.5% |
| PullCubeTool-v1 | 0/40 = 0% | 3/40 = 7.5% |

### 2.2 分析

**为什么 Zero-shot 成功率为 0%？**

1. **State 格式不匹配**
   - Pi0.5 训练数据使用 EEF pose（7D: xyz + quaternion）
   - ManiSkill 提供 Joint positions（9D qpos）

2. **Gripper 始终打开**
   - Pi0.5 输出 gripper = -1（打开状态）
   - 抓取任务需要 gripper = 1（闭合）

3. **无 ManiSkill 微调数据**
   - 云帆的 Pi0 是在 ManiSkill 专家轨迹上微调过的
   - Base 模型没有见过 ManiSkill 的 observation 格式

**解决方案**：需要在 ManiSkill 数据上微调，参考云帆的训练流程。

---

## 3. 环境配置参考

### 3.1 模型下载速度对比

| 来源 | 速度 | 21GB 下载时间 |
|------|------|---------------|
| GCS (gsutil) | ~10-30 MB/s | ~12-35分钟 |
| HuggingFace | ~240KB/s | ~25小时 |
| HF 镜像 (hf-mirror.com) | ~30MB/s | ~12分钟 |

**推荐**：国内用户优先使用 gsutil 或 HF 镜像。

### 3.2 显存需求

| 模型 | 显存使用 | 峰值显存 |
|------|----------|----------|
| pi05_base | 14.49 GB | 14.83 GB |
| pi05_libero | ~12 GB | ~14 GB |

建议使用 16GB+ 显存的 GPU。

### 3.3 Checkpoint 路径

```
checkpoints/
├── pi05_base_hf/           # 14GB - Base 模型
├── pi05_droid_hf/          # 6.8GB - DROID 微调
└── pi05_libero/            # 12GB - LIBERO 微调
    ├── params/
    │   └── params
    └── assets/
        ├── norm_stats.json
        └── model_config.json
```

---

## 4. 常见问题

### Q1: Policy Server 启动失败

```bash
# 检查端口是否被占用
lsof -i :8000

# 如果被占用，杀掉进程
kill -9 <PID>
```

### Q2: LIBERO 环境报错 `No module named 'xxx'`

```bash
# 确保安装了所有依赖
pip install bddl easydict cloudpickle gym tyro tqdm imageio
```

### Q3: 渲染失败 `GLFWError`

```bash
# 使用 EGL 无头渲染
export MUJOCO_GL=egl
```

### Q4: HuggingFace 下载超时

```bash
# 使用镜像
export HF_ENDPOINT="https://hf-mirror.com"

# 或使用 gsutil
gsutil -m cp -r "gs://openpi-assets/checkpoints/pi05_libero/*" checkpoints/pi05_libero/
```

---

## 5. 参考资料

- **OpenPI 官方仓库**: https://github.com/Physical-Intelligence/openpi
- **LIBERO 论文**: https://arxiv.org/abs/2310.08587
- **Pi0 论文**: https://arxiv.org/abs/2410.24164
- **ManiSkill3**: https://github.com/haosulab/ManiSkill
