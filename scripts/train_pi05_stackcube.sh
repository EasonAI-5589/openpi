#!/bin/bash
# Pi0.5 ManiSkill StackCube SFT 训练 - 8卡 H100

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd /share/project/guoyichen/openpi

echo "Starting Pi0.5 SFT training with 8x H100 GPUs..."
echo "Estimated time: ~1.5-2 hours for 30k steps"
echo ""

/share/project/guoyichen/miniconda3/envs/openpi/bin/python -m openpi.training.train \
    pi05_maniskill_stackcube \
    --exp-name=stackcube_sft_v1

echo "Training completed!"
