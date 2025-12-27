#!/usr/bin/env bash
set -euo pipefail

# Usage example:
  # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  # PYTHONPATH=/share/project/yunfan/RL/openpi/src:$PYTHONPATH \
  # bash scripts/run_maniskill_eval.sh \
  #   --config-name pi0_maniskill_stackcube \
  #   --exp-name stackcube \
  #   --env-id StackCube-v1 \
  #   --num-episodes 20 \
  #   --save-video \
  #   --video-dir /share/project/yunfan/RL/caurft/openpi/video_new

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# 如果没有传递参数，使用默认的示例参数（带随机种子）
if [ $# -eq 0 ]; then
  # 使用随机种子进行测试
  RANDOM_SEED=$RANDOM
  echo "No arguments provided, using default example with random seed: $RANDOM_SEED"
  
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/share/project/yunfan/RL/openpi/src:$PYTHONPATH \
  python "$REPO_ROOT/scripts/eval_maniskill.py" \
    --config-name pi0_maniskill_pushcube \
    --exp-name pushcube \
    --env-id PushCube-v1 \
    --checkpoint-step 19999 \
    --num-episodes 30 \
    --save-video True \
    --prompt "Push and move a cube to a goal region in front of it." \
    --seed $RANDOM_SEED \
    --video-dir /share/project/yunfan/RL/caurft/openpi/video_pushcube
else
  # 如果传递了参数，直接使用传递的参数
  python "$REPO_ROOT/scripts/eval_maniskill.py" "$@"
fi
  
  # 如果需要多次随机测试，可以使用循环：
  # for i in {1..5}; do
  #   RANDOM_SEED=$RANDOM
  #   echo "Test $i: Using random seed: $RANDOM_SEED"
  #   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
  #   PYTHONPATH=/share/project/yunfan/RL/openpi/src:$PYTHONPATH \
  #   bash /share/project/yunfan/RL/caurft/openpi/scripts/run_maniskill_eval.sh \
  #     --config-name pi0_maniskill_pushcube\
  #     --exp-name pushcube_${i} \
  #     --env-id PushCube-v1 \
  #     --checkpoint-step 19999 \
  #     --num-episodes 30 \
  #     --save-video True \
  #     --prompt "Push and move a cube to a goal region in front of it." \
  #     --seed $RANDOM_SEED \
  #     --video-dir /share/project/yunfan/RL/caurft/openpi/video_pushcube_${i}
  # done
    

    # "Pick up a orange-white peg and insert the orange end into the box with a hole in it."
    # "Grasp a red cube and move it to a target goal position."
    # "The robot must pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot."
    # "Push and move a cube to a goal region in front of it."
    # "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling"
    # "Place the sphere into the shallow bin"
    # "Pull a cube onto a target."
    # "Given an L-shaped tool that is within the reach of the robot, leverage the tool to pull a cube that is out of it's reach"