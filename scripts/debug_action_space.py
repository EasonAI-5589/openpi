#!/usr/bin/env python3
"""
Debug script to analyze action space mismatch between Pi0.5 and ManiSkill.

This script:
1. Loads Pi0.5 model and runs inference
2. Shows raw action outputs before/after transforms
3. Compares with ManiSkill action space bounds

Author: Claude Code for 郭奕辰
Date: 2025-12-27
"""

import subprocess
import sys
import os

def main():
    project_root = "/share/project/guoyichen/openpi"

    script_content = '''
import sys
import os
sys.path.insert(0, "/share/project/guoyichen/openpi")
sys.path.insert(0, "/share/project/guoyichen/openpi/src")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

print("=" * 60)
print("DEBUG: Action Space Analysis")
print("=" * 60)

# 1. Check ManiSkill action space
print("\\n[1] ManiSkill Action Space")
print("-" * 40)

import gymnasium as gym
import mani_skill.envs

env = gym.make(
    "PickCube-v1",
    obs_mode="rgbd",
    control_mode="pd_ee_delta_pose",
    robot_uids="panda",
    sim_backend="gpu" if torch.cuda.is_available() else "cpu",
    num_envs=1,
)

print(f"Action space: {env.action_space}")
print(f"Action shape: {env.action_space.shape}")
print(f"Action low:  {env.action_space.low}")
print(f"Action high: {env.action_space.high}")

# 2. Load Pi0.5 model and get sample actions
print("\\n[2] Pi0.5 Model Output")
print("-" * 40)

from openpi.training import config as _config
from openpi.policies import policy_config

config = _config.get_config("pi05_maniskill")
policy = policy_config.create_trained_policy(config, "checkpoints/pi05_base_hf")

# Get sample observation
obs, _ = env.reset(seed=42)

# Prepare input
example = {}
images = {}
if "sensor_data" in obs:
    for cam_name, cam_data in obs["sensor_data"].items():
        if "rgb" in cam_data:
            img = cam_data["rgb"]
            if hasattr(img, "cpu"):
                img = img.cpu().numpy()
            if img.ndim == 4:
                img = img[0]
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            images[cam_name] = img

example["images"] = images

# State
if "agent" in obs:
    agent = obs["agent"]
    state_parts = []
    if "qpos" in agent:
        qpos = agent["qpos"]
        if hasattr(qpos, "cpu"):
            qpos = qpos.cpu().numpy()
        if qpos.ndim == 2:
            qpos = qpos[0]
        state_parts.append(qpos)
    if "qvel" in agent:
        qvel = agent["qvel"]
        if hasattr(qvel, "cpu"):
            qvel = qvel.cpu().numpy()
        if qvel.ndim == 2:
            qvel = qvel[0]
        state_parts.append(qvel)
    if state_parts:
        example["state"] = np.concatenate(state_parts).astype(np.float32)

example["prompt"] = "pick up the red cube"

print(f"Input state shape: {example['state'].shape if 'state' in example else 'None'}")
print(f"Input images: {list(example['images'].keys())}")

# Run inference
with torch.no_grad():
    output = policy.infer(example)

actions = output["actions"]
print(f"\\nOutput actions shape: {actions.shape}")
print(f"Output actions dtype: {actions.dtype}")

# Analyze action statistics
print("\\n[3] Action Statistics (after transform)")
print("-" * 40)

print(f"First 5 actions:")
for i in range(min(5, len(actions))):
    print(f"  Step {i}: {actions[i]}")

print(f"\\nAction ranges per dimension:")
for i in range(actions.shape[1]):
    col = actions[:, i]
    print(f"  Dim {i}: min={col.min():.6f}, max={col.max():.6f}, mean={col.mean():.6f}, std={col.std():.6f}")

# Compare with ManiSkill bounds
print("\\n[4] Comparison with ManiSkill Bounds")
print("-" * 40)

env_low = env.action_space.low
env_high = env.action_space.high

for i in range(min(actions.shape[1], len(env_low))):
    action_range = (actions[:, i].min(), actions[:, i].max())
    env_range = (env_low[i], env_high[i])

    # Check if actions are within bounds
    in_bounds = (action_range[0] >= env_range[0]) and (action_range[1] <= env_range[1])

    # Check if actions are too small (< 1% of range)
    env_width = env_range[1] - env_range[0]
    action_width = action_range[1] - action_range[0]
    too_small = action_width < 0.01 * env_width

    status = "OK" if in_bounds else "OUT OF BOUNDS"
    if too_small:
        status += " + TOO SMALL"

    print(f"  Dim {i}: action=[{action_range[0]:.6f}, {action_range[1]:.6f}] vs env=[{env_range[0]:.2f}, {env_range[1]:.2f}] -> {status}")

# 5. Test stepping with actions
print("\\n[5] Test Stepping Environment")
print("-" * 40)

obs, _ = env.reset(seed=42)
total_reward = 0

for step in range(10):
    action = actions[step] if step < len(actions) else actions[-1]

    # Clip to bounds
    action_clipped = np.clip(action, env_low, env_high)

    obs, reward, terminated, truncated, info = env.step(action_clipped)

    if hasattr(reward, "item"):
        reward = reward.item()
    total_reward += reward

    print(f"  Step {step}: action_norm={np.linalg.norm(action[:6]):.4f}, reward={reward:.4f}")

print(f"\\nTotal reward (10 steps): {total_reward:.4f}")

# 6. Summary and recommendations
print("\\n" + "=" * 60)
print("DIAGNOSIS SUMMARY")
print("=" * 60)

# Check if actions are mostly zeros or too small
action_magnitude = np.abs(actions[:, :6]).mean()  # Exclude gripper
print(f"\\nMean action magnitude (pos+rot): {action_magnitude:.6f}")

if action_magnitude < 0.01:
    print("\\n>>> PROBLEM DETECTED: Actions are TOO SMALL!")
    print("    Pi0.5 outputs very small deltas but ManiSkill expects [-1, 1] range.")
    print("\\n    SOLUTION: Increase action scaling in ManiSkillOutputs:")
    print("      - position_scale: try 10.0 - 50.0")
    print("      - rotation_scale: try 5.0 - 20.0")
elif action_magnitude < 0.1:
    print("\\n>>> WARNING: Actions may be too small for effective control.")
    print("    Consider increasing action scales.")
else:
    print("\\n>>> Actions appear to be in reasonable range.")

# Check gripper values
gripper_vals = actions[:, 6]
print(f"\\nGripper values: min={gripper_vals.min():.4f}, max={gripper_vals.max():.4f}")
unique_gripper = np.unique(gripper_vals)
print(f"Unique gripper values: {unique_gripper}")

env.close()
print("\\nDone!")
'''

    script_path = "/tmp/debug_action_space.py"
    with open(script_path, "w") as f:
        f.write(script_content)

    python_bin = "/share/project/guoyichen/miniconda3/envs/openpi/bin/python"
    result = subprocess.run(
        [python_bin, script_path],
        capture_output=True,
        text=True,
        cwd=project_root,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        timeout=300,
    )

    print(result.stdout)
    if result.stderr:
        # Filter common warnings
        for line in result.stderr.split('\n'):
            if 'WARNING' not in line.upper() and 'UserWarning' not in line:
                if line.strip():
                    print(line)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
