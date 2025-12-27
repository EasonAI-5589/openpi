#!/usr/bin/env python3
"""
Debug script to analyze observation alignment between ManiSkill and Pi0.5 expected format.

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
import json

print("=" * 70)
print("DEBUG: Observation Alignment Analysis")
print("=" * 70)

# 1. ManiSkill observation structure
print("\\n[1] ManiSkill Observation Structure")
print("-" * 50)

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

obs, _ = env.reset(seed=42)

def describe_obs(obs, prefix=""):
    """Recursively describe observation structure."""
    if isinstance(obs, dict):
        for k, v in obs.items():
            describe_obs(v, f"{prefix}{k}.")
    elif isinstance(obs, np.ndarray):
        print(f"{prefix[:-1]}: ndarray shape={obs.shape} dtype={obs.dtype}")
    elif hasattr(obs, "shape"):  # torch tensor
        print(f"{prefix[:-1]}: tensor shape={obs.shape} dtype={obs.dtype}")
    else:
        print(f"{prefix[:-1]}: {type(obs).__name__}")

describe_obs(obs)

# 2. Pi0.5 expected input format
print("\\n[2] Pi0.5 Expected Input Format")
print("-" * 50)

# Check what Pi0.5 expects by looking at norm_stats keys
from openpi.training import checkpoints as _checkpoints

norm_stats = _checkpoints.load_norm_stats(
    "/share/project/guoyichen/openpi/checkpoints/pi05_base_hf/assets",
    "maniskill"
)

print("Norm stats keys (what Pi0.5 expects):")
for key in sorted(norm_stats.keys()):
    stats = norm_stats[key]
    print(f"  {key}: shape={stats.mean.shape if hasattr(stats.mean, 'shape') else 'scalar'}")

# 3. What our ManiSkillInputs transform produces
print("\\n[3] What ManiSkillInputs Transform Produces")
print("-" * 50)

from openpi.policies.maniskill_policy import ManiSkillInputs, ManiSkillRepackInputs

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

# State from agent
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

transform = ManiSkillInputs()
transformed = transform(example)

print("After ManiSkillInputs transform:")
for k, v in transformed.items():
    if isinstance(v, dict):
        for k2, v2 in v.items():
            if hasattr(v2, "shape"):
                print(f"  {k}.{k2}: shape={v2.shape} dtype={v2.dtype}")
            else:
                print(f"  {k}.{k2}: {type(v2).__name__}")
    elif hasattr(v, "shape"):
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")
    else:
        print(f"  {k}: {v}")

# 4. State vector analysis
print("\\n[4] State Vector Analysis")
print("-" * 50)

if "state" in example:
    state = example["state"]
    print(f"State vector length: {len(state)}")
    print(f"State values: {state}")

    # Panda robot has:
    # - 9 joint positions (7 arm + 2 gripper fingers)
    # - 9 joint velocities
    print("\\nExpected Panda state structure:")
    print("  qpos[0:7]: arm joint positions")
    print("  qpos[7:9]: gripper finger positions")
    print("  qvel[0:7]: arm joint velocities")
    print("  qvel[7:9]: gripper finger velocities")

    if len(state) >= 18:
        print("\\nActual values:")
        print(f"  Arm joints (qpos[0:7]): {state[0:7]}")
        print(f"  Gripper (qpos[7:9]): {state[7:9]}")
        print(f"  Arm vel (qvel[0:7]): {state[9:16]}")
        print(f"  Gripper vel (qvel[7:9]): {state[16:18]}")

# 5. Compare with DROID expected format
print("\\n[5] DROID Training Data Format")
print("-" * 50)

# Check what DROID format looks like
print("Pi0.5 was trained on DROID dataset which has:")
print("  - 3 camera views (base, left wrist, right wrist)")
print("  - EEF pose (position + quaternion)")
print("  - Gripper state")
print("\\nManiSkill provides:")
print("  - 1-2 camera views (base_camera, possibly hand_camera)")
print("  - Joint positions/velocities (not EEF pose)")
print("\\n>>> POTENTIAL MISMATCH: Pi0.5 expects EEF pose, we're giving joint states!")

# 6. Check if EEF pose is available
print("\\n[6] Checking for EEF Pose Data")
print("-" * 50)

if "extra" in obs:
    print("Extra observation data:")
    describe_obs(obs["extra"], "extra.")
else:
    print("No 'extra' field in observation")

# Check if we can get EEF pose from environment
try:
    from mani_skill.utils.structs import Pose

    # Get controller info
    agent = env.unwrapped.agent
    print(f"\\nAgent type: {type(agent)}")

    # Get end effector pose
    if hasattr(agent, "tcp"):
        tcp = agent.tcp
        if hasattr(tcp, "pose"):
            pose = tcp.pose
            print(f"TCP pose available: {pose}")
except Exception as e:
    print(f"Could not get EEF pose: {e}")

env.close()

print("\\n" + "=" * 70)
print("SUMMARY: Why Pi0.5 may not work on ManiSkill")
print("=" * 70)
print("""
1. DOMAIN GAP: Pi0.5 trained on real DROID data, ManiSkill is simulation
2. OBSERVATION MISMATCH:
   - Pi0.5 expects: EEF pose + multiple camera views
   - ManiSkill gives: Joint positions + 1-2 cameras
3. ACTION SPACE:
   - Pi0.5 outputs: Absolute EEF pose deltas (DROID format)
   - ManiSkill expects: End-effector delta pose (similar but different scale/frame)
4. VISUAL DOMAIN:
   - DROID: Real robot, real objects, real lighting
   - ManiSkill: Simulated, different textures and rendering

To improve success rate, consider:
1. Provide EEF pose instead of joint positions as state
2. Add more camera views if possible
3. Fine-tune on ManiSkill demonstrations (like 云帆 did with Pi0)
4. Adjust action scaling to match ManiSkill's expected magnitudes
""")
'''

    script_path = "/tmp/debug_observation.py"
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
        for line in result.stderr.split('\n'):
            if 'WARNING' not in line.upper() and 'UserWarning' not in line:
                if line.strip():
                    print(line)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
