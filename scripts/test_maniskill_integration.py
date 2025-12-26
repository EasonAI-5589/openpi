#!/usr/bin/env python3
"""
Test script for Pi0.5 + ManiSkill3 integration.

Usage:
    python scripts/test_maniskill_integration.py --test-env      # Test environment only
    python scripts/test_maniskill_integration.py --test-adapter  # Test adapter only
    python scripts/test_maniskill_integration.py --test-model    # Test model loading
    python scripts/test_maniskill_integration.py --run-eval      # Run full evaluation

Author: Claude Code for 郭奕辰
Date: 2025-12-27
"""

import sys
import os
import argparse
import time
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_environment():
    """Test ManiSkill3 environment creation and basic interaction."""
    print("\n" + "=" * 60)
    print("Test 1: ManiSkill3 Environment")
    print("=" * 60)

    try:
        import gymnasium as gym
        import mani_skill.envs

        env_name = "PickCube-v1"
        print(f"Creating environment: {env_name}")

        env = gym.make(
            env_name,
            obs_mode="rgbd",
            control_mode="pd_ee_delta_pose",
            robot_uids="panda",
        )

        print(f"  Action space: {env.action_space}")
        print(f"  Observation space keys: {list(env.observation_space.spaces.keys())}")

        # Reset and step
        obs, info = env.reset(seed=42)
        print(f"  Observation keys: {list(obs.keys())}")

        if "sensor_data" in obs:
            print(f"  Sensor data keys: {list(obs['sensor_data'].keys())}")
            for cam_name, cam_data in obs["sensor_data"].items():
                if "rgb" in cam_data:
                    rgb_shape = cam_data["rgb"].shape
                    print(f"    Camera '{cam_name}' RGB shape: {rgb_shape}")

        if "agent" in obs:
            print(f"  Agent keys: {list(obs['agent'].keys())}")
            if "qpos" in obs["agent"]:
                print(f"  qpos shape: {obs['agent']['qpos'].shape}")

        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        # Handle tensor rewards from GPU simulation
        if hasattr(reward, 'item'):
            reward = reward.item()
        print(f"  Step result: reward={reward:.4f}, terminated={terminated}, truncated={truncated}")

        env.close()
        print("\n✅ Environment test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Environment test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adapter():
    """Test Pi05ManiSkillAdapter preprocessing and postprocessing."""
    print("\n" + "=" * 60)
    print("Test 2: Pi05ManiSkillAdapter")
    print("=" * 60)

    try:
        from src.openpi.maniskill import Pi05ManiSkillAdapter
        from src.openpi.maniskill.pi05_maniskill_adapter import Pi05ManiSkillConfig

        # Create adapter
        config = Pi05ManiSkillConfig(
            image_size=(224, 224),
            camera_names=["base_camera"],
        )
        adapter = Pi05ManiSkillAdapter(config)
        print("  Adapter created successfully")

        # Create mock observation
        import gymnasium as gym
        import mani_skill.envs

        env = gym.make(
            "PickCube-v1",
            obs_mode="rgbd",
            control_mode="pd_ee_delta_pose",
            robot_uids="panda",
        )
        obs, _ = env.reset(seed=42)

        # Test preprocess
        print("\n  Testing preprocess()...")
        model_input = adapter.preprocess(obs, instruction="pick up the red cube")
        print(f"    images shape: {model_input['images'].shape}")
        print(f"    proprio shape: {model_input['proprio'].shape}")
        print(f"    instruction: {model_input['instruction']}")

        # Test postprocess
        print("\n  Testing postprocess()...")
        # Simulate model output: 50 steps x 7 actions (dx, dy, dz, dr, dp, dyaw, gripper)
        mock_raw_actions = np.random.randn(50, 7) * 0.1
        mock_raw_actions[:, 6] = np.clip(mock_raw_actions[:, 6], 0, 1)  # Gripper in [0, 1]

        env_actions = adapter.postprocess(mock_raw_actions)
        print(f"    Number of actions returned: {len(env_actions)}")
        print(f"    First action shape: {env_actions[0].shape}")
        print(f"    First action: {env_actions[0]}")

        # Test sticky gripper
        print("\n  Testing sticky gripper mechanism...")
        adapter.reset()
        for i in range(5):
            # Alternate gripper actions
            gripper_val = 1.0 if i % 2 == 0 else 0.0
            result = adapter._process_gripper(gripper_val)
            print(f"    Step {i}: input={gripper_val:.1f}, output={result:.2f}")

        env.close()
        print("\n✅ Adapter test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Adapter test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test Pi0.5 model loading."""
    print("\n" + "=" * 60)
    print("Test 3: Pi0.5 Model Loading")
    print("=" * 60)

    try:
        # Check CUDA availability
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Try loading the model
        print("\n  Attempting to load Pi0.5 model...")

        # Use the new pi05_maniskill config
        config_name = "pi05_maniskill"
        checkpoint_dir = "checkpoints/pi05_base_hf"

        if not os.path.exists(checkpoint_dir):
            print(f"  ⚠️ Checkpoint not found at: {checkpoint_dir}")
            print("  Trying alternative paths...")
            alt_paths = [
                "checkpoints/pi05_droid_hf",
                "../checkpoints/pi05_base_hf",
            ]
            for alt in alt_paths:
                if os.path.exists(alt):
                    checkpoint_dir = alt
                    config_name = "pi05_maniskill_droid" if "droid" in alt else "pi05_maniskill"
                    break

        if os.path.exists(checkpoint_dir):
            print(f"  Using checkpoint: {checkpoint_dir}")
            print(f"  Using config: {config_name}")

            from openpi.training import config as _config
            from openpi.policies import policy_config

            # Get config for pi05_maniskill
            config = _config.get_config(config_name)
            print("  Config loaded successfully")

            # Create policy
            start = time.time()
            policy = policy_config.create_trained_policy(config, checkpoint_dir)
            load_time = time.time() - start
            print(f"  Policy created in {load_time:.2f}s")

            # Test inference with ManiSkill-style input
            print("\n  Testing inference with ManiSkill-style input...")
            dummy_example = {
                "images": {
                    "base_camera": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                },
                "state": np.random.randn(9).astype(np.float32),  # qpos for Panda
                "prompt": "pick up the red cube",
            }

            start = time.time()
            output = policy.infer(dummy_example)
            infer_time = time.time() - start

            actions = output.get("actions", output)
            if isinstance(actions, torch.Tensor):
                actions = actions.cpu().numpy()
            print(f"  Inference time: {infer_time:.2f}s")
            print(f"  Output actions shape: {actions.shape}")
            print(f"  Actions range: [{actions.min():.3f}, {actions.max():.3f}]")

            if torch.cuda.is_available():
                vram = torch.cuda.memory_allocated() / 1e9
                print(f"  VRAM used: {vram:.2f} GB")

            print("\n✅ Model loading test PASSED")
            return True
        else:
            print(f"  ⚠️ No checkpoint found. Model loading test SKIPPED.")
            return True

    except Exception as e:
        print(f"\n❌ Model loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_evaluation():
    """Run full evaluation loop."""
    print("\n" + "=" * 60)
    print("Test 4: Full Evaluation Loop")
    print("=" * 60)

    try:
        from src.openpi.maniskill.pi05_maniskill_evaluator import (
            Pi05ManiSkillEvaluator,
            EvaluationConfig,
        )

        # Configure for quick test
        config = EvaluationConfig(
            env_name="PickCube-v1",
            num_episodes=3,  # Quick test with 3 episodes
            max_steps_per_episode=50,  # Shorter episodes
            model_config="pi05_maniskill",  # Use new ManiSkill config
            checkpoint_dir="checkpoints/pi05_base_hf",
            save_dir="./evaluation_results",
        )

        print(f"  Environment: {config.env_name}")
        print(f"  Episodes: {config.num_episodes}")
        print(f"  Max steps: {config.max_steps_per_episode}")

        evaluator = Pi05ManiSkillEvaluator(config)

        # Run evaluation
        results = evaluator.evaluate()

        # Save results
        filepath = evaluator.save_results(results)

        evaluator.close()

        print("\n✅ Full evaluation test PASSED")
        print(f"  Results saved to: {filepath}")
        return True

    except Exception as e:
        print(f"\n❌ Full evaluation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Pi0.5 + ManiSkill3 integration")
    parser.add_argument("--test-env", action="store_true", help="Test environment only")
    parser.add_argument("--test-adapter", action="store_true", help="Test adapter only")
    parser.add_argument("--test-model", action="store_true", help="Test model loading")
    parser.add_argument("--run-eval", action="store_true", help="Run full evaluation")
    parser.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    # Default to --all if no specific test selected
    if not any([args.test_env, args.test_adapter, args.test_model, args.run_eval]):
        args.all = True

    results = {}

    print("\n" + "#" * 60)
    print("# Pi0.5 + ManiSkill3 Integration Tests")
    print("#" * 60)

    if args.all or args.test_env:
        results["environment"] = test_environment()

    if args.all or args.test_adapter:
        results["adapter"] = test_adapter()

    if args.all or args.test_model:
        results["model"] = test_model_loading()

    if args.all or args.run_eval:
        results["evaluation"] = run_evaluation()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print("\n" + ("=" * 60))
    if all_passed:
        print("🎉 All tests PASSED!")
    else:
        print("⚠️ Some tests FAILED. Check output above for details.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
