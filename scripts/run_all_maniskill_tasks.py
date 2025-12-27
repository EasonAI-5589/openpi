#!/usr/bin/env python3
"""
Run Pi0.5 evaluation on all ManiSkill3 tasks.

Usage:
    python scripts/run_all_maniskill_tasks.py

Author: Claude Code for 郭奕辰
Date: 2025-12-27
"""

import sys
import os
import json
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables before imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

# ManiSkill tasks to evaluate
MANISKILL_TASKS = [
    "PickCube-v1",
    "StackCube-v1",
    "PegInsertionSide-v1",
    "PlugCharger-v1",
    "PushCube-v1",
    "TurnFaucet-v1",  # Now available after downloading partnet_mobility assets
]

TASK_INSTRUCTIONS = {
    "PickCube-v1": "pick up the red cube",
    "StackCube-v1": "stack the red cube on the green cube",
    "PegInsertionSide-v1": "insert the peg into the hole",
    "PlugCharger-v1": "plug the charger into the socket",
    "PushCube-v1": "push the cube to the target",
    "TurnFaucet-v1": "turn the faucet handle",
}


def check_environment(env_name):
    """Check if environment is registered (without creating it)."""
    try:
        import gymnasium as gym
        import mani_skill.envs

        # Just check if the environment is registered
        spec = gym.spec(env_name)
        return True, None
    except Exception as e:
        return False, str(e)


def run_evaluation_subprocess(env_name, num_episodes=10, max_steps=200):
    """Run evaluation for a single environment in a subprocess to avoid GPU PhysX conflicts."""
    import subprocess
    import tempfile

    print(f"\n{'='*60}")
    print(f"Evaluating: {env_name}")
    print(f"{'='*60}")

    # Create a temporary script for this task
    project_root = "/share/project/guoyichen/openpi"
    script_content = f'''
import sys
import os
sys.path.insert(0, "{project_root}")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import numpy as np

from src.openpi.maniskill.pi05_maniskill_evaluator import (
    Pi05ManiSkillEvaluator,
    EvaluationConfig,
)

config = EvaluationConfig(
    env_name="{env_name}",
    num_episodes={num_episodes},
    max_steps_per_episode={max_steps},
    model_config="pi05_maniskill",
    checkpoint_dir="checkpoints/pi05_base_hf",
    save_dir="./evaluation_results",
)

evaluator = Pi05ManiSkillEvaluator(config)

try:
    results = evaluator.evaluate()
    evaluator.close()
    # Save results to temp file
    results_converted = {{}}
    for k, v in results.items():
        if isinstance(v, np.floating):
            results_converted[k] = float(v)
        elif isinstance(v, np.ndarray):
            results_converted[k] = v.tolist()
        else:
            results_converted[k] = v
    print("RESULTS_JSON:" + json.dumps(results_converted))
except Exception as e:
    import traceback
    traceback.print_exc()
    print("RESULTS_JSON:" + json.dumps({{"env_name": "{env_name}", "error": str(e), "success_rate": 0.0, "num_episodes": 0}}))
'''

    # Write temp script
    script_path = f"/tmp/eval_{env_name.replace('-', '_')}.py"
    with open(script_path, "w") as f:
        f.write(script_content)

    # Run subprocess
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            cwd="/share/project/guoyichen/openpi",
            timeout=1800,  # 30 min timeout per task
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        )

        # Print output
        print(result.stdout)
        if result.stderr:
            # Filter out common warnings
            stderr_lines = result.stderr.split('\n')
            for line in stderr_lines:
                if 'WARNING' not in line and 'UserWarning' not in line:
                    if line.strip():
                        print(line)

        # Extract results from output
        for line in result.stdout.split('\n'):
            if line.startswith("RESULTS_JSON:"):
                results_json = line[len("RESULTS_JSON:"):]
                return json.loads(results_json)

        return {
            "env_name": env_name,
            "error": "No results found in output",
            "success_rate": 0.0,
            "num_episodes": 0,
        }

    except subprocess.TimeoutExpired:
        return {
            "env_name": env_name,
            "error": "Timeout (30 min)",
            "success_rate": 0.0,
            "num_episodes": 0,
        }
    except Exception as e:
        return {
            "env_name": env_name,
            "error": str(e),
            "success_rate": 0.0,
            "num_episodes": 0,
        }
    finally:
        # Clean up temp script
        if os.path.exists(script_path):
            os.remove(script_path)


def main():
    print("\n" + "#" * 60)
    print("# Pi0.5 + ManiSkill3 Full Evaluation")
    print("#" * 60)

    # Check CUDA
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Configuration
    num_episodes = 10  # Episodes per task
    max_steps = 200    # Max steps per episode

    print(f"\nConfiguration:")
    print(f"  Episodes per task: {num_episodes}")
    print(f"  Max steps per episode: {max_steps}")
    print(f"  Tasks to evaluate: {len(MANISKILL_TASKS)}")

    # Check which environments are available
    print("\nChecking environment availability...")
    available_tasks = []
    for task in MANISKILL_TASKS:
        ok, error = check_environment(task)
        if ok:
            print(f"  ✅ {task}")
            available_tasks.append(task)
        else:
            print(f"  ❌ {task}: {error[:50]}...")

    if not available_tasks:
        print("\nNo tasks available! Exiting.")
        return 1

    print(f"\nWill evaluate {len(available_tasks)} tasks")

    # Run evaluations
    all_results = {}
    start_time = time.time()

    for i, task in enumerate(available_tasks):
        print(f"\n[{i+1}/{len(available_tasks)}] Starting {task}...")
        task_start = time.time()

        results = run_evaluation_subprocess(task, num_episodes, max_steps)
        all_results[task] = results

        task_time = time.time() - task_start
        print(f"  Completed in {task_time:.1f}s")

        # Print interim results
        if "success_rate" in results:
            print(f"  Success rate: {results['success_rate']*100:.1f}%")

        # Clear GPU memory between tasks
        torch.cuda.empty_cache()

    total_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Tasks evaluated: {len(all_results)}")
    print()

    # Results table
    print(f"{'Task':<25} {'Success Rate':<15} {'Episodes':<10} {'Avg Reward':<12}")
    print("-" * 62)

    for task, results in all_results.items():
        if "error" in results:
            print(f"{task:<25} {'ERROR':<15} {'-':<10} {'-':<12}")
        else:
            sr = results.get("success_rate", 0) * 100
            eps = results.get("num_episodes", 0)
            avg_r = results.get("average_reward", 0)
            print(f"{task:<25} {sr:>6.1f}%        {eps:<10} {avg_r:>8.2f}")

    # Save consolidated results
    results_dir = "./evaluation_results"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"all_tasks_{timestamp}.json")

    summary = {
        "timestamp": timestamp,
        "total_time_seconds": total_time,
        "num_tasks": len(all_results),
        "configuration": {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "model_config": "pi05_maniskill",
            "checkpoint": "pi05_base_hf",
        },
        "results": all_results,
    }

    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\nResults saved to: {results_file}")

    # Also save a markdown summary
    md_file = os.path.join(results_dir, f"summary_{timestamp}.md")
    with open(md_file, "w") as f:
        f.write("# Pi0.5 + ManiSkill3 Evaluation Results\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Time**: {total_time/60:.1f} minutes\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- Model: pi05_base_hf\n")
        f.write(f"- Episodes per task: {num_episodes}\n")
        f.write(f"- Max steps: {max_steps}\n\n")
        f.write("## Results\n\n")
        f.write("| Task | Success Rate | Episodes | Avg Reward |\n")
        f.write("|------|--------------|----------|------------|\n")
        for task, results in all_results.items():
            if "error" in results:
                f.write(f"| {task} | ERROR | - | - |\n")
            else:
                sr = results.get("success_rate", 0) * 100
                eps = results.get("num_episodes", 0)
                avg_r = results.get("average_reward", 0)
                f.write(f"| {task} | {sr:.1f}% | {eps} | {avg_r:.2f} |\n")
        f.write("\n## Notes\n\n")
        f.write("- This is a zero-shot evaluation (no fine-tuning on ManiSkill tasks)\n")
        f.write("- Low success rates are expected as Pi0.5 was trained on DROID/OXE data\n")
        f.write("- The model demonstrates basic motion capabilities\n")

    print(f"Summary saved to: {md_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
