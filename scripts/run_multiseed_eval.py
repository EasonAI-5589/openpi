#!/usr/bin/env python3
"""
Run Pi0.5 multi-seed evaluation on ManiSkill3 tasks.

Similar to 云帆's Pi0 evaluation format:
- 40 episodes per task
- Multiple random seeds
- Report success rate as X/40

Usage:
    python scripts/run_multiseed_eval.py

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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

# ManiSkill tasks matching 云帆's evaluation
MANISKILL_TASKS = [
    "PickCube-v1",      # pick cube
    "StackCube-v1",     # stack cube
    "PushCube-v1",      # push cube
    "PegInsertionSide-v1",  # precision task
    "PlugCharger-v1",   # plug task
    "TurnFaucet-v1",    # articulated object
]

TASK_INSTRUCTIONS = {
    "PickCube-v1": "pick up the red cube",
    "StackCube-v1": "stack the red cube on the green cube",
    "PushCube-v1": "push the cube to the target",
    "PegInsertionSide-v1": "insert the peg into the hole",
    "PlugCharger-v1": "plug the charger into the socket",
    "TurnFaucet-v1": "turn the faucet handle",
}


def run_single_task(env_name, num_episodes=40, max_steps=200):
    """Run evaluation for a single task with multiple seeds."""
    import subprocess

    print(f"\n{'='*60}")
    print(f"Evaluating: {env_name} ({num_episodes} episodes)")
    print(f"{'='*60}")

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
    seed=42,  # Base seed, will vary per episode
)

evaluator = Pi05ManiSkillEvaluator(config)

try:
    results = evaluator.evaluate()
    evaluator.close()

    # Convert results
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
    print("RESULTS_JSON:" + json.dumps({{"env_name": "{env_name}", "error": str(e), "success_rate": 0.0, "num_episodes": 0, "num_success": 0}}))
'''

    script_path = f"/tmp/eval_multiseed_{env_name.replace('-', '_')}.py"
    with open(script_path, "w") as f:
        f.write(script_content)

    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            cwd="/share/project/guoyichen/openpi",
            timeout=3600,  # 1 hour timeout for 40 episodes
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        )

        print(result.stdout)
        if result.stderr:
            stderr_lines = result.stderr.split('\n')
            for line in stderr_lines:
                if 'WARNING' not in line and 'UserWarning' not in line:
                    if line.strip():
                        print(line)

        for line in result.stdout.split('\n'):
            if line.startswith("RESULTS_JSON:"):
                results_json = line[len("RESULTS_JSON:"):]
                return json.loads(results_json)

        return {
            "env_name": env_name,
            "error": "No results found",
            "success_rate": 0.0,
            "num_episodes": 0,
            "num_success": 0,
        }

    except subprocess.TimeoutExpired:
        return {
            "env_name": env_name,
            "error": "Timeout",
            "success_rate": 0.0,
            "num_episodes": 0,
            "num_success": 0,
        }
    except Exception as e:
        return {
            "env_name": env_name,
            "error": str(e),
            "success_rate": 0.0,
            "num_episodes": 0,
            "num_success": 0,
        }
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)


def main():
    print("\n" + "#" * 60)
    print("# Pi0.5 + ManiSkill3 Multi-Seed Evaluation")
    print("# (Matching 云帆's Pi0 evaluation format)")
    print("#" * 60)

    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    num_episodes = 40  # Match 云帆's evaluation
    max_steps = 200

    print(f"\nConfiguration:")
    print(f"  Episodes per task: {num_episodes}")
    print(f"  Max steps per episode: {max_steps}")
    print(f"  Tasks: {len(MANISKILL_TASKS)}")

    all_results = {}
    start_time = time.time()

    for i, task in enumerate(MANISKILL_TASKS):
        print(f"\n[{i+1}/{len(MANISKILL_TASKS)}] {task}")
        task_start = time.time()

        results = run_single_task(task, num_episodes, max_steps)
        all_results[task] = results

        task_time = time.time() - task_start

        # Print in 云帆's format
        num_success = results.get("num_success", 0)
        num_eps = results.get("num_episodes", num_episodes)
        success_pct = results.get("success_rate", 0) * 100
        print(f"  {task} Success: {num_success}/{num_eps} = {success_pct:.2f}%")
        print(f"  Time: {task_time/60:.1f} min")

        torch.cuda.empty_cache()

    total_time = time.time() - start_time

    # Summary in 云帆's format
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (Pi0.5 on ManiSkill3)")
    print("=" * 60)
    print(f"Total time: {total_time/60:.1f} minutes\n")

    print("Results (matching 云帆's Pi0 format):")
    print("-" * 40)

    for task, results in all_results.items():
        if "error" in results and results.get("num_episodes", 0) == 0:
            print(f"{task}: ERROR - {results.get('error', 'unknown')}")
        else:
            num_success = results.get("num_success", 0)
            num_eps = results.get("num_episodes", 40)
            success_pct = results.get("success_rate", 0) * 100
            print(f"{task} Success: {num_success}/{num_eps} = {success_pct:.2f}%")

    # Save results
    results_dir = "./evaluation_results"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"multiseed_40eps_{timestamp}.json")

    summary = {
        "timestamp": timestamp,
        "model": "pi05_base",
        "total_time_minutes": total_time / 60,
        "episodes_per_task": num_episodes,
        "max_steps": max_steps,
        "results": all_results,
    }

    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\nResults saved to: {results_file}")

    # Comparison with 云帆's Pi0 results
    print("\n" + "=" * 60)
    print("COMPARISON: Pi0.5 (ours) vs Pi0 (云帆)")
    print("=" * 60)
    print(f"{'Task':<25} {'Pi0.5 (ours)':<15} {'Pi0 (云帆)':<15}")
    print("-" * 55)

    yunfan_results = {
        "PickCube-v1": "1/40 = 2.50%",
        "StackCube-v1": "24/40 = 60.00%",
        "PushCube-v1": "28/40 = 70.00%",
    }

    for task in MANISKILL_TASKS:
        if task in all_results:
            r = all_results[task]
            num_success = r.get("num_success", 0)
            num_eps = r.get("num_episodes", 40)
            our_result = f"{num_success}/{num_eps} = {r.get('success_rate', 0)*100:.2f}%"
        else:
            our_result = "N/A"

        yunfan = yunfan_results.get(task, "N/A")
        print(f"{task:<25} {our_result:<15} {yunfan:<15}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
