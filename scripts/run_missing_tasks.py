#!/usr/bin/env python3
"""
Run Pi0.5 evaluation on the 3 missing tasks that 云帆 evaluated.

Tasks:
- PlaceSphere-v1: 11/40 = 27.50% (云帆 Pi0)
- PullCube-v1: 35/40 = 87.50% (云帆 Pi0)
- PullCubeTool-v1: 3/40 = 7.50% (云帆 Pi0)

Usage:
    python scripts/run_missing_tasks.py

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

# The 3 missing tasks
MISSING_TASKS = [
    "PlaceSphere-v1",
    "PullCube-v1",
    "PullCubeTool-v1",
]

TASK_INSTRUCTIONS = {
    "PlaceSphere-v1": "place the sphere at the target location",
    "PullCube-v1": "pull the cube towards you",
    "PullCubeTool-v1": "use the tool to pull the cube",
}

# 云帆 Pi0 results for comparison
YUNFAN_RESULTS = {
    "PlaceSphere-v1": "11/40 = 27.50%",
    "PullCube-v1": "35/40 = 87.50%",
    "PullCubeTool-v1": "3/40 = 7.50%",
}


def run_single_task(env_name, num_episodes=40, max_steps=200):
    """Run evaluation for a single task with multiple seeds."""
    import subprocess

    print(f"\n{'='*60}")
    print(f"Evaluating: {env_name} ({num_episodes} episodes)")
    print(f"云帆 Pi0 result: {YUNFAN_RESULTS.get(env_name, 'N/A')}")
    print(f"{'='*60}")

    project_root = "/share/project/guoyichen/openpi"
    python_bin = "/share/project/guoyichen/miniconda3/envs/openpi/bin/python"

    script_content = f'''
import sys
import os
sys.path.insert(0, "{project_root}")
sys.path.insert(0, "{project_root}/src")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import numpy as np

from openpi.maniskill.pi05_maniskill_evaluator import (
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
    seed=42,
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

    script_path = f"/tmp/eval_missing_{env_name.replace('-', '_')}.py"
    with open(script_path, "w") as f:
        f.write(script_content)

    try:
        result = subprocess.run(
            [python_bin, script_path],
            capture_output=True,
            text=True,
            cwd=project_root,
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
    print("# Pi0.5 Evaluation: Missing Tasks")
    print("# (Aligning with 云帆's Pi0 evaluation)")
    print("#" * 60)

    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    num_episodes = 40
    max_steps = 200

    print(f"\nConfiguration:")
    print(f"  Episodes per task: {num_episodes}")
    print(f"  Max steps per episode: {max_steps}")
    print(f"  Tasks: {len(MISSING_TASKS)}")

    all_results = {}
    start_time = time.time()

    for i, task in enumerate(MISSING_TASKS):
        print(f"\n[{i+1}/{len(MISSING_TASKS)}] {task}")
        task_start = time.time()

        results = run_single_task(task, num_episodes, max_steps)
        all_results[task] = results

        task_time = time.time() - task_start

        num_success = results.get("num_success", 0)
        num_eps = results.get("num_episodes", num_episodes)
        success_pct = results.get("success_rate", 0) * 100
        print(f"  {task} Success: {num_success}/{num_eps} = {success_pct:.2f}%")
        print(f"  Time: {task_time/60:.1f} min")

        torch.cuda.empty_cache()

    total_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (Missing Tasks)")
    print("=" * 60)
    print(f"Total time: {total_time/60:.1f} minutes\n")

    print(f"{'Task':<20} {'Pi0.5 (ours)':<20} {'Pi0 (云帆)':<20}")
    print("-" * 60)

    for task in MISSING_TASKS:
        if task in all_results:
            r = all_results[task]
            if "error" in r and r.get("num_episodes", 0) == 0:
                our_result = f"ERROR: {r.get('error', 'unknown')[:15]}"
            else:
                num_success = r.get("num_success", 0)
                num_eps = r.get("num_episodes", 40)
                our_result = f"{num_success}/{num_eps} = {r.get('success_rate', 0)*100:.2f}%"
        else:
            our_result = "N/A"

        yunfan = YUNFAN_RESULTS.get(task, "N/A")
        print(f"{task:<20} {our_result:<20} {yunfan:<20}")

    # Save results
    results_dir = "./evaluation_results"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"missing_tasks_{timestamp}.json")

    summary = {
        "timestamp": timestamp,
        "model": "pi05_base",
        "total_time_minutes": total_time / 60,
        "episodes_per_task": num_episodes,
        "max_steps": max_steps,
        "results": all_results,
        "yunfan_pi0_results": YUNFAN_RESULTS,
    }

    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\nResults saved to: {results_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
