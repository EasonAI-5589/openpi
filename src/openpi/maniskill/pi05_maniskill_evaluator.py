"""
Pi0.5 + ManiSkill3 Evaluator

This module provides the evaluation loop for Pi0.5 model on ManiSkill3 environments.
Handles model loading, environment creation, rollout execution, and result logging.

Author: Claude Code for 郭奕辰
Date: 2025-12-27
"""

import os
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
from tqdm import tqdm

try:
    import gymnasium as gym
    import mani_skill.envs  # Register ManiSkill environments
except ImportError:
    print("Warning: ManiSkill3 not installed. Run: pip install mani-skill")

from .pi05_maniskill_adapter import Pi05ManiSkillAdapter, Pi05ManiSkillConfig


@dataclass
class EvaluationConfig:
    """Configuration for Pi0.5 ManiSkill evaluation."""

    # Environment settings
    env_name: str = "PickCube-v1"
    num_episodes: int = 10
    max_steps_per_episode: int = 200
    seed: int = 42

    # Robot settings
    robot_uid: str = "panda"  # "panda" or "xmate3_robotiq"
    control_mode: str = "pd_ee_delta_pose"  # End-effector delta pose control
    obs_mode: str = "rgbd"  # "rgbd", "state", "state_dict"
    render_mode: str = None  # "human", "rgb_array", or None

    # GPU settings
    sim_backend: str = "gpu"  # "gpu" or "cpu"
    num_envs: int = 1  # Number of parallel environments

    # Model settings
    model_config: str = "pi05_base"  # or "pi05_droid"
    checkpoint_dir: str = None  # Path to checkpoint
    device: str = "cuda"

    # Adapter settings
    adapter_config: Pi05ManiSkillConfig = field(default_factory=Pi05ManiSkillConfig)

    # Logging settings
    save_dir: str = "./evaluation_results"
    save_videos: bool = False
    log_interval: int = 1

    # Task instructions (environment -> instruction mapping)
    task_instructions: Dict[str, str] = field(default_factory=lambda: {
        "PickCube-v1": "pick up the red cube",
        "StackCube-v1": "stack the red cube on the green cube",
        "PegInsertionSide-v1": "insert the peg into the hole",
        "PlugCharger-v1": "plug the charger into the socket",
        "PushCube-v1": "push the cube to the target",
        "TurnFaucet-v1": "turn the faucet handle",
    })


class Pi05ManiSkillEvaluator:
    """
    Evaluator for Pi0.5 model on ManiSkill3 environments.

    Usage:
        evaluator = Pi05ManiSkillEvaluator(config)
        results = evaluator.evaluate()
        evaluator.save_results(results)
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Initialize adapter
        self.adapter = Pi05ManiSkillAdapter(config.adapter_config)

        # Create environment
        self.env = self._create_env()

        # Model will be loaded lazily
        self.model = None
        self.policy = None

        # Results storage
        self.results: List[Dict[str, Any]] = []

        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)

    def _create_env(self) -> gym.Env:
        """Create ManiSkill3 environment."""
        env_kwargs = {
            "obs_mode": self.config.obs_mode,
            "control_mode": self.config.control_mode,
            "render_mode": self.config.render_mode,
            "robot_uids": self.config.robot_uid,
        }

        # Add GPU simulation if available
        if self.config.sim_backend == "gpu" and torch.cuda.is_available():
            env_kwargs["sim_backend"] = "gpu"
            env_kwargs["num_envs"] = self.config.num_envs

        env = gym.make(self.config.env_name, **env_kwargs)
        return env

    def load_model(self) -> None:
        """Load Pi0.5 model for inference."""
        if self.policy is not None:
            return  # Already loaded

        print(f"Loading Pi0.5 model: {self.config.model_config}")
        start_time = time.time()

        try:
            # Import OpenPI components
            from openpi.training import config as _config
            from openpi.policies import policy_config

            # Get configuration
            config = _config.get_config(self.config.model_config)

            # Create policy from checkpoint
            if self.config.checkpoint_dir:
                checkpoint_dir = self.config.checkpoint_dir
            else:
                # Use default checkpoint path
                checkpoint_dir = f"checkpoints/{self.config.model_config}_hf"

            self.policy = policy_config.create_trained_policy(config, checkpoint_dir)
            print(f"Model loaded in {time.time() - start_time:.2f}s")

        except ImportError as e:
            print(f"Warning: Could not import OpenPI: {e}")
            print("Using mock model for testing...")
            self.policy = MockPolicy()

    def _get_instruction(self) -> str:
        """Get task instruction for current environment."""
        return self.config.task_instructions.get(
            self.config.env_name,
            "complete the task"
        )

    def run_episode(self, episode_idx: int) -> Dict[str, Any]:
        """Run a single evaluation episode."""
        # Reset environment and adapter
        obs, info = self.env.reset(seed=self.config.seed + episode_idx)
        self.adapter.reset()

        episode_data = {
            "episode_idx": episode_idx,
            "success": False,
            "total_reward": 0.0,
            "steps": 0,
            "actions": [],
            "observations": [],
        }

        instruction = self._get_instruction()
        action_queue = []  # Buffer for action chunks

        for step in range(self.config.max_steps_per_episode):
            # Get action from model
            if len(action_queue) == 0:
                # Prepare input directly for OpenPI policy (transforms are in policy)
                example = self._prepare_policy_input_direct(obs, instruction)

                # Run inference
                with torch.no_grad():
                    if self.policy is not None:
                        policy_output = self.policy.infer(example)
                        raw_actions = policy_output.get("actions", policy_output)

                        if isinstance(raw_actions, torch.Tensor):
                            raw_actions = raw_actions.cpu().numpy()
                    else:
                        # Fallback: random actions
                        raw_actions = np.random.randn(50, 7) * 0.1

                # Actions from policy already go through ManiSkillOutputs transform
                # Just need to handle the action chunk
                if raw_actions.ndim == 1:
                    raw_actions = raw_actions[np.newaxis, :]

                # Add all actions to queue (policy outputs 7D actions after transforms)
                for i in range(raw_actions.shape[0]):
                    action_queue.append(raw_actions[i])

            # Execute action
            action = action_queue.pop(0) if action_queue else np.zeros(7)

            # Ensure action has correct shape for environment
            action = self._format_action(action)

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Handle tensor rewards
            if hasattr(reward, 'item'):
                reward = reward.item()

            # Update episode data
            episode_data["total_reward"] += float(reward)
            episode_data["steps"] = step + 1
            episode_data["actions"].append(action.tolist() if isinstance(action, np.ndarray) else action)

            # Check success - handle tensor
            success = info.get("success", False)
            if hasattr(success, 'item'):
                success = success.item()
            if success:
                episode_data["success"] = True
                break

            # Handle tensor terminated/truncated
            if hasattr(terminated, 'item'):
                terminated = terminated.item()
            if hasattr(truncated, 'item'):
                truncated = truncated.item()

            if terminated or truncated:
                break

        return episode_data

    def _prepare_policy_input_direct(
        self,
        obs: Dict[str, Any],
        instruction: str
    ) -> Dict[str, Any]:
        """Prepare input dict for OpenPI policy from ManiSkill observation.

        This method converts ManiSkill observation format to the format
        expected by our ManiSkillInputs transform in the policy.
        """
        example = {}

        # Extract images from sensor_data
        images = {}
        if "sensor_data" in obs:
            for cam_name, cam_data in obs["sensor_data"].items():
                if "rgb" in cam_data:
                    img = cam_data["rgb"]
                    if hasattr(img, 'cpu'):  # Handle torch tensors
                        img = img.cpu().numpy()
                    if img.ndim == 4:  # [B, H, W, C]
                        img = img[0]
                    # Ensure uint8
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    images[cam_name] = img

        example["images"] = images

        # Extract state from agent
        state = None
        if "agent" in obs:
            agent = obs["agent"]
            state_parts = []

            if "qpos" in agent:
                qpos = agent["qpos"]
                if hasattr(qpos, 'cpu'):
                    qpos = qpos.cpu().numpy()
                if qpos.ndim == 2:
                    qpos = qpos[0]
                state_parts.append(qpos)

            if "qvel" in agent:
                qvel = agent["qvel"]
                if hasattr(qvel, 'cpu'):
                    qvel = qvel.cpu().numpy()
                if qvel.ndim == 2:
                    qvel = qvel[0]
                state_parts.append(qvel)

            if state_parts:
                state = np.concatenate(state_parts).astype(np.float32)

        if state is not None:
            example["state"] = state

        # Add prompt
        example["prompt"] = instruction

        return example

    def _prepare_policy_input(
        self,
        model_input: Dict[str, torch.Tensor],
        obs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare input dict for OpenPI policy. (Legacy - not used)"""
        example = {}

        # Images - OpenPI expects specific keys based on training config
        images = model_input["images"]
        if images.ndim == 4:  # [B, C, H, W]
            images = images[0]  # Remove batch dim -> [C, H, W]

        # Convert to numpy HWC format as expected by OpenPI
        if isinstance(images, torch.Tensor):
            images = images.permute(1, 2, 0).cpu().numpy()  # [H, W, C]

        # Try different image key patterns used by OpenPI
        example["observation/exterior_image_1_left"] = images
        example["observation/image"] = images

        # Proprioception
        proprio = model_input["proprio"]
        if isinstance(proprio, torch.Tensor):
            proprio = proprio.cpu().numpy()
        if proprio.ndim == 2:
            proprio = proprio[0]

        example["observation/proprio"] = proprio

        # Language instruction
        example["prompt"] = model_input["instruction"]

        return example

    def _format_action(self, action: np.ndarray) -> np.ndarray:
        """Format action for ManiSkill environment."""
        action = np.array(action, dtype=np.float32)

        # ManiSkill expects action shape based on control mode
        # pd_ee_delta_pose: [dx, dy, dz, dax, day, daz, gripper] = 7D
        # Some robots have 8D (two gripper fingers)

        expected_dim = self.env.action_space.shape[-1]

        if action.shape[-1] < expected_dim:
            # Pad with zeros (usually gripper extension)
            padding = np.zeros(expected_dim - action.shape[-1])
            action = np.concatenate([action, padding])
        elif action.shape[-1] > expected_dim:
            # Truncate
            action = action[:expected_dim]

        # Clip to action space bounds
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = np.clip(action, low, high)

        return action

    def evaluate(self) -> Dict[str, Any]:
        """Run full evaluation loop."""
        print(f"\n{'='*60}")
        print(f"Pi0.5 ManiSkill Evaluation")
        print(f"{'='*60}")
        print(f"Environment: {self.config.env_name}")
        print(f"Model: {self.config.model_config}")
        print(f"Episodes: {self.config.num_episodes}")
        print(f"Max steps: {self.config.max_steps_per_episode}")
        print(f"{'='*60}\n")

        # Load model
        self.load_model()

        # Run episodes
        self.results = []
        successes = 0

        for ep in tqdm(range(self.config.num_episodes), desc="Evaluating"):
            episode_data = self.run_episode(ep)
            self.results.append(episode_data)

            if episode_data["success"]:
                successes += 1

            if (ep + 1) % self.config.log_interval == 0:
                success_rate = successes / (ep + 1) * 100
                tqdm.write(
                    f"Episode {ep + 1}/{self.config.num_episodes} | "
                    f"Success: {episode_data['success']} | "
                    f"Steps: {episode_data['steps']} | "
                    f"Reward: {episode_data['total_reward']:.2f} | "
                    f"Rate: {success_rate:.1f}%"
                )

        # Compute aggregate metrics
        aggregate = self._compute_metrics()
        print(f"\n{'='*60}")
        print("Final Results:")
        print(f"  Success Rate: {aggregate['success_rate']:.1f}%")
        print(f"  Avg Steps: {aggregate['avg_steps']:.1f}")
        print(f"  Avg Reward: {aggregate['avg_reward']:.2f}")
        print(f"{'='*60}\n")

        return aggregate

    def _compute_metrics(self) -> Dict[str, float]:
        """Compute aggregate metrics from results."""
        if not self.results:
            return {}

        successes = sum(1 for r in self.results if r["success"])
        total_steps = sum(r["steps"] for r in self.results)
        total_reward = sum(r["total_reward"] for r in self.results)

        return {
            "success_rate": successes / len(self.results) * 100,
            "num_success": successes,
            "num_episodes": len(self.results),
            "avg_steps": total_steps / len(self.results),
            "avg_reward": total_reward / len(self.results),
            "env_name": self.config.env_name,
            "model_config": self.config.model_config,
        }

    def save_results(self, aggregate: Dict[str, Any]) -> str:
        """Save evaluation results to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        env_short = self.config.env_name.replace("-v1", "").replace("-", "_")

        filename = f"{env_short}_{self.config.model_config}_{timestamp}.json"
        filepath = Path(self.config.save_dir) / filename

        results_data = {
            "config": {
                "env_name": self.config.env_name,
                "model_config": self.config.model_config,
                "num_episodes": self.config.num_episodes,
                "max_steps": self.config.max_steps_per_episode,
                "robot_uid": self.config.robot_uid,
                "control_mode": self.config.control_mode,
            },
            "aggregate": aggregate,
            "episodes": self.results,
            "timestamp": timestamp,
        }

        with open(filepath, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"Results saved to: {filepath}")
        return str(filepath)

    def close(self):
        """Clean up resources."""
        if self.env is not None:
            self.env.close()


class MockPolicy:
    """Mock policy for testing when model is not available."""

    def infer(self, example: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Return random actions for testing."""
        # Pi0.5 outputs 50-step action chunks with 7D actions
        actions = np.random.randn(50, 7) * 0.05
        actions[:, 6] = 0.5  # Gripper middle position
        return {"actions": actions}


def evaluate_single_env(
    env_name: str,
    model_config: str = "pi05_base",
    checkpoint_dir: str = None,
    num_episodes: int = 10,
    save_dir: str = "./evaluation_results",
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a single environment.

    Args:
        env_name: ManiSkill3 environment name
        model_config: Pi0.5 configuration name
        checkpoint_dir: Path to model checkpoint
        num_episodes: Number of evaluation episodes
        save_dir: Directory to save results

    Returns:
        Aggregate evaluation metrics
    """
    config = EvaluationConfig(
        env_name=env_name,
        model_config=model_config,
        checkpoint_dir=checkpoint_dir,
        num_episodes=num_episodes,
        save_dir=save_dir,
    )

    evaluator = Pi05ManiSkillEvaluator(config)
    try:
        results = evaluator.evaluate()
        evaluator.save_results(results)
        return results
    finally:
        evaluator.close()


def evaluate_all_envs(
    model_config: str = "pi05_base",
    checkpoint_dir: str = None,
    num_episodes: int = 10,
    save_dir: str = "./evaluation_results",
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate Pi0.5 on all supported ManiSkill environments.

    Args:
        model_config: Pi0.5 configuration name
        checkpoint_dir: Path to model checkpoint
        num_episodes: Number of episodes per environment
        save_dir: Directory to save results

    Returns:
        Dict mapping environment names to their results
    """
    envs = [
        "PickCube-v1",
        "StackCube-v1",
        "PegInsertionSide-v1",
        "PushCube-v1",
    ]

    all_results = {}

    for env_name in envs:
        print(f"\n{'#'*60}")
        print(f"# Evaluating: {env_name}")
        print(f"{'#'*60}")

        try:
            results = evaluate_single_env(
                env_name=env_name,
                model_config=model_config,
                checkpoint_dir=checkpoint_dir,
                num_episodes=num_episodes,
                save_dir=save_dir,
            )
            all_results[env_name] = results
        except Exception as e:
            print(f"Error evaluating {env_name}: {e}")
            all_results[env_name] = {"error": str(e)}

    # Print summary table
    print(f"\n{'='*60}")
    print("Summary of All Environments")
    print(f"{'='*60}")
    print(f"{'Environment':<25} {'Success Rate':>15} {'Avg Steps':>12}")
    print("-" * 60)

    for env_name, results in all_results.items():
        if "error" in results:
            print(f"{env_name:<25} {'ERROR':>15}")
        else:
            sr = results.get("success_rate", 0)
            steps = results.get("avg_steps", 0)
            print(f"{env_name:<25} {sr:>14.1f}% {steps:>12.1f}")

    print("=" * 60)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Pi0.5 on ManiSkill3")
    parser.add_argument(
        "--env", type=str, default="PickCube-v1",
        help="Environment name (default: PickCube-v1)"
    )
    parser.add_argument(
        "--model", type=str, default="pi05_base",
        help="Model config (default: pi05_base)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of episodes (default: 10)"
    )
    parser.add_argument(
        "--all-envs", action="store_true",
        help="Evaluate on all supported environments"
    )
    parser.add_argument(
        "--save-dir", type=str, default="./evaluation_results",
        help="Directory to save results"
    )

    args = parser.parse_args()

    if args.all_envs:
        evaluate_all_envs(
            model_config=args.model,
            checkpoint_dir=args.checkpoint,
            num_episodes=args.episodes,
            save_dir=args.save_dir,
        )
    else:
        evaluate_single_env(
            env_name=args.env,
            model_config=args.model,
            checkpoint_dir=args.checkpoint,
            num_episodes=args.episodes,
            save_dir=args.save_dir,
        )
