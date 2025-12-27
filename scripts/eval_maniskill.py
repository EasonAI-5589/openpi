#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np

from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config


def _ensure_maniskill_registered():
    """Import ManiSkill envs so they register with Gymnasium.

    Raises a helpful error if ManiSkill is not installed.
    """
    try:
        import mani_skill  # noqa: F401
        import mani_skill.envs  # noqa: F401  # triggers environment registration
    except Exception as e:
        raise RuntimeError(
            "ManiSkill is not installed or failed to import. Install it (e.g., 'pip install -e /path/to/ManiSkill' or 'pip install mani_skill') and try again."
        ) from e


def _find_checkpoint_dir(base_dir: Path, checkpoint_step: Optional[int] = None) -> Path:
    """Return a concrete checkpoint step directory under base_dir.

    If checkpoint_step is specified, look for that specific step number.
    If base_dir itself is a step directory (contains 'params' or 'model.safetensors'), return base_dir.
    Otherwise, pick the numerically largest step subdirectory.
    """
    base_dir = base_dir.resolve()

    def _is_valid_checkpoint(d: Path) -> bool:
        """Check if directory contains a valid checkpoint (params or HF format)."""
        return (d / "params").exists() or (d / "model.safetensors").exists()

    # If specific step is requested, look for it
    if checkpoint_step is not None:
        step_dir = base_dir / str(checkpoint_step)
        if step_dir.exists() and _is_valid_checkpoint(step_dir):
            print(f"Found specified checkpoint step: {checkpoint_step}")
            return step_dir
        else:
            raise FileNotFoundError(f"Checkpoint step {checkpoint_step} not found under: {base_dir}")

    # Check if base_dir itself is a valid checkpoint (params or HuggingFace format)
    if _is_valid_checkpoint(base_dir):
        return base_dir

    # Select largest numeric subdirectory
    candidates = []
    for child in base_dir.iterdir():
        if child.is_dir() and child.name.isdigit() and _is_valid_checkpoint(child):
            try:
                step = int(child.name)
                candidates.append((step, child))
            except ValueError:
                continue
    if not candidates:
        raise FileNotFoundError(f"No checkpoint step directories with 'params' or 'model.safetensors' found under: {base_dir}")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _extract_policy_inputs_from_obs(raw_obs: dict, prompt: str) -> dict:
    """Map ManiSkill raw obs to OpenPI policy input dict expected by Libero transforms.

    Produces keys:
      - 'observation/image' (H, W, 3)
      - 'observation/wrist_image' (H, W, 3) if available, else zero image matching base
      - 'observation/state' (8,) using first 8 dims of agent qpos
      - 'prompt' (str)
    """
    # Images: ManiSkill returns float32 in [0,1] with an alpha channel and a batch dim (B=1)
    def _to_numpy(x):
        try:
            import torch  # type: ignore
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        try:
            return np.asarray(x)
        except Exception:
            return np.array(x)

    base = raw_obs["sensor_data"]["base_camera"]["Color"][..., :3]
    if base.ndim == 4 and base.shape[0] == 1:
        base = base[0]
    base = _to_numpy(base)

    wrist = None
    if "hand_camera" in raw_obs.get("sensor_data", {}):
        wrist = raw_obs["sensor_data"]["hand_camera"]["Color"][..., :3]
        if wrist.ndim == 4 and wrist.shape[0] == 1:
            wrist = wrist[0]
        wrist = _to_numpy(wrist)

    # State: use first 8 dims of qpos, squeeze batch dim if present
    qpos = raw_obs["agent"]["qpos"]
    if qpos.ndim == 2 and qpos.shape[0] == 1:
        qpos = qpos[0]
    qpos = _to_numpy(qpos)
    state8 = np.asarray(qpos[:8], dtype=np.float32)

    inputs = {
        "observation/image": np.asarray(base),
        "observation/state": state8,
        "prompt": prompt,
    }
    if wrist is not None:
        inputs["observation/wrist_image"] = np.asarray(wrist)
    else:
        # Use a zero image fallback matching base resolution if wrist cam not available
        h, w = base.shape[:2]
        inputs["observation/wrist_image"] = np.zeros((h, w, 3), dtype=np.uint8)
    return inputs


def _ensure_uint8_image(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    if np.issubdtype(frame.dtype, np.floating):
        # Assume [0,1] floats
        return np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return frame.astype(np.uint8)


def _maybe_write_video(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    if len(frames) < 2:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio

        frames_uint8 = [_ensure_uint8_image(f) for f in frames]
        imageio.mimsave(out_path.as_posix(), frames_uint8, fps=fps)
    except Exception as e:
        print(f"[WARN] Failed to save video at {out_path}: {e}")


def evaluate(
    config_name: str,
    checkpoint_base: Optional[str],
    exp_name: Optional[str],
    checkpoint_step: Optional[int],
    env_id: str,
    num_episodes: int,
    prompt: str,
    sim_backend: str,
    control_mode: str,
    save_video: bool,
    video_dir: Optional[str],
    seed: Optional[int],
    action_horizon_steps: Optional[int],
    render_fps: int,
    success_metric: str,
    reward_success_threshold: float,
):
    # Resolve checkpoint directory
    if checkpoint_base is None:
        if exp_name is None:
            raise ValueError("Either --checkpoint-dir or --exp-name must be provided")
        train_cfg = _config.get_config(config_name)
        base = Path(train_cfg.checkpoint_base_dir) / train_cfg.name / exp_name
    else:
        base = Path(checkpoint_base)
    ckpt_dir = _find_checkpoint_dir(base, checkpoint_step)
    print(f"Using checkpoint: {ckpt_dir}")

    # Load config and trained policy
    train_cfg = _config.get_config(config_name)
    sample_kwargs = {}
    if action_horizon_steps is not None:
        sample_kwargs["num_steps"] = int(action_horizon_steps)
    policy = _policy_config.create_trained_policy(
        train_cfg,
        ckpt_dir,
        default_prompt=prompt,
        sample_kwargs=sample_kwargs or None,
    )

    # Ensure ManiSkill envs are registered with Gymnasium
    _ensure_maniskill_registered()

    # Build ManiSkill environment
    env_kwargs = dict(
        id=env_id,
        num_envs=1,
        obs_mode="sensor_data",
        control_mode=control_mode,
        sim_backend=sim_backend,
        # Render frames for video if requested
        render_mode="rgb_array" if save_video else None,
        max_episode_steps=500,
        sim_config={"sim_freq": 500, "control_freq": 10},
        sensor_configs={"shader_pack": "default", "width":256, "height":256},
    )
    # Remove None values to avoid gym errors
    env_kwargs = {k: v for k, v in env_kwargs.items() if v is not None}
    # import ipdb;ipdb.set_trace()
    env = gym.make(**env_kwargs)

    successes = 0
    episode_lengths = []
    episode_times = []

    for ep in range(num_episodes):
        ep_seed = None if seed is None else seed
        raw_obs, _ = env.reset(seed=ep_seed)
        done = False
        t0 = time.time()
        step_count = 0
        frames = []

        while not done:
            # Prepare inputs and run policy inference to obtain an action chunk
            inputs = _extract_policy_inputs_from_obs(raw_obs, prompt)
            import cv2
            import os
            # import ipdb; ipdb.set_trace()
            rgb_base = (inputs['observation/image'] * 255).clip(0, 255).astype(np.uint8)
            bgr_base = cv2.cvtColor(rgb_base, cv2.COLOR_RGB2BGR)
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "debug_images")
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, "base_image.jpg"), bgr_base)
            rgb_wrist = (inputs['observation/wrist_image'] * 255).clip(0, 255).astype(np.uint8)
            bgr_wrist = cv2.cvtColor(rgb_wrist, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, "wrist_image.jpg"), bgr_wrist)
            

            outputs = policy.infer(inputs)
            actions_chunk = np.asarray(outputs["actions"])  # (H, 7)

            # Execute the chunk sequentially until episode terminates or chunk ends
            max_actions_to_execute = 20
            if actions_chunk.shape[0] > max_actions_to_execute:
                actions_to_execute = actions_chunk[:max_actions_to_execute]
            else:
                actions_to_execute = actions_chunk
                
            for a in actions_to_execute:
                # Ensure shape (7,)
                action = np.asarray(a, dtype=np.float32)
                raw_obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1
                done = bool(terminated or truncated)

                if save_video:
                    frame = env.render()
                    if frame is not None:
                        if hasattr(frame, "detach") and hasattr(frame, "cpu"):
                            frame = frame.detach().cpu().numpy()
                        if frame.ndim == 4 and frame.shape[0] == 1:
                            frame = frame[0]
                        frames.append(_ensure_uint8_image(frame))

                if done:
                    break

        dt = time.time() - t0
        episode_times.append(dt)
        episode_lengths.append(step_count)

        # Success determination
        ep_success = False
        if success_metric == "info":
            if isinstance(info, dict):
                for k in ("success", "succeed", "is_success", "episode_success"):
                    if bool(info.get(k, False)):
                        ep_success = True
                        break
                if not ep_success and isinstance(info.get("episode"), dict):
                    ep_success = bool(info["episode"].get("success", False))
        elif success_metric == "reward_pos":
            ep_success = bool(reward > 0)
        elif success_metric == "reward_thresh":
            ep_success = bool(reward >= reward_success_threshold)
        successes += int(ep_success)

        print(
            f"Episode {ep+1}/{num_episodes}: steps={step_count}, time={dt:.2f}s, success={ep_success}"
        )

        if save_video:
            out_dir = Path(video_dir) if video_dir is not None else (ckpt_dir / "videos")
            out_path = out_dir / f"{env_id}_ep{ep+1}_{'success' if ep_success else 'failure'}.mp4"
            _maybe_write_video(frames, out_path, fps=render_fps)

    success_rate = successes / max(1, num_episodes)
    print("=" * 70)
    print(f"Success: {successes}/{num_episodes} = {success_rate * 100:.2f}%")
    if episode_lengths:
        print(f"Avg steps: {np.mean(episode_lengths):.2f}")
    if episode_times:
        print(f"Avg time: {np.mean(episode_times):.2f}s")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate an OpenPI pi0 policy in ManiSkill.")
    parser.add_argument("--config-name", default="pi0_maniskill", help="Train config name used for the model")
    parser.add_argument("--checkpoint-dir", default=None, help="Path to checkpoint step dir or its parent.")
    parser.add_argument("--checkpoint-step", type=int, default=None, help="Specific checkpoint step number to load")
    parser.add_argument("--exp-name", default=None, help="Experiment name, used if checkpoint-dir not provided.")
    parser.add_argument("--env-id", default="StackCube-v1", help="ManiSkill environment ID")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument(
        "--prompt",
        default=(
            "Grasp a red cube and move it to a target goal position."
        ),
        help="Instruction prompt",
    )
    parser.add_argument("--sim-backend", default="cpu", help="ManiSkill sim backend: 'cpu' or 'gpu'")
    parser.add_argument("--control-mode", default="pd_ee_delta_pose", help="ManiSkill control mode")
    parser.add_argument("--save-video", type=str, default="True", 
                       help="Whether to save videos. Use 'True' or 'False' (default: True)")
    parser.add_argument("--video-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of diffusion steps for pi0 sampling (defaults to model's default)",
    )
    parser.add_argument("--render-fps", type=int, default=30)
    parser.add_argument(
        "--success-metric",
        choices=("info", "reward_pos", "reward_thresh"),
        default="info",
        help="How to decide success: 'info' uses env info flags (recommended), 'reward_pos' uses reward>0, 'reward_thresh' uses final reward >= threshold.",
    )
    parser.add_argument(
        "--reward-success-threshold",
        type=float,
        default=0.99,
        help="Threshold for success when --success-metric=reward_thresh.",
    )

    args = parser.parse_args()

    # Honor XLA env vars if set; do not override here
    print(f"Prompt is : {args.prompt}")
    # Convert string to boolean for save_video
    save_video_bool = args.save_video.lower() in ('true', '1', 'yes', 'on')
    # import ipdb; ipdb.set_trace()
    evaluate(
        config_name=args.config_name,
        checkpoint_base=args.checkpoint_dir,
        exp_name=args.exp_name,
        checkpoint_step=args.checkpoint_step,
        env_id=args.env_id,
        num_episodes=args.num_episodes,
        prompt=args.prompt,
        sim_backend=args.sim_backend,
        control_mode=args.control_mode,
        save_video=save_video_bool,
        video_dir=args.video_dir,
        seed=args.seed,
        action_horizon_steps=args.num_steps,
        render_fps=args.render_fps,
        success_metric=args.success_metric,
        reward_success_threshold=args.reward_success_threshold,
    )


if __name__ == "__main__":
    # Ensure openpi src is importable if script is executed directly
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if src_path.exists() and src_path.as_posix() not in sys.path:
        sys.path.insert(0, src_path.as_posix())
    main()


