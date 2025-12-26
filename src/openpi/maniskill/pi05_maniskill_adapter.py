"""
Pi0.5 + ManiSkill3 Adapter

This module provides the adapter layer between Pi0.5 (openpi) and ManiSkill3 environments.
Based on open-pi-zero's SimplerAdapter pattern.

Architecture:
    ManiSkill3 Observation
        ↓
    Pi05ManiSkillAdapter.preprocess()
        ↓ (images, proprio, instruction)
    Pi0.5 Model Inference
        ↓ (raw_actions)
    Pi05ManiSkillAdapter.postprocess()
        ↓ (env_actions)
    ManiSkill3 env.step()

Author: Claude Code for 郭奕辰
Date: 2025-12-27
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import cv2
from collections import deque
from transforms3d.euler import euler2axangle, mat2euler, quat2mat


@dataclass
class Pi05ManiSkillConfig:
    """Configuration for Pi0.5 ManiSkill adapter."""

    # Image settings
    image_size: Tuple[int, int] = (224, 224)
    num_cameras: int = 1  # Number of cameras to use
    camera_names: List[str] = None  # Camera names to use, None = auto-detect

    # Proprioception settings
    proprio_dim: int = 8  # [x, y, z, qx, qy, qz, qw, gripper] or [x, y, z, roll, pitch, yaw, gripper]
    use_euler: bool = True  # Use Euler angles instead of quaternions

    # Action settings
    action_dim: int = 7  # [dx, dy, dz, droll, dpitch, dyaw, gripper]
    action_scale: float = 1.0
    horizon_steps: int = 50  # Pi0.5 outputs 50-step action chunks
    act_steps: int = 1  # Execute 1 step at a time

    # Gripper settings
    sticky_gripper: bool = True  # Use sticky gripper mechanism
    sticky_gripper_num_repeat: int = 15
    gripper_threshold: float = 0.5

    # Normalization (will be loaded from dataset statistics)
    proprio_mean: Optional[np.ndarray] = None
    proprio_std: Optional[np.ndarray] = None
    action_mean: Optional[np.ndarray] = None
    action_std: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.camera_names is None:
            self.camera_names = ["base_camera"]


class Pi05ManiSkillAdapter:
    """
    Adapter for Pi0.5 model to work with ManiSkill3 environments.

    Handles:
    1. Observation preprocessing (images + proprioception)
    2. Action postprocessing (rotation conversion, gripper handling)
    3. Normalization/denormalization
    """

    def __init__(self, config: Pi05ManiSkillConfig = None):
        self.config = config or Pi05ManiSkillConfig()

        # Gripper state management
        self.previous_gripper_action = None
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0

        # Action history for temporal smoothing (optional)
        self.action_history = deque(maxlen=self.config.horizon_steps)

        # Image history for context (if needed by model)
        self.image_history = deque(maxlen=2)

    def reset(self):
        """Reset adapter state for new episode."""
        self.previous_gripper_action = None
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.action_history.clear()
        self.image_history.clear()

    def preprocess(
        self,
        obs: Dict[str, Any],
        instruction: str = "pick up the object",
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess ManiSkill3 observation for Pi0.5 model.

        Args:
            obs: ManiSkill3 observation dict
            instruction: Language instruction

        Returns:
            Dict with keys:
                - images: torch.Tensor [B, C, H, W] or [B, N_cams, C, H, W]
                - proprio: torch.Tensor [B, proprio_dim]
                - instruction: str
        """
        # Extract images from sensor_data
        images = self._extract_images(obs)

        # Extract proprioception
        proprio = self._extract_proprio(obs)

        # Normalize if statistics available
        if self.config.proprio_mean is not None:
            proprio = self._normalize(
                proprio,
                self.config.proprio_mean,
                self.config.proprio_std,
            )

        return {
            "images": images,
            "proprio": proprio,
            "instruction": instruction,
        }

    def _extract_images(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Extract and preprocess images from observation."""
        images = []

        sensor_data = obs.get("sensor_data", {})

        for cam_name in self.config.camera_names:
            if cam_name not in sensor_data:
                # Try to find any available camera
                available_cams = list(sensor_data.keys())
                if available_cams:
                    cam_name = available_cams[0]
                else:
                    raise ValueError(f"No camera data found in observation")

            cam_data = sensor_data[cam_name]
            rgb = cam_data.get("rgb")

            if rgb is None:
                raise ValueError(f"No RGB data for camera {cam_name}")

            # Convert to numpy if tensor
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.cpu().numpy()

            # Handle batch dimension
            if rgb.ndim == 4:  # [B, H, W, C]
                rgb = rgb[0]  # Take first batch

            # Resize to target size
            if rgb.shape[:2] != self.config.image_size:
                rgb = cv2.resize(
                    rgb,
                    self.config.image_size,
                    interpolation=cv2.INTER_LANCZOS4,
                )

            # Convert to tensor [C, H, W]
            rgb_tensor = torch.as_tensor(rgb, dtype=torch.uint8)
            if rgb_tensor.shape[-1] == 3:  # [H, W, C] -> [C, H, W]
                rgb_tensor = rgb_tensor.permute(2, 0, 1)

            images.append(rgb_tensor)

        # Stack cameras
        if len(images) == 1:
            return images[0].unsqueeze(0)  # [1, C, H, W]
        else:
            return torch.stack(images, dim=0).unsqueeze(0)  # [1, N_cams, C, H, W]

    def _extract_proprio(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Extract proprioceptive state from observation."""
        agent = obs.get("agent", {})
        extra = obs.get("extra", {})

        # Get joint positions (includes gripper)
        qpos = agent.get("qpos")
        if qpos is None:
            raise ValueError("No qpos in agent observation")

        if isinstance(qpos, torch.Tensor):
            qpos = qpos.cpu().numpy()

        # Handle batch dimension
        if qpos.ndim == 2:
            qpos = qpos[0]

        # Get TCP (tool center point) pose if available
        tcp_pose = extra.get("tcp_pose")
        if tcp_pose is not None:
            if isinstance(tcp_pose, torch.Tensor):
                tcp_pose = tcp_pose.cpu().numpy()
            if tcp_pose.ndim == 2:
                tcp_pose = tcp_pose[0]

            # tcp_pose: [x, y, z, qx, qy, qz, qw]
            pos = tcp_pose[:3]
            quat = tcp_pose[3:7]  # xyzw format

            if self.config.use_euler:
                # Convert quaternion to Euler angles
                rot_mat = quat2mat([quat[3], quat[0], quat[1], quat[2]])  # wxyz for transforms3d
                euler = mat2euler(rot_mat)  # returns (roll, pitch, yaw)

                # Gripper: last joint position or normalized
                gripper = qpos[-1] if len(qpos) > 7 else 0.04  # Default open
                gripper_normalized = np.clip(gripper / 0.04, 0, 1)  # Normalize to [0, 1]

                proprio = np.concatenate([pos, euler, [gripper_normalized]])
            else:
                # Use quaternion directly
                gripper = qpos[-1] if len(qpos) > 7 else 0.04
                gripper_normalized = np.clip(gripper / 0.04, 0, 1)
                proprio = np.concatenate([pos, quat, [gripper_normalized]])
        else:
            # Fallback: use joint positions
            proprio = qpos

        return torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)  # [1, proprio_dim]

    def _normalize(
        self,
        data: torch.Tensor,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> torch.Tensor:
        """Normalize data using mean and std."""
        mean = torch.as_tensor(mean, dtype=data.dtype, device=data.device)
        std = torch.as_tensor(std, dtype=data.dtype, device=data.device)
        return (data - mean) / (std + 1e-8)

    def _denormalize(
        self,
        data: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> np.ndarray:
        """Denormalize data using mean and std."""
        return data * std + mean

    def postprocess(
        self,
        raw_actions: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Postprocess Pi0.5 model output to ManiSkill3 action format.

        Args:
            raw_actions: Model output [horizon_steps, action_dim]
                         Format: [dx, dy, dz, droll, dpitch, dyaw, gripper]

        Returns:
            List of environment actions, each [7] or [8] depending on robot
            Format: [dx, dy, dz, rot_ax_x, rot_ax_y, rot_ax_z, gripper]
        """
        if raw_actions.ndim == 1:
            raw_actions = raw_actions.reshape(1, -1)

        # Denormalize if statistics available
        if self.config.action_mean is not None:
            raw_actions = self._denormalize(
                raw_actions,
                self.config.action_mean,
                self.config.action_std,
            )

        env_actions = []

        for i, raw_action in enumerate(raw_actions[:self.config.act_steps]):
            # Position delta
            pos_delta = raw_action[:3] * self.config.action_scale

            # Rotation: Euler angles -> Axis-angle
            roll, pitch, yaw = raw_action[3:6]
            try:
                rot_ax, rot_angle = euler2axangle(roll, pitch, yaw)
                rot_axangle = rot_ax * rot_angle * self.config.action_scale
            except Exception:
                # Fallback for small rotations
                rot_axangle = np.array([roll, pitch, yaw]) * self.config.action_scale

            # Gripper
            gripper_raw = raw_action[6] if len(raw_action) > 6 else 0.0
            gripper_action = self._process_gripper(gripper_raw)

            # Combine into environment action
            # ManiSkill3 expects [pos_delta(3), rot_axangle(3), gripper(1)] = 7D
            # But some robots have 8D action space (gripper has 2 fingers)
            env_action = np.concatenate([
                pos_delta,
                rot_axangle,
                [gripper_action],
            ])

            env_actions.append(env_action)

        return env_actions

    def _process_gripper(self, gripper_raw: float) -> float:
        """
        Process gripper action with sticky mechanism.

        Args:
            gripper_raw: Raw gripper action from model, typically [0, 1]
                         0 = close, 1 = open

        Returns:
            Processed gripper action for ManiSkill, typically [-1, 1]
            -1 = close, 1 = open
        """
        # Convert from [0, 1] to [-1, 1]
        gripper_normalized = (gripper_raw * 2) - 1  # Now: -1=close, 1=open

        if not self.config.sticky_gripper:
            # Simple binary threshold
            return 1.0 if gripper_normalized > 0 else -1.0

        # Sticky gripper mechanism
        current_gripper = gripper_normalized

        if self.previous_gripper_action is None:
            self.previous_gripper_action = current_gripper
            return current_gripper

        # Compute relative change
        relative_gripper = self.previous_gripper_action - current_gripper

        # Trigger sticky action on significant change
        if abs(relative_gripper) > self.config.gripper_threshold and not self.sticky_action_is_on:
            self.sticky_action_is_on = True
            self.sticky_gripper_action = np.sign(relative_gripper)

        # Apply sticky action
        if self.sticky_action_is_on:
            self.gripper_action_repeat += 1
            result = self.sticky_gripper_action

            # Release sticky after N repeats
            if self.gripper_action_repeat >= self.config.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0
        else:
            result = 0.0  # No change

        self.previous_gripper_action = current_gripper
        return result


def create_adapter_for_env(env_name: str) -> Pi05ManiSkillAdapter:
    """
    Factory function to create appropriate adapter for environment.

    Args:
        env_name: ManiSkill3 environment name (e.g., "PickCube-v1")

    Returns:
        Configured Pi05ManiSkillAdapter
    """
    # Default config
    config = Pi05ManiSkillConfig()

    # Environment-specific configurations
    if "Pick" in env_name:
        config.sticky_gripper = True
        config.sticky_gripper_num_repeat = 15
    elif "Push" in env_name:
        config.sticky_gripper = False  # No gripper for pushing
    elif "Stack" in env_name:
        config.sticky_gripper = True
        config.sticky_gripper_num_repeat = 20  # Longer for stacking

    return Pi05ManiSkillAdapter(config)
