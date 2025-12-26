"""
ManiSkill policy transforms for Pi0.5 integration.

This module provides input/output transforms to convert between ManiSkill
observation/action formats and the format expected by Pi0.5 model.

Author: Claude Code for 郭奕辰
Date: 2025-12-27
"""

import dataclasses
from typing import Any

import numpy as np

import openpi.models.model as _model
import openpi.transforms as transforms


@dataclasses.dataclass
class ManiSkillInputs(transforms.DataTransformFn):
    """Transform ManiSkill observations to Pi0.5 model inputs.

    Expected input format from ManiSkill:
        {
            "images": {"base_camera": np.ndarray [H, W, 3]},
            "state": np.ndarray [proprio_dim],
            "prompt": str,
        }

    Output format for Pi0.5:
        {
            "image": {"cam_name": np.ndarray [H, W, 3], ...},  # Dict of images
            "state": np.ndarray [proprio_dim],
            "prompt": str,
        }
    """

    model_type: _model.ModelType = _model.ModelType.PI05

    # Camera names mapping: ManiSkill camera name -> Pi0.5 camera name
    # Pi0.5 expects: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
    camera_name_map: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "base_camera": "base_0_rgb",
        "hand_camera": "left_wrist_0_rgb",
    })

    # Default image size
    image_size: tuple[int, int] = (224, 224)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        result = dict(data)

        # Process images - output as dict for Pi0.5
        image_dict = {}
        image_mask_dict = {}

        if "images" in data:
            images = data["images"]
            for cam_name, pi0_cam_name in self.camera_name_map.items():
                if cam_name in images:
                    img = images[cam_name]
                    # Ensure correct format: [H, W, 3] uint8
                    if isinstance(img, np.ndarray):
                        if img.dtype != np.uint8:
                            if img.max() <= 1.0:
                                img = (img * 255).astype(np.uint8)
                            else:
                                img = img.astype(np.uint8)
                        image_dict[pi0_cam_name] = img
                        image_mask_dict[pi0_cam_name] = True
                else:
                    # Create placeholder for missing cameras
                    image_dict[pi0_cam_name] = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
                    image_mask_dict[pi0_cam_name] = False

        # Ensure all required images exist (Pi0.5 expects 3 cameras)
        required_cameras = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        for cam_name in required_cameras:
            if cam_name not in image_dict:
                # Create placeholder for missing cameras
                image_dict[cam_name] = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
                image_mask_dict[cam_name] = False

        result["image"] = image_dict
        result["image_mask"] = image_mask_dict

        # Process state
        if "state" in data:
            result["state"] = np.array(data["state"], dtype=np.float32)
        elif "proprio" in data:
            result["state"] = np.array(data["proprio"], dtype=np.float32)

        # Ensure prompt exists
        if "prompt" not in result:
            result["prompt"] = "complete the task"

        # Clean up intermediate keys
        result.pop("images", None)
        result.pop("proprio", None)

        return result


@dataclasses.dataclass
class ManiSkillOutputs(transforms.DataTransformFn):
    """Transform Pi0.5 model outputs to ManiSkill action format.

    Expected input format from Pi0.5:
        {
            "actions": np.ndarray [horizon, action_dim],  # 32-dim actions
        }

    Output format for ManiSkill:
        {
            "actions": np.ndarray [horizon, 7],  # [dx, dy, dz, dax, day, daz, gripper]
        }

    The Pi0.5 32-dim action format (for DROID-style):
        - [0:3]: EEF position delta (dx, dy, dz)
        - [3:6]: EEF rotation delta (Euler angles or axis-angle)
        - [6]: Gripper action (continuous)
        - [7:32]: Padding/unused
    """

    # Action dimensions to extract from 32-dim output
    position_dims: tuple[int, int] = (0, 3)    # dx, dy, dz
    rotation_dims: tuple[int, int] = (3, 6)    # rotation
    gripper_dim: int = 6                        # gripper

    # Output action dimension (ManiSkill pd_ee_delta_pose)
    output_dim: int = 7

    # Action scaling
    position_scale: float = 1.0
    rotation_scale: float = 1.0
    gripper_threshold: float = 0.5  # Above this -> close, below -> open

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        result = dict(data)

        if "actions" in data:
            raw_actions = np.array(data["actions"])

            # Handle different input shapes
            if raw_actions.ndim == 1:
                raw_actions = raw_actions[np.newaxis, :]  # [1, action_dim]

            horizon = raw_actions.shape[0]
            output_actions = np.zeros((horizon, self.output_dim), dtype=np.float32)

            # Extract position
            pos_start, pos_end = self.position_dims
            output_actions[:, 0:3] = raw_actions[:, pos_start:pos_end] * self.position_scale

            # Extract rotation
            rot_start, rot_end = self.rotation_dims
            output_actions[:, 3:6] = raw_actions[:, rot_start:rot_end] * self.rotation_scale

            # Extract gripper
            gripper = raw_actions[:, self.gripper_dim]
            # Convert to binary: 1.0 = close, -1.0 = open (ManiSkill convention)
            output_actions[:, 6] = np.where(gripper > self.gripper_threshold, 1.0, -1.0)

            result["actions"] = output_actions

        return result


@dataclasses.dataclass
class ManiSkillRepackInputs(transforms.DataTransformFn):
    """Repack ManiSkill-style inputs to match OpenPI expected format.

    This handles the conversion from various ManiSkill observation formats
    to a standardized format that can be processed by ManiSkillInputs.
    """

    # Camera names to include
    camera_names: list[str] = dataclasses.field(default_factory=lambda: ["base_camera"])

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        result = {}

        # Handle nested observation structure
        if "observation" in data:
            obs = data["observation"]
        else:
            obs = data

        # Extract images from sensor_data
        if "sensor_data" in obs:
            result["images"] = {}
            for cam_name in self.camera_names:
                if cam_name in obs["sensor_data"]:
                    cam_data = obs["sensor_data"][cam_name]
                    if "rgb" in cam_data:
                        img = cam_data["rgb"]
                        if hasattr(img, 'cpu'):  # Handle torch tensors
                            img = img.cpu().numpy()
                        if img.ndim == 4:  # [B, H, W, C]
                            img = img[0]
                        result["images"][cam_name] = img

        # Extract proprioception from agent
        if "agent" in obs:
            agent = obs["agent"]
            state_parts = []

            # Joint positions
            if "qpos" in agent:
                qpos = agent["qpos"]
                if hasattr(qpos, 'cpu'):
                    qpos = qpos.cpu().numpy()
                if qpos.ndim == 2:
                    qpos = qpos[0]
                state_parts.append(qpos)

            # Joint velocities (optional)
            if "qvel" in agent:
                qvel = agent["qvel"]
                if hasattr(qvel, 'cpu'):
                    qvel = qvel.cpu().numpy()
                if qvel.ndim == 2:
                    qvel = qvel[0]
                state_parts.append(qvel)

            if state_parts:
                result["state"] = np.concatenate(state_parts)

        # Pass through prompt
        if "prompt" in data:
            result["prompt"] = data["prompt"]

        # Pass through actions if present
        if "actions" in data:
            result["actions"] = data["actions"]

        return result


def create_maniskill_transforms(model_type: _model.ModelType = _model.ModelType.PI05) -> transforms.Group:
    """Create a transforms group for ManiSkill integration."""
    return transforms.Group(
        inputs=[
            ManiSkillRepackInputs(),
            ManiSkillInputs(model_type=model_type),
        ],
        outputs=[
            ManiSkillOutputs(),
        ],
    )
