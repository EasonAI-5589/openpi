"""
Minimal example script for converting ManiSkill dataset to LeRobot format.

We use the ManiSkill dataset (stored in HDF5) for this example.

Usage:
python examples/maniskill/convert_maniskill_data_to_lerobot.py --h5_file /path/to/your/data.h5 --task_name stackcube --task_prompt "Pick up a red cube and stack it on top of a green cube"

Note: to run the script, you need to install h5py:
`pip install h5py`

The resulting dataset will get saved to /share/project/guoyichen/maniskill_lerobot/maniskill_<task_name>
"""

import shutil
import h5py
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import numpy as np

OUTPUT_DIR = Path("/share/project/guoyichen/maniskill_lerobot")  # Custom output directory

# Task prompts for different ManiSkill tasks
TASK_PROMPTS = {
    "stackcube": "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling",
    "pickcube": "Grasp a red cube and move it to a target goal position.",
    "pushcube": "Push and move a cube to a goal region in front of it.",
    "placesphere": "Place the sphere into the shallow bin.",
    "peginsertionside": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
    "plugcharger": "Insert the charger into the receptacle.",
    "pullcube": "Pull a cube onto a target.",
    "pullcubetool": "Given an L-shaped tool that is within the reach of the robot, leverage the tool to pull a cube that is out of it's reach",
}


def main(h5_file: str, task_name: str = "stackcube", task_prompt: str | None = None, *, push_to_hub: bool = False):
    # Determine the task prompt
    if task_prompt is None:
        task_prompt = TASK_PROMPTS.get(task_name.lower(), f"Complete the {task_name} task.")

    repo_name = f"maniskill_{task_name.lower()}"

    # Clean up any existing dataset in the output directory
    output_path = OUTPUT_DIR / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Converting {task_name} data to LeRobot format...")
    print(f"  Input: {h5_file}")
    print(f"  Output: {output_path}")
    print(f"  Task prompt: {task_prompt}")

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        root=output_path,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=0,
        image_writer_processes=0,
    )

    # Load HDF5 file and write episodes to the LeRobot dataset
    with h5py.File(h5_file, 'r') as f:
        # Get all trajectory keys (traj_0, traj_1, etc.)
        traj_keys = [key for key in f.keys() if key.startswith('traj_')]
        
        print(f"Found {len(traj_keys)} trajectories in the HDF5 file")
        
        for traj_key in traj_keys:
            traj = f[traj_key]
            
            # Extract data
            qpos = traj['obs']['agent']['qpos'][:]  # shape: (T, 9)
            actions = traj['actions'][:]  # shape: (T-1, 7)
            base_camera = traj['obs']['sensor_data']['base_camera']['Color'][:]  # shape: (T, 256, 256, 4)

            # Use hand camera if available, otherwise zeros
            if 'hand_camera' in traj['obs']['sensor_data']:
                hand_camera = traj['obs']['sensor_data']['hand_camera']['Color'][:]  # shape: (T, 256, 256, 4)
            else:
                hand_camera = np.zeros_like(base_camera)
            
            # Get the minimum length to handle potential mismatches
            min_len = min(len(qpos), len(actions) + 1, len(base_camera), len(hand_camera))
            
            # Process each timestep (excluding the last one since actions has T-1 length)
            for t in range(min_len - 1):
                dataset.add_frame(
                    {
                        "image": base_camera[t, :, :, :3],  # Remove alpha channel, keep RGB
                        "wrist_image": hand_camera[t, :, :, :3],  # Remove alpha channel, keep RGB
                        "state": qpos[t, :8],  # Take first 8 dimensions
                        "actions": actions[t],  # shape: (7,)
                        "task": task_prompt,
                    }
                )
            
            # Save the episode
            dataset.save_episode()
            print(f"Processed {traj_key}: {min_len - 1} frames")

    # Optionally push to the Hugging Face Hub
    # if push_to_hub:
    #     dataset.push_to_hub(
    #         tags=["maniskill", "stackcube", "panda"],
    #         private=False,
    #         push_videos=True,
    #         license="apache-2.0",
    #     )


if __name__ == "__main__":
    tyro.cli(main)