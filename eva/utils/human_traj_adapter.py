"""
Human Trajectory Adapter for EVA Replayer

Converts human retargeted trajectory data (eef_pose + gripper) into 
a format compatible with the Replayer controller.

Human data format:
    - eef_pose.npy: (N, 7) - position xyz + quaternion xyzw
    - retarget_gripper_action.npy: (N,) - gripper values

Replayer format:
    - trajectory: (N, 7) - position xyz + euler xyz + gripper
"""

import os
import numpy as np
from eva.utils.geometry_utils import quat_to_euler, quat_to_rmat


BASE_PATH = '/home/franka/eva_tony/human_data'
GRIPPER_OFFSET = np.array([0, 0, 0.062])  # Offset from wrist to gripper tip


def load_human_trajectory(task: str, traj_num: int, 
                          apply_gripper_offset: bool = True,
                          gripper_threshold: float = None,
                          base_path: str = BASE_PATH) -> np.ndarray:
    """
    Load human trajectory data and convert to Replayer-compatible format.
    
    Args:
        task: Task name (e.g., 'open_drawer')
        traj_num: Trajectory number (e.g., 1)
        apply_gripper_offset: Whether to apply the gripper offset correction
        gripper_threshold: If provided, binarize gripper values (> threshold -> 1, else 0)
        base_path: Base path to human data directory
        
    Returns:
        trajectory: (N, 7) array with [x, y, z, euler_x, euler_y, euler_z, gripper]
    """
    data_path = os.path.join(base_path, task, f"traj_{traj_num}", "processed_3d")
    
    eef_pose_path = os.path.join(data_path, "eef_pose.npy")
    gripper_path = os.path.join(data_path, "retarget_gripper_action.npy")
    
    eef_poses = np.load(eef_pose_path)  # (N, 7): xyz + quat xyzw
    gripper_actions = np.load(gripper_path)  # (N,)
    
    assert len(eef_poses) == len(gripper_actions), \
        f"Mismatch: eef_poses {len(eef_poses)} vs gripper {len(gripper_actions)}"
    
    n_steps = len(eef_poses)
    trajectory = np.zeros((n_steps, 7))
    
    for i in range(n_steps):
        position = eef_poses[i, :3].copy()
        quaternion = eef_poses[i, 3:]
        position[0] += 0.2
        
        # Apply gripper offset correction (transform from gripper tip to wrist)
        if apply_gripper_offset:
            rotation_matrix = quat_to_rmat(quaternion)
            position -= rotation_matrix @ GRIPPER_OFFSET
        
        # Convert quaternion to euler angles
        euler = quat_to_euler(quaternion)
        
        trajectory[i, :3] = position
        trajectory[i, 3:6] = euler
    
    # Process gripper values
    if gripper_threshold is not None:
        gripper_actions = np.where(gripper_actions > gripper_threshold, 1.0, 0.0)
    
    trajectory[:, 6] = gripper_actions
    
    return trajectory


def save_replayable_trajectory(trajectory: np.ndarray, output_path: str):
    """
    Save trajectory in a format compatible with Replayer.
    
    Args:
        trajectory: (N, 7) array with [x, y, z, euler_x, euler_y, euler_z, gripper]
        output_path: Path to save the trajectory (.npy or .npz)
    """
    if output_path.endswith('.npz'):
        np.savez(output_path, actions_pos=trajectory)
    else:
        np.save(output_path, trajectory)
    print(f"Saved replayable trajectory to {output_path}")


def process_human_trajectory(task: str, traj_num: int, 
                             output_dir: str = None,
                             apply_gripper_offset: bool = True,
                             gripper_threshold: float = None,
                             base_path: str = BASE_PATH) -> str:
    """
    Full pipeline: load human trajectory, convert, and save.
    
    Args:
        task: Task name (e.g., 'open_drawer')
        traj_num: Trajectory number (e.g., 1)
        output_dir: Directory to save output (default: same as input)
        apply_gripper_offset: Whether to apply the gripper offset correction
        gripper_threshold: If provided, binarize gripper values
        base_path: Base path to human data directory
        
    Returns:
        output_path: Path to the saved trajectory file
    """
    trajectory = load_human_trajectory(
        task=task,
        traj_num=traj_num,
        apply_gripper_offset=apply_gripper_offset,
        gripper_threshold=gripper_threshold,
        base_path=base_path
    )
    
    if output_dir is None:
        output_dir = os.path.join(base_path, task, f"traj_{traj_num}", "processed_3d")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "replayable_traj.npy")
    
    save_replayable_trajectory(trajectory, output_path)
    
    print(f"Processed trajectory: {task}/traj_{traj_num}")
    print(f"  Shape: {trajectory.shape}")
    print(f"  Position range: [{trajectory[:, :3].min(axis=0)}, {trajectory[:, :3].max(axis=0)}]")
    print(f"  Gripper range: [{trajectory[:, 6].min():.3f}, {trajectory[:, 6].max():.3f}]")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert human trajectory for Replayer")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., open_drawer)")
    parser.add_argument("--traj", type=int, required=True, help="Trajectory number")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--no-offset", action="store_true", help="Skip gripper offset correction")
    parser.add_argument("--gripper-threshold", type=float, default=0.5, 
                        help="Binarize gripper values with this threshold")
    parser.add_argument("--base-path", type=str, default=BASE_PATH, help="Base path to human data")
    
    args = parser.parse_args()
    
    output_path = process_human_trajectory(
        task=args.task,
        traj_num=args.traj,
        output_dir=args.output_dir,
        apply_gripper_offset=not args.no_offset,
        gripper_threshold=args.gripper_threshold,
        base_path=args.base_path
    )
    
    print(f"\nTo use with Replayer:")
    print(f'  replayer = Replayer("{output_path}")')

