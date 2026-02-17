"""Compare planned trajectory with actually executed trajectory."""

import numpy as np
import argparse

def compare_trajectories(planned_path: str, executed_path: str):
    # Load trajectories
    planned = np.load(planned_path)
    executed = np.load(executed_path)
    
    # Get planned trajectory
    if isinstance(planned, np.lib.npyio.NpzFile):
        plan_traj = planned["actions_pos"]
    else:
        plan_traj = planned
    
    # Get actual robot states (first 6: cartesian_position, 7th: gripper)
    states = executed["states"]
    actual_traj = np.zeros((len(states), 7))
    actual_traj[:, :6] = states[:, :6]  # cartesian_position
    actual_traj[:, 6] = states[:, 6]     # gripper_position
    
    print(f"Planned: {plan_traj.shape}, Actual states: {actual_traj.shape}")
    
    min_len = min(len(plan_traj), len(actual_traj))
    plan_traj = plan_traj[:min_len]
    actual_traj = actual_traj[:min_len]
    
    # Compute errors
    pos_error = np.linalg.norm(plan_traj[:, :3] - actual_traj[:, :3], axis=1)
    rot_error = np.linalg.norm(plan_traj[:, 3:6] - actual_traj[:, 3:6], axis=1)
    gripper_error = np.abs(plan_traj[:, 6] - actual_traj[:, 6])
    
    print(f"\n=== Planned vs Actual Robot States ({min_len} steps) ===")
    print(f"Position error (m):   mean={pos_error.mean():.4f}, max={pos_error.max():.4f}")
    print(f"Rotation error (rad): mean={rot_error.mean():.4f}, max={rot_error.max():.4f}")
    print(f"Gripper error:        mean={gripper_error.mean():.4f}, max={gripper_error.max():.4f}")
    print(f"Total RMSE: {np.sqrt(np.mean((plan_traj - actual_traj)**2)):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--planned", default="human_data/open_drawer/traj_1/processed_3d/replayable_traj.npy")
    parser.add_argument("--executed", default="data/eval/2026-01-13/2026-01-13_23-24-28/trajectory.npz")
    args = parser.parse_args()
    
    compare_trajectories(args.planned, args.executed)

