#!/usr/bin/env python3

import argparse
import time
from eva.manager import load_runner
from tqdm import tqdm
def main():
    parser = argparse.ArgumentParser(description="Run Pi0 policy with EVA infrastructure")
    parser.add_argument("-n", "--n_traj", type=int, default=10, help="Number of trajectories to run")
    parser.add_argument('--external_camera', type=str, default="right", choices=['left', 'right'],
                       help="Which external camera to use")
    args = parser.parse_args()

    total_rollouts = 3000

    runner = load_runner(manager=False, controller='mixed', record_depth=False, record_pcd=False, post_process=True)
    # start with spacemouse 
    
    # Start interactive session
    print("\nPi0 Policy Runner")
    print("=================")
    print("Controls:")
    print("  Space: Toggle movement")
    print("  =: switch between PI0 and SpaceMouse")
    print("  y: Mark as success")
    print("  n: Mark as failure")
    print("  Ctrl+C: Exit")
    
    for i in tqdm(range(args.n_traj), desc="Running trajectories", disable=(args.n_traj == 1)):
        print(f"\n MAIN --- Trajectory {i+1}/{args.n_traj} ---")
            
        runner.run_trajectory(mode="collect", reset_robot=True, wait_for_controller=True)
            
        # Wait for success or failure or user interrupt
        print("Running... (press 'y' for success, 'n' for failure, or Ctrl+C to stop)")
        try:
            for i in tqdm(range(total_rollouts), desc="Waiting for success or failure"):
                controller_info = runner.get_controller_info()
                if controller_info["success"] or controller_info["failure"]:
                    result = "Success" if controller_info["success"] else "Failure"
                    print(f"MAIN === \n{result}! Execution complete, please reset environment..\n\n\n")
                    time.sleep(1)
                    break
        except KeyboardInterrupt:
            print("\nExecution interrupted by user.")
            
        print("MAIN === Resetting robot...")
        runner.reset_robot()
        
    print("\nMAIN === Exiting...")

if __name__ == "__main__":
    main() 