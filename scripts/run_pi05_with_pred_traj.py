
import argparse
from tqdm import tqdm
import time
from eva.runner import Runner
from eva.manager import load_runner
import os
import json
# run_pi05.py -n 1 -c pi0_policy
                
def evaluate_policy(runner: Runner, controller=None, n_traj=1, practice=False):
    print("Evaluating pi05 with controller:", controller)
    runner.set_controller(controller) 
    
    for _ in tqdm(range(n_traj), disable=(n_traj == 1)):
        current_instr = getattr(runner.controller, 'current_instruction', 'None')
        print(f"\nCurrent instruction: {current_instr}")
        new_instruction = input("Enter new instruction (press Enter to keep current): ").strip()
        if new_instruction:
            runner.controller.set_instruction(new_instruction)
       
        
        # Ask for horizon
        current_horizon = getattr(runner.controller, 'open_loop_horizon', 'None')
        print(f"Current horizon: {current_horizon}")
        new_horizon = input("Enter new horizon (press Enter to keep current): ").strip()
        if new_horizon:
            if hasattr(runner.controller, 'set_horizon'):
                runner.controller.set_horizon(new_horizon)
            else:
                runner.controller.open_loop_horizon = int(new_horizon)
                print(f"Set open loop horizon to {runner.controller.open_loop_horizon}")

        print("DEBUG: instruction = ", getattr(runner.controller, 'current_instruction', 'None'))

        start_time = time.time()
        runner.run_trajectory(mode="collect") 
        print("\033[91mReady to reset, press any controller button...\033[0m")

        controller_info = runner.get_controller_info()
        if controller_info["success"] or controller_info["failure"]:
             print("\n\n\033[91mEVAL == Total steps: ", controller_info["t_step"], "\033[0m")

        end_time = time.time()
        print(f"\033[91mTime taken: {end_time - start_time:.2f} seconds\033[0m")
        runner.reset_robot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_traj", type=int, default=10)
    parser.add_argument("--practice", action="store_true")
    parser.add_argument("--traj_version", type=int, choices=[0, 1], default=0)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--gemini-model", default="gemini-robotics-er-1.5-preview")
    parser.add_argument("--data-path", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    parser.add_argument("--instruction_cache_path", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "preprocess_cache", "instruction_cache.json"))
    
    args = parser.parse_args()
    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(os.path.join(args.data_path, "preprocess_cache"), exist_ok=True)

    if os.path.exists(args.instruction_cache_path):
        with open(args.instruction_cache_path, "r") as f:
            instruction_cache = json.load(f)
    else:
        instruction_cache = {}

    runner = load_runner(manager=False, controller="pi0_policy", record_depth=False, \
                         record_pcd=False, post_process=True)
    
    evaluate_policy(runner, controller="pi0_policy", n_traj=args.n_traj, practice=args.practice)
