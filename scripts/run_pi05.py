
import argparse
from tqdm import tqdm
import time
from eva.runner import Runner
from eva.manager import load_runner
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
    args = parser.parse_args()

    runner = load_runner(manager=False, controller="pi0_policy", record_depth=False, \
                         record_pcd=False, post_process=True)
    
    evaluate_policy(runner, controller="pi0_policy", n_traj=args.n_traj, practice=args.practice)
