
import argparse
from tqdm import tqdm
import time
from eva.runner import Runner
from eva.manager import load_runner
# run_eva_policy.py -n 1 -c pi0_policy
# jie wang, 04/17/2025 V1 start from collect_trajectory.py
# TODO: 
# 1. add more controllers
# 2. add more evaluation metrics
                
def evaluate_policy(runner: Runner, controller=None, n_traj=1, practice=False):
    print("Evaluating policy with controller:", controller)
    if controller is not None and controller != "mixed":
        runner.set_controller(controller) 
    
        
    for _ in tqdm(range(n_traj), disable=(n_traj == 1)):
        start_time = time.time()
        runner.run_trajectory(mode="collect") # mode: collect, evaluate, practice
        print("\033[91mReady to reset, press any controller button...\033[0m")
        for _ in tqdm(range(500), desc="Collecting rollouts"):
            controller_info = runner.get_controller_info()
            if controller_info["success"] or controller_info["failure"]:
                print("DEBUG: contoller_info.keys() = ", controller_info.keys())
                print(controller_info.keys())
                # print("\n\n\033[91mEVAL == SWITCH AT: ", controller_info["switch_at"], "\033[0m")
                print("\n\n\033[91mEVAL == Total steps: ", controller_info["t_step"], "\033[0m")
                breakpoint
                break
        end_time = time.time()
        print(f"\033[91mTime taken: {end_time - start_time:.2f} seconds\033[0m")
        runner.reset_robot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_traj", type=int, default=10)
    parser.add_argument("-c","--controller", default=None, choices=["occulus", "keyboard", "gello", "spacemouse" ,
                                                                    "policy", "pi0_policy",
                                                                   "keyboard_pi0", "aawr_pi0", "mixed", "replay_pi0"])  
    parser.add_argument("--practice", action="store_true")
    args = parser.parse_args()

    runner = load_runner(manager=False, controller=args.controller, record_depth=False, \
                         record_pcd=False, post_process=True)
    
    evaluate_policy(runner, controller=args.controller, n_traj=args.n_traj, practice=args.practice)
