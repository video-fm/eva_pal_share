
import argparse
from tqdm import tqdm

from eva.runner import Runner
from eva.manager import load_runner


def collect_trajectory(runner: Runner, controller=None, n_traj=10, practice=False):
    if controller is not None and controller != "mixed":
        runner.set_controller(controller) 
    runner.reset_robot()
    
    for _ in tqdm(range(n_traj), disable=(n_traj == 1)):
        runner.run_trajectory(mode="collect") # mode: collect, evaluaste, practice

        runner.print("Ready to reset, press any controller button...")
        for _ in tqdm(range(1000), desc="Collecting rollouts"):
            controller_info = runner.get_controller_info()
            if controller_info["success"] or controller_info["failure"]:
                break
        runner.reset_robot()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_traj", type=int, default=10)
    parser.add_argument("-c", "--controller", default=None, choices=["occulus", "keyboard", "gello", "spacemouse", \
                                                                     "replay_pi0", "aawr_pi0", "pi0_policy", "mixed", "policy"])  
    parser.add_argument("--practice", action="store_true")
    args = parser.parse_args()

    runner = load_runner(manager=False, controller=args.controller, record_depth=False, record_pcd=False, post_process=True)
    collect_trajectory(runner, controller=args.controller, n_traj=args.n_traj, practice=args.practice)
