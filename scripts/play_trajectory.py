
import argparse

from eva.runner import Runner
from eva.manager import load_runner


def play_trajectory(runner: Runner, traj_path: str, action_space: str, autoplay=False, skip_reset=False):
    runner.run_trajectory(mode="evaluate", wait_for_controller=not autoplay, reset_robot=not skip_reset)
    if not autoplay:
        print("Ready to reset, press any controller button...")
        while True:
            controller_info = runner.get_controller_info()
            if controller_info["success"] or controller_info["failure"]:
                break
        runner.reset_robot()
    runner.set_prev_controller()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    TRAJ_FULL_PATH = "human_data/open_drawer/traj_3/processed_3d/replayable_traj.npy"
    # TRAJ_FULL_PATH = "human_data/open_cabinet/traj_0/processed_3d/replayable_traj.npy"
    # TRAJ_FULL_PATH = "human_data/open_drawer/traj_1/processed_3d/replayabele_traj_withoffset.npy"
    # TRAJ_FULL_PATH = "human_data/open_drawer/traj_1/processed_3d/replayable_traj_withoffset.npy"
    parser.add_argument("--traj_path", default=TRAJ_FULL_PATH, type=str)
    parser.add_argument("--action_space", default="cartesian_position")
    parser.add_argument("--autoplay", action="store_true") 
    parser.add_argument("--skip_reset", action="store_true")
    args = parser.parse_args()
    args.traj_path = TRAJ_FULL_PATH

    runner = load_runner(manager=False, controller=None, record_depth=False, record_pcd=False, post_process=True)
    runner.set_controller("replayer", traj_path=args.traj_path, action_space=args.action_space)
    play_trajectory(runner, args.traj_path, args.action_space, args.autoplay, args.skip_reset)
