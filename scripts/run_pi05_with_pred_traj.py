
import argparse
from tqdm import tqdm
import time
from eva.runner_augmented import RunnerAugmented
from eva.manager_augmented import load_runner
from eva.controllers.pi0_policy import Pi0PolicyConfig
import os
import json
# run_pi05.py -n 1 -c pi0_policy

def evaluate_policy(runner: RunnerAugmented, controller=None, controller_kwargs=None, n_traj=1, practice=False):
    print("Evaluating pi05 with controller:", controller)
    runner.set_controller(controller, **(controller_kwargs or {}))
    
    for _ in tqdm(range(n_traj), disable=(n_traj == 1)):
        current_instr = getattr(runner.controller, 'current_instruction', None)
        print(f"\nCurrent instruction: {current_instr}")
        new_instruction = input("Enter new instruction (press Enter to keep current): ").strip()
        if new_instruction:
            runner.controller.set_instruction(new_instruction)
            current_instr = new_instruction

        if current_instr:
            runner.preprocess_instruction(current_instr)
            if runner.steps:
                runner.controller.set_instruction(runner.steps[0]["step"])
        
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
    parser.add_argument("-n", "--n_traj", type=int, default=5)
    parser.add_argument("--practice", action="store_true")
    parser.add_argument("--traj_version", type=int, choices=[0, 1], default=1)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--gemini-model", default="gemini-robotics-er-1.6-preview")
    parser.add_argument("--data-path", type=str, default="/home/franka/eva_jiani/data/test_traj")
    parser.add_argument("--instruction_cache_path", default=None)
    parser.add_argument("--max-plan-count", type=int, default=20,
                        help="Maximum number of replanning calls per trajectory")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Disable trajectory overlay on model input (planning images are still saved)")
    parser.add_argument("--no-debug-per-frame", action="store_true",
                        help="Disable per-env-step debug saving (overlay images + JSON + stitched MP4)")
    parser.add_argument("--debug-video-fps", type=int, default=15,
                        help="FPS for the stitched per-frame debug video (default: 15)")
    parser.add_argument("--remote-host", default=None,
                        help="pi0 policy server host (overrides Pi0PolicyConfig default)")
    parser.add_argument("--remote-port", type=int, default=None,
                        help="pi0 policy server port (overrides Pi0PolicyConfig default)")
    parser.add_argument("--open-loop-horizon", type=int, default=None,
                        help="Number of actions to execute per policy query (overrides default 8)")
    parser.add_argument("--instruction", default=None,
                        help="Initial instruction to pass to the policy (otherwise interactive prompt)")

    args = parser.parse_args()
    os.makedirs(args.data_path, exist_ok=True)
    if args.instruction_cache_path is None:
        args.instruction_cache_path = os.path.join(args.data_path, "instruction_cache.json")

    if os.path.exists(args.instruction_cache_path):
        with open(args.instruction_cache_path, "r") as f:
            instruction_cache = json.load(f)
    else:
        instruction_cache = {}

    runner = load_runner(
        manager=False,
        controller="pi0_policy",
        record_depth=False,
        record_pcd=False,
        post_process=True,
        use_annotated_camera=not args.no_overlay,
        gpt_model=args.model,
        gemini_model=args.gemini_model,
        save_trajectory_img_dir=args.data_path,
        max_plan_count=args.max_plan_count,
        debug_per_frame=not args.no_debug_per_frame,
        debug_video_fps=args.debug_video_fps,
    )
    runner.instruction_cache = instruction_cache
    runner.instruction_cache_path = args.instruction_cache_path

    pi0_cfg_overrides = {}
    if args.remote_host is not None:
        pi0_cfg_overrides["remote_host"] = args.remote_host
    if args.remote_port is not None:
        pi0_cfg_overrides["remote_port"] = args.remote_port
    if args.open_loop_horizon is not None:
        pi0_cfg_overrides["open_loop_horizon"] = args.open_loop_horizon
    if args.instruction is not None:
        pi0_cfg_overrides["instruction"] = args.instruction
    controller_kwargs = {}
    if pi0_cfg_overrides:
        controller_kwargs["config"] = Pi0PolicyConfig(**pi0_cfg_overrides)

    evaluate_policy(
        runner,
        controller="pi0_policy",
        controller_kwargs=controller_kwargs,
        n_traj=args.n_traj,
        practice=args.practice,
    )

    with open(args.instruction_cache_path, "w") as f:
        json.dump(runner.instruction_cache, f, indent=2)
