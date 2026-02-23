from multiprocessing.managers import BaseManager
from collections import defaultdict

from eva.env import FrankaEnv
from eva.runner_augmented import RunnerAugmented


class RunnerAugmentedManager(BaseManager):
    pass


EXPOSED_METHODS = [
    "get_camera_feed",
    "get_controller_info",
    "apply_action",
    "reset_robot",
    "run_trajectory",
    "get_obs",
    "get_state",
    "set_action_space",
    "set_controller",
    "set_prev_controller",
    "set_controller_instruction",
    "print",
    "preprocess_instruction",
    "get_pred_traj",
    "enable_annotated_camera",
    "disable_annotated_camera",
    "set_annotated_camera_enabled",
]


def init(
    controller,
    record_depth,
    record_pcd,
    post_process,
    horizon=None,
    use_annotated_camera=False,
    gpt_model="gpt-4o-mini",
    gemini_model="gemini-robotics-er-1.5-preview",
    plan_freq=10,
    max_plan_count=10,
    save_trajectory_img_dir=None,
    augment_camera_ids=None,
):
    camera_kwargs = defaultdict(
        lambda: {"depth": record_depth, "pointcloud": record_pcd}
    )
    env = FrankaEnv(camera_kwargs=camera_kwargs)
    runner = RunnerAugmented(
        env=env,
        controller=controller,
        post_process=post_process,
        horizon=horizon,
        use_annotated_camera=use_annotated_camera,
        gpt_model=gpt_model,
        gemini_model=gemini_model,
        plan_freq=plan_freq,
        max_plan_count=max_plan_count,
        save_trajectory_img_dir=save_trajectory_img_dir,
        augment_camera_ids=augment_camera_ids or ["varied_camera_1"],
    )
    return runner


def start_runner(
    controller="occulus",
    record_depth=False,
    record_pcd=False,
    post_process=False,
    horizon=None,
    use_annotated_camera=False,
    gpt_model="gpt-4o-mini",
    gemini_model="gemini-robotics-er-1.5-preview",
    plan_freq=10,
    max_plan_count=10,
    save_trajectory_img_dir=None,
    augment_camera_ids=None,
):
    runner = init(
        controller,
        record_depth,
        record_pcd,
        post_process,
        horizon=horizon,
        use_annotated_camera=use_annotated_camera,
        gpt_model=gpt_model,
        gemini_model=gemini_model,
        plan_freq=plan_freq,
        max_plan_count=max_plan_count,
        save_trajectory_img_dir=save_trajectory_img_dir,
        augment_camera_ids=augment_camera_ids,
    )
    RunnerAugmentedManager.register(
        "Runner", lambda: runner, exposed=EXPOSED_METHODS
    )
    manager = RunnerAugmentedManager(
        address=("localhost", 50000), authkey=b"franka_runner"
    )
    server = manager.get_server()
    print("Starting augmented runner on localhost:50000...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down augmented runner...")
        server.shutdown()
        server.server_close()
        runner.close()


def load_runner(manager=True, **kwargs):
    if not manager:
        print("Loading augmented runner in standalone mode...")
        runner = init(**kwargs)
        return runner
    print("Loading augmented runner via manager connection ...")
    RunnerAugmentedManager.register("Runner")
    manager = RunnerAugmentedManager(
        address=("localhost", 50000), authkey=b"franka_runner"
    )
    try:
        manager.connect()
        return manager.Runner()
    except ConnectionRefusedError:
        print(
            "ERROR: Failed to connect to augmented runner server, "
            "please make sure start_runner.py is running."
        )
        raise
