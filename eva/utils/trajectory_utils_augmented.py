"""
Augmented trajectory utilities with replanning hooks.

Provides run_trajectory_augmented(), which extends the base run_trajectory loop
with support for injecting analysis (e.g. predicted trajectory updates) on the
first step and periodically every ``plan_freq`` steps.
"""

import time
import numpy as np

from eva.utils.trajectory_utils import TrajectoryWriter
from eva.utils.misc_utils import time_ms, yellow_print
from eva.remote_timer import RemoteTimer

timer = RemoteTimer()


def run_trajectory_augmented(
    env,
    controller,
    horizon=None,
    save_filepath=None,
    metadata=None,
    wait_for_controller=False,
    obs_pointer=None,
    recording_folderpath=False,
    randomize_reset=False,
    reset_robot=True,
    post_process=False,
    obs_transform=None,
    # --- augmented parameters ---
    analysis_fn=None,
    on_analysis_complete=None,
    plan_freq=10,
    max_plan_count=10,
):
    """Run a trajectory with periodic analysis / replanning hooks.

    Identical to :func:`eva.utils.trajectory_utils.run_trajectory` except for
    the extra keyword arguments below.

    Parameters
    ----------
    analysis_fn : callable or None
        ``analysis_fn(obs, step_num, plan_count)`` is invoked on the **first**
        step (step 0) and then every ``plan_freq`` steps, up to
        ``max_plan_count`` total invocations.  Use this to query an LLM for a
        predicted trajectory, run a detector, etc.  The return value is passed
        through to ``on_analysis_complete``.
    on_analysis_complete : callable or None
        ``on_analysis_complete(analysis_result, obs, step_num)`` is called
        immediately after ``analysis_fn`` returns.  Use this to overlay the
        predicted trajectory on observations, update the controller, etc.
    plan_freq : int
        Re-run ``analysis_fn`` every this many steps after the initial call.
    max_plan_count : int
        Maximum number of times ``analysis_fn`` will be called per trajectory.
    """
    if post_process:
        assert save_filepath is not None, "Must save data to post process"

    controller.reset_state()
    env.camera_reader.set_trajectory_mode()

    if save_filepath:
        traj_writer = TrajectoryWriter(save_filepath, metadata=metadata, post_process=post_process)
    if recording_folderpath:
        env.camera_reader.start_recording(recording_folderpath)

    num_steps = 0
    plan_count = 0
    if reset_robot:
        env.reset()

    timer.reset()
    timer.toggle("Running Inference...")

    while True:
        controller_info = controller.get_info()
        skip_action = wait_for_controller and (not controller_info["movement_enabled"])
        control_timestamps = {"step_start": time_ms()}

        obs = env.get_observation()
        if obs_pointer is not None:
            obs_pointer.update(obs)
        obs["controller_info"] = controller_info
        obs["timestamp"]["skip_action"] = skip_action

        # ---- Analysis / replanning hook (runs on raw obs) ----
        should_analyse = (
            analysis_fn is not None
            and plan_count < max_plan_count
            and (num_steps == 0 or (plan_freq > 0 and num_steps % plan_freq == 0))
        )
        if should_analyse:
            yellow_print(
                f"[augmented] Running analysis at step {num_steps} "
                f"(plan {plan_count + 1}/{max_plan_count})"
            )
            analysis_result = analysis_fn(obs, num_steps, plan_count)
            plan_count += 1

            if on_analysis_complete is not None:
                on_analysis_complete(analysis_result, obs, num_steps)

        # ---- Apply obs transform after analysis so overlay reflects latest trajectory ----
        if obs_transform is not None:
            obs = obs_transform(obs)

        # ---- Policy forward pass ----
        control_timestamps["policy_start"] = time_ms()
        action, controller_action_info = controller.forward(obs)
        if controller.get_name() == "aawr-pi0":
            controller.save_grid(save_filepath)

        control_timestamps["sleep_start"] = time_ms()
        comp_time = time_ms() - control_timestamps["step_start"]
        sleep_left = (1 / env.control_hz) - (comp_time / 1000)
        if sleep_left > 0:
            time.sleep(sleep_left)

        # ---- Step environment ----
        control_timestamps["control_start"] = time_ms()
        if skip_action:
            action_info = env.create_action_dict(np.zeros_like(action))
        else:
            action_info = env.step(action)
        action_info.update(controller_action_info)

        control_timestamps["step_end"] = time_ms()
        obs["timestamp"]["control"] = control_timestamps
        timestep = {"observation": obs, "action": action_info}
        if save_filepath:
            traj_writer.write_timestep(timestep)

        # ---- Termination check ----
        num_steps += 1
        if horizon is not None:
            end_traj = horizon == num_steps
        else:
            end_traj = controller_info["success"] or controller_info["failure"]

        if end_traj:
            timer.reset()
            yellow_print(
                f"[augmented] Trajectory ended after {num_steps} steps, "
                f"{plan_count} analysis calls"
            )
            if recording_folderpath:
                env.camera_reader.stop_recording()
            if save_filepath:
                traj_writer.close(metadata=controller_info)
            return controller_info
