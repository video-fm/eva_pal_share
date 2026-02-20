"""
RunnerAugmented: A Runner subclass that can pass annotated camera images to the model.

When use_annotated_camera is enabled, observations are transformed via annotate_observation()
before being passed to the controller. Override annotate_observation() to implement
custom annotation logic (e.g., bounding boxes, keypoints, segmentation overlays).
"""

import os
from copy import deepcopy
from datetime import datetime
import shutil

import numpy as np
from PIL import Image
from openai import OpenAI

from eva.runner import Runner
from eva.detectors.trajectory_v1 import (
    query_target_objects,
    query_target_location,
    query_trajectory,
    encode_pil_image,
)
from eva.detectors.traj_vis_utils import add_arrow
from eva.utils.trajectory_utils_augmented import run_trajectory_augmented
from eva.utils.calibration_utils import save_calibration_info
from eva.utils.parameters import (
    robot_type,
    robot_serial_number,
    code_version,
    varied_camera_1_id,
)
from eva.utils.misc_utils import yellow_print


class RunnerAugmented(Runner):
    """Runner with optional annotated camera feed for model input."""

    def __init__(
        self,
        env,
        controller,
        use_annotated_camera=False,
        openai_client=None,
        gpt_model="gpt-4o-mini",
        gemini_model="gemini-robotics-er-1.5-preview",
        plan_freq=10,
        max_plan_count=10,
        save_trajectory_img_path=None,
        augment_camera_ids = ["varied_camera_1"],
        **kwargs,
    ):
        super().__init__(env, controller, **kwargs)
        self._use_annotated_camera = use_annotated_camera
        self.pred_traj = None
        self.instruction_cache = None
        self._openai_client = openai_client or OpenAI()
        self._gpt_model = gpt_model
        self._gemini_model = gemini_model
        self.steps = []
        self.step = 0
        self.save_trajectory_img_path = save_trajectory_img_path
        self.augment_camera_ids = augment_camera_ids
        
        # TODO: replan for each 10 roll outs has been executed
        self.plan_freq = plan_freq
        # TODO: max plan count is the maximum number of plans to be executed
        self.max_plan_count = max_plan_count
        
        
        
    def annotate_observation(self, obs):
        """
        Overlay predicted trajectory arrow onto camera images listed in
        ``self.augment_camera_ids``.  Other cameras are left untouched.
        """
        if self.pred_traj is None or "trajectory" not in self.pred_traj:
            return obs
        pts = [tuple(p) for p in self.pred_traj["trajectory"]]
        if len(pts) < 2:
            return obs

        if "image" not in obs:
            return obs

        for cam_id in list(obs["image"].keys()):
            if not any(aid in str(cam_id) for aid in self.augment_camera_ids):
                continue
            img_arr = obs["image"][cam_id]
            annotated = add_arrow(img_arr, pts, color="red", line_width=3)
            obs["image"][cam_id] = np.array(annotated.convert("RGB"))

        return obs

    def preprocess_instruction(self, instruction: str):
        """Extract target objects and steps from instruction using trajectory_v1 pipeline."""
        target = query_target_objects(
            self._openai_client, instruction, model=self._gpt_model
        )
        self.steps = target.get("steps", [])
        self.step = 0

    def _image_to_pil(self, curr_image, max_size=1024):
        """Convert curr_image (numpy array or PIL Image) to resized PIL Image."""
        if isinstance(curr_image, np.ndarray):
            img = Image.fromarray(curr_image).convert("RGB")
        elif isinstance(curr_image, Image.Image):
            img = curr_image.convert("RGB")
        else:
            raise TypeError(f"curr_image must be np.ndarray or PIL.Image, got {type(curr_image)}")
        w, h = img.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
        return img

    def check_traj_distance(self, pred_traj, curr_image):
        raise NotImplementedError("Not implemented")
    
    def check_step_completion(self, pred_traj, curr_image):
        raise NotImplementedError("Not implemented")
    
    def get_pred_traj(
        self,
        curr_image,
        gemini_model_name=None,
    ):
        """
        Get predicted trajectory for the current step from curr_image.

        Args:
            curr_image: numpy array (H×W×C) or PIL Image from camera.
            gemini_model_name: Model for object detection (default: self._gemini_model).
            save_trajectory_img_path: Optional path to save visualization.

        Returns:
            Trajectory dict with "trajectory" key (list of [x,y] points), or None on failure.
        """
        if not self.steps or self.step >= len(self.steps):
            return None

        step_data = self.steps[self.step]
        manipulating_object = step_data["manipulating_object"]
        target_related_object = step_data["target_related_object"]
        target_location = step_data["target_location"]
        target_objects = list(set([manipulating_object, target_related_object]))

        img = self._image_to_pil(curr_image)
        gemini_model = gemini_model_name or self._gemini_model

        object_locations = query_target_location(
            img,
            target_objects,
            model_name=gemini_model,
            visualize=False,
        )
        if object_locations is None:
            return None

        manipulating_object_point = object_locations.get(manipulating_object)
        target_related_object_point = object_locations.get(target_related_object)
        if manipulating_object_point is None or target_related_object_point is None:
            return None

        img_encoded = encode_pil_image(img)

        trajectory = query_trajectory(
            self._openai_client,
            img=img,
            img_encoded=img_encoded,
            task=step_data["step"],
            manipulating_object=manipulating_object,
            manipulating_object_point=manipulating_object_point,
            target_related_object=target_related_object,
            target_related_object_point=target_related_object_point,
            target_location=target_location,
            model_name=self._gpt_model,
        )

        if self.save_trajectory_img_path is not None and trajectory and "trajectory" in trajectory:
            pts = [tuple(p) for p in trajectory["trajectory"]]
            if len(pts) >= 2:
                img_with_arrow = add_arrow(img, pts, color="red", line_width=3)
                img_with_arrow.convert("RGB").save(self.save_trajectory_img_path)

        self.pred_traj = trajectory
    
    def set_annotated_camera_enabled(self, enabled: bool):
        """Enable or disable passing annotated camera images to the model."""
        self._use_annotated_camera = enabled

    def enable_annotated_camera(self):
        """Enable annotated camera mode."""
        self._use_annotated_camera = True

    def disable_annotated_camera(self):
        """Disable annotated camera mode (use raw images)."""
        self._use_annotated_camera = False

    @property
    def use_annotated_camera(self) -> bool:
        """Whether annotated camera images are passed to the model."""
        return self._use_annotated_camera

    def _extract_camera_image(self, obs):
        """Pull the varied-camera image from an observation dict for analysis."""
        if "image" not in obs:
            return None
        image_dict = obs["image"]
        for cam_id in image_dict:
            if varied_camera_1_id in str(cam_id):
                return image_dict[cam_id]
        return next(iter(image_dict.values()), None)

    def _build_analysis_fn(self):
        """Return the analysis callback for run_trajectory_augmented."""
        def analysis_fn(obs, step_num, plan_count):
            img = self._extract_camera_image(obs)
            if img is None:
                yellow_print(f"[augmented] No camera image at step {step_num}, skipping analysis")
                return None
            self.get_pred_traj(img)
            return self.pred_traj
        return analysis_fn

    def _build_on_analysis_complete(self):
        """Return the post-analysis callback for run_trajectory_augmented."""
        def on_analysis_complete(analysis_result, obs, step_num):
            if analysis_result is not None:
                yellow_print(
                    f"[augmented] Updated predicted trajectory at step {step_num}"
                )
        return on_analysis_complete

    def run_trajectory(self, mode, reset_robot=True, wait_for_controller=True):
        def _obs_transform(obs):
            return self.annotate_observation(deepcopy(obs))

        obs_transform = _obs_transform if self._use_annotated_camera else None

        info = dict(
            time=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            robot_serial_number=f"{robot_type}-{robot_serial_number}",
            version_number=code_version,
            controller=self.controller.get_name(),
        )

        if hasattr(self.controller, "current_instruction"):
            info["instruction"] = self.controller.current_instruction

        if hasattr(self.controller, "open_loop_horizon"):
            info["open_loop_horizon"] = self.controller.open_loop_horizon

        traj_name = info["time"]

        if mode == "collect":
            save_dir = os.path.join(self.failure_logdir, traj_name)
        elif mode == "evaluate":
            save_dir = os.path.join(self.eval_logdir, traj_name)
        elif mode == "practice":
            save_dir, recording_dir, save_filepath = None, None, None

        if save_dir is not None:
            if len(self.full_cam_ids) != 6:
                raise ValueError("WARNING: User is trying to collect data without all three cameras running!")
            recording_dir = os.path.join(save_dir, "recordings")
            save_filepath = os.path.join(save_dir, "trajectory.h5")
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(recording_dir, exist_ok=True)
            save_calibration_info(os.path.join(save_dir, "calibration.json"))

            if hasattr(self.controller, "current_instruction"):
                instr_file = os.path.join(save_dir, "instruction.txt")
                with open(instr_file, "w") as f:
                    f.write(self.controller.current_instruction)
                yellow_print(f"Saved instruction to {instr_file}")

        yellow_print("Saving policy name")
        policy_name = self.controller.get_policy_name()
        if save_dir is not None:
            with open(os.path.join(save_dir, "policy.md"), "w") as f:
                f.write(f"# Policy\n\n{policy_name}")

        has_steps = bool(self.steps)
        analysis_fn = self._build_analysis_fn() if has_steps else None
        on_analysis_complete = self._build_on_analysis_complete() if has_steps else None

        self.traj_running = True
        self.env._robot.establish_connection()
        controller_info = run_trajectory_augmented(
            self.env,
            controller=self.controller,
            horizon=self.horizon,
            metadata=info,
            obs_pointer=self.obs_pointer,
            reset_robot=reset_robot,
            recording_folderpath=recording_dir,
            save_filepath=save_filepath,
            post_process=self.post_process,
            wait_for_controller=wait_for_controller,
            obs_transform=obs_transform,
            analysis_fn=analysis_fn,
            on_analysis_complete=on_analysis_complete,
            plan_freq=self.plan_freq,
            max_plan_count=self.max_plan_count,
        )
        self.traj_running = False
        self.obs_pointer = {}

        if mode == "collect" and save_filepath is not None:
            if controller_info["success"]:
                new_save_dir = os.path.join(self.success_logdir, traj_name)
                shutil.move(save_dir, new_save_dir)
                save_dir = new_save_dir
