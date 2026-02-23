"""
RunnerAugmented: A Runner subclass that can pass annotated camera images to the model.

When use_annotated_camera is enabled, observations are transformed via annotate_observation()
before being passed to the controller. Override annotate_observation() to implement
custom annotation logic (e.g., bounding boxes, keypoints, segmentation overlays).
"""

import json
import os
from copy import deepcopy
from datetime import datetime
import shutil

import cv2
import numpy as np
from PIL import Image
from openai import OpenAI

from eva.runner import Runner
from eva.detectors.trajectory_v1 import (
    query_target_objects,
    query_target_location,
    query_trajectory,
    query_step_completion,
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
    camera_string_to_id_dict,
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
        max_plan_count=20,
        save_trajectory_img_dir=None,
        augment_camera_ids = ["varied_camera_1"],
        **kwargs,
    ):
        super().__init__(env, controller, **kwargs)
        self._use_annotated_camera = use_annotated_camera
        self.pred_traj = None
        self.instruction_cache = None
        self.instruction_cache_path = None
        self._openai_client = openai_client or OpenAI()
        self._gpt_model = gpt_model
        self._gemini_model = gemini_model
        self.steps = []
        self.step = 0
        self.current_end_point = None
        self.img_history = []
        self.img_encoded_history = []
        self.save_trajectory_img_dir = save_trajectory_img_dir
        self.augment_camera_ids = [
            camera_string_to_id_dict.get(aid, aid) for aid in augment_camera_ids
        ]
        
        self.plan_freq = plan_freq
        self.max_plan_count = max_plan_count
        
    def annotate_observation(self, obs):
        """
        Overlay predicted trajectory arrow onto camera images listed in
        ``self.augment_camera_ids``.  Other cameras are left untouched.

        Returns:
            The (mutated) obs dict.  ``obs["_annotated_cameras"]`` is set to the
            list of camera IDs that were annotated (empty list when trajectory
            is unavailable).
        """
        if "image" not in obs:
            obs["_annotated_cameras"] = []
            return obs

        can_annotate = (
            self.pred_traj is not None
            and "trajectory" in self.pred_traj
            and len(self.pred_traj["trajectory"]) >= 2
        )

        annotated_ids = []
        raw_ids = []
        for cam_id in list(obs["image"].keys()):
            cam_str = str(cam_id)
            is_target = (
                "left" in cam_str
                and any(aid in cam_str for aid in self.augment_camera_ids)
            )
            if is_target and can_annotate:
                pts = [tuple(p) for p in self.pred_traj["trajectory"]]
                img_arr = obs["image"][cam_id]
                img_rgb = self._cam_img_to_rgb(img_arr)
                annotated = add_arrow(img_rgb, pts, color="red", line_width=3)
                annotated_rgb = np.array(annotated.convert("RGB"))
                if img_arr.shape[-1] == 4:
                    obs["image"][cam_id] = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGRA)
                else:
                    obs["image"][cam_id] = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                annotated_ids.append(cam_id)
            else:
                raw_ids.append(cam_id)

        obs["_annotated_cameras"] = annotated_ids
        return obs

    def _reset_step_state(self):
        """Reset per-step tracking state (trajectory, end point, image history)."""
        self.pred_traj = None
        self.current_end_point = None
        self.img_history = []
        self.img_encoded_history = []
        self._last_object_locations = None
        self._last_trajectory_raw = None

    def preprocess_instruction(self, instruction: str):
        """Extract target objects and steps from instruction using trajectory_v1 pipeline.

        If ``self.instruction_cache`` is a dict and already contains
        *instruction*, the cached result is reused instead of calling the API.
        New results are written back to the cache dict (caller is responsible
        for persisting to disk).
        """
        self._reset_step_state()

        if isinstance(self.instruction_cache, dict) and instruction in self.instruction_cache:
            cached = self.instruction_cache[instruction]
            self.steps = cached.get("steps", [])
            self.step = 0
            yellow_print(f"[cache hit] Reusing cached instruction: {instruction!r}")
            return

        img_encoded = None
        try:
            cam_obs, _ = self.env.read_cameras()
            cam_img = self._extract_camera_image(cam_obs)
            if cam_img is not None:
                pil_img, _ = self._image_to_pil(cam_img)
                img_encoded = encode_pil_image(pil_img)
        except Exception:
            yellow_print("[preprocess] Could not capture camera image, proceeding without")

        target = query_target_objects(
            self._openai_client, instruction, model=self._gpt_model,
            img_encoded=img_encoded,
        )
        self.steps = target.get("steps", [])
        self.step = 0

        if isinstance(self.instruction_cache, dict):
            self.instruction_cache[instruction] = target
            if self.instruction_cache_path:
                with open(self.instruction_cache_path, "w") as f:
                    json.dump(self.instruction_cache, f, indent=2)

    def _image_to_pil(self, curr_image, max_size=1024):
        """Convert curr_image to a resized PIL Image.

        Returns:
            (resized_img, original_size) where original_size is (width, height)
            before resizing.
        """
        if isinstance(curr_image, np.ndarray):
            img = Image.fromarray(curr_image).convert("RGB")
        elif isinstance(curr_image, Image.Image):
            img = curr_image.convert("RGB")
        else:
            raise TypeError(f"curr_image must be np.ndarray or PIL.Image, got {type(curr_image)}")
        original_size = img.size
        w, h = original_size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
        return img, original_size

    @staticmethod
    def _rescale_trajectory(trajectory, resized_size, original_size):
        """Rescale trajectory coordinates from resized-image space to original-image space.

        Args:
            trajectory: dict with "trajectory", "start_point", "end_point" keys.
            resized_size: (width, height) of the resized image used for prediction.
            original_size: (width, height) of the original camera image.

        Returns:
            A new trajectory dict with all coordinates scaled to original_size.
        """
        rw, rh = resized_size
        ow, oh = original_size
        if rw == ow and rh == oh:
            return trajectory

        sx = ow / rw
        sy = oh / rh

        def scale_pt(pt):
            return [pt[0] * sx, pt[1] * sy]

        out = dict(trajectory)
        if "trajectory" in out:
            out["trajectory"] = [scale_pt(p) for p in out["trajectory"]]
        if "start_point" in out:
            out["start_point"] = scale_pt(out["start_point"])
        if "end_point" in out:
            out["end_point"] = scale_pt(out["end_point"])
        return out

    def check_traj_distance(self, pred_traj, curr_image):
        raise NotImplementedError("Not implemented")

    def check_step_completion(self):
        """Check whether the current step is complete using accumulated image history.

        Returns:
            Dict with "is_complete" and "reasoning" keys, or None if not enough
            state is available for the check.
        """
        if not self.steps or self.step >= len(self.steps):
            return None
        if not self.img_history:
            return None

        step_data = self.steps[self.step]
        return query_step_completion(
            [self.img_history[-1]],
            step=step_data["step"],
            model_name=self._gemini_model,
            max_images=1,
        )
    
    def get_pred_traj(
        self,
        curr_image,
        gemini_model_name=None,
        target_location_point=None,
    ):
        """
        Get predicted trajectory for the current step from curr_image.

        Args:
            curr_image: numpy array (H*W*C) or PIL Image from camera.
            gemini_model_name: Model for object detection (default: self._gemini_model).
            target_location_point: If provided, re-plan toward this known end
                point (in resized-image coordinates) instead of letting the
                model choose one freely.

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

        img, original_size = self._image_to_pil(curr_image)
        gemini_model = gemini_model_name or self._gemini_model

        object_locations = query_target_location(
            img,
            target_objects,
            model_name=gemini_model,
            visualize=False,
        )
        self._last_object_locations = object_locations
        if object_locations is None:
            yellow_print("[get_pred_traj] query_target_location returned None")
            return None

        manipulating_object_point = object_locations.get(manipulating_object)
        target_related_object_point = object_locations.get(target_related_object)
        if manipulating_object_point is None or target_related_object_point is None:
            missing = []
            if manipulating_object_point is None:
                missing.append(manipulating_object)
            if target_related_object_point is None:
                missing.append(target_related_object)
            yellow_print(
                f"[get_pred_traj] Object(s) not found: {missing}. "
                f"Detected: {list(object_locations.keys())}"
            )
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
            target_location_point=target_location_point,
        )

        if self.save_trajectory_img_dir is not None and trajectory and "trajectory" in trajectory:
            pts = [tuple(p) for p in trajectory["trajectory"]]
            save_trajectory_img_path = os.path.join(self.save_trajectory_img_dir, f"{self.step:05d}_trajectory.jpg")
            if len(pts) >= 2:
                img_with_arrow = add_arrow(img, pts, color="red", line_width=3)
                img_with_arrow.convert("RGB").save(save_trajectory_img_path)

        self._last_trajectory_raw = trajectory

        if trajectory and "end_point" in trajectory and self.current_end_point is None:
            self.current_end_point = trajectory["end_point"]

        if trajectory and "trajectory" in trajectory:
            trajectory = self._rescale_trajectory(trajectory, img.size, original_size)

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

    @staticmethod
    def _cam_img_to_rgb(img):
        """Convert a raw camera image (BGRA/BGR) to a 3-channel RGB numpy array."""
        if img.ndim == 3 and img.shape[-1] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        if img.ndim == 3 and img.shape[-1] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _extract_camera_image(self, obs):
        """Pull the varied-camera left image from an observation dict as RGB."""
        if "image" not in obs:
            return None
        image_dict = obs["image"]
        for cam_id in image_dict:
            cam_str = str(cam_id)
            if varied_camera_1_id in cam_str and "left" in cam_str:
                return self._cam_img_to_rgb(image_dict[cam_id])
        raw = next(iter(image_dict.values()), None)
        return self._cam_img_to_rgb(raw) if raw is not None else None

    def _build_analysis_fn(self, run_save_dir=None, trajectory_log=None):
        """Return the analysis callback for run_trajectory_augmented.

        The callback implements the full step-completion / re-planning loop:

        1. First invocation (no existing trajectory): query an initial
           trajectory and record its ``end_point``.
        2. Subsequent invocations: check whether the current step is complete.
           - **Complete** -- advance ``self.step``, reset per-step state, and
             query a fresh trajectory for the next step.
           - **Not complete** -- re-plan toward the stored ``end_point`` so
             the model can correct for drift.

        Args:
            run_save_dir: If set, saves raw frames and trajectory-annotated
                frames to this directory at each analysis step.
            trajectory_log: If set, a mutable list that will be appended with
                trajectory metadata dicts for each analysis step.
        """
        def analysis_fn(obs, step_num, plan_count):
            img = self._extract_camera_image(obs)
            if img is None:
                yellow_print(f"[augmented] No camera image at step {step_num}, skipping analysis")
                return None

            if run_save_dir is not None:
                frame_path = os.path.join(run_save_dir, f"frame_step_{step_num:03d}.jpg")
                Image.fromarray(img).convert("RGB").save(frame_path)

            pil_img, _ = self._image_to_pil(img)
            self.img_history.append(pil_img)
            self.img_encoded_history.append(encode_pil_image(pil_img))

            if self.steps and self.step < len(self.steps):
                self.controller.set_instruction(self.steps[self.step]["step"])

            completion = None
            if len(self.img_history) >= 1:
                completion = self.check_step_completion()

            if completion and completion.get("is_complete", False):
                yellow_print(
                    f"[augmented] Step {self.step} "
                    f"(\"{self.steps[self.step]['step']}\") completed at env step {step_num}"
                )
                self.step += 1
                self._reset_step_state()

                if self.step >= len(self.steps):
                    yellow_print("[augmented] All steps completed!")
                    if run_save_dir is not None:
                        query_log = {
                            "step_num": step_num,
                            "plan_count": plan_count + 1,
                            "step_index": self.step,
                            "step_description": None,
                            "completion_check": completion,
                            "object_locations": {},
                            "trajectory_raw": None,
                            "trajectory_rescaled": None,
                            "all_steps_completed": True,
                        }
                        query_path = os.path.join(run_save_dir, f"queries_step_{step_num:03d}.json")
                        with open(query_path, "w") as f:
                            json.dump(query_log, f, indent=2)
                    return None

                self.controller.set_instruction(self.steps[self.step]["step"])
                self.controller.reset_state()
                pil_img_new, _ = self._image_to_pil(img)
                self.img_history = [pil_img_new]
                self.img_encoded_history = [encode_pil_image(pil_img_new)]

                self.get_pred_traj(img)
            else:
                if self.pred_traj is not None:
                    yellow_print(
                        f"[augmented] Step {self.step} not complete, "
                        f"re-planning at env step {step_num}"
                    )
                    self.get_pred_traj(img, target_location_point=self.current_end_point)
                else:
                    self.get_pred_traj(img)

            if run_save_dir is not None:
                query_log = {
                    "step_num": step_num,
                    "plan_count": plan_count + 1,
                    "step_index": self.step,
                    "step_description": self.steps[self.step]["step"] if self.step < len(self.steps) else None,
                    "completion_check": completion,
                    "object_locations": {
                        k: list(v) if v is not None else None
                        for k, v in (self._last_object_locations or {}).items()
                    },
                    "trajectory_raw": self._last_trajectory_raw,
                    "trajectory_rescaled": self.pred_traj,
                }
                query_path = os.path.join(run_save_dir, f"queries_step_{step_num:03d}.json")
                with open(query_path, "w") as f:
                    json.dump(query_log, f, indent=2)

            if run_save_dir is not None and self.pred_traj and "trajectory" in self.pred_traj:
                pts = [tuple(p) for p in self.pred_traj["trajectory"]]
                if len(pts) >= 2:
                    traj_path = os.path.join(run_save_dir, f"traj_step_{step_num:03d}.jpg")
                    pil_img_save = Image.fromarray(img).convert("RGB")
                    img_with_arrow = add_arrow(pil_img_save, pts, color="red", line_width=3)
                    img_with_arrow.convert("RGB").save(traj_path)

                    overlay_path = os.path.join(run_save_dir, f"overlay_step_{step_num:03d}.jpg")
                    annotated_obs = self.annotate_observation(deepcopy(obs))
                    annotated_obs.pop("_annotated_cameras", None)
                    for cam_key in annotated_obs.get("image", {}):
                        cam_str = str(cam_key)
                        if "left" in cam_str and any(aid in cam_str for aid in self.augment_camera_ids):
                            overlay_rgb = self._cam_img_to_rgb(annotated_obs["image"][cam_key])
                            Image.fromarray(overlay_rgb).save(overlay_path)
                            break

                if trajectory_log is not None:
                    trajectory_log.append({
                        "step_num": step_num,
                        "plan_count": plan_count + 1,
                        "step_index": self.step,
                        "frame_file": f"frame_step_{step_num:03d}.jpg",
                        "traj_image_file": f"traj_step_{step_num:03d}.jpg",
                        "overlay_file": f"overlay_step_{step_num:03d}.jpg",
                        "trajectory": self.pred_traj,
                    })

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
        _logged_first = [False]

        def _obs_transform(obs):
            transformed = self.annotate_observation(deepcopy(obs))
            annotated_ids = transformed.pop("_annotated_cameras", [])
            if not _logged_first[0] and len(annotated_ids) >= 1:
                all_cams = list(transformed.get("image", {}).keys())
                raw_ids = [c for c in all_cams if c not in annotated_ids]
                yellow_print(
                    f"[obs_transform] annotated={annotated_ids}, raw={raw_ids}"
                )
                _logged_first[0] = True
            return transformed

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

        run_save_dir = None
        trajectory_log = []
        if self.save_trajectory_img_dir is not None:
            run_save_dir = os.path.join(self.save_trajectory_img_dir, traj_name)
            os.makedirs(run_save_dir, exist_ok=True)

        has_steps = bool(self.steps)
        analysis_fn = self._build_analysis_fn(
            run_save_dir=run_save_dir,
            trajectory_log=trajectory_log,
        ) if has_steps else None
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

        if run_save_dir is not None and trajectory_log:
            cache = {
                "instruction": getattr(self.controller, "current_instruction", None),
                "steps": self.steps,
                "run_id": traj_name,
                "gpt_model": self._gpt_model,
                "gemini_model": self._gemini_model,
                "plan_freq": self.plan_freq,
                "max_plan_count": self.max_plan_count,
                "trajectories": trajectory_log,
            }
            cache_path = os.path.join(run_save_dir, "cache.json")
            with open(cache_path, "w") as f:
                json.dump(cache, f, indent=2)
            yellow_print(f"Saved trajectory cache to {cache_path}")

        if mode == "collect" and save_filepath is not None:
            if controller_info["success"]:
                new_save_dir = os.path.join(self.success_logdir, traj_name)
                shutil.move(save_dir, new_save_dir)
                save_dir = new_save_dir
