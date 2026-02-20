"""
Unit tests for RunnerAugmented.

All hardware-dependent parts (env, controller, robot, cameras, OpenAI/Gemini)
are mocked so the tests run without any physical setup.

Strategy: We replace eva.runner and other heavy-import modules with lightweight
fakes *before* importing RunnerAugmented, so that the deep hardware import
chain (spacemouse, polymetis, pi0, aruco, etc.) is never triggered.
"""

import json
import os
import sys
import types
import unittest
from datetime import datetime
from unittest.mock import patch, MagicMock
from copy import deepcopy

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Pre-import stubs
# ---------------------------------------------------------------------------

# 1. Fake Runner base class (avoids the entire controller import tree)
_fake_runner_mod = types.ModuleType("eva.runner")
class _FakeRunner:
    def __init__(self, *args, **kwargs):
        pass
_fake_runner_mod.Runner = _FakeRunner
sys.modules["eva.runner"] = _fake_runner_mod

# 2. Stub calibration_utils (uses cv2.aruco APIs removed in OpenCV 4.8+)
_fake_calib = types.ModuleType("eva.utils.calibration_utils")
_fake_calib.save_calibration_info = MagicMock()
sys.modules["eva.utils.calibration_utils"] = _fake_calib

# 3. Stub parameters (also touches cv2.aruco at module level)
_fake_params = types.ModuleType("eva.utils.parameters")
_fake_params.robot_type = "fr3"
_fake_params.robot_serial_number = ""
_fake_params.code_version = "2.0"
_fake_params.varied_camera_1_id = "26368109"
_fake_params.hand_camera_id = "14436910"
sys.modules["eva.utils.parameters"] = _fake_params

# 4. Stub trajectory_utils_augmented
_fake_traj_aug = types.ModuleType("eva.utils.trajectory_utils_augmented")
_fake_traj_aug.run_trajectory_augmented = MagicMock()
sys.modules["eva.utils.trajectory_utils_augmented"] = _fake_traj_aug

# 5. Stub misc_utils
_fake_misc = types.ModuleType("eva.utils.misc_utils")
_fake_misc.yellow_print = lambda *a, **kw: None
_fake_misc.data_dir = "/tmp/eva_test_data"
sys.modules["eva.utils.misc_utils"] = _fake_misc

# 6. Stub google genai unless the real package AND GEMINI_API_KEY are both
#    available.  trajectory_v1.py calls genai.Client(api_key=...) at import
#    time, which crashes if the key is None with the real package.
_google_genai_available = False
try:
    if not os.getenv("GEMINI_API_KEY"):
        raise ImportError("GEMINI_API_KEY not set")
    from google import genai as _test_genai  # noqa: F401
    from google.genai import types as _test_types  # noqa: F401
    _google_genai_available = True
except (ImportError, ModuleNotFoundError, ValueError):
    _mock_genai = MagicMock()
    sys.modules["google.genai"] = _mock_genai
    sys.modules["google.genai.types"] = _mock_genai.types
    if "google" not in sys.modules:
        sys.modules["google"] = MagicMock()

from eva.runner_augmented import RunnerAugmented  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner(**overrides):
    """Build a RunnerAugmented with Runner.__init__ fully bypassed."""
    defaults = dict(
        use_annotated_camera=False,
        openai_client=MagicMock(),
        gpt_model="gpt-test",
        gemini_model="gemini-test",
        plan_freq=10,
        max_plan_count=10,
        save_trajectory_img_path=None,
        augment_camera_ids=["varied_camera_1"],
    )
    defaults.update(overrides)
    return RunnerAugmented(
        env=MagicMock(),
        controller=MagicMock(),
        **defaults,
    )


TEST_FRAMES_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, "data", "test_frames"
)
CAMERAS = ["varied_camera_1", "varied_camera_2", "hand_camera"]


class FrameLoader:
    """Sequentially loads frames from data/test_frames, cycling through cameras.

    Each call to ``next()`` returns the next frame (as a uint8 RGB numpy
    array) and advances the internal counter.  Frames are drawn round-robin
    from the three camera directories so every camera gets exercised.  The
    frame index increments continuously from 000.jpg onward.
    """

    def __init__(self):
        self._index = 0
        frame_counts = []
        for cam in CAMERAS:
            cam_dir = os.path.join(TEST_FRAMES_DIR, cam)
            n = len([f for f in os.listdir(cam_dir) if f.endswith(".jpg")])
            frame_counts.append(n)
        self._max_index = min(frame_counts)

    def next(self, camera=None):
        """Return the next frame. ``camera`` overrides the round-robin pick."""
        cam = camera or CAMERAS[self._index % len(CAMERAS)]
        idx = self._index % self._max_index
        self._index += 1
        path = os.path.join(TEST_FRAMES_DIR, cam, f"{idx:03d}.jpg")
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


frames = FrameLoader()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAnnotateObservation(unittest.TestCase):

    def test_noop_when_no_trajectory(self):
        runner = _make_runner()
        runner.pred_traj = None
        obs = {"image": {"varied_camera_1_abc": frames.next()}}
        result = runner.annotate_observation(obs)
        np.testing.assert_array_equal(
            result["image"]["varied_camera_1_abc"],
            obs["image"]["varied_camera_1_abc"],
        )

    def test_noop_when_trajectory_key_missing(self):
        runner = _make_runner()
        runner.pred_traj = {"something_else": []}
        obs = {"image": {"varied_camera_1_abc": frames.next()}}
        result = runner.annotate_observation(obs)
        np.testing.assert_array_equal(
            result["image"]["varied_camera_1_abc"],
            obs["image"]["varied_camera_1_abc"],
        )

    def test_noop_when_fewer_than_2_points(self):
        runner = _make_runner()
        runner.pred_traj = {"trajectory": [[100, 200]]}
        obs = {"image": {"varied_camera_1_abc": frames.next()}}
        original = obs["image"]["varied_camera_1_abc"].copy()
        result = runner.annotate_observation(obs)
        np.testing.assert_array_equal(
            result["image"]["varied_camera_1_abc"], original
        )

    def test_noop_when_no_image_key(self):
        runner = _make_runner()
        runner.pred_traj = {"trajectory": [[10, 20], [30, 40]]}
        obs = {"state": np.zeros(7)}
        result = runner.annotate_observation(obs)
        self.assertNotIn("image", result)

    @patch("eva.runner_augmented.add_arrow")
    def test_annotates_matching_camera(self, mock_add_arrow):
        fake_annotated = Image.fromarray(frames.next(camera="varied_camera_1"))
        mock_add_arrow.return_value = fake_annotated

        runner = _make_runner(augment_camera_ids=["varied_camera_1"])
        runner.pred_traj = {"trajectory": [[10, 20], [30, 40], [50, 60]]}

        img = frames.next(camera="varied_camera_1")
        hand_img = frames.next(camera="hand_camera")
        obs = {
            "image": {
                "varied_camera_1_123": img,
                "hand_camera_456": hand_img,
            }
        }
        original_hand = hand_img.copy()

        result = runner.annotate_observation(obs)

        mock_add_arrow.assert_called_once_with(
            img, [(10, 20), (30, 40), (50, 60)], color="red", line_width=3
        )
        self.assertEqual(result["image"]["varied_camera_1_123"].shape[2], 3)
        np.testing.assert_array_equal(
            result["image"]["hand_camera_456"], original_hand
        )

    @patch("eva.runner_augmented.add_arrow")
    def test_skips_non_matching_cameras(self, mock_add_arrow):
        runner = _make_runner(augment_camera_ids=["varied_camera_1"])
        runner.pred_traj = {"trajectory": [[10, 20], [30, 40]]}
        obs = {"image": {"hand_camera_789": frames.next(camera="hand_camera")}}
        runner.annotate_observation(obs)
        mock_add_arrow.assert_not_called()


class TestToggleMethods(unittest.TestCase):

    def test_initial_disabled(self):
        self.assertFalse(_make_runner(use_annotated_camera=False).use_annotated_camera)

    def test_initial_enabled(self):
        self.assertTrue(_make_runner(use_annotated_camera=True).use_annotated_camera)

    def test_enable(self):
        runner = _make_runner(use_annotated_camera=False)
        runner.enable_annotated_camera()
        self.assertTrue(runner.use_annotated_camera)

    def test_disable(self):
        runner = _make_runner(use_annotated_camera=True)
        runner.disable_annotated_camera()
        self.assertFalse(runner.use_annotated_camera)

    def test_set_enabled(self):
        runner = _make_runner()
        runner.set_annotated_camera_enabled(True)
        self.assertTrue(runner.use_annotated_camera)
        runner.set_annotated_camera_enabled(False)
        self.assertFalse(runner.use_annotated_camera)


class TestImageToPil(unittest.TestCase):

    def test_from_ndarray(self):
        runner = _make_runner()
        arr = frames.next()
        h, w = arr.shape[:2]
        pil = runner._image_to_pil(arr, max_size=max(w, h))
        self.assertIsInstance(pil, Image.Image)
        self.assertEqual(pil.size, (w, h))

    def test_from_pil_image(self):
        runner = _make_runner()
        img = Image.fromarray(frames.next())
        pil = runner._image_to_pil(img)
        self.assertIsInstance(pil, Image.Image)
        self.assertEqual(pil.mode, "RGB")

    def test_resizes_large_image(self):
        runner = _make_runner()
        arr = frames.next()
        pil = runner._image_to_pil(arr, max_size=128)
        self.assertLessEqual(max(pil.size), 128)

    def test_preserves_small_image_under_limit(self):
        runner = _make_runner()
        arr = frames.next()
        h, w = arr.shape[:2]
        pil = runner._image_to_pil(arr, max_size=max(w, h) + 100)
        self.assertEqual(pil.size, (w, h))

    def test_rejects_invalid_type(self):
        with self.assertRaises(TypeError):
            _make_runner()._image_to_pil("not_an_image")


class TestExtractCameraImage(unittest.TestCase):

    @patch("eva.runner_augmented.varied_camera_1_id", "26368109")
    def test_extracts_varied_camera(self):
        img = frames.next(camera="varied_camera_1")
        obs = {"image": {"26368109": img, "other_cam": frames.next(camera="varied_camera_2")}}
        result = _make_runner()._extract_camera_image(obs)
        np.testing.assert_array_equal(result, img)

    def test_returns_none_when_no_image_key(self):
        result = _make_runner()._extract_camera_image({"state": np.zeros(7)})
        self.assertIsNone(result)

    @patch("eva.runner_augmented.varied_camera_1_id", "99999999")
    def test_falls_back_to_first_camera(self):
        img = frames.next(camera="hand_camera")
        obs = {"image": {"some_cam": img}}
        result = _make_runner()._extract_camera_image(obs)
        np.testing.assert_array_equal(result, img)


class TestPreprocessInstruction(unittest.TestCase):

    @patch("eva.runner_augmented.query_target_objects")
    def test_sets_steps_from_query(self, mock_query):
        mock_query.return_value = {
            "steps": [
                {
                    "step": "pick up the pineapple",
                    "manipulating_object": "pineapple",
                    "target_related_object": "bowl",
                    "target_location": "inside the bowl",
                }
            ]
        }
        runner = _make_runner()
        runner.preprocess_instruction("put the pineapple in the bowl")

        mock_query.assert_called_once_with(
            runner._openai_client, "put the pineapple in the bowl", model="gpt-test"
        )
        self.assertEqual(len(runner.steps), 1)
        self.assertEqual(runner.step, 0)
        self.assertEqual(runner.steps[0]["manipulating_object"], "pineapple")

    @patch("eva.runner_augmented.query_target_objects")
    def test_handles_empty_response(self, mock_query):
        mock_query.return_value = {}
        runner = _make_runner()
        runner.preprocess_instruction("do something")
        self.assertEqual(runner.steps, [])
        self.assertEqual(runner.step, 0)


class TestGetPredTraj(unittest.TestCase):

    def test_returns_none_when_no_steps(self):
        runner = _make_runner()
        runner.steps = []
        runner.step = 0
        self.assertIsNone(runner.get_pred_traj(frames.next()))

    def test_returns_none_when_step_exceeds_length(self):
        runner = _make_runner()
        runner.steps = [
            {"step": "x", "manipulating_object": "a",
             "target_related_object": "b", "target_location": "c"}
        ]
        runner.step = 5
        self.assertIsNone(runner.get_pred_traj(frames.next()))

    @patch("eva.runner_augmented.query_trajectory")
    @patch("eva.runner_augmented.encode_pil_image", return_value="encoded")
    @patch("eva.runner_augmented.query_target_location")
    def test_returns_none_when_detection_fails(self, mock_loc, _enc, mock_traj):
        mock_loc.return_value = None
        runner = _make_runner()
        runner.steps = [
            {"step": "pick", "manipulating_object": "a",
             "target_related_object": "b", "target_location": "c"}
        ]
        runner.step = 0
        self.assertIsNone(runner.get_pred_traj(frames.next()))
        mock_traj.assert_not_called()

    @patch("eva.runner_augmented.query_trajectory")
    @patch("eva.runner_augmented.encode_pil_image", return_value="encoded")
    @patch("eva.runner_augmented.query_target_location")
    def test_returns_none_when_object_not_detected(self, mock_loc, _enc, _traj):
        mock_loc.return_value = {"pineapple": (100, 200)}  # missing "bowl"
        runner = _make_runner()
        runner.steps = [
            {"step": "pick", "manipulating_object": "pineapple",
             "target_related_object": "bowl", "target_location": "c"}
        ]
        runner.step = 0
        self.assertIsNone(runner.get_pred_traj(frames.next()))

    @patch("eva.runner_augmented.query_trajectory")
    @patch("eva.runner_augmented.encode_pil_image", return_value="encoded")
    @patch("eva.runner_augmented.query_target_location")
    def test_successful_trajectory(self, mock_loc, _enc, mock_traj):
        mock_loc.return_value = {"pineapple": (100, 200), "bowl": (300, 400)}
        mock_traj.return_value = {
            "trajectory": [[100, 200], [200, 300], [300, 400]]
        }
        runner = _make_runner()
        runner.steps = [
            {
                "step": "put pineapple into the bowl",
                "manipulating_object": "pineapple",
                "target_related_object": "bowl",
                "target_location": "inside bowl",
            }
        ]
        runner.step = 0
        runner.get_pred_traj(frames.next())

        self.assertIsNotNone(runner.pred_traj)
        self.assertEqual(len(runner.pred_traj["trajectory"]), 3)


class TestBuildCallbacks(unittest.TestCase):

    def test_analysis_fn_returns_none_when_no_image(self):
        runner = _make_runner()
        fn = runner._build_analysis_fn()
        self.assertIsNone(fn({"state": np.zeros(7)}, step_num=0, plan_count=0))

    @patch("eva.runner_augmented.varied_camera_1_id", "cam1")
    def test_analysis_fn_calls_get_pred_traj(self):
        runner = _make_runner()
        runner.get_pred_traj = MagicMock()
        runner.pred_traj = {"trajectory": [[1, 2], [3, 4]]}

        img = frames.next(camera="varied_camera_1")
        fn = runner._build_analysis_fn()
        result = fn({"image": {"cam1": img}}, step_num=5, plan_count=1)

        runner.get_pred_traj.assert_called_once()
        self.assertEqual(result, runner.pred_traj)

    def test_on_analysis_complete_does_not_raise(self):
        cb = _make_runner()._build_on_analysis_complete()
        cb({"trajectory": [[1, 2]]}, {}, 0)
        cb(None, {}, 0)


# ---------------------------------------------------------------------------
# Integration tests — real API calls for trajectory queries
# ---------------------------------------------------------------------------

_HAS_LIVE_APIS = (
    _google_genai_available
    and bool(os.getenv("OPENAI_API_KEY"))
    and bool(os.getenv("GEMINI_API_KEY"))
)


@unittest.skipUnless(
    _HAS_LIVE_APIS,
    "Requires google-genai package, OPENAI_API_KEY, and GEMINI_API_KEY",
)
class TestTrajectoryQueryIntegration(unittest.TestCase):
    """Integration test that makes real API calls for trajectory prediction.

    Exercises the analysis-hook code path of ``run_trajectory_augmented``
    (lines 92-107 in trajectory_utils_augmented.py) by feeding sequential
    frames from ``data/test_frames`` through the live query pipeline.

    Skipped automatically when API keys or ``google-genai`` are missing.
    """

    def _make_obs(self):
        """Build an observation dict with real camera images."""
        return {
            "image": {
                "26368109": frames.next(camera="varied_camera_1"),
                "14436910": frames.next(camera="hand_camera"),
                "25455306": frames.next(camera="varied_camera_2"),
            },
        }

    SAVE_DIR = os.path.join(
        os.path.dirname(__file__), os.pardir, "data", "test_traj"
    )

    def test_full_pipeline_with_analysis_loop(self):
        """Preprocess instruction then run the analysis scheduling loop.

        Mirrors ``run_trajectory_augmented`` lines 92-107: feeds sequential
        frames as observations and triggers ``analysis_fn`` at step 0 and
        every ``plan_freq`` steps, exactly as the real loop does.

        Saves trajectory images and a JSON cache under data/test_traj/.
        """
        from openai import OpenAI
        from eva.detectors.traj_vis_utils import add_arrow

        # -- Set up output directory --
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join(self.SAVE_DIR, run_id)
        os.makedirs(out_dir, exist_ok=True)

        runner = _make_runner(
            openai_client=OpenAI(),
            gpt_model="gpt-4o-mini",
            gemini_model="gemini-robotics-er-1.5-preview",
            plan_freq=15,
            max_plan_count=5,
        )

        # -- Step 1: real instruction preprocessing (OpenAI) --
        instruction = "place the pineapple in the bowl"
        runner.preprocess_instruction(instruction)
        self.assertGreater(len(runner.steps), 0, "Should extract at least one step")
        step_data = runner.steps[0]
        self.assertIn("manipulating_object", step_data)
        self.assertIn("target_related_object", step_data)
        self.assertIn("target_location", step_data)

        # Save preprocessing cache
        cache = {
            "instruction": instruction,
            "steps": runner.steps,
            "run_id": run_id,
            "gpt_model": "gpt-4o-mini",
            "gemini_model": "gemini-robotics-er-1.5-preview",
            "plan_freq": runner.plan_freq,
            "max_plan_count": runner.max_plan_count,
            "trajectories": [],
        }

        # -- Step 2: build real analysis callbacks --
        analysis_fn = runner._build_analysis_fn()
        on_analysis_complete = runner._build_on_analysis_complete()

        # -- Step 3: simulate the run_trajectory_augmented scheduling loop --
        plan_freq = runner.plan_freq
        max_plan_count = runner.max_plan_count
        plan_count = 0
        trajectories = []

        num_simulation_steps = plan_freq + 1  # triggers analysis at step 0 and step plan_freq
        for num_steps in range(num_simulation_steps):
            obs = self._make_obs()

            should_analyse = (
                analysis_fn is not None
                and plan_count < max_plan_count
                and (num_steps == 0 or (plan_freq > 0 and num_steps % plan_freq == 0))
            )
            if should_analyse:
                # Save the raw input frame before analysis
                cam_img = obs["image"]["26368109"]
                frame_path = os.path.join(out_dir, f"frame_step_{num_steps:03d}.jpg")
                Image.fromarray(cam_img).save(frame_path)

                # Point runner's built-in save path at this step
                traj_img_path = os.path.join(out_dir, f"traj_step_{num_steps:03d}.jpg")
                runner.save_trajectory_img_path = traj_img_path

                result = analysis_fn(obs, num_steps, plan_count)
                plan_count += 1
                if on_analysis_complete is not None:
                    on_analysis_complete(result, obs, num_steps)

                if result is not None and "trajectory" in result:
                    trajectories.append((num_steps, result))

                    # If runner didn't save (e.g. image conversion), draw and save manually
                    if not os.path.exists(traj_img_path):
                        pts = [tuple(p) for p in result["trajectory"]]
                        if len(pts) >= 2:
                            pil_img = Image.fromarray(cam_img).convert("RGB")
                            annotated = add_arrow(pil_img, pts, color="red", line_width=3)
                            annotated.convert("RGB").save(traj_img_path)

                    # Append to cache
                    cache["trajectories"].append({
                        "step_num": num_steps,
                        "plan_count": plan_count,
                        "frame_file": f"frame_step_{num_steps:03d}.jpg",
                        "traj_image_file": f"traj_step_{num_steps:03d}.jpg",
                        "trajectory": result,
                    })

        # -- Step 4: save full cache --
        cache_path = os.path.join(out_dir, "cache.json")
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)

        # -- Step 5: verify we got real trajectories --
        self.assertGreater(
            len(trajectories), 0,
            "Should produce at least one trajectory from real API calls",
        )
        for step_num, traj in trajectories:
            pts = traj["trajectory"]
            self.assertGreaterEqual(
                len(pts), 2,
                f"Trajectory at step {step_num} should have >= 2 points",
            )
            for pt in pts:
                self.assertEqual(len(pt), 2, "Each point should be [x, y]")

        # Verify files were saved
        self.assertTrue(os.path.exists(cache_path), "cache.json should exist")
        saved_files = os.listdir(out_dir)
        self.assertTrue(
            any(f.startswith("traj_step_") for f in saved_files),
            "At least one trajectory image should be saved",
        )
        print(f"\n  Saved {len(saved_files)} files to {out_dir}")


if __name__ == "__main__":
    unittest.main()
