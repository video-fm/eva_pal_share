#!/usr/bin/env python3
"""Standalone SpaceMouse teleop demo for tuning control parameters.

All SpaceMouse hyperparams are exposed as CLI args. Use this script to
experiment with sensitivity / deadzone / smoothing before committing
values to parameters.py.

When SPACEMOUSE_OVERRIDE_CONFIG=True in parameters.py, spacemouse_config
dict is used as the base. CLI args always override on top.

Usage:
    # Use parameters.py defaults (if SPACEMOUSE_OVERRIDE_CONFIG=True)
    python scripts/spacemouse_teleop.py

    # Override specific values for tuning
    python scripts/spacemouse_teleop.py --pos_sensitivity 12 --action_scale 0.15
"""

import argparse
import cv2
import numpy as np
from dataclasses import dataclass

from eva.controllers.spacemouse import (
    SpaceMouseInterface,
    SpaceMouseConfig,
    DEFAULT_KEY_MACROS,
    _MACRO_DEFS,
)
from eva.env import FrankaEnv
import eva.utils.parameters as params


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class TeleopConfig:
    # Camera
    left_camera_id: str = params.varied_camera_1_id
    right_camera_id: str = params.varied_camera_2_id
    wrist_camera_id: str = params.hand_camera_id
    external_camera: str = "right"

    # SpaceMouse tuning (all knobs exposed here)
    max_lin_vel: float = 5.0
    max_rot_vel: float = 5.0
    max_gripper_vel: float = 5.0
    pos_sensitivity: float = 8.0
    rot_sensitivity: float = 8.0
    action_scale: float = 0.1
    deadzone: float = 0.05
    smoothing: float = 0.3


# ── Image helpers ────────────────────────────────────────────────────────────

def extract_images(config: TeleopConfig, obs_dict):
    """Pull left / right / wrist images from observation dict."""
    image_obs = obs_dict["image"]
    images = {"left_image": None, "right_image": None, "wrist_image": None}

    for key, frame in image_obs.items():
        if "left" not in key:
            continue
        if config.left_camera_id in key:
            images["left_image"] = frame[..., :3][..., ::-1]
        elif config.right_camera_id in key:
            images["right_image"] = frame[..., :3][..., ::-1]
        elif config.wrist_camera_id in key:
            images["wrist_image"] = frame[..., :3][..., ::-1]

    return images


def render_gui(images, config: TeleopConfig, instruction: str):
    """Show wrist + external camera side-by-side with instruction overlay."""
    ext_key = f"{config.external_camera}_image"
    wrist = images.get("wrist_image")
    external = images.get(ext_key)

    if wrist is None or external is None:
        return

    h = min(wrist.shape[0], external.shape[0])
    wrist_resized = cv2.resize(wrist, (int(wrist.shape[1] * h / wrist.shape[0]), h))
    ext_resized = cv2.resize(external, (int(external.shape[1] * h / external.shape[0]), h))

    combined = np.concatenate([wrist_resized, ext_resized], axis=1)
    combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

    # Draw instruction overlay
    if instruction:
        overlay = combined.copy()
        text_h = 36
        cv2.rectangle(overlay, (0, 0), (combined.shape[1], text_h + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, combined, 0.4, 0, combined)
        cv2.putText(
            combined, f"Instruction: {instruction}",
            (10, text_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

    cv2.imshow("SpaceMouse Teleop", combined)


# ── Macro helper ─────────────────────────────────────────────────────────────

def make_macro_action(name, sign):
    """Build a 7-DoF action vector from a named macro."""
    prefix = name.split("_")[0]
    if prefix not in _MACRO_DEFS:
        raise ValueError(f"Unknown macro: {name}")
    sl, axis, mag = _MACRO_DEFS[prefix]
    vec = np.zeros(7, dtype=np.float32)
    vec[sl] = axis * mag * sign * 10
    return vec


# ── Main teleop loop ────────────────────────────────────────────────────────

class InteractiveBot:
    def __init__(self, config: TeleopConfig):
        self.config = config
        camera_kwargs = {
            "hand_camera":     {"depth": False, "pointcloud": False},
            "varied_camera_1": {"depth": False, "pointcloud": False},
            "varied_camera_2": {"depth": False, "pointcloud": False},
        }
        self.env = FrankaEnv(camera_kwargs=camera_kwargs)
        self.instruction = ""
        self._macro_queue = []

    def reset(self):
        self.env.reset()

    def _prompt_instruction(self):
        cv2.destroyWindow("SpaceMouse Teleop")
        print("\n--- Enter new instruction (type in terminal, press Enter) ---")
        new_text = input("Instruction: ").strip()
        if new_text:
            self.instruction = new_text
            print(f"Instruction set to: {self.instruction}")
        else:
            print("Instruction unchanged.")

    def run_teleop(self):
        interface = SpaceMouseInterface(
            pos_sensitivity=self.config.pos_sensitivity,
            rot_sensitivity=self.config.rot_sensitivity,
            action_scale=self.config.action_scale,
        )
        interface.start_control()

        cfg = self.config
        print("\n── SpaceMouse Teleop ──")
        print(f"  pos_sensitivity={cfg.pos_sensitivity}  rot_sensitivity={cfg.rot_sensitivity}")
        print(f"  action_scale={cfg.action_scale}  deadzone={cfg.deadzone}  smoothing={cfg.smoothing}")
        print(f"  max_lin_vel={cfg.max_lin_vel}  max_rot_vel={cfg.max_rot_vel}  max_gripper_vel={cfg.max_gripper_vel}")
        print()
        print("  Mouse:  move = translate, twist = rotate")
        print("  Left button:  toggle gripper")
        print("  Right button: reset robot")
        print("  Keyboard:")
        print("    q / ESC : quit")
        print("    g       : toggle gripper")
        print("    r       : reset robot")
        print("    i       : set instruction (displayed on GUI)")
        print("    space   : reset SpaceMouse origin")
        print("  Macros:")
        print("    w/s     : forward / backward")
        print("    a/d     : left / right")
        print("    1/2     : tilt up / down")
        print("    5/6     : roll CCW / CW")
        print("    7/8     : rotate left / right")
        print()

        gripper_open = 1

        while True:
            # ── Macro step ──
            if self._macro_queue:
                action = self._macro_queue.pop(0)
                action[6] = 1 - int(interface.control_gripper)
                action = np.clip(action, -1.0, 1.0)
                self.env.step(action)
            else:
                # ── SpaceMouse input ──
                data = interface.get_controller_state()

                if data["lock"]:
                    self.reset()
                    continue

                dpos = data["dpos"]
                drot = data["raw_drotation"]
                gripper_open = int(1 - float(data["grasp"]))

                if np.linalg.norm(dpos) > 0.001 or np.linalg.norm(drot) > 0.001 or data["hold"]:
                    action = np.concatenate([dpos, drot, [gripper_open]])
                    action = np.clip(action, -1.0, 1.0)
                    self.env.step(action)

            # ── GUI ──
            obs = self.env.get_observation()
            images = extract_images(self.config, obs)
            render_gui(images, self.config, self.instruction)

            key = cv2.waitKey(20)
            if key == -1 or key == 255:
                continue
            elif key == ord("q") or key == 27:
                print("Exiting teleop.")
                break
            elif key == ord("g"):
                gripper_open = 1 - gripper_open
                print(f"Gripper {'open' if gripper_open else 'closed'}")
            elif key == ord("r"):
                self.reset()
                print("Robot reset")
            elif key == ord("i"):
                self._prompt_instruction()
            elif key == ord(" "):
                interface._reset_internal_state()
                print("SpaceMouse origin reset")
            elif key in DEFAULT_KEY_MACROS:
                name, sign = DEFAULT_KEY_MACROS[key]
                vec = make_macro_action(name, sign)
                self._macro_queue.append(vec)
                print(f"[Macro] {name}")

        cv2.destroyAllWindows()


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SpaceMouse teleop — tune hyperparams here, then commit to parameters.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--external_camera", type=str, default="right", choices=["left", "right"])

    # All SpaceMouse tuning knobs (CLI args override parameters.py / local defaults)
    parser.add_argument("--max_lin_vel",      type=float, default=None)
    parser.add_argument("--max_rot_vel",      type=float, default=None)
    parser.add_argument("--max_gripper_vel",  type=float, default=None)
    parser.add_argument("--pos_sensitivity",  type=float, default=None)
    parser.add_argument("--rot_sensitivity",  type=float, default=None)
    parser.add_argument("--action_scale",     type=float, default=None)
    parser.add_argument("--deadzone",         type=float, default=None)
    parser.add_argument("--smoothing",        type=float, default=None)

    args = parser.parse_args()

    # Build SpaceMouseConfig: respect SPACEMOUSE_OVERRIDE_CONFIG flag, CLI args always win
    cli_overrides = {
        k: v for k, v in vars(args).items()
        if k in SpaceMouseConfig.__dataclass_fields__ and v is not None
    }

    if params.SPACEMOUSE_OVERRIDE_CONFIG:
        sm_config = SpaceMouseConfig.from_params(**cli_overrides)
        print(f"Loaded SpaceMouse config from parameters.py (+ {len(cli_overrides)} CLI overrides)")
    else:
        sm_config = SpaceMouseConfig(**cli_overrides) if cli_overrides else SpaceMouseConfig()
        print("Using local SpaceMouseConfig defaults" + (f" (+ {len(cli_overrides)} CLI overrides)" if cli_overrides else ""))

    config = TeleopConfig(
        external_camera=args.external_camera,
        max_lin_vel=sm_config.max_lin_vel,
        max_rot_vel=sm_config.max_rot_vel,
        max_gripper_vel=sm_config.max_gripper_vel,
        pos_sensitivity=sm_config.pos_sensitivity,
        rot_sensitivity=sm_config.rot_sensitivity,
        action_scale=sm_config.action_scale,
        deadzone=sm_config.deadzone,
        smoothing=sm_config.smoothing,
    )

    bot = InteractiveBot(config)
    bot.reset()
    bot.run_teleop()


if __name__ == "__main__":
    main()
