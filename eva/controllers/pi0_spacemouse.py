import numpy as np
import time
import threading
from dataclasses import dataclass

from eva.controllers.pi0_policy import Pi0Policy, Pi0PolicyConfig
from eva.controllers.spacemouse import SpaceMouse, SpaceMouseConfig
from eva.utils.misc_utils import create_info_dict
import eva.utils.parameters as params


@dataclass
class SpaceMousePi0Config:
    # PI0 config
    init_instruction: str = "pick up the object"
    remote_host: str = "10.102.212.31"
    remote_port: int = 8000
    action_space: str = "joint_velocity"
    gripper_action_space: str = "position"
    left_camera_id: str = params.varied_camera_1_id
    right_camera_id: str = params.varied_camera_2_id
    wrist_camera_id: str = params.hand_camera_id
    external_camera: str = "left"
    open_loop_horizon: int = 8

    # SpaceMouse config — when SPACEMOUSE_OVERRIDE_CONFIG=True in parameters.py,
    # parameters.spacemouse_config is used as the base and these values override on top
    max_lin_vel: float = 5.0
    max_rot_vel: float = 5.0
    max_gripper_vel: float = 5.0
    pos_sensitivity: float = 8.0
    rot_sensitivity: float = 8.0
    action_scale: float = 0.1
    deadzone: float = 0.05
    smoothing: float = 0.3


class SpaceMousePi0:
    """SpaceMousePi0 Controller with runtime switching."""

    def __init__(self, config: SpaceMousePi0Config = SpaceMousePi0Config(), on_switch_callback=None):
        print("Initializing SpaceMousePi0 Controller")
        self._on_switch_callback = on_switch_callback
        self.init_instruction = config.init_instruction
        self._cartesian_velocity_actions = []

        # Initialize PI0 controller
        pi0_config = Pi0PolicyConfig(
            remote_host=config.remote_host,
            remote_port=config.remote_port,
            action_space=config.action_space,
            gripper_action_space=config.gripper_action_space,
            left_camera_id=config.left_camera_id,
            right_camera_id=config.right_camera_id,
            wrist_camera_id=config.wrist_camera_id,
            external_camera=config.external_camera,
            open_loop_horizon=config.open_loop_horizon,
        )
        self.pi0_controller = Pi0Policy(pi0_config)

        # Initialize SpaceMouse controller
        sm_fields = dict(
            max_lin_vel=config.max_lin_vel,
            max_rot_vel=config.max_rot_vel,
            max_gripper_vel=config.max_gripper_vel,
            pos_sensitivity=config.pos_sensitivity,
            rot_sensitivity=config.rot_sensitivity,
            action_scale=config.action_scale,
            deadzone=config.deadzone,
            smoothing=config.smoothing,
        )
        if params.SPACEMOUSE_OVERRIDE_CONFIG:
            sm_config = SpaceMouseConfig.from_params(**sm_fields)
        else:
            sm_config = SpaceMouseConfig(**sm_fields)
        self.spacemouse_controller = SpaceMouse(config=sm_config)

        self._switch_lock = threading.Lock()
        self._switching_in_progress = False

        self.reset_state()
        print("Mixed Controller Controls:")
        print("  '=' key: Switch between PI0 and SpaceMouse")
        print(f"  Current controller: {self.get_current_controller().get_name()}")

    @property
    def action_space(self):
        return self.get_current_controller().action_space

    @property
    def gripper_action_space(self):
        return self.get_current_controller().gripper_action_space

    def get_name(self):
        return "SpaceMousePi0"

    def reset_state(self):
        self.pi0_controller.reset_state()
        self.spacemouse_controller.reset_state()
        self.pi0_controller.stop_policy()
        self._cartesian_velocity_actions = []

        self._state = {
            "success": False,
            "failure": False,
            "movement_enabled": True,
            "controller_on": True,
            "switch_label": 1,        # 0: PI0, 1: SpaceMouse
            "current_controller": 1,
        }

        if self._on_switch_callback:
            ctrl = self.spacemouse_controller
            self._on_switch_callback(ctrl.action_space, ctrl.gripper_action_space)

        self.set_instruction(self.init_instruction)

    def set_instruction(self, instruction):
        self.pi0_controller.set_instruction(instruction)

    def register_key(self, key):
        if key == ord("="):
            self._handle_switch()

        # Forward to active controller
        if self._state["current_controller"] == 0:
            self.pi0_controller.register_key(key)
        else:
            self.spacemouse_controller.register_key(key)

        # Sync shared state
        info = self.get_current_controller().get_info()
        self._state["success"] = info["success"]
        self._state["failure"] = info["failure"]
        self._state["movement_enabled"] = info["movement_enabled"]
        self._state["controller_on"] = info["controller_on"]

    def _handle_switch(self):
        with self._switch_lock:
            if self._switching_in_progress:
                return
            self._switching_in_progress = True

            if self._state["current_controller"] == 0:
                # PI0 → SpaceMouse
                self.pi0_controller.stop_policy()
                self._state["switch_label"] = 1.0
                self._state["current_controller"] = 1
                print("Switched to SpaceMouse controller")
            else:
                # SpaceMouse → PI0
                self._state["switch_label"] = 0.0
                time.sleep(0.1)
                self.pi0_controller.start_policy()
                self._state["current_controller"] = 0
                print("Switched to PI0 controller")

            if self._on_switch_callback:
                ctrl = self.get_current_controller()
                self._on_switch_callback(ctrl.action_space, ctrl.gripper_action_space)

            self._switching_in_progress = False

    def get_info(self):
        return self._state

    def close(self):
        self.pi0_controller.close()
        self.spacemouse_controller.close()

    def get_current_controller(self):
        if self._state["current_controller"] == 0:
            return self.pi0_controller
        return self.spacemouse_controller

    def forward(self, observation):
        with self._switch_lock:
            if self._state["current_controller"] == 0:
                action, info = self.pi0_controller.forward(observation)
                info["joint_velocity"] = action[:7]
                info["gripper_velocity"] = action[-1]
            else:
                action, info = self.spacemouse_controller.forward(observation)
                info["cartesian_velocity"] = action[:6]
                info["gripper_velocity"] = action[-1]
                if self._state["movement_enabled"]:
                    self._cartesian_velocity_actions.append(action)

            action = np.array(action, dtype=np.float32)
            action = np.clip(action, -1, 1)
            return action, info

