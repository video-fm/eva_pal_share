import threading
import time
from collections import namedtuple, deque
from dataclasses import dataclass

import numpy as np
from eva.utils.geometry_utils import rotation_matrix
from eva.utils.misc_utils import create_info_dict, yellow_print

try:
    import os
    os.environ["LD_LIBRARY_PATH"] = os.getcwd()
    import hid
except ModuleNotFoundError as exc:
    raise ImportError(
        "Unable to load module hid, please check if SpaceMouse is connected. \n"
        "Type 'lsusb' and compare pid with 0x256f:0xc635. \n"
    ) from exc


# ── HID helpers ──────────────────────────────────────────────────────────────

AxisSpec = namedtuple("AxisSpec", ["channel", "byte1", "byte2", "scale"])

SPACE_MOUSE_SPEC = {
    "x":     AxisSpec(channel=1, byte1=1,  byte2=2,  scale=1),
    "y":     AxisSpec(channel=1, byte1=3,  byte2=4,  scale=-1),
    "z":     AxisSpec(channel=1, byte1=5,  byte2=6,  scale=-1),
    "roll":  AxisSpec(channel=1, byte1=7,  byte2=8,  scale=-1),
    "pitch": AxisSpec(channel=1, byte1=9,  byte2=10, scale=-1),
    "yaw":   AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
}


def _to_int16(y1, y2):
    """Convert two 8-bit bytes to a signed 16-bit integer."""
    x = y1 | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


def _scale_to_control(x, axis_scale=350.0, min_v=-1.0, max_v=1.0):
    """Normalize raw HID reading to [min_v, max_v]."""
    x = x / axis_scale
    return min(max(x, min_v), max_v)


def _convert(b1, b2):
    """Convert two raw HID bytes to a scaled control value."""
    return _scale_to_control(_to_int16(b1, b2))


# ── SpaceMouseInterface (low-level HID driver) ──────────────────────────────

class SpaceMouseInterface:
    """Minimalistic HID driver for 3Dconnexion SpaceMouse Compact.

    Use ``hid.enumerate()`` to list connected HID devices and verify the
    vendor / product IDs before running.

    See https://zhuyifengzju.github.io/deoxys_docs/html/tutorials/using_teleoperation_devices.html
    """

    def __init__(
        self,
        vendor_id=0x256f,
        product_id=0xc635,
        pos_sensitivity=10,
        rot_sensitivity=10,
        action_scale=0.08,
        deadzone=0.05,
        smoothing=0.5,
        sample_hz=15,
    ):
        print("Opening SpaceMouse device")
        self.device = hid.device()
        self.device.open(vendor_id, product_id)

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.action_scale = action_scale
        self.deadzone = deadzone
        self.smoothing = smoothing

        self._prev_dpos = np.zeros(3)
        self._prev_drot = np.zeros(3)

        self.gripper_is_closed = False
        print("Manufacturer: %s" % self.device.get_manufacturer_string())
        print("Product: %s" % self.device.get_product_string())

        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self.single_click_and_hold = False
        self.elapsed_time = 0

        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.lock_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

        self._sample_dt = 1.0 / sample_hz
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def _reset_internal_state(self):
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0
        self._control = np.zeros(6)
        self.single_click_and_hold = False
        self.t_last_click = time.time()

    def start_control(self):
        self._reset_internal_state()
        self.lock_state = 0

    def get_controller_state(self):
        """Return current 6-DoF state of the SpaceMouse."""
        raw_dpos = np.array(self.control[:3]) * self.action_scale
        raw_drot = np.array(self.control[3:]) * self.action_scale

        dpos = np.array([raw_dpos[1], raw_dpos[0], -raw_dpos[2]])
        drot = np.array([-raw_drot[1], raw_drot[0], raw_drot[2]])

        self._prev_dpos = dpos
        self._prev_drot = drot

        roll, pitch, yaw = raw_drot
        drot1 = rotation_matrix(angle=-pitch, direction=[1.0, 0, 0])[:3, :3]
        drot2 = rotation_matrix(angle=roll,   direction=[0, 1.0, 0])[:3, :3]
        drot3 = rotation_matrix(angle=yaw,    direction=[0, 0, 1.0])[:3, :3]
        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=drot,
            grasp=self.control_gripper,
            hold=self.single_click_and_hold,
            lock=self.lock_state,
        )

    def _apply_deadzone(self, values, threshold):
        """Apply deadzone filter to control values."""
        magnitude = np.linalg.norm(values)
        if magnitude < threshold:
            return np.zeros_like(values)
        scale = (magnitude - threshold) / (magnitude * (1 - threshold))
        return values * scale

    def _smooth_control(self, current, previous, smoothing_factor):
        """Apply exponential smoothing to control values."""
        return smoothing_factor * previous + (1 - smoothing_factor) * current

    def _apply_response_curve(self, values, curve_factor=2.0):
        """Apply exponential response curve for better fine control."""
        signs = np.sign(values)
        magnitudes = np.abs(values)
        return signs * (magnitudes ** curve_factor)

    def run(self):
        """Listener thread that continuously reads HID packets."""
        while True:
            try:
                d = self.device.read(13)
                if d is None or len(d) < 2:
                    continue

                if d[0] == 1:  # Translation
                    x = _convert(d[1], d[2]) if len(d) > 2 else 0.0
                    y = _convert(d[3], d[4]) if len(d) > 4 else 0.0
                    z = _convert(d[5], d[6]) if len(d) > 6 else 0.0
                    self._control[0] = x * self.pos_sensitivity
                    self._control[1] = y * self.pos_sensitivity
                    self._control[2] = z * self.pos_sensitivity

                elif d[0] == 2:  # Rotation
                    roll  = _convert(d[1], d[2]) if len(d) > 2 else 0.0
                    pitch = _convert(d[3], d[4]) if len(d) > 4 else 0.0
                    yaw   = _convert(d[5], d[6]) if len(d) > 6 else 0.0
                    self._control[3] = roll  * self.rot_sensitivity
                    self._control[4] = pitch * self.rot_sensitivity
                    self._control[5] = yaw   * self.rot_sensitivity

                elif d[0] == 3:  # Buttons
                    if len(d) >= 2:
                        button_state = d[1]
                        # Left button — toggle gripper
                        if button_state & 1:
                            t_click = time.time()
                            if not hasattr(self, "t_last_click"):
                                self.t_last_click = t_click
                            self.elapsed_time = t_click - self.t_last_click
                            self.t_last_click = t_click
                            if self.elapsed_time > 0.5:
                                self.gripper_is_closed = not self.gripper_is_closed
                                print(f"Gripper: {'closed' if self.gripper_is_closed else 'open'}")
                            self.single_click_and_hold = True
                        else:
                            self.single_click_and_hold = False
                        # Right button — reset
                        if button_state & 2:
                            self.lock_state = 1
                        else:
                            self.lock_state = 0

            except Exception as e:
                print(f"SpaceMouse HID error: {e}")
                self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                continue

    @property
    def control(self):
        return np.array(self._control)

    @property
    def control_gripper(self):
        return self.gripper_is_closed

    def get_action(self):
        if sum(abs(self.control)) > 0.0 or self.control_gripper is not None:
            return (self.action_scale * self.control, self.control_gripper, self.lock_state)
        return None, self.control_gripper, self.lock_state


# ── SpaceMouse (high-level controller) ──────────────────────────────────────

@dataclass
class SpaceMouseConfig:
    max_lin_vel: float = 3.0
    max_rot_vel: float = 3.0
    max_gripper_vel: float = 3.0
    pos_sensitivity: float = 8.0
    rot_sensitivity: float = 8.0
    action_scale: float = 0.15
    deadzone: float = 0.05
    smoothing: float = 0.3

    @classmethod
    def from_params(cls, **overrides):
        """Load from parameters.spacemouse_config dict, then apply overrides."""
        import eva.utils.parameters as params
        base = cls(**params.spacemouse_config)
        for k, v in overrides.items():
            setattr(base, k, v)
        return base


# Keyboard macro bindings: key → (macro_name, sign)
DEFAULT_KEY_MACROS = {
    ord("1"): ("tilt_up",       +1),
    ord("2"): ("tilt_down",     -1),
    ord("w"): ("move_forward",  +1),
    ord("s"): ("move_backward", -1),
    ord("a"): ("side_left",     +1),
    ord("d"): ("side_right",    -1),
    ord("5"): ("roll_ccw",      +1),
    ord("6"): ("roll_cw",       -1),
    ord("7"): ("rotate_left",   +1),
    ord("8"): ("rotate_right",  -1),
}

# Macro definitions: name_prefix → (axis_index_range, magnitude_per_unit)
_MACRO_DEFS = {
    "tilt":     (slice(3, 6), np.array([0, 1, 0]), 0.175),
    "move":     (slice(0, 3), np.array([1, 0, 0]), 0.053),
    "side":     (slice(0, 3), np.array([0, 1, 0]), 0.053),
    "roll":     (slice(3, 6), np.array([0, 0, 1]), 0.233),
    "rotate":   (slice(3, 6), np.array([1, 0, 0]), 0.175),
}


class SpaceMouse:
    def __init__(self, config: SpaceMouseConfig = SpaceMouseConfig(), **kwargs):
        # Accept kwargs for backward compat (MixedController passes individual params)
        if kwargs:
            for field in SpaceMouseConfig.__dataclass_fields__:
                if field in kwargs:
                    setattr(config, field, kwargs[field])

        self.action_space = "cartesian_velocity"
        self.gripper_action_space = "position"
        self.max_lin_vel = config.max_lin_vel
        self.max_rot_vel = config.max_rot_vel
        self.max_gripper_vel = config.max_gripper_vel

        self.interface = SpaceMouseInterface(
            pos_sensitivity=config.pos_sensitivity,
            rot_sensitivity=config.rot_sensitivity,
            action_scale=config.action_scale,
            deadzone=config.deadzone,
            smoothing=config.smoothing,
        )

        self._state = {
            "success": False,
            "failure": False,
            "movement_enabled": True,
            "controller_on": True,
        }

        # Macro config
        self.queue_size = 3
        self._macro_queue = deque(maxlen=self.queue_size)
        self._macro_step = None
        self._key2macro = dict(DEFAULT_KEY_MACROS)

        self._display_controls()

    def get_name(self):
        return "spacemouse"

    def get_policy_name(self):
        return "spacemouse-data"

    @staticmethod
    def _display_controls():
        print("\nSpaceMouse controls:")
        print("- Move to control position")
        print("- Twist to control orientation")
        print("- Left button: toggle gripper")
        print("- Right button: None")
        print("- 'y' key: success")
        print("- 'n' key: failure")
        print("- 'space' key: reset origin")

    def reset_state(self):
        self.interface._reset_internal_state()
        self._state = {
            "success": False,
            "failure": False,
            "movement_enabled": True,
            "controller_on": True,
        }
        self.first_action = True

    def set_state(self, key):
        self._state["success"] = key == "y"
        self._state["failure"] = key == "n"

    def _limit_velocity(self, lin_vel, rot_vel, gripper_vel):
        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)
        gripper_vel_norm = np.abs(gripper_vel)
        if lin_vel_norm > self.max_lin_vel:
            lin_vel = lin_vel * self.max_lin_vel / lin_vel_norm
        if rot_vel_norm > self.max_rot_vel:
            rot_vel = rot_vel * self.max_rot_vel / rot_vel_norm
        if gripper_vel_norm > self.max_gripper_vel:
            gripper_vel = gripper_vel * self.max_gripper_vel / gripper_vel_norm
        return lin_vel, rot_vel, gripper_vel

    def _apply_control_curve(self, values, exponent=2.0):
        signs = np.sign(values)
        magnitudes = np.abs(values)
        return signs * (magnitudes ** exponent)

    def get_info(self):
        return self._state.copy()

    def register_key(self, key):
        if key == ord(" "):
            print("Resetting origin")
            self.interface._reset_internal_state()
        elif key == ord("y"):
            print("Spacemouse teleop success")
            self._state["success"] = True
            self.interface._reset_internal_state()
        elif key == ord("n"):
            print("Spacemouse teleop failure")
            self._state["failure"] = True
            self.interface._reset_internal_state()
        elif key in self._key2macro:
            name, sign = self._key2macro[key]
            macro = self._make_macro(name, sign)
            self._macro_queue.append(macro)
            print(f"[Macro] queued {name} ({len(self._macro_queue)}/{self.queue_size})")
            if len(self._macro_queue) == self.queue_size:
                print("[Macro] queue full — oldest macro will be dropped on next add")

    # ── Macro helpers ──

    def _make_macro(self, name, sign):
        """Return (action_vec, repeat_steps) for the named macro."""
        H = 1  # chunk horizon
        prefix = name.split("_")[0]
        if prefix not in _MACRO_DEFS:
            raise ValueError(f"Unknown macro: {name}")
        sl, axis, mag = _MACRO_DEFS[prefix]
        vec = np.zeros(7, dtype=np.float32)
        vec[sl] = axis * mag * sign * 10
        return vec, H

    # ── Forward ──

    def forward(self, obs_dict):
        """Return 7-DoF action (cartesian velocity) and info dict."""
        if self.first_action:
            self.first_action = False
            print("SPACEMOUSE WORKING!")
            input("Press Enter to continue...")
        time.sleep(0.01) # Tony: Set based on your needs!!

        # Macro override
        if self._macro_step is not None or self._macro_queue:
            if self._macro_step is None:
                self._macro_step = list(self._macro_queue.popleft())
            vec, remaining = self._macro_step
            action = vec.copy()
            self._macro_step[1] -= 1
            if self._macro_step[1] <= 0:
                self._macro_step = None

            data = self.interface.get_controller_state()
            gripper_action = 1.0 if data["grasp"] else -1.0
            grip_vel = gripper_action
            action[6] = grip_vel

            info_dict = create_info_dict(obs_dict, self._state, gripper_action, grip_vel)
            return action, info_dict

        # Normal tele-op
        data = self.interface.get_controller_state()
        dpos = data["dpos"]
        drot = data["raw_drotation"]
        gripper_action = 1.0 if data["grasp"] else -1.0

        lin_vel = self._apply_control_curve(dpos, exponent=1)
        rot_vel = self._apply_control_curve(drot, exponent=1)
        grip_vel = gripper_action

        action = np.concatenate([lin_vel, rot_vel, [grip_vel]])
        action = np.clip(action, -1, 1)

        info_dict = create_info_dict(obs_dict, self._state, gripper_action, grip_vel)
        return action, info_dict

    def close(self):
        yellow_print("Closing SpaceMouse")
