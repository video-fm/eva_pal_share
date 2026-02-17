
import numpy as np

from eva.utils.geometry_utils import add_angles, euler_to_quat, quat_diff, quat_to_euler, rmat_to_quat

class Keyboard:
    def __init__(
        self,
        max_lin_vel: float = 1,
        max_rot_vel: float = 1,
        max_gripper_vel: float = 1,
        spatial_coeff: float = 1,
        pos_action_gain: float = 5,
        rot_action_gain: float = 2,
        gripper_action_gain: float = 3,
    ):
        self.action_space = "cartesian_velocity"
        self.gripper_action_space = "velocity"
        self.max_lin_vel = max_lin_vel
        self.max_rot_vel = max_rot_vel
        self.max_gripper_vel = max_gripper_vel
        self.spatial_coeff = spatial_coeff
        self.pos_action_gain = pos_action_gain
        self.rot_action_gain = rot_action_gain
        self.gripper_action_gain = gripper_action_gain

        self.pressed_keys = set()
        self._state = {
            "success": False,
            "failment_enabled": False,
            "conture": False,
            "moveroller_on": True,
        }
        self.reset_state()

        print("Initialized keyboard controller (keep camera feed window in focus)")
        print(
            """
            u  i  o       +x
            j     l    +y    -y
            m  ,  .       -x
            y     n    +z    -z

            q  w  e       +rx
            a     d    +ry    -ry
            z  x  c       -rx
            r     v    +rz    -rz

            [: open gripper
            ]: close gripper

            space: toggle control
            enter: success
            backspace: failure
            """
        )
        print("Press space to unlock control...")

    def get_name(self):
        return "keyboard"

    def reset_state(self):
        self.pressed_keys.clear()
        self.reset_origin = True
        self.robot_origin = None
        self.keyboard_origin = None
        self.keyboard_state = None
        self._state = {
            "success": False,
            "failure": False,
            "movement_enabled": False,
            "controller_on": True,
        }
    
    def register_key(self, key):
        if key == ord(" "):
            self._state["movement_enabled"] = not self._state["movement_enabled"]
            print("Movement enabled:", self._state["movement_enabled"])
        elif key == 13:
            self._state["success"] = True
        elif key == 8:
            print("Failure")
            self._state["failure"] = True
        else:
            self.pressed_keys.add(chr(key))
    
    def get_info(self):
        return self._state.copy()
    
    def _process_keys(self):
        dx, dy, dz = 0, 0, 0
        droll, dpitch, dyaw = 0, 0, 0
        if len(self.pressed_keys) > 0:
            print(self.pressed_keys)
        for key in self.pressed_keys:
            match key:
                case "u":
                    dx += 0.01
                    dy += 0.01
                case "i":
                    dx += 0.01
                case "o":
                    dx += 0.01
                    dy -= 0.01
                case "j":
                    dy += 0.01
                case "l":
                    dy -= 0.01
                case "m":
                    dx -= 0.01
                    dy += 0.01
                case ",":
                    dx -= 0.01
                case ".":
                    dx -= 0.01
                    dy -= 0.01
                case "y":
                    dz += 0.01
                case "n":
                    dz -= 0.01
                case "q":
                    droll += 0.01
                    dpitch += 0.01
                case "w":
                    droll += 0.01
                case "e":
                    droll += 0.01
                    dpitch -= 0.01
                case "a":
                    dpitch += 0.01
                case "d":
                    dpitch -= 0.01
                case "z":
                    droll -= 0.01
                    dpitch += 0.01
                case "x":
                    droll -= 0.01
                case "c":
                    droll -= 0.01
                    dpitch -= 0.01
                case "r":
                    dyaw += 0.01
                case "v":
                    dyaw -= 0.01
                case "[":
                    self.keyboard_state["gripper"] = 0
                case "]":
                    self.keyboard_state["gripper"] = 1
        if dx != 0 or dy != 0 or dz != 0 or droll != 0 or dpitch != 0 or dyaw != 0:
            if not self._state["movement_enabled"]:
                print("Movement disabled, press space to enable!")
        self.keyboard_state["pos"] = self.keyboard_state["pos"] + np.array([dx, dy, dz]) * self.spatial_coeff
        self.keyboard_state["quat"] = euler_to_quat(add_angles(quat_to_euler(self.keyboard_state["quat"]), np.array([droll, dpitch, dyaw])))
        self.pressed_keys.clear()

    def forward(self, obs_dict):
        if self.keyboard_state:
            self._process_keys()
        return self._calculate_action(obs_dict["robot_state"])
    
    def _limit_velocity(self, lin_vel, rot_vel, gripper_vel):
        """Scales down the linear and angular magnitudes of the action"""
        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)
        gripper_vel_norm = np.linalg.norm(gripper_vel)
        if lin_vel_norm > self.max_lin_vel:
            lin_vel = lin_vel * self.max_lin_vel / lin_vel_norm
        if rot_vel_norm > self.max_rot_vel:
            rot_vel = rot_vel * self.max_rot_vel / rot_vel_norm
        if gripper_vel_norm > self.max_gripper_vel:
            gripper_vel = gripper_vel * self.max_gripper_vel / gripper_vel_norm
        return lin_vel, rot_vel, gripper_vel

    def _calculate_action(self, state_dict):
        # Read Observation
        robot_pos = np.array(state_dict["cartesian_position"][:3])
        robot_euler = state_dict["cartesian_position"][3:]
        robot_quat = euler_to_quat(robot_euler)
        robot_gripper = state_dict["gripper_position"]

        # Reset Origin On Release #
        if self.reset_origin:
            self.robot_origin = {"pos": robot_pos, "quat": robot_quat}
            self.keyboard_state = {"pos": robot_pos, "quat": robot_quat, "gripper": 0}
            self.keyboard_origin = {"pos": self.keyboard_state["pos"], "quat": self.keyboard_state["quat"]}
            self.reset_origin = False

        # Calculate Positional Action #
        robot_pos_offset = robot_pos - self.robot_origin["pos"]
        target_pos_offset = self.keyboard_state["pos"] - self.keyboard_origin["pos"]
        pos_action = target_pos_offset - robot_pos_offset

        # Calculate Euler Action #
        robot_quat_offset = quat_diff(robot_quat, self.robot_origin["quat"])
        target_quat_offset = quat_diff(self.keyboard_state["quat"], self.keyboard_origin["quat"])
        quat_action = quat_diff(target_quat_offset, robot_quat_offset)
        euler_action = quat_to_euler(quat_action)

        # Calculate Gripper Action #
        gripper_action = self.keyboard_state["gripper"] - robot_gripper

        # Calculate Desired Pose #
        target_pos = pos_action + robot_pos
        target_euler = add_angles(euler_action, robot_euler)
        target_cartesian = np.concatenate([target_pos, target_euler])
        target_gripper = self.keyboard_state["gripper"]

        # Scale Appropriately #
        pos_action *= self.pos_action_gain
        euler_action *= self.rot_action_gain
        gripper_action *= self.gripper_action_gain
        lin_vel, rot_vel, gripper_vel = self._limit_velocity(pos_action, euler_action, gripper_action)

        # Prepare Return Values #
        info_dict = {"target_cartesian_position": target_cartesian, "target_gripper_position": target_gripper}
        action = np.concatenate([lin_vel, rot_vel, [gripper_vel]])
        action = action.clip(-1, 1)

        return action, info_dict
    
    def close(self):
        self.running = False
