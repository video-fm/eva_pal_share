
import gym
import numpy as np
from copy import deepcopy

from eva.cameras.multi_camera_wrapper import MultiCameraWrapper
from eva.robot.server_interface import ServerInterface
from eva.utils.calibration_utils import load_calibration_info
from eva.utils.parameters import camera_type_dict, hand_camera_id, nuc_ip
from eva.utils.geometry_utils import change_pose_frame
from eva.utils.misc_utils import time_ms

class FrankaEnv(gym.Env):
    def __init__(self, action_space="cartesian_position", gripper_action_space="velocity", camera_kwargs={}, do_reset=True):
        # Initialize Gym Environment
        super().__init__()

        self.set_action_space(action_space)
        self.set_gripper_action_space(gripper_action_space)

        # Robot Configuration
        self.reset_joints = np.array([0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0])
        # self.reset_joints = np.array([0, -1 / 2 * np.pi, 0, -7 / 8 * np.pi, 0, 5 / 12 * np.pi, 0.0])  # For vlm_trajectory project
        # self.reset_joints = np.array([-0.017, -0.269, 0.040, -1.916, -0.0015, 1.569, 0.015]) # For tiptop project
        # self.reset_joints = np.array([-0.0167203, -0.22184323, 0.01463179, -2.4473877, -0.01777307, 3.62010765, -0.0041602])
        print(f"Initializing joints at: {self.reset_joints}")
        self.control_hz = 15

        if nuc_ip is None:
            from eva.robot.controller import FrankaController
            self._robot = FrankaController()
        else:
            self._robot = ServerInterface(ip_address=nuc_ip)

        # Create Cameras
        self.camera_reader = MultiCameraWrapper(camera_kwargs)
        self.calibration_dict = load_calibration_info()
        self.camera_type_dict = camera_type_dict

        # Reset Robot
        if do_reset:
            self.reset()

    def step(self, action):
        # Check Action
        assert len(action) == self.DoF, f"Provided action dimension ({len(action)}) does not match expected ({self.DoF}) for action space {self.action_space}!"
        if self.check_action_range:
            assert (action.max() <= 1) and (action.min() >= -1)

        # Update Robot
        action_info = self.update_robot(
            action,
            action_space=self.action_space,
            gripper_action_space=self.gripper_action_space,
        )

        # Return Action Info
        return action_info

    def reset(self):
        self._robot.update_gripper(0, velocity=False, blocking=True)
        self._robot.update_joints(self.reset_joints, velocity=False, blocking=True, cartesian_noise=None)

    def change_reset_joints(self):
        print("6666")
        state_dict, _ = self.get_state()
        
        joints = state_dict['robot_state']['joint_positions']
        self._robot.update_gripper(0, velocity=False, blocking=True)
        print("7777")
        self._robot.update_joints(joints, velocity=False, blocking=True, cartesian_noise=None)
        print("8888")
        self.reset_joints = joints

    def update_robot(self, action, action_space="cartesian_velocity", gripper_action_space="velocity", blocking=False):
        action_info = self._robot.update_command(
            action,
            action_space=action_space,
            gripper_action_space=gripper_action_space,
            blocking=blocking
        )
        return action_info

    def create_action_dict(self, action, action_space=None, gripper_action_space="velocity", robot_state=None):
        if action_space is None:
            action_space = self.action_space
        return self._robot.create_action_dict(action, action_space, gripper_action_space, robot_state)

    def read_cameras(self):
        return self.camera_reader.read_cameras()

    def get_state(self):
        read_start = time_ms()
        state_dict, timestamp_dict = self._robot.get_robot_state()
        timestamp_dict["read_start"] = read_start
        timestamp_dict["read_end"] = time_ms()
        return state_dict, timestamp_dict

    def get_camera_extrinsics(self, state_dict):
        # Adjust gripper camera by current pose
        extrinsics = deepcopy(self.calibration_dict)
        for cam_id in self.calibration_dict:
            if hand_camera_id not in cam_id:
                continue
            gripper_pose = state_dict["cartesian_position"] 
            extrinsics[cam_id + "_gripper_offset"] = extrinsics[cam_id] # Q: what is this offset?
            extrinsics[cam_id] = change_pose_frame(extrinsics[cam_id], gripper_pose)
        return extrinsics

    def get_observation(self):
        obs_dict = {"timestamp": {}}

        # Robot State #
        state_dict, timestamp_dict = self.get_state()
        obs_dict["robot_state"] = state_dict
        obs_dict["timestamp"]["robot_state"] = timestamp_dict

        # Camera Readings #
        camera_obs, camera_timestamp = self.read_cameras()
        obs_dict.update(camera_obs)
        obs_dict["timestamp"]["cameras"] = camera_timestamp

        # Camera Info #
        obs_dict["camera_type"] = deepcopy(self.camera_type_dict)
        extrinsics = self.get_camera_extrinsics(state_dict)
        obs_dict["camera_extrinsics"] = extrinsics

        intrinsics = {}
        for cam in self.camera_reader.camera_dict.values():
            cam_intr_info = cam.get_intrinsics()
            for (full_cam_id, info) in cam_intr_info.items():
                intrinsics[full_cam_id] = info["cameraMatrix"]
        obs_dict["camera_intrinsics"] = intrinsics

        return obs_dict
    
    def set_action_space(self, action_space):
        print(f"Set action space to {action_space}")
        assert action_space in ["cartesian_position", "joint_position", "cartesian_velocity", "joint_velocity"]
        self.action_space = action_space
        self.check_action_range = "velocity" in action_space
        self.DoF = 7 if ("cartesian" in action_space) else 8
    
    def set_gripper_action_space(self, gripper_action_space):
        self.gripper_action_space = gripper_action_space
    
    def reload_calibration(self):
        self.calibration_dict = load_calibration_info()

    def close(self):
        self._robot.server.close()
