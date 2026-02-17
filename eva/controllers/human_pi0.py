import numpy as np
import time
import os
import cv2
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from eva.utils.misc_utils import run_threaded_command, create_info_dict, print_datadict_tree
from eva.utils.misc_utils import yellow_print, blue_print
from dataclasses import dataclass

import eva.utils.parameters as params
from eva.utils.geometry_utils import quat_to_rmat, quat_to_euler
from scipy.spatial.transform import Rotation as Rot

'''
Human Pi0 controller for EVA framework
Author: Jie Wang
Version: 
   2026-01-09: initiate from pi0 policy controller
   2026-01-10: add demodiffusion style retargetting logic
   2026-02-17: clean up for release

'''
@dataclass
class DemoDiffusionConfig:
    remote_host: str = "10.102.212.31"
    remote_port: int = 8000
    action_space: str = "joint_velocity"
    gripper_action_space: str = "position"
    left_camera_id: str = params.varied_camera_1_id
    right_camera_id: str = params.varied_camera_2_id
    wrist_camera_id: str = params.hand_camera_id
    external_camera: str = "left"
    open_loop_horizon: int = 8
    instruction: str = "open the cabinet door"
    model_name: str = "pi0-droid"
    task: str = "open_cabinet"
    traj: int = 0
    gripper_threshold: float = 0.3

BASE_PATH = '/home/franka/eva_tony/human_data'
RETARGET_MODE = "cartesian_position"

open_loop_horizon = 8
predict_action_horizon = 10 # 10 for pi0, 15 for pi05
full_action_dim = 32  # Pi0 uses 32-dim action space internally
droid_action_dim = 8  # DROID only uses first 8 dims

def load_trajectory_data(config):
    data_path = os.path.join(BASE_PATH, config.task, f"traj_{config.traj}")
    eef_pose_list = np.load(os.path.join(data_path, "processed_3d", "eef_pose.npy"))
    retarget_gripper_action_list = np.load(os.path.join(data_path, "processed_3d", "retarget_gripper_action.npy"))
    retarget_gripper_action_list = np.where(retarget_gripper_action_list > config.gripper_threshold, 1, 0).astype(float)
    
    for i in range(len(eef_pose_list)):
        eef_pose = eef_pose_list[i]    
        # eef_pose[0] += 0.12
        eef_pose[2] += 0.05

        eef_rotation = quat_to_rmat(eef_pose[3:])

        eef_pose_list[i][:3] -= eef_rotation @ np.array([0., 0, 0.062]).T # gripper tip to wrist offset
        # print(f"Loaded {len(eef_pose_list)} poses from trajectory {config.traj}")
        # print(f"EEF pose shape: {eef_pose_list.shape}")
        # print(f"Gripper action shape: {retarget_gripper_action_list.shape}")
    return data_path, eef_pose_list, retarget_gripper_action_list



class DemoDiffusionPolicy:
    def __init__(self, config: DemoDiffusionConfig = DemoDiffusionConfig(), on_switch_callback=None):
        """Initialize DemoDiffusion-Pi0 policy controller but don't start querying"""
        print("DemoDiffusion-Pi0 init")
        self.action_space = config.action_space
        self.gripper_action_space = config.gripper_action_space
        self.config = config
        # Callback to runner for switching action spaces
        self._on_switch_callback = on_switch_callback
        
        # Camera settings
        self.left_camera_id = config.left_camera_id
        self.right_camera_id = config.right_camera_id
        self.wrist_camera_id = config.wrist_camera_id
        self.external_camera = config.external_camera
        self.current_instruction = config.instruction
        self.model_name = config.model_name
        assert self.external_camera in ["left", "right"], f"External camera must be 'left' or 'right', got {self.external_camera}"
        
        # Policy server settings
        self.remote_host = config.remote_host
        self.remote_port = config.remote_port
        self._policy_client = None
        self._is_running = True
        
        # Action execution settings
        self.open_loop_horizon = config.open_loop_horizon
        self.actions_from_chunk_completed = 0
        self.policy_query_count = 0

        self.pred_action_chunk = None

        # Demo Diffusion settings
        self.data_path, self.eef_pose_list, self.retarget_gripper_action_list = load_trajectory_data(config)
        self.time_denoise = 0.2
        print(f"Loaded trajectory data from {self.data_path}")
        # Robot controller reference - set by Runner via set_env()
        self._env = None

        self.reset_state()
    
    def set_env(self, env):
        """Set environment reference for robot controller access.
        Called by Runner after controller creation."""
        self._env = env
        
    def get_name(self):
        return self.model_name

    def get_policy_name(self):
        return str("demodiffusion\n" + self.model_name + "\n" + self.data_path)

    def save_human_data(self):
        pass
        
    def reset_state(self):
        self.actions_from_chunk_completed = 0
        self.policy_query_count = 0
        self.pred_action_chunk = None
        self._state = {
            "success": False,
            "failure": False,
            "movement_enabled": True,
            "controller_on": True,
            "switch_label": 0, 
            "t_step": 0,
            "at_start_position": False
        }
        # traj_num = input("Please input the trajectory number: ")
        # self.config.traj = int(traj_num) if traj_num.isdigit() else 0
        # print(f"Set trajectory number: {self.config.traj}")
        # self.data_path, self.eef_pose_list, self.retarget_gripper_action_list = load_trajectory_data(self.config)

    def set_state(self, key):
        self._state["success"] = key == "y"
        self._state["failure"] = key == "n"
            
    def _update_internal_state(self):
        pass
            
    def _process_reading(self):
        pass
    
    def set_instruction(self, instruction):
        self.current_instruction = instruction
        print(f"Set instruction: {self.current_instruction}")
    
    def set_horizon(self, horizon):
        self.open_loop_horizon = int(horizon)
        print(f"Set open loop horizon: {self.open_loop_horizon}")

    def register_key(self, key):
        """Handle key presses."""
        if key == ord(" "):
            pass
            # self._state["movement_enabled"] = not self._state["movement_enabled"]
            # print("Movement enabled:", self._state["movement_enabled"])
        elif key == ord("y"):
            self._state["success"] = True
            print("Success!")
        elif key == ord("n"):
            self._state["failure"] = True
            print("Failure!")
    

    def get_info(self):
        ret = self._state.copy()
        return ret

    def _extract_images(self, obs_dict):
        image_observations = obs_dict["image"]
        
        left_image, right_image, wrist_image = None, None, None
        for key in image_observations:
            # Note the "left" below refers to the left camera in the stereo pair.
            # The model is only trained on left stereo cams, so we only feed those.
            if self.left_camera_id in key and "left" in key:
                left_image = image_observations[key]
            elif self.right_camera_id in key and "left" in key:
                right_image = image_observations[key]
            elif self.wrist_camera_id in key and "left" in key:
                wrist_image = image_observations[key]
        
        if left_image is not None:
            left_image = left_image[..., :3]  # Drop alpha channel
            left_image = left_image[..., ::-1]  # Convert BGR to RGB
        
        if right_image is not None:
            right_image = right_image[..., :3]
            right_image = right_image[..., ::-1]
        
        if wrist_image is not None:
            wrist_image = wrist_image[..., :3]
            wrist_image = wrist_image[..., ::-1]
            
        return {
            "left_image": left_image,
            "right_image": right_image,
            "wrist_image": wrist_image
        }

    
    def _ensure_connected(self):
        """Ensure connection to policy server only if running"""
        if self._is_running and self._policy_client is None:
            print(f"Connecting to policy server at {self.remote_host}:{self.remote_port}")
            self._policy_client = websocket_client_policy.WebsocketClientPolicy(self.remote_host, self.remote_port)
            
            # Try to get policy name from server metadata
            try:
                metadata = self._policy_client.get_server_metadata()
                print(f"Server metadata: {metadata}")
                if "policy_name" in metadata:
                    self.model_name = metadata["policy_name"]
                    print(f"Updated model name from server: {self.model_name}")
            except Exception as e:
                print(f"Failed to get server metadata: {e}")
    
    def start_policy(self):
        """Start policy querying"""
        print("Starting PI0 policy...")
        self._is_running = True
        self._ensure_connected()
        
    def stop_policy(self):
        """Stop policy querying"""
        print("Stopping PI0 policy...")
        self._is_running = False
        # self._policy_client = None  # Let the garbage collector handle the cleanup
        self.reset_state()


    def move_to_start_position(self, current_cartesian_position):
        """Move robot to start position using cartesian_position action space.
        Returns (action, reached_start) where action is 7-dim cartesian pose + gripper."""
        start_eef_pose = self.eef_pose_list[0]
        start_position = start_eef_pose[:3]
        start_euler = quat_to_euler(start_eef_pose[3:])
        start_gripper = self.retarget_gripper_action_list[0]
        time.sleep(0.3)
        # Apply Z-axis 180° rotation for upside down gripper
        rot_z_180 = np.array([[-1, 0.0, 0.0],
                              [0.0, -1.0, 0.0],
                              [0, 0, 1]])
        start_rotation = Rot.from_euler("xyz", start_euler).as_matrix()
        start_rotation = start_rotation @ rot_z_180
        # start_euler = Rot.from_matrix(start_rotation).as_euler("xyz")
        
        # Check if close enough to start position
        position_error = np.linalg.norm(current_cartesian_position[:3] - start_position)
        orientation_error = np.linalg.norm(current_cartesian_position[3:] - start_euler)
        
        position_threshold = 0.2
        orientation_threshold = 0.1
        
        reached_start = position_error < position_threshold and orientation_error < orientation_threshold
        
        if reached_start:
            blue_print(f"Reached start position! pos_err={position_error:.4f}, ori_err={orientation_error:.4f}")
        else:
            print(f"Moving to start: pos_err={position_error:.4f}, ori_err={orientation_error:.4f}")
        
        # Return cartesian position action: [x, y, z, rx, ry, rz, gripper]
        action = np.concatenate([start_position, start_euler, [start_gripper]])
        return action, reached_start

    def compute_demo_actions(self, timestep, eef_pose_list, retarget_gripper_action_list):
        actions_demo = np.zeros((predict_action_horizon, full_action_dim))
        
        if self._env is None:
            print("\033[91mError: Environment not set. Call set_env() first.\033[0m")
            return actions_demo
        
        if timestep + predict_action_horizon < len(eef_pose_list):
            actions_demo[:, 7] = retarget_gripper_action_list[timestep:timestep + predict_action_horizon]
        else:
            actions_demo[:, 7] = retarget_gripper_action_list[-predict_action_horizon:]
        
        # Extract EEF poses for arm actions (next poses to move toward)
        if timestep + 1 + predict_action_horizon <= len(eef_pose_list):
            eef_poses_next = eef_pose_list[timestep + 1:timestep + 1 + predict_action_horizon]
        else:
            eef_poses_next = eef_pose_list[-predict_action_horizon:]
        
        # Populate arm actions with cartesian positions (position + euler angles)
        actions_demo[:, :3] = eef_poses_next[:, :3]  # Position
        actions_demo[:, 3:6] = quat_to_euler(eef_poses_next[:, 3:])  # Quaternion to Euler
        
        robot_state = None
        for i in range(predict_action_horizon):            
            feed = actions_demo[i]
            robot_state = None if i == 0 else robot_state
            action_dict = self._env.create_action_dict(
                action=feed, 
                action_space=RETARGET_MODE, 
                gripper_action_space="velocity", 
                robot_state=robot_state
            )
            robot_state = action_dict.pop("robot_state")
            for key in action_dict:
                robot_state[key] = action_dict[key]
            actions_demo[i, :7] = action_dict["joint_velocity"]
        
        return actions_demo


    def forward(self, observation):
        """Return: 8 dof action step on joint velocity space"""
        time.sleep(0.3)

        cartesian_position = np.array(observation["robot_state"]["cartesian_position"])
        joint_position = np.array(observation["robot_state"]["joint_positions"])
        gripper_position = np.array([observation["robot_state"]["gripper_position"]])
        
        if not self._is_running:
            # Return safe action when not running
            info_dict = create_info_dict(observation, self._state)
            action = np.concatenate([joint_position, [gripper_position[0]]])
            
            return np.zeros_like(action), info_dict

        self._ensure_connected()
        
        # At beginning of trajectory, move to start position first
        if not self._state["at_start_position"]:
            # Switch to cartesian_position action space via runner callback
            if self._on_switch_callback:
                self._on_switch_callback("cartesian_position", "position")
            
            action, reached_start = self.move_to_start_position(cartesian_position)
            info_dict = create_info_dict(observation, self._state)
            
            if reached_start:
                self._state["at_start_position"] = True
                blue_print("Start position reached - switching back to joint_velocity")
                # Switch back to joint_velocity via runner callback
                if self._on_switch_callback:
                    self._on_switch_callback(self.action_space, self.gripper_action_space)
            else:
                return action, info_dict
        
        # if self._state["t_step"] == 0:
        #     self.time_denoise = input("Plase input time_denoise between 0~1: ")
        #     if not self.time_denoise or float(self.time_denoise) < 0.0 or float(self.time_denoise) > 1.0:
        #         self.time_denoise = 0.2
        #         yellow_print("input error, set time denoise to 0.2")

        self._state["t_step"] += 1

        # Extract camera images
        images = self._extract_images(observation)
        # TODO: refactor this into utils function
        if self.policy_query_count == 0:
            for name, img in images.items():
                if img is not None:
                    img_rgb = img[..., ::-1] # Convert BGR to RGB
                    os.makedirs("debug/demodiffusion", exist_ok=True)
                    instruction_formatted = self.current_instruction.lower().replace(" ", "_")
                    cv2.imwrite(f"debug/demodiffusion/{instruction_formatted}_{name}.jpg", img_rgb)
        
        info_dict = create_info_dict(observation, self._state)

        actions_demo = self.compute_demo_actions(self._state["t_step"], self.eef_pose_list, self.retarget_gripper_action_list)

        if self._state["t_step"] > len(self.eef_pose_list) - predict_action_horizon:
            print("\033[92mDemo Diffusion: Reached end of trajectory - stopping robot\033[0m")
            print(f"\033[92mCompleted {self._state['t_step']} steps\033[0m")
            # Signal success and return zero velocities to stop the robot
            rollout_result=input("Please press y if the rollout is successful, n if it is not: ")
            if rollout_result == "y":
                self._state["success"] = True
            else:
                self._state["failure"] = True
            zero_action = np.zeros(droid_action_dim)  # 7 zero joint velocities + 1 gripper
            zero_action[-1] = gripper_position[0]  # Maintain current gripper state
            return zero_action, info_dict
        
        if self.actions_from_chunk_completed == 0 or self.actions_from_chunk_completed >= self.open_loop_horizon:
            self.actions_from_chunk_completed = 0
            # Check if we have the required images
            if self.external_camera == "left" and images["left_image"] is None:
                print("Error: Left camera image not available")
                return np.concatenate([joint_position, [gripper_position[0]]]), info_dict
            elif self.external_camera == "right" and images["right_image"] is None:
                print("Error: Right camera image not available")
                return np.concatenate([joint_position, [gripper_position[0]]]), info_dict
            elif images["wrist_image"] is None:
                print("Error: Wrist camera image not available")
                return np.concatenate([joint_position, [gripper_position[0]]]), info_dict
            
            # Get exterior image based on selected camera
            ext_image = images["left_image"] if self.external_camera == "left" else images["right_image"]


            request_data = {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(ext_image, 224, 224),
                "observation/wrist_image_left": image_tools.resize_with_pad(images["wrist_image"], 224, 224),
                "observation/joint_position": joint_position,
                "observation/gripper_position": gripper_position,
                "prompt": self.current_instruction,
                "retargeted_actions": actions_demo,
                "time": float(self.time_denoise),  # Must be float, not int (JAX type checking)
                "action_dim_used": 8
            }
            
            # Query policy server
            print(f"Querying policy server: {self.current_instruction}")
            start_time = time.time()
            try:
                self.policy_query_count += 1
                print(f"PI0 query {self.policy_query_count}")
                # Slice to first 8 dims: 7 joint velocities + 1 gripper (DROID action space)
                self.pred_action_chunk = self._policy_client.infer(request_data)["actions"][:, :droid_action_dim]
                end_time = time.time()
                print(f"Policy inference took: {end_time - start_time:.6f} seconds")
                print(f"Received action chunk, shape: {self.pred_action_chunk.shape}")
            except Exception as e:
                print(f"Error querying policy server: {e}")
                print(f"staying same joint position:{joint_position}")
                # Return safe action in case of error
                action = np.zeros_like(joint_position)
                action = np.concatenate([action, [gripper_position[0]]])
                print(f"Fallback action: {action}")
                return action, info_dict
        
        #  Forward: valid action chunk
        if self.pred_action_chunk is not None:
            action = self.pred_action_chunk[self.actions_from_chunk_completed]
            self.actions_from_chunk_completed += 1
            print(f"PI0 updating action {self.actions_from_chunk_completed}")
            # Binarize gripper action # TODO check from pi0 data
            if action[-1].item() > 0.5:
                action = np.concatenate([action[:-1], np.ones((1,))])
            else:
                action = np.concatenate([action[:-1], np.zeros((1,))])
            
            # clip all dimensions of action to [-1, 1]
            action = np.clip(action, -1, 1)
                 
            info_dict["cartesian_position"] = cartesian_position  # Essential for timestep_processor
            info_dict["gripper_position"] = action[-1]
            info_dict["joint_velocity"] = action[:-1]

            # print(f"PI0 Forwarding Action: {action}")
            return action, info_dict
        else:
            # Fallback action - maintain current position
            action = np.concatenate([joint_position, [gripper_position[0]]])
            print(f"Fallback action: {action}")
            return action, info_dict
    
    def close(self):
        """Clean up resources."""
        self.stop_policy()