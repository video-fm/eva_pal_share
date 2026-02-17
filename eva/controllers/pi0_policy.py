import numpy as np
import time
import threading
import os
import cv2
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from eva.utils.misc_utils import run_threaded_command, create_info_dict, print_datadict_tree
from dataclasses import dataclass

import eva.utils.parameters as params


'''
Pi0 Policy controller for EVA framework
Author: Jie Wang
Version: 
   2025-04-09: rewrite pi0 inference under EVA 
   2025-04-13: add switch controller logic 
   2025-04-14: fix initialization error

'''
@dataclass
class Pi0PolicyConfig:
    remote_host: str = "10.102.212.31"
    remote_port: int = 8000
    action_space: str = "joint_velocity"
    gripper_action_space: str = "position"
    left_camera_id: str = params.varied_camera_1_id
    right_camera_id: str = params.varied_camera_2_id
    wrist_camera_id: str = params.hand_camera_id
    external_camera: str = "left"
    open_loop_horizon: int = 8
    instruction: str = "pick up the pineapple in the drawer"
    model_name: str = "pi05-fm"

class Pi0Policy:
    def __init__(self, config: Pi0PolicyConfig = Pi0PolicyConfig()):
        """Initialize PI0 policy controller but don't start querying"""
        print("PI0 init")
        self.action_space = config.action_space
        self.gripper_action_space = config.gripper_action_space
        
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
        # self.current_instruction = "find the pineapple on the bookshelf" # Default instruction
        self.reset_state()
        
    def get_name(self):
        return self.model_name
        
    def get_policy_name(self):
        return self.model_name

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
            "t_step": 0
        }

    def set_state(self, key):
        self._state["success"] = key == "y"
        self._state["failure"] = key == "n"
            
    def _update_internal_state(self):
        pass
        # while self.running:
        #     time.sleep(1 / 50)
        #     self._process_reading()
            
    def _process_reading(self):
        pass
    
    def set_instruction(self, instruction):
        self.current_instruction = instruction
        # Set instrucion
        # instruction = "find the pineapple and place into the bowl"
        # self.current_instruction = instruction.replace("pineapple", "yellow duck")
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

    def forward(self, observation):
        """Return: 8 dof action step on joint velocity space"""
        
        cartesian_position = np.array(observation["robot_state"]["cartesian_position"])
        joint_position = np.array(observation["robot_state"]["joint_positions"])
        gripper_position = np.array([observation["robot_state"]["gripper_position"]])
        
        if not self._is_running:
            # Return safe action when not running
            info_dict = create_info_dict(observation, self._state)
            action = np.concatenate([joint_position, [gripper_position[0]]])
            
            return np.zeros_like(action), info_dict

        self._ensure_connected()
        
        # if self._state["t_step"] == 0:
        #     input("Press Enter to continue...")
        
        self._state["t_step"] += 1

        # Extract camera images
        images = self._extract_images(observation)
        if self.policy_query_count == 0:
            for name, img in images.items():
                if img is not None:
                    img_rgb = img[..., ::-1] # Convert BGR to RGB
                    os.makedirs("debug", exist_ok=True)
                    cv2.imwrite(f"debug/pi0_{self.current_instruction}_{name}.jpg", img_rgb)
        
        info_dict = create_info_dict(observation, self._state)

        # If we need a new action chunk
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
            
            # Prepare request data
            request_data = {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(ext_image, 224, 224),
                "observation/wrist_image_left": image_tools.resize_with_pad(images["wrist_image"], 224, 224),
                "observation/joint_position": joint_position,
                "observation/gripper_position": gripper_position,
                "prompt": self.current_instruction,
            }
            
            # Query policy server
            print(f"Querying policy server: {self.current_instruction}")
            start_time = time.time()
            try:
                self.policy_query_count += 1
                print(f"PI0 query {self.policy_query_count}")
                self.pred_action_chunk = self._policy_client.infer(request_data)["actions"]
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