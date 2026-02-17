import cv2, os, time
import numpy as np
import threading
from dataclasses import dataclass
from eva.controllers.pi0_policy import Pi0Policy, Pi0PolicyConfig
from eva.controllers.aawr import AAWRPolicy

import eva.utils.parameters as params

from eva.detectors.dinox_detector import DINOX
from PIL import Image

'''
AAWR Pi0 - Active Perception Policy controller for EVA framework
Author: Jie Wang
Version: 
   2025-04-22: initiate from pi0-spacemouse policy controller
   2026-02-15: refactor as AAWR policy
'''

oob_bounds = {
    "x": [-0.1, 1],
    "y": [-0.5, 0.5],
    "z": [-0.1, 1]
}


@dataclass
class AAWRControllerConfig:
    # PI0 config
    init_instruction: str = "Find the target object in the drawer and pick it up"
    remote_host: str = "10.102.212.31"
    remote_port: int = 8000
    action_space: str = "joint_velocity"
    gripper_action_space: str = "position"
    left_camera_id: str = params.varied_camera_1_id
    right_camera_id: str = params.varied_camera_2_id
    wrist_camera_id: str = params.hand_camera_id
    external_camera: str = "left"
    open_loop_horizon: int = 8
    policy_path: str = "models/AWR/vert/1.pt"

    target_name: str = "yellow pineapple toy"
    detection_prompt: str = "yellow pineapple toy . toy"  # Prompt for DINOX
    detection_confidence: float = 0.25  # Minimum confidence for detection

SWITCH_INTERVAL = 100

class AAWRPi0Controller:
    def __init__(self, config: AAWRControllerConfig = AAWRControllerConfig(), on_switch_callback=None):
        """Initialize both controllers and set up switching logic"""
        print("Initializing AAWR Controller")
        print("AAWR == CONFIG == \n notice you need to change it under eva/controllers/aawr_pi0.py:\n", config)

        # Callback for notifying action space changes
        self._on_switch_callback = on_switch_callback
        self.init_instruction = config.init_instruction
        self.target_name = config.target_name
        self.config = config

        # Initialize DINOX detector
        self.dinox_detector = DINOX()
        self.temp_image_dir = "dinox_results"
        os.makedirs(self.temp_image_dir, exist_ok=True)
        self.dinox_switch_counter = 1
        self.detect_streak = 0

        # Initialize PI0 controller (but don't start querying yet)
        pi0_config = Pi0PolicyConfig(
            remote_host=config.remote_host,
            remote_port=config.remote_port,
            action_space=config.action_space,
            gripper_action_space=config.gripper_action_space,
            left_camera_id=config.left_camera_id,
            right_camera_id=config.right_camera_id,
            wrist_camera_id=config.wrist_camera_id,
            external_camera=config.external_camera,
            open_loop_horizon=config.open_loop_horizon
        )
        self.pi0_controller = Pi0Policy(pi0_config)

        self.aawr_controller = AAWRPolicy(policy_path=config.policy_path)
        # Controller switching lock
        self._switch_lock = threading.Lock()
        self._switching_in_progress = False
        
        self.reset_state() # initialize aawr controller's internal state
        print("\n\n ==== AAWR Controller Controls: ====")
        print("- '=' key: Switch between AAWR & PI0 ")
        print("Current controller: AAWR ")
        
        print("AAWR_PI0 == INIT FINISHED ==")
        
    @property
    def action_space(self):
        """Dynamically return current controller's action space"""
        return self.get_current_controller().action_space
    
    @property
    def gripper_action_space(self):
        """Dynamically return current controller's gripper action space"""
        return self.get_current_controller().gripper_action_space
        
    def get_name(self):
        return "aawr-pi0"
        
    def get_policy_name(self):
        return self.aawr_controller.get_policy_name()
    
    def save_grid(self, save_filepath):
        if self._state["current_controller"] == 1:
            self.aawr_controller.save_grid(save_filepath)
            # print(f"AAWRPI0 == Occupancy grid saved to {save_filepath}")
        else:
            pass            

    def reset_state(self):
        """Reset both controllers and internal state"""
        self.pi0_controller.reset_state()
        self.aawr_controller.reset_state()
        self.pi0_controller.stop_policy()
        self.dinox_switch_counter = 1
        self.detect_streak = 0
        self._switching_in_progress = False

        self._state = {
            "success": False,
            "failure": False,
            "movement_enabled": True,
            "controller_on": True,
            "switch_label": 1,  # 0: PI0, 1: AAWR, 0.5: Transitioning
            "current_controller": 1,   #
            "t_step": 0,
            "switch_at": 0
        }
        
        if self._on_switch_callback:
            current_controller = self.aawr_controller
            print(f"Current controller: {current_controller.get_name()}")
            print(f"Action space: {current_controller.action_space}")
            print(f"Gripper action space: {current_controller.gripper_action_space}")
            self._on_switch_callback(
                current_controller.action_space,
                current_controller.gripper_action_space
            )
        
        self.set_instruction(self.init_instruction)

    def set_instruction(self, instruction):
        """Set instruction for PI0 controller"""
        self.pi0_controller.set_instruction(instruction)


    def detect_target(self, image_path) -> bool:
        """switch to pi0 if detect target object in one checks"""
        predictions = self.dinox_detector.get_dinox(image_path, self.config.detection_prompt)
        for obj in predictions:
            class_name = obj.get('category', '').lower()
            confidence = obj.get('score', 0.0)
            if (self.target_name in class_name and confidence >= self.config.detection_confidence):
                print(f"{self.target_name} detected with confidence {confidence:.2f}")
                return True
        
        return False 
    
    def switch_controller(self):
        """Switch between PI0 and Replayer controllers"""
        with self._switch_lock:
            if not self._switching_in_progress:
                self._switching_in_progress = True
                
                # Switch controller
                if self._state["current_controller"] == 0:
                    print("Stopping PI0 policy...")
                    print("Preparing to switch to Replayer controller...")
                    time.sleep(0.1)  # Small delay to ensure clean switch
                    
                    self._state["switch_label"] = 1.0
                    self.pi0_controller.stop_policy()
                    self._state["current_controller"] = 1

                    print("Switched to Replayer controller")
                else:
                    self._state["switch_label"] = 0.0
                    print("Preparing to switch to PI0 controller...")
                    time.sleep(0.1)  # Small delay to ensure clean switch
                    
                    print("Starting PI0 policy...")
                    self.pi0_controller.start_policy()
                    
                    self._state["current_controller"] = 0
                    # self.pi0_controller.set_instruction(input("Enter command for PI0: "))
                    print("Switched to PI0 controller")
                    self._state["switch_at"] = self._state["t_step"]
                    
                # Notify runner about action space change
                if self._on_switch_callback:
                    current_controller = self.get_current_controller()
                    print(f"Current controller: {current_controller.get_name()}")
                    print(f"Action space: {current_controller.action_space}")
                    print(f"Gripper action space: {current_controller.gripper_action_space}")
                    self._on_switch_callback(
                        current_controller.action_space,
                        current_controller.gripper_action_space
                    )
                
                self._switching_in_progress = False

    def register_key(self, key):
        """Handle key presses for both controllers"""
        if key == ord("="):
            self.switch_controller()
          
        # Forward key press to current controller
        if self._state["current_controller"] == 0:
            self.pi0_controller.register_key(key)
        else:
            self.aawr_controller.register_key(key)
            
    # Update shared state
        controller_state = self.get_current_controller().get_info()
        self._state["success"] = controller_state["success"]
        self._state["failure"] = controller_state["failure"]
        self._state["movement_enabled"] = controller_state["movement_enabled"]
        self._state["controller_on"] = controller_state["controller_on"]
    
    def get_info(self):
        """Get combined controller info"""
        return self._state
    
    def get_current_controller(self):
        """Helper to get current active controller"""
        return self.pi0_controller if self._state["current_controller"] == 0 else self.aawr_controller
    

    def forward(self, observation):
        """Forward observation to current controller"""
        if self._state["t_step"] == 0:
            print("AAWR == Starting...")
        # ------------------DINOX SWITCH LOGIC ---------------------------
        if self.dinox_switch_counter % SWITCH_INTERVAL  == 0 and self._state["current_controller"] == 1:
            img_array = observation["image"][f"{self.config.wrist_camera_id}_left"]
            # Convert BGR to RGB before creating PIL Image
            # OpenCV uses BGR by default, PIL expects RGB
            img_array_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_array_rgb)
            temp_pth = "./temp.jpg"
            img.save(temp_pth)
            
            if self.detect_target(temp_pth):                   # ① interval success
                self.detect_streak += 1                          #   bump streak
                print(f"[DETECT] success (streak={self.detect_streak})")

                if self.detect_streak == 2:                      # ② two in a row
                    print("***** TWO HITS - SWITCH! *****")
                    self.switch_controller()
                    self.detect_streak = 0                       # ③ reset streak
            else:                                                # ④ interval fail
                print("[DETECT] fail -streak reset")
                self.detect_streak = 0                           #   wipe streak

        print("dinox_switch_counter: ", self.dinox_switch_counter)
        self.dinox_switch_counter += 1
        
        # -------------------DINOX SWITCH LOGIC----------------------
        with self._switch_lock:
            if self._state["current_controller"] == 0: # PI0 controller， 8 joint + 1 gripper
                # print('pi00000000')
                
                action, info = self.pi0_controller.forward(observation)
                info["joint_velocity"] = action[:7]
                info["gripper_velocity"] = action[-1]

            else: # AAWR controller, 3D translation + 3D rotation + 1 gripper
                # print('aawr00000000')
                action, info = self.aawr_controller.forward(observation)
                # CLIP cart vel action if OOB
                cartesian_position = observation["robot_state"]["cartesian_position"] 

                # add dummy zero in gripper position
                cartesian_position = np.concatenate([cartesian_position[:6], [0]])
                new_cartesian_position = cartesian_position + action
                
                # # TODO: @XINGFAG, Check and clip boundaries for x, y, z
                for axis in ['x', 'y', 'z']:
                    idx = {'x': 0, 'y': 1, 'z': 2}[axis]
                    min_bound, max_bound = oob_bounds[axis]
                    if new_cartesian_position[idx] < min_bound:
                        action[idx] = min_bound - cartesian_position[idx]
                    elif new_cartesian_position[idx] > max_bound:
                        action[idx] = max_bound - cartesian_position[idx]

                info["cartesian_velocity"] = action[:6]
                info["gripper_velocity"] = action[-1]

            if info is None:
                print("AAWR == INFO IS NONE == ")
            action = np.array(action, dtype=np.float32)
            action = np.clip(action, -1, 1)
            self._state["t_step"] += 1
            
            return action, info
    
    def close(self):
        """Clean up both controllers"""
        self.pi0_controller.close()
        self.aawr_controller.close() 
        if os.path.exists("./temp.jpg"):
            os.remove("./temp.jpg")

# Backward-compatibility alias
AAWRController = AAWRPi0Controller