import numpy as np
import time
import threading
import os, cv2
from PIL import Image

from openpi_client import image_tools
from openpi_client import websocket_client_policy
from eva.utils.misc_utils import print_datadict_shape, print_datadict_tree, create_info_dict
from dataclasses import dataclass
from eva.controllers.pi0_policy import Pi0Policy, Pi0PolicyConfig
from eva.controllers.replayer import Replayer
from eva.detectors.dinox_detector import DINOX
import eva.utils.parameters as params
from eva.utils.misc_utils import blue_print

'''
 Replayer Pi0 controller for EVA framework
Author: Jie Wang
Version:
   2025-04-25: initiate from pi0-spacemouse mixed controller
   2025-05-xx: added pineapple detection based switching
   2026-02-15: clean up for release
'''
@dataclass
class ReplayConfig:
    # PI0 config
    init_instruction: str = "pick up the yellow pineapple toy and place it on the table"
    remote_host: str = "10.102.212.31"
    remote_port: int = 8000
    action_space: str = "joint_velocity"
    gripper_action_space: str = "position"
    left_camera_id: str = params.varied_camera_1_id
    right_camera_id: str = params.varied_camera_2_id
    wrist_camera_id: str = params.hand_camera_id
    external_camera: str = "right"
    open_loop_horizon: int = 8
    target_name: str = "yellow pineapple toy"
    traj_path: str = "models/Replay/hard/2025-05-11_21-27-56/trajectory.npz"
    detection_prompt: str = "yellow pineapple toy . toy"  # Prompt for DINOX
    detection_confidence: float = 0.3  # Minimum confidence for detection
    

class ReplayPi0Controller:
    def __init__(self, config: ReplayConfig = ReplayConfig(), on_switch_callback=None):
        """Initialize both controllers and set up switching logic"""
        blue_print("Initializing Replay Controller (PI0 + Replayer)")
        blue_print("ReplayPi0 == CONFIG == \n notice you need to change it under eva/controllers/replay_pi0.py:\n", config)
        
        # Callback for notifying action space changes
        self._on_switch_callback = on_switch_callback
        self.init_instruction = config.init_instruction
        self.config = config
        
        # Initialize DINOX detector
        self.dinox_detector = DINOX()
        self.temp_image_dir = "dinox_results"
        os.makedirs(self.temp_image_dir, exist_ok=True)
        self.dinox_switch_counter = 1 # todo: set as 0 in hard task
        self.switch_interval = 1000
        
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
        
        self.replayer_controller = Replayer(config.traj_path)
        
        # Controller switching lock
        self._switch_lock = threading.Lock()
        self._switching_in_progress = False
        
        self.reset_state() # initialize mix controller's internal state
        blue_print("\n\n ==== ReplayPi0 Controls: ====")
        blue_print("- '=' key: Switch between PI0 and Replayer")
        blue_print("- Auto-switching when target is detected")
        blue_print("Current controller: PI0")
        
        # Start_state with PI0
        # self.pi0_controller.start_policy()
        blue_print("ReplayPi0 == INIT FINISHED ==")
        
    @property
    def action_space(self):
        """Dynamically return current controller's action space"""
        return self.get_current_controller().action_space
    
    @property
    def gripper_action_space(self):
        """Dynamically return current controller's gripper action space"""
        return self.get_current_controller().gripper_action_space
        
    def get_name(self):
        return "pi0-replayer-mixed"
        
    def reset_state(self):
        """Reset both controllers and internal state"""
        self.pi0_controller.reset_state()
        self.replayer_controller.reset_state()
        self.pi0_controller.stop_policy()
        self.dinox_switch_counter = 1
        self._switching_in_progress = False
        self.detect_streak = 0
        
        self._state = {
            "success": False,
            "failure": False,
            "movement_enabled": True,
            "controller_on": True,
            "switch_label": 1,  # 0: PI0, 1: Replayer, 0.5: Transitioning
            "current_controller": 1,  
            "t_step": 0,
            "switch_at": 0
        }
        
        if self._on_switch_callback:
            current_controller = self.replayer_controller
            blue_print(f"Current controller: {current_controller.get_name()}")
            blue_print(f"Action space: {current_controller.action_space}")
            blue_print(f"Gripper action space: {current_controller.gripper_action_space}")
            self._on_switch_callback(
                current_controller.action_space,
                current_controller.gripper_action_space
            )
        
        self.set_instruction(self.init_instruction)

    def set_instruction(self, instruction):
        """Set instruction for PI0 controller"""
        self.pi0_controller.set_instruction(instruction)
        
    def detect_target(self, image_path):
        """Run DINOX detector on image to detect target object"""
        predictions = self.dinox_detector.get_dinox(image_path, self.config.detection_prompt)
        for obj in predictions:
            class_name = obj.get('category', '').lower()
            confidence = obj.get('score', 0.0)
            if (self.config.target_name in class_name and confidence >= self.config.detection_confidence):
                blue_print(f"Target detected with confidence {confidence:.2f}")
                return True

        return False
    
    def switch_controller(self):
        """Switch between PI0 and Replayer controllers"""
        with self._switch_lock:
            if not self._switching_in_progress:
                self._switching_in_progress = True
                
                # Switch controller
                if self._state["current_controller"] == 0:
                    blue_print("Stopping PI0 policy...")
                    blue_print("Preparing to switch to Replayer controller...")
                    time.sleep(0.1)  # Small delay to ensure clean switch
                    
                    self._state["switch_label"] = 1.0
                    self.pi0_controller.stop_policy()
                    self._state["current_controller"] = 1

                    blue_print("Switched to Replayer controller")
                else:
                    blue_print("Preparing to switch to PI0 controller...")
                    time.sleep(0.1)  # Small delay to ensure clean switch
                    
                    self._state["switch_label"] = 0.0
                    blue_print("Starting PI0 policy...")
                    self.pi0_controller.start_policy()
                    
                    self._state["current_controller"] = 0
                    blue_print("Switched to PI0 controller")
                    self._state["switch_at"] = self._state["t_step"]

                # Notify runner about action space change
                if self._on_switch_callback:
                    current_controller = self.get_current_controller()
                    blue_print(f"Current controller: {current_controller.get_name()}")
                    blue_print(f"Action space: {current_controller.action_space}")
                    blue_print(f"Gripper action space: {current_controller.gripper_action_space}")
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
            self.replayer_controller.register_key(key)
            
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
        return self.pi0_controller if self._state["current_controller"] == 0 else self.replayer_controller
    
    def forward(self, observation):
        """Forward observation to current controller"""
        # blue_print(self.dinox_switch_counter, "================")
        # blue_print_datadict_shape(observation, indent=4, save_data=True)
        if self._state["t_step"] == 0:
            print("ReplayPi0 == Starting...")
        blue_print("dinox_switch_counter===", self.dinox_switch_counter)
        if self.dinox_switch_counter % self.switch_interval == 0 and self._state["current_controller"] == 1:
            # Get the image from observation
            # import ipdb; ipdb.set_trace()
            # blue_print("!!!!!DINOX switch triggered!!!!!")
            img_array = observation["image"][f"{self.config.wrist_camera_id}_left"]
            # Convert BGR to RGB before creating PIL Image
            # OpenCV uses BGR by default, PIL expects RGB
            img_array_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_array_rgb)
            temp_pth = "./temp.jpg"
            img.save(temp_pth)
            if self.detect_target(temp_pth):                   # ① interval success
                self.detect_streak += 1                          #   bump streak
                blue_print(f"[DETECT] success (streak={self.detect_streak})")

                if self.detect_streak == 2:                      # ② two in a row
                    blue_print("***** TWO HITS - SWITCH! *****")
                    self.switch_controller()
                    self.detect_streak = 0                       # ③ reset streak
            else:                                                # ④ interval fail
                blue_print("[DETECT] fail -streak reset")
                self.detect_streak = 0                           #   wipe streak



           
        self.dinox_switch_counter += 1
            
        with self._switch_lock:
            if self._state["current_controller"] == 0: # PI0 controller， 8 joint + 1 gripper
                # blue_print('pi00000000')
                
                action, info = self.pi0_controller.forward(observation)
                info["joint_velocity"] = action[:7]
                info["gripper_velocity"] = action[-1]
                action = np.clip(action, -1, 1)

            else: # Replayer controller, cartesian position + 1 gripper
                time.sleep(0.5)
                action, info = self.replayer_controller.forward(observation)
                # blue_print("REPLAY_PI0 == replay action & info", action, info)
                info["cartesian_position"] = action[:6]
                info["gripper_position"] = action[-1]

            if info is None:
                blue_print("ReplayPi0 == INFO IS NONE == ")
            
            # Ensure action is float32
            action = np.array(action, dtype=np.float32)

            # blue_print_datadict_shape(info, indent=4, save_data=True)
            # blue_print("process action", action)
            self._state["t_step"] += 1
            return action, info
    
    def close(self):
        """Clean up both controllers"""
        self.pi0_controller.close()
        self.replayer_controller.close()
        if os.path.exists("./temp.jpg"):
            os.remove("./temp.jpg")