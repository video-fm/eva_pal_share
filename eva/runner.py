import os
import time
from copy import deepcopy
from datetime import datetime
import cv2
import h5py
from pathlib import Path
import shutil
import threading
import numpy as np
# teleop 
from eva.controllers.occulus import Occulus
from eva.controllers.spacemouse import SpaceMouse
from eva.controllers.keyboard import Keyboard
from eva.controllers.gello import Gello
# policy
from eva.controllers.replayer import Replayer
from eva.controllers.pi0_policy import Pi0Policy
from eva.controllers.human_pi0 import DemoDiffusionPolicy

# Active Perception Series
from eva.controllers.policy import Policy # currently fixed as avg pooling aawr policy
from eva.controllers.aawr import AAWRPolicy
from eva.controllers.pi0_spacemouse import MixedController
from eva.controllers.aawr_pi0 import AAWRPi0Controller
from eva.controllers.replay_pi0 import ReplayPi0Controller

# writer & utils
from eva.utils.trajectory_utils import run_trajectory
from eva.utils.calibration_utils import calibrate_camera, check_calibration, check_calibration_info, save_calibration_info
from eva.utils.misc_utils import data_dir, run_threaded_command, print_datadict_tree
from eva.utils.parameters import hand_camera_id, code_version, robot_serial_number, robot_type

from eva.utils.misc_utils import yellow_print

class Runner:
    def __init__(self, env, controller, save_data=False, post_process=False, horizon=None):
        yellow_print("RUN === CONTROLLER === ", controller)
        self.env = env
        self.controller = None
        self.set_controller(controller)

        self.traj_running = False
        self.obs_pointer = {}
        self.horizon = horizon
        # Get Camera Info #
        self.cam_ids = list(env.camera_reader.camera_dict.keys())
        self.cam_ids.sort()

        _, full_cam_ids = self.get_camera_feed()
        self.num_cameras = len(full_cam_ids)
        self.full_cam_ids = full_cam_ids
        self.advanced_calibration = False

        self.stop_camera_feed = None
        self.display_thread = None

        # Make Sure Log Directorys Exist #
        self.success_logdir = os.path.join(data_dir, "success", datetime.now().strftime("%Y-%m-%d"))
        self.failure_logdir = os.path.join(data_dir, "failure", datetime.now().strftime("%Y-%m-%d"))
        self.eval_logdir = os.path.join(data_dir, "eval", datetime.now().strftime("%Y-%m-%d"))
        if not os.path.isdir(self.success_logdir):
            os.makedirs(self.success_logdir)
        if not os.path.isdir(self.failure_logdir):
            os.makedirs(self.failure_logdir)
        self.save_data = save_data
        self.post_process = post_process

        self.display_camera_feed()

    def reset_robot(self):
        self.env._robot.establish_connection() # Why do this?
        self.controller.reset_state()
        self.env.reset()

    def apply_action(self, action): # TODO check with will
        self.env.step(action)

    def get_controller_info(self):
        info = self.controller.get_info()
        return deepcopy(info)

    def enable_advanced_calibration(self):
        self.advanced_calibration = True
        self.env.camera_reader.enable_advanced_calibration()

    def disable_advanced_calibration(self):
        self.advanced_calibration = False
        self.env.camera_reader.disable_advanced_calibration()

    def set_calibration_mode(self, cam_id):
        self.env.camera_reader.set_calibration_mode(cam_id)

    def set_trajectory_mode(self):
        self.env.camera_reader.set_trajectory_mode()

    def run_trajectory(self, mode, reset_robot=True, wait_for_controller=True):
        info = dict(
            time=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            robot_serial_number=f"{robot_type}-{robot_serial_number}",
            version_number=code_version,
            controller=self.controller.get_name(),
        )

        if hasattr(self.controller, "current_instruction"):
            info["instruction"] = self.controller.current_instruction

        if hasattr(self.controller, "open_loop_horizon"):
            info["open_loop_horizon"] = self.controller.open_loop_horizon

        traj_name = info["time"]

        if mode == "collect":
            # Assume failure first, move to success post-run
            save_dir = os.path.join(self.failure_logdir, traj_name)
        elif mode == "evaluate":
            save_dir = os.path.join(self.eval_logdir, traj_name)
        elif mode == "practice":
            save_dir, recording_dir, save_filepath = None, None, None
        
        if save_dir is not None:
            if len(self.full_cam_ids) != 6:
                raise ValueError("WARNING: User is trying to collect data without all three cameras running!")
            recording_dir = os.path.join(save_dir, "recordings")
            save_filepath = os.path.join(save_dir, "trajectory.h5")
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(recording_dir, exist_ok=True)
            save_calibration_info(os.path.join(save_dir, "calibration.json"))

            # Save instruction to a text file if available
            if hasattr(self.controller, "current_instruction"):
                instr_file = os.path.join(save_dir, "instruction.txt")
                with open(instr_file, "w") as f:
                    f.write(self.controller.current_instruction)
                yellow_print(f"Saved instruction to {instr_file}")

        yellow_print("Saving policy name")
        policy_name = self.controller.get_policy_name()
        with open(os.path.join(save_dir, f"policy.md"), "w") as f:
            f.write(f"# Policy\n\n{policy_name}")


        self.traj_running = True
        self.env._robot.establish_connection()
        controller_info = run_trajectory( # This is from trajectory_utils.py
            self.env,
            controller=self.controller,
            horizon=self.horizon,
            metadata=info,
            obs_pointer=self.obs_pointer,
            reset_robot=reset_robot,
            recording_folderpath=recording_dir,
            save_filepath=save_filepath,
            post_process=self.post_process,
            wait_for_controller=wait_for_controller,
        )
        self.traj_running = False
        self.obs_pointer = {}

        if mode == "collect" and save_filepath is not None:
            if controller_info["success"]:
                new_save_dir = os.path.join(self.success_logdir, traj_name)
                shutil.move(save_dir, new_save_dir)
                save_dir = new_save_dir
    
    def calibrate_camera(self, cam_id, reset_robot=True):
        self.traj_running = True
        self.env._robot.establish_connection()
        success = calibrate_camera(
            self.env,
            cam_id,
            controller=self.controller,
            obs_pointer=self.obs_pointer,
            wait_for_controller=True,
            reset_robot=reset_robot,
        )
        self.traj_running = False
        self.obs_pointer = {}
        return success

    def check_calibration(self, reset_robot=True):
        self.traj_running = True
        self.env._robot.establish_connection()
        success = check_calibration(
            self.env,
            controller=self.controller,
            obs_pointer=self.obs_pointer,
            wait_for_controller=True,
            reset_robot=reset_robot
        )
        self.traj_running = False
        self.obs_pointer = {}
        return success

    def check_calibration_info(self, remove_hand_camera=False):
        info_dict = check_calibration_info(self.full_cam_ids)
        if remove_hand_camera:
            info_dict["old"] = [cam_id for cam_id in info_dict["old"] if (hand_camera_id not in cam_id)]
        return info_dict
        
    def display_camera_feed(self, camera_id=None):
        self.stop_camera_feed = threading.Event()

        self.overlay_mode = 1

        def display_thread():
            cv2.namedWindow("eva", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("eva", 1920, 720)
            cv2.setWindowProperty("eva", cv2.WND_PROP_TOPMOST, 1)
            while not self.stop_camera_feed.is_set():
                try:
                    self.camera_feed, self.cam_ids = self.get_camera_feed()
                    if camera_id is not None:
                        self.camera_feed = [feed for i, feed in enumerate(self.camera_feed) if str(camera_id) in self.cam_ids[i] ]
                except Exception as e:
                    # print("Failed to get camera feed:", e)
                    time.sleep(0.1)
                    continue
                                
                from PIL import Image
                overlay_imgs = [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ]
                for i in range(len(self.camera_feed)):
                    if overlay_imgs[i] is None:
                        continue
                    img = self.camera_feed[i]
                    overlay_img = Image.open(overlay_imgs[i])
                    overlay_img = np.array(overlay_img)
                    if self.overlay_mode == 0:
                        continue
                    elif self.overlay_mode == 1:
                        self.camera_feed[i] = cv2.addWeighted(img, 0.5, overlay_img, 0.5, 0)
                    elif self.overlay_mode == 2:
                        self.camera_feed[i] = overlay_img
                # self.camera_feed = self.camera_feed[2:]

                cols = [np.vstack(self.camera_feed[i:i+2]) for i in range(0, len(self.camera_feed), 2)]
                grid = np.hstack(cols)
                cv2.imshow("eva", cv2.cvtColor(cv2.resize(grid, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR))

                key = cv2.waitKey(1) & 0xFF
                if self.controller is not None and key != 255:
                    self.controller.register_key(key)
                    if key == ord('o'):
                        self.overlay_mode = (self.overlay_mode + 1) % 3

            cv2.destroyAllWindows()
        self.display_thread = run_threaded_command(display_thread)


    def get_gui_imgs(self, obs):
        all_cam_ids = list(obs["image"].keys())
        all_cam_ids.sort()

        gui_images = []
        for cam_id in all_cam_ids:
            img = cv2.cvtColor(obs["image"][cam_id], cv2.COLOR_BGRA2RGB)
            gui_images.append(img)
        

        return gui_images, all_cam_ids

    def get_camera_feed(self):
        if self.traj_running:
            if "image" not in self.obs_pointer:
                raise ValueError
            obs = deepcopy(self.obs_pointer)
        else:
            obs = self.env.read_cameras()[0]
        gui_images, cam_ids = self.get_gui_imgs(obs)
        return gui_images, cam_ids
    
    def get_obs(self):
        if self.traj_running:
            yellow_print("Traj mode")
            if "image" not in self.obs_pointer:
                raise ValueError
            obs = deepcopy(self.obs_pointer)
        else:
            yellow_print("Not traj mode")
            obs = self.env.read_cameras()[0]
        return obs
    
    def get_state(self):
        # TODO check what is inside
        obs = self.env.get_observation()
        return obs
        
    def get_robot_state(self): # Written by Tony
        state_dict, _ = self.env._robot.get_robot_state()
        return state_dict

    def close_camera_feed(self):
        if self.stop_camera_feed is not None and self.display_thread is not None:
            self.stop_camera_feed.set()
            self.display_thread.join()
            self.stop_camera_feed = None
            self.display_thread = None
    
    def set_action_space(self, action_space):
        self.env.set_action_space(action_space)
    
    def set_controller(self, controller, **kwargs):
        yellow_print("Setting controller:", controller)
        if controller is None:
            return
        
        def update_action_spaces(action_space, gripper_action_space):
            yellow_print(f"RUNNER == Updating action spaces - Action: {action_space}, Gripper: {gripper_action_space} ==========")
            self.env.set_action_space(action_space)
            self.env.set_gripper_action_space(gripper_action_space)
        
        
        self.prev_controller = self.controller
        if controller == "occulus":
            self.controller = Occulus()
        elif controller == "keyboard":
            self.controller = Keyboard()
        elif controller == "gello":
            self.controller = Gello()
        elif controller == "spacemouse": 
            self.controller = SpaceMouse()
        elif controller == "policy":
            self.controller = Policy(policy_path="XXX",**kwargs)
        elif controller == "replayer":
            self.controller = Replayer(**kwargs)
        elif controller == "pi0_policy":
            self.controller = Pi0Policy(**kwargs)
        elif controller == "demodiffusion_pi0":
            kwargs['on_switch_callback'] = update_action_spaces
            self.controller = DemoDiffusionPolicy(**kwargs)
        elif controller == "aawr_pi0":
            yellow_print('using aawr_pi0')
            kwargs['on_switch_callback'] = update_action_spaces
            self.controller = AAWRPi0Controller(**kwargs)
        elif controller == "replay_pi0":
            kwargs['on_switch_callback'] = update_action_spaces
            self.controller = ReplayPi0Controller(**kwargs)
        elif controller == "mixed":
            #  TODO: rename after ddl
            kwargs['on_switch_callback'] = update_action_spaces
            self.controller = MixedController(**kwargs)
        elif controller == "keyboard_pi0":
            from eva.controllers.keyboard_pi0 import KeyboardPi0
            kwargs['on_switch_callback'] = update_action_spaces
            self.controller = KeyboardPi0(**kwargs)
        else:
            raise ValueError(f"Controller {controller} not recognized!")
        yellow_print("Controller set to", self.controller.get_name(), "\n=================\n")

        # Pass env to controller if it needs robot access (e.g., for IK/action conversion)
        if hasattr(self.controller, 'set_env'):
            self.controller.set_env(self.env)

        self.env.set_action_space(self.controller.action_space)
        self.env.set_gripper_action_space(self.controller.gripper_action_space)
    


    def set_prev_controller(self):
        self.controller = self.prev_controller
        self.env.set_action_space(self.controller.action_space)
        self.env.set_gripper_action_space(self.controller.gripper_action_space)
    
    def reload_calibration(self):
        self.env.reload_calibration()
    
    def yellow_print(self, string):
        # This is used by scripts to yellow_print to the runner console instead of the script console
        # In general, we want to yellow_print everything to the runner console
        yellow_print(string)

    def set_controller_instruction(self, instruction):
        """Set instruction for the controller if it supports it."""
        if hasattr(self.controller, 'set_instruction'):
            self.controller.set_instruction(instruction)
            return True
        return False

    def close(self):
        self.reset_robot()
        self.close_camera_feed()
        self.env.close()
        self.controller.close()#
