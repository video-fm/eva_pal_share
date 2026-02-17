
import numpy as np
import time
from eva.utils.trajectory_utils import TrajectoryReader


class Replayer:
    def __init__(self, traj_path, action_space="cartesian_velocity", gripper_action_space="position"):
        self.action_space = "cartesian_position"
        self.gripper_action_space = "position"
        self.traj_path = traj_path
        if traj_path.endswith(".npz"):
            print(traj_path, "+++++++++++++++++++++++")
            if self.action_space == "cartesian_velocity":
                print("cartesian_velocity loading from npz")
                self.traj = np.load(traj_path)["actions_vel"]
                
            elif self.action_space == "cartesian_position":
                print("cartesian_position loading from npz")
                self.traj = np.load(traj_path)["actions_pos"]
                # import ipdb; ipdb.set_trace()

        elif traj_path.endswith(".npy"):
            print("npy loading")
            self.traj = np.load(traj_path)

        elif traj_path.endswith(".h5"):
            
            print(traj_path, "+h55555+")
            traj_reader = TrajectoryReader(traj_path, read_images=False)
            self.traj = []
            for i in range(traj_reader.length()):
                timestep = traj_reader.read_timestep()
                arm_action = timestep["observation"]["robot_state"]["cartesian_position"]
                gripper_action = timestep["observation"]["robot_state"]["gripper_position"]
                if 1: #not timestep["observation"]["timestamp"]["skip_action"]:
                    self.traj.append(np.concatenate([arm_action, [gripper_action]]))
            self.traj = np.array(self.traj)
        else:
            raise ValueError(f"Invalid trajectory format: {traj_path}")
        
        self.traj_len = self.traj.shape[0]
        self.delay = 0
        self.t = 0
        self._state = {
            "success": False,
            "failure": False,
            "movement_enabled": True,
            "controller_on": True,
            "t_step": 0,
        }
        self.traj_len = self.traj.shape[0]

    def get_name(self):
        return "Replayer"
    
    def get_policy_name(self):
        return str(self.traj_path)
        
    def reset_state(self):
        self.t = 0
        self.delay = 0
        self._state["success"] = False
        self._state["failure"] = False
        self._state["movement_enabled"] = True
        self._state["controller_on"] = True
        self._state["t_step"] = 0
         
    
    def register_key(self, key):
        if key == ord(" "):
            pass
            # self._state["movement_enabled"] = not self._state["movement_enabled"]
            # print("Movement enabled:", self._state["movement_enabled"])
        elif key == ord("y"):
            self._state["success"] = True
        elif key == ord("n"):
            self._state["failure"] = True
            
    def get_info(self):
        return self._state

    def forward(self, observation):
        # print("Movement enabled:", self._state["movement_enabled"])
        cur_ee_pos = np.zeros((7,))
        cur_ee_pos[:6] = observation["robot_state"]["cartesian_position"]
        time.sleep(0.3)
        if self.delay > 0 or self.t >= self.traj_len:
            # TODO implement cartesian movement
            if self.action_space == "cartesian_velocity":
                action = np.zeros((7,))
            elif self.action_space == "cartesian_position":
                action = cur_ee_pos
                
            self.delay -= 1
        else:
            action = self.traj[self.t]
            self.t += 1
            if self.t >= self.traj_len:
                self._state["success"] = True
            
        # if self.t % 10 == 0:
        #     input(f"REPLAYER == Press Enter to continue... {self.t}/{self.traj_len}")
        # Print current end-effector pose (xyz position)
        print(f"EEF Position (xyz): [{cur_ee_pos[0]:.4f}, {cur_ee_pos[1]:.4f}, {cur_ee_pos[2]:.4f}]")
        return action, {}

    
    def close(self):

        pass