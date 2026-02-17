import numpy as np
from env import FrankaEnv

"""
This script is used to test the basic control of the Franka robot using the FrankaEnv class.
It includes testing the reset functionality, robot state observation, camera system, and simple action execution.
- 01/25 tonyw v1 
"""

def test_basic_control():
    # Create environment instance using cartesian velocity control mode
    env = FrankaEnv(
        action_space="cartesian_velocity",
        camera_kwargs={},  # Camera can be configured as needed
        do_reset=True  # Reset robot on startup
    )
    
    # Test reset functionality
    print("Testing reset...")
    env.reset(randomize=False)
    
    # Get robot state
    print("\nTesting state observation...")
    state_dict, timestamps = env.get_state()
    print("Robot state:", state_dict)
    print("Timestamps:", timestamps)
    
    # Test camera system
    print("\nTesting camera system...")
    obs_dict = env.get_observation()
    print("Camera types:", obs_dict["camera_type"])
    print("Available cameras:", list(obs_dict["camera_extrinsics"].keys()))
    
    # Test simple action execution
    print("\nTesting action execution...")
    # Create a small cartesian velocity action (x, y, z, rx, ry, rz, gripper)
    action = np.zeros(7)  # Zero velocity command
    action[2] = 0.1  # Small motion in z direction
    
    # Execute action
    action_info = env.step(action)
    print("Action result:", action_info)

if __name__ == "__main__":
    test_basic_control()