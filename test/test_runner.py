import time
import numpy as np
from eva.runner import Runner
from eva.env import FrankaEnv
from collections import defaultdict

def main():
    print("Initializing environment...")
    camera_kwargs = defaultdict(lambda: {"depth": False, "pointcloud": False})
    env = FrankaEnv(camera_kwargs=camera_kwargs)
    
    print("Initializing runner...")
    runner = Runner(env=env, controller="spacemouse", save_data=False, post_process=False)
    try:
        print("\nSetting action space...")
        action_space = "joint_position"  # or "joint_position", "cartesian_velocity", "joint_velocity"
        runner.set_action_space(action_space)
        print("\n1. Getting robot state...")
        state = runner.get_state()
        print(f"Robot state: {state}")
        
        print("\n2. Getting observations...")
        obs = runner.get_obs()
        print(f"Observations keys: {obs.keys()}")
        
        print("\n3. Testing action execution...")
        # Create a zero action based on action space
        dof = 7 if "cartesian" in action_space else 8
        zero_action = np.zeros(dof, dtype=np.float32)
        
        print("Action type:", type(zero_action))
        print("Action shape:", zero_action.shape)
        print("Action:", zero_action)
        
        print("Applying zero action...")
        runner.apply_action(zero_action)
        
        print("\n4. Testing camera feed...")
        images, cam_ids = runner.get_camera_feed()
        print(f"Number of cameras: {len(images)}")
        print(f"Camera IDs: {cam_ids}")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nClosing runner...")
        runner.close()

if __name__ == "__main__":
    main() 