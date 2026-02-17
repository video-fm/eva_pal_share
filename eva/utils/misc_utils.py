import time
import multiprocessing
import subprocess
import threading
from pathlib import Path
import torch
import os
import glob
import numpy as np
import datetime

data_dir = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data"))

def time_ms():
    return time.time_ns() // 1_000_000

def now_hms():
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

def run_terminal_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True, executable="/bin/bash", encoding="utf8")
    return process

def run_threaded_command(command, args=(), daemon=True):
    thread = threading.Thread(target=command, args=args, daemon=daemon)
    thread.start()
    return thread

def run_multiprocessed_command(command, args=()):
    process = multiprocessing.Process(target=command, args=args)
    process.start()
    return process

def get_latest_trajectory():
    data_dirs = glob.glob(str(data_dir) + "*/**/", recursive=True)
    data_dirs = [d for d in data_dirs if os.path.exists(os.path.join(d, "trajectory.h5"))]
    data_dirs.sort(key=os.path.getmtime)
    data_dirs = data_dirs[-1:]
    return data_dirs[0]

def get_latest_image():
    data_dirs = glob.glob(str(data_dir) + "/images/*")
    data_dirs.sort(key=os.path.getmtime)
    data_dirs = data_dirs[-1:]
    return data_dirs[0]

def print_datadict_tree(datadict, indent=4, save_data=False):
    """Pretty prints a nested dictionary structure showing robot observation data.
    
    Args:
        datadict: Dictionary containing robot observation data
        indent: Current indentation level (default: 0)
    """
    for key, value in datadict.items():
        print(" " * indent + str(key))
        if isinstance(value, dict):
            print_datadict_tree(value, indent + 2)
        elif isinstance(value, (list, tuple, set)):
            print(" " * (indent + 2) + str(type(value).__name__) + " of length " + str(len(value)))
        else:
            print(" " * (indent + 2) + str(type(value).__name__) + ": " + str(value))
    if save_data:
        with open("data_tree.txt", "w") as f:
            f.write(str(datadict))
    
def print_datadict_shape(datadict, indent=4, save_data=False):
    for key, value in datadict.items():
        print(" " * indent + str(key))
        if isinstance(value, dict):
            print_datadict_shape(value, indent + 2)
        elif isinstance(value, (list, tuple, set)):
            print(" " * (indent + 2) + str(type(value).__name__) + " of length " + str(len(value)))
        elif isinstance(value, (np.ndarray, torch.Tensor)):
            print(" " * (indent + 2) + str(type(value).__name__) + ": " + str(value.shape))
        else:
            print(" " * (indent + 2) + str(type(value).__name__))

    if save_data:
        with open("data_shape.txt", "w") as f:
            f.write(str(datadict))


def create_info_dict(obs_dict, state, gripper_action = 0, grip_vel = 0):
    info_dict ={
        "cartesian_position": obs_dict["robot_state"]["cartesian_position"],
        "joint_position": obs_dict["robot_state"]["joint_positions"],
        "cartesian_velocity": np.zeros_like(obs_dict["robot_state"]["cartesian_position"]).astype(np.float32), #TODO check if this is correct
        "joint_velocity": np.zeros_like(obs_dict["robot_state"]["joint_positions"]).astype(np.float32),
        "gripper_position": float(gripper_action),
        "gripper_velocity": float(grip_vel), 
        "gripper_delta": 0.0, # TODO check if this is correct 
        "controller_info": {
            "success": bool(state["success"]),
            "failure": bool(state["failure"]),
            "movement_enabled": bool(state["movement_enabled"]),
            "controller_on": bool(state["controller_on"]),
        },
        "timestamp": {"skip_action": False} # Only negative under obs.timestampe.skip_action 
    }
    return info_dict
        

# ANSI Color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
PURPLE = '\033[95m'
CYAN = '\033[96m'
END_COLOR = '\033[0m'

def colored_print(color, *args, **kwargs):
    """Print function that outputs text in specified color
    
    Args:
        color: ANSI color code to use
        *args: Arguments to pass to print
        **kwargs: Keyword arguments to pass to print
    """
    print(f"{color}", end="")
    print(*args, **kwargs)
    print(f"{END_COLOR}", end="")

def red_print(*args, **kwargs):
    colored_print(RED, *args, **kwargs)

def green_print(*args, **kwargs):
    colored_print(GREEN, *args, **kwargs)

def yellow_print(*args, **kwargs):
    colored_print(YELLOW, *args, **kwargs)

def blue_print(*args, **kwargs):
    colored_print(BLUE, *args, **kwargs)

def purple_print(*args, **kwargs):
    colored_print(PURPLE, *args, **kwargs)

def cyan_print(*args, **kwargs):
    colored_print(CYAN, *args, **kwargs)