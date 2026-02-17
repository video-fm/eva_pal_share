
import argparse
import threading
import numpy as np
import cv2

from eva.controllers.occulus import Occulus
from eva.env import FrankaEnv
from eva.runner import Runner
from eva.utils.misc_utils import run_threaded_command
from eva.manager import load_runner

def calibrate_camera(runner: Runner, camera_id, advanced_calibration=False):
    if advanced_calibration:
        runner.enable_advanced_calibration()
    runner.set_calibration_mode(camera_id)
    success = runner.calibrate_camera(camera_id, reset_robot=False)
    if success:
        print("Calibration complete!")
    else:
        print("Calibration failed")
    if advanced_calibration:
        runner.disable_advanced_calibration()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera_id", type=str)
    parser.add_argument("--advanced", action="store_true")
    args = parser.parse_args()

    runner = load_runner()
    calibrate_camera(runner, args.camera_id, args.advanced)
