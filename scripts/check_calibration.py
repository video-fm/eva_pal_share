
import argparse
import threading
import numpy as np
import cv2

from eva.controllers.occulus import Occulus
from eva.env import FrankaEnv
from eva.runner import Runner
from eva.utils.misc_utils import run_threaded_command
from eva.manager import load_runner


def check_calibration(runner: Runner):
    print("Annotating end effector pose in camera feed...")
    runner.reload_calibration()
    runner.check_calibration()


if __name__ == "__main__":
    runner = load_runner()
    check_calibration(runner)
