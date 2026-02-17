
from eva.manager import load_runner
import sys
if __name__ == "__main__":
    runner = load_runner(manager=False, controller="keyboard", record_depth=False, record_pcd=False, post_process=False)
    runner.reset_robot()