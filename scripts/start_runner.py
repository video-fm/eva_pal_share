import argparse

import eva
from eva.manager import start_runner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=str, default="occulus", choices=["occulus", "keyboard", "gello", "spacemouse"])
    parser.add_argument("--record_depth", action="store_true")
    parser.add_argument("--record_pcd", action="store_true")
    parser.add_argument("--post_process", action="store_true", help="Saves data in expanded format, akin to process_trajectory")
    args = parser.parse_args()

    start_runner(args.controller, args.record_depth, args.record_pcd, args.post_process)
