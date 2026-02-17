# Eva Franka Infrastructure

Eva is a simple, modular, and extendable Franka infrastructure built on [DROID](https://github.com/droid-dataset/droid). Key features include:
- Modular design with atomic components, making it configurable and extendable.
- Lightweight and simple interface via the terminal and a camera feed window.
- Clean and organized code, streamlining future development.

## Installation
The DROID software and hardware setup form the foundation for Eva. Please install them following the instructions [here](https://droid-dataset.github.io/droid/).

## Usage

Following the DROID setup, Eva runs on two machines:
- NUC: Handles low-level control of the Franka Emika with a server built on [Polymetis](https://facebookresearch.github.io/fairo/polymetis/).
- Laptop: Handles high-level logic (policy inference, teleoperation, etc) with a runner that executes user scripts.

We recommend the following tmux setup:
```
+-------------------------+-------------------------+
|                         |                         |
|      Server (NUC)       |     Runner (Laptop)     |
+-------------------------+                         |
|    Scripts (Laptop)     |                         |
|                         |                         |
+-------------------------+-------------------------+
```

### Startup

1. On the NUC, run
```bash
cd eva/eva/robot
./launch_server.sh
```
2. On the laptop, run
```bash
conda activate eva
cd eva/scripts
python start_runner.py
```

### Scripts

After the server and runner are started, you can execute scripts found in `eva/scripts/`. Some of the main functions include:
- `collect_trajectory.py`: Collects teleoperated trajectories saved in `eva/data/`.
- `play_trajectory.py`: Replays a selected trajectory.
- `process_trajectory.py`: Processes the compressed trajectory data into a more usable format.
- `calibrate_camera.py`: Calibrates a camera using the Charuco board.
- `check_calibration.py`: Overlays a gripper annotation on the camera feed.
- `take_pictures.py`: Saves camera pictures to `eva/data/images`.
- `reset_robot.py`: Resets the robot pose to default.


### Controller Support

Eva supports the following controllers:
#### Oculus Quest 2 VR
- Classic DROID controller, map actions to the left controller.
- 
#### Keyboard
#### Gello
#### 3Dconnexion SpaceMouse
    - Move to control position
    - Twist to control orientation
    - Left button: toggle gripper
    - Right button: None
    - 'y' key: success
    - 'n' key: failure
    - 'space' key: reset origin
### Development

Code development should be entirely done on the laptop, and to sync the codebase with the NUC, run `./sync_infra.sh`. Remember to restart the server or runner if code changes affect them.

If you are using Eva and plan to make significant changes, **please work in a copy of this directory** (eg, `eva_wliang`).

### Data Transfer between Franka Laptop and EXX

To train model on data collected from the Franka laptop, run the following command:
```bash
./send_data_to_exx.sh /path/to/data
```

