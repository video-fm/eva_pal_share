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
- TODO: add Gello support

#### 3Dconnexion SpaceMouse

**Hardware:** 3Dconnexion SpaceMouse Compact (VID `0x256f`, PID `0xc635`). Verify with `lsusb`.

**Dependency:** `hidapi` (included in pixi.toml). The HID device requires exclusive access — only one process can open it at a time.

**Controls:**
| Input | Action |
|---|---|
| Move mouse | Translate end-effector |
| Twist mouse | Rotate end-effector |
| Left button (hold >0.5s) | Toggle gripper |
| Right button | Reset robot |
| `y` key | Mark trajectory as success |
| `n` key | Mark trajectory as failure |
| `space` key | Reset SpaceMouse origin |

**Keyboard macros** (available in both teleop and data collection):
| Key | Macro |
|---|---|
| `w` / `s` | Forward / backward |
| `a` / `d` | Left / right |
| `1` / `2` | Tilt up / down |
| `5` / `6` | Roll CCW / CW |
| `7` / `8` | Rotate left / right |

**Tuning parameters with the teleop script:**

Use `scripts/spacemouse_teleop.py` to interactively tune SpaceMouse hyperparameters before committing them to `parameters.py`. The script shows a live GUI with wrist + external camera feeds. Notice this script is only used for tuning, not for data collection.

```bash
# Use default config from parameters.py (when SPACEMOUSE_OVERRIDE_CONFIG=True)
python scripts/spacemouse_teleop.py

# All available tuning flags
python scripts/spacemouse_teleop.py \
    --pos_sensitivity 8.0 \
    --rot_sensitivity 8.0 \
    --action_scale 0.1 \
    --deadzone 0.05 \
    --smoothing 0.3 \
    --max_lin_vel 5.0 \
    --max_rot_vel 5.0 \
    --max_gripper_vel 5.0 \
    --external_camera right
```

Additional teleop keys: `g` toggle gripper, `r` reset robot, `i` set instruction overlay, `q`/`ESC` quit.

Config priority: `parameters.py` base (if `SPACEMOUSE_OVERRIDE_CONFIG=True`) → CLI args override on top.

**Collecting data with SpaceMouse + Pi0 mixed controller:**

`scripts/collect_pi0_spacemouse.py` runs a mixed-mode controller that supports runtime switching between a Pi0 policy and SpaceMouse teleoperation.

```bash
# Collect 10 trajectories (default)
python scripts/collect_pi0_spacemouse.py

# Collect N trajectories
python scripts/collect_pi0_spacemouse.py --n 20
```

Controls: `=` switch between Pi0 and SpaceMouse, `space` toggle movement, `y` success, `n` failure.

### Development

Code development should be entirely done on the laptop, and to sync the codebase with the NUC, run `./sync_infra.sh`. Remember to restart the server or runner if code changes affect them.

If you are using Eva and plan to make significant changes, **please work in a copy of this directory** (eg, `eva_wliang`).

### Data Transfer between Franka Laptop and GPU Cluster

To train model on data collected from the Franka laptop, run the following command:
```bash
./send_data_to_cluster.sh /path/to/data
```

