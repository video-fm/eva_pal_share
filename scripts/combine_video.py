
import os
import cv2
import numpy as np
import argparse
import h5py
from tqdm import tqdm
import imageio

PATH_TO_DATA = "data/success/2026-01-29"
    
def draw_text(img, text_lines, font_scale=1.0, thickness=2, color=(255, 255, 255)):
    """Draws multiline text on an image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 50  # Start y position
    line_height = int(40 * font_scale)
    
    for line in text_lines:
        cv2.putText(img, line, (30, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += line_height
    return img

def process_episode(episode_path):
    print(f"Processing: {episode_path}")
    
    rec_path = os.path.join(episode_path, "recordings")
    traj_path = os.path.join(episode_path, "trajectory.h5")
    instr_path = os.path.join(episode_path, "instruction.txt")
    
    video_paths = {
        "tl": os.path.join(rec_path, "varied_camera_1.mp4"),
        "tr": os.path.join(rec_path, "varied_camera_2.mp4"),
        "bl": os.path.join(rec_path, "hand_camera.mp4")
    }
    
    instruction = "N/A"
    horizon = "N/A"
    policy_name = "N/A"
    total_steps = "N/A"

    if os.path.exists(instr_path):
        with open(instr_path, 'r') as f:
            instruction = f.read().strip()
    elif os.path.exists(traj_path):
        try:
            with h5py.File(traj_path, 'r') as f:
                if 'instruction' in f.attrs:
                    instruction = f.attrs['instruction']
        except Exception:
            pass

    if os.path.exists(traj_path):
        try:
            with h5py.File(traj_path, 'r') as f:
                if 'open_loop_horizon' in f.attrs:
                    horizon = str(f.attrs['open_loop_horizon'])
                if 'controller' in f.attrs:
                    policy_name = str(f.attrs['controller'])
                if 't_step' in f.attrs:
                    total_steps = str(f.attrs['t_step'])
        except Exception as e:
            print(f"Error reading HDF5: {e}")

    if not os.path.exists(video_paths["tl"]):
        print("Skipping: Main video not found")
        return

    cap = cv2.VideoCapture(video_paths["tl"])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    info_bg = np.zeros((height, width, 3), dtype=np.uint8)

    if "success" in episode_path.lower():
        status = "SUCCESS"
        status_color = (0, 255, 0)
    elif "failure" in episode_path.lower():
        status = "FAILURE"
        status_color = (0, 0, 255) # Red in BGR
    else:
        status = "UNKNOWN"
        status_color = (255, 255, 255)

    lines = [
        f"Status: {status}",
        "Controller Info",
        f"Policy: {policy_name}",
        f"Horizon: {horizon}",
        f"Total Steps: {total_steps}",
        "",
        "Instruction:",
    ]
    max_char = 40
    for i in range(0, len(instruction), max_char):
        lines.append(instruction[i:i+max_char])
    
    info_frame = draw_text(info_bg.copy(), lines, font_scale=1.5, thickness=2)
    
    info_frame = cv2.cvtColor(info_frame, cv2.COLOR_BGR2RGB)

    status_str = status.lower()
    
    safe_policy = policy_name.replace("-", "_").replace(" ", "_")
    safe_policy = "".join([c for c in safe_policy if c.isalnum() or c == "_"])

    safe_instruction = instruction.replace(" ", "_")
    safe_instruction = "".join([c for c in safe_instruction if c.isalnum() or c == "_"])

    if len(safe_instruction) > 80:
        safe_instruction = safe_instruction[:80]

    timestamp = os.path.basename(episode_path)

    filename = f"{status_str}_{safe_policy}_{safe_instruction}_{timestamp}.mp4"
    output_path = os.path.join(episode_path, filename)
    
    if os.path.exists(output_path):
        print(f"Skipping {episode_path}: {filename} already exists.")
        return

    try:
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p')
    except Exception as e:
        print(f"Failed to init libx264 writer, trying mjpeg: {e}")
        writer = imageio.get_writer(output_path, fps=fps, codec='mjpeg')

    caps = {k: cv2.VideoCapture(p) if os.path.exists(p) else None for k, p in video_paths.items()}

    pbar = tqdm(total=total_frames, desc=f"Rendering {os.path.basename(episode_path)}")
    
    while True:
        frames = {}
        ret_all = True
        
        for key in ["tl", "tr", "bl"]:
            if caps[key] is not None:
                ret, frame = caps[key].read()
                if not ret:
                    ret_all = False
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                else:
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                    # Convert BGR to RGB for imageio
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames[key] = frame
            else:
                frames[key] = np.zeros((height, width, 3), dtype=np.uint8)

        if not caps["tl"].isOpened() or not ret_all:
             if not caps["tl"].isOpened() or caps["tl"].get(cv2.CAP_PROP_POS_FRAMES) >= total_frames:
                 break

        # Assemble Grid (Swapped BL/Info)
        # Top: Varied 1 | Varied 2
        top_row = np.hstack((frames["tl"], frames["tr"]))
        # Bottom: Info | Hand
        bottom_row = np.hstack((info_frame, frames["bl"])) 
        
        grid = np.vstack((top_row, bottom_row))
        
        writer.append_data(grid)
        pbar.update(1)

    pbar.close()
    for cap in caps.values():
        if cap: cap.release()
    writer.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine episode videos into a 2x2 grid.")
    path = PATH_TO_DATA
    
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        exit(1)
        
    if os.path.exists(os.path.join(path, "trajectory.h5")):
        process_episode(path)
    else:
        subdirs = sorted([os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
        
        count = 0
        for d in subdirs:
            if os.path.exists(os.path.join(d, "trajectory.h5")):
                process_episode(d)
                count += 1
        
        if count == 0:
            print("No valid episodes found (looking for folders containing trajectory.h5)")
