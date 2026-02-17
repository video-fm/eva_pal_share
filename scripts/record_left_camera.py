#!/usr/bin/env python3
"""
Record video from the left camera (ID: 26368109)

Usage:
    python scripts/record_left_camera.py                    # Record for 60 seconds (default)
    python scripts/record_left_camera.py --duration 30      # Record for 30 seconds
    python scripts/record_left_camera.py --output video.mp4 # Custom output file
    python scripts/record_left_camera.py --svo              # Record as SVO2 (ZED native format)

Press 'q' to stop recording early.
"""

import argparse
import cv2
import time
import os
from datetime import datetime
from pathlib import Path

import pyzed.sl as sl

# Left camera ID
LEFT_CAMERA_ID = "26368109"


def find_camera_by_serial(serial_number: str):
    """Find a ZED camera by its serial number."""
    cameras = sl.Camera.get_device_list()
    for cam in cameras:
        if str(cam.serial_number) == serial_number:
            return cam
    return None


def record_to_mp4(output_path: str, duration: float, fps: int = 10, resolution: str = "720"):
    """Record video from the left camera to MP4 format."""
    
    # Initialize camera
    camera = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_serial_number(int(LEFT_CAMERA_ID))
    
    # Set resolution
    if resolution == "720":
        init_params.camera_resolution = sl.RESOLUTION.HD720
    elif resolution == "1080":
        init_params.camera_resolution = sl.RESOLUTION.HD1080
    elif resolution == "2k":
        init_params.camera_resolution = sl.RESOLUTION.HD2K
    else:
        init_params.camera_resolution = sl.RESOLUTION.HD720
    
    init_params.camera_fps = fps
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # No depth needed for recording
    init_params.camera_image_flip = sl.FLIP_MODE.ON  # Flip 180 degrees (camera is mounted upside down)
    
    # Open camera
    status = camera.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open camera {LEFT_CAMERA_ID}: {status}")
        return False
    
    # Get actual resolution
    cam_info = camera.get_camera_information()
    width = cam_info.camera_configuration.resolution.width
    height = cam_info.camera_configuration.resolution.height
    actual_fps = cam_info.camera_configuration.fps
    
    print(f"Camera opened: {width}x{height} @ {actual_fps} FPS")
    print(f"Recording to: {output_path}")
    print(f"Duration: {duration} seconds (press 'q' to stop early)")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, actual_fps, (width, height))
    
    # Initialize image container
    left_image = sl.Mat()
    runtime = sl.RuntimeParameters()
    
    # Recording loop
    start_time = time.time()
    frame_count = 0
    
    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                print(f"\nRecording complete: {duration} seconds")
                break
            
            # Grab frame
            if camera.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                camera.retrieve_image(left_image, sl.VIEW.LEFT)
                frame = left_image.get_data()
                
                # Convert BGRA to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                video_writer.write(frame_bgr)
                frame_count += 1
                
                # Display preview
                preview = cv2.resize(frame_bgr, (640, 360))
                cv2.putText(preview, f"Recording: {elapsed:.1f}s / {duration}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(preview, f"Frames: {frame_count}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Recording - Press 'q' to stop", preview)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(f"\nRecording stopped by user at {elapsed:.1f}s")
                    break
                    
    except KeyboardInterrupt:
        print("\nRecording interrupted")
    finally:
        video_writer.release()
        camera.close()
        cv2.destroyAllWindows()
    
    print(f"Saved {frame_count} frames to {output_path}")
    return True


def record_to_svo(output_path: str, duration: float, fps: int = 30, resolution: str = "720"):
    """Record video from the left camera to SVO2 format (ZED native)."""
    
    # Ensure correct extension
    if not output_path.endswith(".svo2"):
        output_path = output_path.rsplit(".", 1)[0] + ".svo2"
    
    # Initialize camera
    camera = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_serial_number(int(LEFT_CAMERA_ID))
    
    # Set resolution
    if resolution == "720":
        init_params.camera_resolution = sl.RESOLUTION.HD720
    elif resolution == "1080":
        init_params.camera_resolution = sl.RESOLUTION.HD1080
    elif resolution == "2k":
        init_params.camera_resolution = sl.RESOLUTION.HD2K
    else:
        init_params.camera_resolution = sl.RESOLUTION.HD720
    
    init_params.camera_fps = fps
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.camera_image_flip = sl.FLIP_MODE.ON  # Flip 180 degrees (camera is mounted upside down)
    
    # Open camera
    status = camera.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open camera {LEFT_CAMERA_ID}: {status}")
        return False
    
    # Get actual resolution
    cam_info = camera.get_camera_information()
    width = cam_info.camera_configuration.resolution.width
    height = cam_info.camera_configuration.resolution.height
    actual_fps = cam_info.camera_configuration.fps
    
    print(f"Camera opened: {width}x{height} @ {actual_fps} FPS")
    print(f"Recording to: {output_path}")
    print(f"Duration: {duration} seconds (press 'q' to stop early)")
    
    # Start recording
    recording_params = sl.RecordingParameters(output_path, sl.SVO_COMPRESSION_MODE.H265)
    status = camera.enable_recording(recording_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to start recording: {status}")
        camera.close()
        return False
    
    # Initialize image container for preview
    left_image = sl.Mat()
    runtime = sl.RuntimeParameters()
    
    # Recording loop
    start_time = time.time()
    frame_count = 0
    
    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                print(f"\nRecording complete: {duration} seconds")
                break
            
            # Grab frame (this also records it)
            if camera.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                frame_count += 1
                
                # Display preview
                camera.retrieve_image(left_image, sl.VIEW.LEFT)
                frame = left_image.get_data()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                preview = cv2.resize(frame_bgr, (640, 360))
                cv2.putText(preview, f"Recording SVO: {elapsed:.1f}s / {duration}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(preview, f"Frames: {frame_count}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Recording SVO - Press 'q' to stop", preview)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(f"\nRecording stopped by user at {elapsed:.1f}s")
                    break
                    
    except KeyboardInterrupt:
        print("\nRecording interrupted")
    finally:
        camera.disable_recording()
        camera.close()
        cv2.destroyAllWindows()
    
    print(f"Saved {frame_count} frames to {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Record video from left camera (26368109)")
    parser.add_argument("--duration", "-d", type=float, default=30.0,
                        help="Recording duration in seconds (default: 30)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path (default: auto-generated with timestamp)")
    parser.add_argument("--svo", action="store_true",
                        help="Record to SVO2 format instead of MP4")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second (default: 30)")
    parser.add_argument("--resolution", "-r", type=str, default="720",
                        choices=["720", "1080", "2k"],
                        help="Video resolution (default: 720)")
    
    args = parser.parse_args()
    
    # Check if camera is available
    cam_info = find_camera_by_serial(LEFT_CAMERA_ID)
    if cam_info is None:
        print(f"Error: Left camera (ID: {LEFT_CAMERA_ID}) not found!")
        print("Available cameras:")
        for cam in sl.Camera.get_device_list():
            print(f"  - {cam.serial_number}")
        return
    
    print(f"Found left camera: {LEFT_CAMERA_ID}")
    
    # Generate output path if not provided
    if args.output is None:
        output_dir = Path("/home/franka/eva_tony/data/recordings")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ext = "svo2" if args.svo else "mp4"
        args.output = str(output_dir / f"left_camera_{timestamp}.{ext}")
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print("start recording now")
    # Record
    if args.svo:
        record_to_svo(args.output, args.duration, args.fps, args.resolution)
    else:
        record_to_mp4(args.output, args.duration, args.fps, args.resolution)


if __name__ == "__main__":
    main()

