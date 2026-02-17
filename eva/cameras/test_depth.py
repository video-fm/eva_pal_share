import json
import open3d as o3d
from pathlib import Path
from eva.cameras.zed_camera import gather_zed_cameras, ZedCamera

def load_calibration():
    """Load camera calibration data from JSON file."""
    calib_path = Path("eva/utils/calibration.json")
    with open(calib_path, 'r') as f:
        return json.load(f)

def depth_to_pointcloud(depth_image, intrinsics, scale=1000.0):
    """Convert depth image to point cloud using camera intrinsics."""
    # Create coordinate maps
    height, width = depth_image.shape
    fx = intrinsics["cameraMatrix"][0][0]
    fy = intrinsics["cameraMatrix"][1][1]
    cx = intrinsics["cameraMatrix"][0][2]
    cy = intrinsics["cameraMatrix"][1][2]
    
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    x_map, y_map = np.meshgrid(x, y)
    
    # Convert depth to 3D points
    Z = depth_image / scale  # Convert to meters
    X = (x_map - cx) * Z / fx
    Y = (y_map - cy) * Z / fy
    
    # Stack and reshape
    points = np.stack([X, Y, Z], axis=-1)
    points = points.reshape(-1, 3)
    
    # Remove invalid points (zero depth)
    mask = Z.reshape(-1) > 0
    points = points[mask]
    
    return points

def main():
    # Load calibration data
    calib_data = load_calibration()
    print("Loaded calibration data for cameras:", [k.split('_')[0] for k in calib_data.keys() if k.endswith('_left')])
    
    # Initialize cameras
    cameras = gather_zed_cameras()
    print(f"Found {len(cameras)} ZED cameras")
    
    for cam in cameras:
        print(f"\nProcessing camera {cam.serial_number}")
        
        # Configure camera for depth capture
        cam.set_reading_parameters(
            image=True,
            depth=True,
            pointcloud=True,
            concatenate_images=False,
            resolution=(1280, 720),  # HD720 resolution
            resize_func=None
        )
        
        # Set to trajectory mode (using standard parameters)
        cam.set_trajectory_mode()
        
        # Capture one frame
        data, timestamps = cam.read_camera()
        if data is None:
            print(f"Failed to read from camera {cam.serial_number}")
            continue
            
        # Get depth data
        print(data.keys())
        if 'pointcloud' in data:
            points = data['pointcloud'][f"{cam.serial_number}_left"][:, :, :3].reshape(-1, 3)
            print(points.shape)
            pcd = o3d.geometry.PointCloud()
            # import ipdb; ipdb.set_trace()
            pcd.points = o3d.utility.Vector3dVector(points)
            import ipdb; ipdb.set_trace()
            output_path = f"camera_{cam.serial_number}_pointcloud.pcd"
            o3d.io.write_point_cloud(output_path, pcd)


        # if 'depth' in data:
        #     raw_depth = data['depth'][f"{cam.serial_number}_left"]
        #     print(raw_depth.shape)
        #     depth_array = np.nan_to_num(raw_depth, nan=0.0, posinf=0.0, neginf=0.0)
        #     depth_min = np.min(depth_array)
        #     depth_max = np.max(depth_array)
        #     normalized_depth = depth_array
        #     # normalized_depth = (depth_array - depth_min) / (depth_max - depth_min) * 255
        #     # Convert to uint8 type
        #     import matplotlib.pyplot as plt
        #     plt.imshow(depth_array, cmap='viridis')
        #     plt.title(cam.serial_number)
        #     plt.show()
        #     intrinsics = cam.get_intrinsics()[f"{cam.serial_number}_right"]
            
        #     # Convert depth to point cloud
        #     points = depth_to_pointcloud(depth_array, intrinsics)
            
        #     # Create and save point cloud
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(points)
            
        #     # Save point cloud
        #     output_path = f"camera_{cam.serial_number}_pointcloud_right.pcd"
        #     o3d.io.write_point_cloud(output_path, pcd)
        #     print(f"Saved point cloud to {output_path}")
            
        #     # Print some statistics
        #     print(f"Point cloud contains {len(points)} points")
        #     print(f"Raw Depth range: {np.min(raw_depth):.2f} to {np.max(raw_depth):.2f}")
        #     print(f"Normalized Depth range: {np.min(normalized_depth):.2f} to {np.max(normalized_depth):.2f}")
        # # Disable camera
        cam.disable_camera()

if __name__ == "__main__":
    main()