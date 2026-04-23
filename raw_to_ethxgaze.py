import os
import csv
import numpy as np
import cv2
from tqdm import tqdm

import data_rectification as dpc
from data_matcher import match_regular_to_regular
from data_loader import GazeDataLoader
import config


# --- Helper Functions (From your original pipeline) ---

def convert_to_opencv_basis(v):
    """
    Converts from the standard robotics basis (X-forward, Y-left, Z-up)
    to the OpenCV basis (X-right, Y-down, Z-forward).
    """
    v = np.asarray(v)
    return np.array([-v[1], -v[2], v[0]])

def flip_yaw_180(rvec):
    """Flips the yaw angle."""
    R, _ = cv2.Rodrigues(rvec)
    R_flip_yaw = np.array([
        [-1,  0,  0],
        [ 0,  1,  0],
        [ 0,  0, -1]
    ])
    R_new = R_flip_yaw @ R
    rvec_new, _ = cv2.Rodrigues(R_new)
    return rvec_new.flatten()

def rotate_rvec_z_axis(rvec, degrees=-90):
    """Rotates the rotation vector (rvec) around the Z-axis."""
    angle_rad = np.deg2rad(degrees)
    R_z = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0,                 0,                 1]
    ])
    R_orig, _ = cv2.Rodrigues(rvec)
    R_new = R_z @ R_orig
    new_rvec, _ = cv2.Rodrigues(R_new)
    return new_rvec.flatten()


# --- Core Dataset Creation Script ---

def create_dataset_for_experiment(exp_dir: str, output_dataset_dir: str):
    """
    Reads raw Gaze4HRI data from exp_dir, rectifies images, and saves
    the dataset (images + labels.csv) into output_dataset_dir.
    """
    # 1. Setup Output Architecture
    images_dir = os.path.join(output_dataset_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    csv_path = os.path.join(output_dataset_dir, "labels.csv")

    # 2. Init DataLoader
    dataloader = GazeDataLoader(
        root_dir=exp_dir,
        target_period=config.get_target_period(),
        camera_pose_period=config.get_camera_pose_period(),
        time_diff_max=config.get_time_diff_max(),
        get_latest_subdirectory_by_name=True
    )
    actual_cwd = dataloader.get_cwd()

    print(f"Loading data from {actual_cwd}...")
    rgb_frames = dataloader.load_rgb_video(as_numpy=True)
    timestamps = dataloader.load_rgb_timestamps()
    eye_positions = dataloader.load_eye_positions()
    target_positions = dataloader.load_target_positions()
    head_poses = dataloader.load_head_poses(frame="world")
    camera_poses = dataloader.load_camera_poses(frame="world")

    # Load and format Camera Intrinsics
    intrinsics_path = os.path.join(actual_cwd, "camera_intrinsics.npy")
    camera_intrinsics = np.load(intrinsics_path, allow_pickle=True).item()
    K = np.array([
        [float(camera_intrinsics["fx"]), 0.0, float(camera_intrinsics["cx"])],
        [0.0, float(camera_intrinsics["fy"]), float(camera_intrinsics["cy"])],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # 3. Time Synchronization
    print("Synchronizing data to RGB timestamps...")
    rgb_fps = config.get_rgb_fps()
    mocap_freq = config.get_mocap_freq()
    
    rgb_period_ms = 1000.0 / rgb_fps if rgb_fps > 0 else 33.33
    mocap_period_ms = 1000.0 / mocap_freq if mocap_freq > 0 else 10.0

    _, index_array = match_regular_to_regular(
        tuple1=(timestamps.reshape(-1, 1), rgb_period_ms),
        tuple2=(eye_positions, mocap_period_ms),
        max_match_diff_ms=20.0
    )

    # 4. Rectification & Saving Routine
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "pitch", "yaw", "timestamp"])

        for i, frame in enumerate(tqdm(rgb_frames, desc="Creating Dataset")):
            if frame is None or frame.size == 0:
                continue
                
            if i >= len(index_array):
                break
                
            idx_mocap = index_array[i]

            # A. Transform Geometries from World to Camera Frame
            cam_T_world = camera_poses[idx_mocap, 1:].reshape(4, 4)
            world_T_cam = np.linalg.inv(cam_T_world)

            eye_world = np.append(eye_positions[idx_mocap, 1:4], 1.0)
            eye_cam = (world_T_cam @ eye_world)[:3]

            target_world = np.append(target_positions[idx_mocap, 1:4], 1.0)
            target_cam = (world_T_cam @ target_world)[:3]

            head_T_world = head_poses[idx_mocap, 1:].reshape(4, 4)
            head_T_cam = world_T_cam @ head_T_world
            rvec_head, _ = cv2.Rodrigues(head_T_cam[:3, :3])
            rvec_head = rvec_head.flatten()

            # B. Move to OpenCV Basis (Rectification logic requirements)
            center_i = convert_to_opencv_basis(eye_cam) * 1000.0
            target_i = convert_to_opencv_basis(target_cam) * 1000.0
            rvec_i = rotate_rvec_z_axis(flip_yaw_180(convert_to_opencv_basis(rvec_head)), -90)

            try:
                # C. Compute Normalization Math
                N = dpc.norm(
                    center=center_i,
                    gazetarget=target_i,
                    headrotvec=rvec_i,
                    imsize=(224, 224),
                    camparams=K
                )

                # D. Extract normalized frame and rectified 3D gaze
                norm_img = N.GetImage(frame) 
                g_rect_cv = N.GetGaze(scale=True)  # Native rectified normalized 3D gaze

                # E. Calculate final ETH-X Pitch & Yaw
                # dpc.GazeTo2d returns [yaw, pitch] according to ETH-X spherical representation
                yaw, pitch = dpc.GazeTo2d(g_rect_cv)

                # F. Write out the results
                ts = float(timestamps[i])
                img_name = f"frame_{i:06d}.jpg"
                img_path = os.path.join(images_dir, img_name)
                
                cv2.imwrite(img_path, norm_img)
                writer.writerow([img_name, pitch, yaw, ts])
                
            except Exception as e:
                # In case dpc.norm fails due to math instabilities (very rare edge cases)
                continue

    print(f"\nDataset generation complete! Output located at: {output_dataset_dir}")

if __name__ == "__main__":
    # Example usage execution block
    experiment_dir = os.path.join(config.get_dataset_base_directory(), "2025-08-06/subj_xxxx/line_movement_fast/horizontal")
    output_target_dir = "./rectified_hri_dataset"
    
    create_dataset_for_experiment(experiment_dir, output_target_dir)