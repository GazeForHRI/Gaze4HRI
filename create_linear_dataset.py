import os
import csv
import cv2
import math
import numpy as np
from tqdm import tqdm

import config
from data_loader import GazeDataLoader
from data_matcher import match_regular_to_regular
import data_rectification as dpc
from frame_db import _parse_exp_dir, _load_subject_meta

# --- Geometry Helper Functions ---
def convert_to_opencv_basis(v):
    v = np.asarray(v)
    return np.array([-v[1], -v[2], v[0]])

def flip_yaw_180(rvec):
    R, _ = cv2.Rodrigues(rvec)
    R_flip_yaw = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    R_new = R_flip_yaw @ R
    rvec_new, _ = cv2.Rodrigues(R_new)
    return rvec_new.flatten()

def rotate_rvec_z_axis(rvec, degrees=-90):
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

def _vec_to_pitch_yaw(x: float, y: float, z: float) -> tuple[float, float]:
    """Calculates pitch and yaw exactly as in pitch_yaw_stats.py"""
    n = math.sqrt(x*x + y*y + z*z)
    if n < 1e-12:
        return float("nan"), float("nan")
    x, y, z = x/n, y/n, z/n

    pitch_rad = math.atan2(-y, math.sqrt(x*x + z*z))

    u, v = x, z
    eps = 1e-12
    if abs(v) > eps:
        base = math.atan(u / v)
        if v > 0:
            yaw_rad = base
        else:
            yaw_rad = base + (math.pi if u >= 0 else -math.pi)
    else:
        yaw_rad = (math.pi/2) if u > 0 else ((-math.pi/2) if u < 0 else float("nan"))

    if math.isfinite(yaw_rad):
        yaw_rad = (yaw_rad + math.pi) % (2*math.pi) - math.pi

    yaw_rad += math.pi
    yaw_rad = (yaw_rad + math.pi) % (2*math.pi) - math.pi

    return -pitch_rad, -yaw_rad

# --- Main Dataset Generation ---
def create_linear_dataset(subject_dir_rel: str, output_dir: str, N: int = 7):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "dataset_manifest.csv")
    
    # Required columns requested + 4 raw Gaze4HRI attributes
    columns = [
        "subject_dir", "exp_type", "point", "gender", "birthyear",
        "glasses", "blink", "frame_idx", "exp_dir", 
        "gaze_pitch", "gaze_yaw", # same format with ETH-X-Gaze (i.e. rectified)
        "gaze4hri_raw_gaze_pitch", "gaze4hri_raw_gaze_yaw", # in the raw gaze4hri format, which is a plain vector in the camera frame, unlike the rectified ETH-X-Gaze format.
        "gaze4hri_raw_head_pitch", "gaze4hri_raw_head_yaw"
    ]
    
    # Initialize CSV
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
    
    base_dir = config.get_dataset_base_directory()
    exp_dirs = config.get_all_exp_directories_under_a_subject_directory(subject_dir_rel)
    
    total_frames_saved = 0
    
    for exp_dir_rel in tqdm(exp_dirs, desc=f"Processing {subject_dir_rel}"):
        exp_dir_abs = os.path.join(base_dir, exp_dir_rel)
        
        if not os.path.exists(exp_dir_abs):
            continue
        if config.is_experiment_directory_excluded_from_eval(exp_dir_abs):
            continue
            
        try:
            dataloader = GazeDataLoader(
                root_dir=exp_dir_abs,
                target_period=config.get_target_period(),
                camera_pose_period=config.get_camera_pose_period(),
                time_diff_max=config.get_time_diff_max(),
                get_latest_subdirectory_by_name=True
            )
            actual_cwd = dataloader.get_cwd()
            if not os.path.exists(actual_cwd):
                continue
                
            rgb_frames = dataloader.load_rgb_video(as_numpy=True)
            if not rgb_frames or len(rgb_frames) < N:
                continue
                
            timestamps = dataloader.load_rgb_timestamps()
            eye_positions = dataloader.load_eye_positions()
            target_positions = dataloader.load_target_positions()
            head_poses = dataloader.load_head_poses(frame="world")
            camera_poses = dataloader.load_camera_poses(frame="world")
            
            try:
                blink_ann = dataloader.get_blink_annotations()
            except FileNotFoundError:
                blink_ann = np.zeros(len(rgb_frames), dtype=np.int16)

            intrinsics_path = os.path.join(actual_cwd, "camera_intrinsics.npy")
            camera_intrinsics = np.load(intrinsics_path, allow_pickle=True).item()
            K = np.array([
                [float(camera_intrinsics["fx"]), 0.0, float(camera_intrinsics["cx"])],
                [0.0, float(camera_intrinsics["fy"]), float(camera_intrinsics["cy"])],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)

            # Time Synchronization
            rgb_fps = config.get_rgb_fps()
            mocap_freq = config.get_mocap_freq()
            rgb_period_ms = 1000.0 / rgb_fps if rgb_fps > 0 else 33.33
            mocap_period_ms = 1000.0 / mocap_freq if mocap_freq > 0 else 10.0

            _, index_array = match_regular_to_regular(
                tuple1=(timestamps.reshape(-1, 1), rgb_period_ms),
                tuple2=(eye_positions, mocap_period_ms),
                max_match_diff_ms=20.0
            )

            exp_images_dir = os.path.join(output_dir, "images", exp_dir_rel)
            os.makedirs(exp_images_dir, exist_ok=True)
            
            subject_dir, exp_type, point = _parse_exp_dir(exp_dir_abs)
            meta = _load_subject_meta(subject_dir)
            
            rows_to_write = []
            
            for i, frame in enumerate(rgb_frames):
                if frame is None or frame.size == 0 or i >= len(index_array):
                    continue
                    
                idx_mocap = index_array[i]

                # Geometry Transformations
                cam_T_world = camera_poses[idx_mocap, 1:].reshape(4, 4)
                world_T_cam = np.linalg.inv(cam_T_world)

                eye_world = np.append(eye_positions[idx_mocap, 1:4], 1.0)
                eye_cam = (world_T_cam @ eye_world)[:3]

                target_world = np.append(target_positions[idx_mocap, 1:4], 1.0)
                target_cam = (world_T_cam @ target_world)[:3]

                head_T_world = head_poses[idx_mocap, 1:].reshape(4, 4)
                head_T_cam = world_T_cam @ head_T_world
                
                # --- Gaze4HRI RAW Gaze & Head Computation ---
                # Raw Gaze Vector in Camera Frame
                raw_gaze_vec = target_cam - eye_cam
                raw_gaze_vec /= np.linalg.norm(raw_gaze_vec)
                gx, gy, gz = float(raw_gaze_vec[0]), float(raw_gaze_vec[1]), float(raw_gaze_vec[2])
                # Axis swap matching pitch_yaw_stats.py
                gx, gy, gz = gy, gz, gx
                raw_gaze_pitch, raw_gaze_yaw = _vec_to_pitch_yaw(gx, gy, gz)
                
                # Raw Head Vector in Camera Frame
                R_head_raw = head_T_cam[:3, :3]
                # Forward vector in raw Gaze4HRI is the X-axis [1.0, 0.0, 0.0]
                hx, hy, hz = R_head_raw @ np.array([1.0, 0.0, 0.0])
                nrm = float(np.linalg.norm([hx, hy, hz]))
                if nrm > 0:
                    hx, hy, hz = hx / nrm, hy / nrm, hz / nrm
                # Axis swap matching pitch_yaw_stats.py
                hx, hy, hz = hy, hz, hx
                raw_head_pitch, raw_head_yaw = _vec_to_pitch_yaw(hx, hy, hz)

                # --- ETH-X Rectification Computation ---
                rvec_head, _ = cv2.Rodrigues(R_head_raw)
                rvec_head = rvec_head.flatten()

                center_i = convert_to_opencv_basis(eye_cam) * 1000.0
                target_i = convert_to_opencv_basis(target_cam) * 1000.0
                rvec_i = rotate_rvec_z_axis(flip_yaw_180(convert_to_opencv_basis(rvec_head)), -90)

                try:
                    N_norm = dpc.norm(
                        center=center_i,
                        gazetarget=target_i,
                        headrotvec=rvec_i,
                        imsize=(224, 224),
                        camparams=K
                    )

                    norm_img = N_norm.GetImage(frame)
                    
                    # ETH-X Gaze Pitch & Yaw
                    g_rect_cv = N_norm.GetGaze(scale=True)
                    gaze_yaw, gaze_pitch = dpc.GazeTo2d(g_rect_cv)
                    
                    # Blink Logic
                    lab = int(blink_ann[i]) if i < len(blink_ann) else 0
                    blink_val = 1 if lab in [2, 3] else 0
                    
                    # Save Image
                    img_name = f"frame_{i:06d}.jpg"
                    img_path = os.path.join(exp_images_dir, img_name)
                    cv2.imwrite(img_path, norm_img)
                    
                    # Prepare Row Data
                    rows_to_write.append([
                        subject_dir, exp_type, point, meta.get("gender", ""), 
                        meta.get("birthyear", ""), meta.get("glasses", ""), 
                        blink_val, i, exp_dir_rel, 
                        gaze_pitch, gaze_yaw, 
                        raw_gaze_pitch, raw_gaze_yaw, 
                        raw_head_pitch, raw_head_yaw
                    ])
                    total_frames_saved += 1
                    
                except Exception as e:
                    continue
            
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows_to_write)
                
        except Exception as e:
            print(f"Skipping {exp_dir_rel} due to error: {e}")

    print(f"\nCompleted! Saved {total_frames_saved} frames to {output_dir}")

if __name__ == "__main__":
    create_linear_dataset(
        subject_dir_rel="2025-08-06/subj_xxxx",
        output_dir="./gaze4hri_debug_dataset4",
        N=7 
    )