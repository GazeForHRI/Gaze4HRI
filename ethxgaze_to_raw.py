import os
import csv
import numpy as np

import config
from data_loader import GazeDataLoader
from unrectification import unrectify_gaze_vectors


def pitch_yaw_to_unit_vector(pitch, yaw):
    """Step 2: Convert pitch and yaw to a 3D unit vector."""
    x = -np.cos(pitch) * np.sin(yaw)
    y = -np.sin(pitch)
    z = -np.cos(pitch) * np.cos(yaw)
    return np.array([x, y, z], dtype=np.float64)


def transform_gaze_to_custom_basis(vec: np.ndarray) -> np.ndarray:
    """Step 3: Do basis transformation as in PureGaze."""
    vec = vec / np.linalg.norm(vec)
    return np.array([vec[2], -vec[0], -vec[1]])  # [z, -x, -y]


def ethxgaze_to_raw(exp_dir: str, dataset_csv_path: str, model_name="ethxgaze_to_raw"):
    """
    Reads the dataset CSV, converts labels back to rectified 3D gaze, 
    unrectifies it, and saves the output as a model estimation for evaluation.
    """
    # Load dataset labels
    gaze_rectified_list = []
    print(f"Loading dataset labels from {dataset_csv_path}...")
    with open(dataset_csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row["timestamp"])
            pitch = float(row["pitch"])
            yaw = float(row["yaw"])

            # Step 2
            vec_3d = pitch_yaw_to_unit_vector(pitch, yaw)
            
            # Step 3
            vec_custom = transform_gaze_to_custom_basis(vec_3d)

            gaze_rectified_list.append([ts, vec_custom[0], vec_custom[1], vec_custom[2]])

    gaze_estimations_rectified = np.array(gaze_rectified_list, dtype=np.float64)

    # Initialize DataLoader to fetch raw mocap data
    dataloader = GazeDataLoader(
        root_dir=exp_dir,
        target_period=config.get_target_period(),
        camera_pose_period=config.get_camera_pose_period(),
        time_diff_max=config.get_time_diff_max(),
        get_latest_subdirectory_by_name=True
    )

    actual_cwd = dataloader.get_cwd()
    print(f"Loading geometry data from {actual_cwd}...")
    
    eye_positions_world = dataloader.load_eye_positions()
    head_poses_world = dataloader.load_head_poses(frame="world")
    camera_poses_world = dataloader.load_camera_poses(frame="world")

    intrinsics_path = os.path.join(actual_cwd, "camera_intrinsics.npy")
    camera_intrinsics = np.load(intrinsics_path, allow_pickle=True).item()

    rgb_fps = config.get_rgb_fps()
    mocap_freq = config.get_mocap_freq()
    target_period_ms = 1000.0 / rgb_fps if rgb_fps > 0 else 33.33
    mocap_period_ms = 1000.0 / mocap_freq if mocap_freq > 0 else 10.0

    print("Running Unrectification (Step 4)...")
    # Step 4: Unrectify PureGaze's output
    gaze_unrectified = unrectify_gaze_vectors(
        gaze_estimations_rectified=gaze_estimations_rectified,
        head_poses_world=head_poses_world,
        eye_positions_world=eye_positions_world,
        camera_poses_world=camera_poses_world,
        camera_intrinsics=camera_intrinsics,
        target_period_ms=target_period_ms,
        mocap_period_ms=mocap_period_ms
    )

    # Save outputs as a mock "model" estimation
    output_dir = os.path.join(actual_cwd, "gaze_estimations", model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "gaze_directions.npy")
    np.save(output_path, gaze_unrectified)
    
    indices_path = os.path.join(output_dir, "gaze_directions_indices.npy")
    valid_indices = np.arange(len(gaze_unrectified))
    np.save(indices_path, valid_indices)

    print(f"Saved {len(gaze_unrectified)} unrectified vectors to {output_dir}")


if __name__ == "__main__":
    # Example execution
    experiment_dir = os.path.join(config.get_dataset_base_directory(), "2025-08-06/subj_xxxx/line_movement_fast/horizontal")
    csv_file_path = "./rectified_hri_dataset/labels.csv"
    
    ethxgaze_to_raw(experiment_dir, csv_file_path, model_name="ethxgaze_to_raw")