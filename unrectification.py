import numpy as np
import cv2
import os
from tqdm import tqdm

import data_rectification as dpc
from data_matcher import match_regular_to_regular
from data_loader import GazeDataLoader
import config

# --------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------

def convert_to_opencv_basis(v):
    """
    Converts a vector from the standard robotics/experiment basis (X-forward, Y-left, Z-up)
    to the OpenCV basis (X-right, Y-down, Z-forward).
    Transformation: [x, y, z] -> [-y, -z, x]
    """
    v = np.asarray(v)
    return np.array([-v[1], -v[2], v[0]])

def convert_from_opencv_basis(v):
    """
    Converts a vector from the OpenCV basis (X-right, Y-down, Z-forward)
    back to the standard robotics/experiment basis.
    Transformation: [x, y, z] -> [z, -x, -y] 
    """
    v = np.asarray(v)
    return np.array([v[2], -v[0], -v[1]])

def flip_yaw_180(rvec):
    """
    Rotates the rotation vector rvec by 180° around the Y-axis.
    """
    R, _ = cv2.Rodrigues(rvec)
    R_flip_yaw = np.array([
        [-1, 0,  0],
        [ 0, 1,  0],
        [ 0, 0, -1]
    ])
    R_new = R_flip_yaw @ R
    new_rvec, _ = cv2.Rodrigues(R_new)
    return new_rvec.flatten()

def rotate_rvec_z_axis(rvec, degrees=-90):
    """
    Rotates the rotation vector (rvec) around the Z-axis by a specified angle.
    """
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

def _as_camparams(ci):
    if not isinstance(ci, dict):
        raise TypeError(f"Expected dict for camera intrinsics, got {type(ci)}")
    
    fx = float(ci["fx"])
    fy = float(ci["fy"])
    cx = float(ci["cx"])
    cy = float(ci["cy"])
    
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    dist = np.array(ci.get("distortion_coeffs", np.zeros(5)), dtype=np.float64).ravel()
    return {"mtx": K, "dist": dist}

# --------------------------------------------------------------------------------
# Core Logic
# --------------------------------------------------------------------------------

def unrectify_gaze_vectors(
    gaze_estimations_rectified: np.ndarray,
    head_poses_world: np.ndarray,
    eye_positions_world: np.ndarray,
    camera_poses_world: np.ndarray,
    camera_intrinsics: dict,
    target_period_ms: float,
    mocap_period_ms: float
) -> np.ndarray:
    """
    Applies the INVERSE of the data rectification transformation to gaze vectors.
    """
    
    # 1. Match Mocap Data
    gaze_ts = gaze_estimations_rectified[:, 0].reshape(-1, 1)
    
    # Map from gaze_row_index -> mocap_index
    # Note: match_regular_to_regular uses periods only for stable start detection if needed,
    # but primarily relies on timestamp matching.
    _, idx_eye = match_regular_to_regular((gaze_ts, target_period_ms), (eye_positions_world, mocap_period_ms), max_match_diff_ms=20.0)
    _, idx_head = match_regular_to_regular((gaze_ts, target_period_ms), (head_poses_world, mocap_period_ms), max_match_diff_ms=20.0)
    _, idx_cam = match_regular_to_regular((gaze_ts, target_period_ms), (camera_poses_world, mocap_period_ms), max_match_diff_ms=20.0)
    
    map_eye = {g: m for g, m in zip(_, idx_eye)}
    map_head = {g: m for g, m in zip(_, idx_head)}
    map_cam = {g: m for g, m in zip(_, idx_cam)}

    output_gaze = []
    cam_params_formatted = _as_camparams(camera_intrinsics)
    
    # Use tqdm only if batch processing calls this directly, but usually batch handles tqdm.
    # We'll print a summary line instead.
    # print(f"  Unrectifying {len(gaze_estimations_rectified)} frames...")
    
    for i in range(len(gaze_estimations_rectified)):
        if i not in map_eye or i not in map_head or i not in map_cam:
            continue
            
        # Get raw mocap data
        eye_world = eye_positions_world[map_eye[i], 1:4]
        head_world_flat = head_poses_world[map_head[i], 1:]
        cam_world_flat = camera_poses_world[map_cam[i], 1:]
        
        # 2. Transform Geometry to Camera Frame
        # cam_world_flat is Cam_T_World (Pose of Camera in World).
        # We need World_T_Cam (Matrix to transform World points to Camera).
        cam_T_world_mat = cam_world_flat.reshape(4, 4)
        transformation_mat = np.linalg.inv(cam_T_world_mat) 
        
        # Eye in Camera
        eye_h = np.append(eye_world, 1.0)
        eye_cam = (transformation_mat @ eye_h)[:3]
        
        # Head in Camera
        head_T_world_mat = head_world_flat.reshape(4, 4)
        head_T_cam_mat = transformation_mat @ head_T_world_mat
        
        # Head Rotation Vector in Camera
        R_head = head_T_cam_mat[:3, :3]
        rvec_head, _ = cv2.Rodrigues(R_head)
        rvec_head = rvec_head.flatten()
        
        # 3. Setup dpc.norm Inputs (Must be in OpenCV Basis)
        # Convert Eye Camera (ROS) -> Eye Camera (CV) and meters -> mm
        center_i = convert_to_opencv_basis(eye_cam) * 1000.0
        
        # Convert Head Rotation (ROS) -> Head Rotation (CV) and apply rectification adjustments
        rvec_opencv = convert_to_opencv_basis(rvec_head)
        rvec_flipped = flip_yaw_180(rvec_opencv)
        rvec_i = rotate_rvec_z_axis(rvec_flipped, -90)
        
        # Dummy Target (Not used for M matrix calculation)
        target_dummy = np.array([0., 0., 1000.]) 
        
        try:
            # 4. Get M Matrix
            norm_obj = dpc.norm(
                center=center_i,
                gazetarget=target_dummy,
                headrotvec=rvec_i,
                imsize=(224, 224),
                camparams=cam_params_formatted["mtx"]
            )
            M = norm_obj.M_mat
            
            try:
                M_inv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                M_inv = np.linalg.pinv(M)
            
            # 5. Inverse Transform Gaze
            # Input: gaze_estimations_rectified is in ROS basis (relative to Rectified Frame).
            g_rect_ros = gaze_estimations_rectified[i, 1:4]
            
            # Step A: Convert ROS -> CV (because M operates on CV vectors)
            g_rect_cv = convert_to_opencv_basis(g_rect_ros)
            
            # Step B: Un-rectify using M_inv
            g_cam_cv = M_inv @ g_rect_cv
            g_cam_cv = g_cam_cv / (np.linalg.norm(g_cam_cv) + 1e-8)
            
            # Step C: Convert CV -> ROS (Result should be in ROS Camera Frame)
            g_cam_ros = convert_from_opencv_basis(g_cam_cv)
            
            out_row = np.concatenate(([gaze_estimations_rectified[i, 0]], g_cam_ros))
            output_gaze.append(out_row)
            
        except Exception as e:
            continue

    return np.array(output_gaze)

# --------------------------------------------------------------------------------
# Wrapper Function (Single Experiment)
# --------------------------------------------------------------------------------

def unrectify_experiment_model(data_loader: GazeDataLoader, model_name: str):
    """
    Loads rectified predictions for a single experiment/model, applies inverse transform, and saves.
    """
    # 1. Load Rectified Estimates
    try:
        gaze_rectified = data_loader.load_gaze_estimations(model=model_name, frame="camera")
    except FileNotFoundError:
        # Silent return often better for batch processing logs, or verbose if needed
        return False

    print(f"[{model_name}] Unrectifying in {data_loader.get_cwd()}...")

    # 2. Load Geometry Data
    eye_pos = data_loader.load_eye_positions()
    head_poses = data_loader.load_head_poses(frame="world")
    camera_poses = data_loader.load_camera_poses(frame="world")
    
    # Load Intrinsics
    intrinsics_path = os.path.join(data_loader.get_cwd(), "camera_intrinsics.npy")
    if not os.path.exists(intrinsics_path):
        print(f"  [Warn] Camera intrinsics not found in {data_loader.get_cwd()}, skipping.")
        return False
    
    camera_intrinsics = np.load(intrinsics_path, allow_pickle=True).item()

    # 3. Calculate Periods Dynamically
    rgb_fps = config.get_rgb_fps()
    mocap_freq = config.get_mocap_freq()
    
    # Estimates are at RGB frequency
    gaze_est_period = 1000.0 / rgb_fps if rgb_fps > 0 else 33.33
    # Mocap data is at Mocap frequency
    mocap_period = 1000.0 / mocap_freq if mocap_freq > 0 else 10.0

    # 4. Perform Inverse Transform
    gaze_camera_frame = unrectify_gaze_vectors(
        gaze_estimations_rectified=gaze_rectified,
        head_poses_world=head_poses,
        eye_positions_world=eye_pos,
        camera_poses_world=camera_poses,
        camera_intrinsics=camera_intrinsics,
        target_period_ms=gaze_est_period,
        mocap_period_ms=mocap_period
    )

    # 5. Save
    output_dir = os.path.join(data_loader.get_cwd(), "gaze_estimations", f"{model_name}_unrectified")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "gaze_directions.npy")
    np.save(output_path, gaze_camera_frame)
    
    indices_path = os.path.join(output_dir, "gaze_directions_indices.npy")
    try:
        orig_indices = data_loader.load_gaze_estimation_valid_indices(model_name)
        np.save(indices_path, orig_indices)
    except:
        pass
        
    return True

# --------------------------------------------------------------------------------
# Batch Processing
# --------------------------------------------------------------------------------

def unrectify_batch(subject_dirs: list[str], gaze_models: list[str]):
    """
    Iterates over all experiments for the given subjects and unrectifies the specified models.
    """
    total_processed = 0
    
    for subject_dir in subject_dirs:
        # Get all experiment directories for this subject
        exp_dirs = config.get_all_exp_directories_under_a_subject_directory(subject_dir)
        
        # Iterate over experiments with progress bar
        for exp_dir in tqdm(exp_dirs, desc=f"Unrectifying {os.path.basename(subject_dir)}"):
            if not os.path.exists(exp_dir):
                continue
                
            if config.is_experiment_directory_excluded_from_eval(exp_dir):
                continue
                
            # Initialize DataLoader for this experiment
            # Note: GazeDataLoader takes the parent folder and finds the latest timestamped subdir.
            # exp_dir from config usually points to ".../point_name" (e.g., .../lighting_10/p1)
            # So we pass exp_dir directly.
            try:
                dataloader = GazeDataLoader(
                    root_dir=exp_dir,
                    target_period=config.get_target_period(),
                    camera_pose_period=config.get_camera_pose_period(),
                    time_diff_max=config.get_time_diff_max(),
                    get_latest_subdirectory_by_name=True
                )
                
                # Check if we successfully loaded a timestamped dir
                # (Extra safety if GazeDataLoader defaults to empty cwd if not found)
                if not os.path.exists(dataloader.get_cwd()):
                    continue

                for model in gaze_models:
                    success = unrectify_experiment_model(dataloader, model)
                    if success:
                        total_processed += 1
                        
            except Exception as e:
                print(f"Error processing {exp_dir}: {e}")
                continue

    print(f"\nBatch Unrectification Complete. Processed {total_processed} model-experiment pairs.")

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    # Define the models that need unrectification
    # Typically these are the ones with '_rectification' in their name
    models_to_unrectify = [
        "puregaze_rectification",
        # "gazetr_rectification",
        # "l2cs_padding0fixed_rectification_isrgb_False_resize_448",
        # "mcgaze_clip_size_7_rectification",
        # "gaze3d_clip_len_8_rectification"
    ]
    
    # Get all subjects
    # Use rnd=False to process everyone
    subjects = config.get_dataset_subject_directories(rnd=False)
    
    print(f"Starting unrectification for {len(subjects)} subjects and {len(models_to_unrectify)} models.")
    
    unrectify_batch(subjects, models_to_unrectify)