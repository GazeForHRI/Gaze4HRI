import os
import pandas as pd
import numpy as np
import cv2
import h5py
from tqdm import tqdm

import config
from data_loader import GazeDataLoader
from data_matcher import match_regular_to_regular
import data_rectification as dpc

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

# --- Main Direct-to-HDF5 Generator ---
def create_h5_dataset(N: int = 20):
    base_dir = config.get_dataset_base_directory()
    mapping_path = os.path.join(base_dir, "subject_id_mapping.csv")
    output_dir = os.path.join(base_dir, "gaze4hri_torch_dataset")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found at {mapping_path}. Run generate_subject_mapping first.")
        
    mapping_df = pd.read_csv(mapping_path)
    
    for _, row in mapping_df.iterrows():
        subject_id = row["subject_id"]
        subject_dir_rel = row["subject_dir"]
        
        h5_path = os.path.join(output_dir, f"{subject_id}.h5")
        exp_dirs = config.get_all_exp_directories_under_a_subject_directory(subject_dir_rel)
        
        # Open HDF5 file and create resizable datasets
        with h5py.File(h5_path, 'w') as h5:
            img_ds = h5.create_dataset("images", shape=(0, 224, 224, 3), maxshape=(None, 224, 224, 3), dtype='uint8', chunks=(1, 224, 224, 3))
            gaze_ds = h5.create_dataset("gaze_pitch_yaw", shape=(0, 2), maxshape=(None, 2), dtype='float32')
            blink_ds = h5.create_dataset("is_blink", shape=(0,), maxshape=(None,), dtype='uint8')
            
            boundaries = []
            
            for exp_dir_rel in tqdm(exp_dirs, desc=f"Packing {subject_id}"):
                exp_dir_abs = os.path.join(base_dir, exp_dir_rel)
                
                if not os.path.exists(exp_dir_abs) or config.is_experiment_directory_excluded_from_eval(exp_dir_abs):
                    continue
                    
                try:
                    dataloader = GazeDataLoader(
                        root_dir=exp_dir_abs,
                        target_period=config.get_target_period(),
                        camera_pose_period=config.get_camera_pose_period(),
                        time_diff_max=config.get_time_diff_max(),
                        get_latest_subdirectory_by_name=True
                    )
                    
                    if not os.path.exists(dataloader.get_cwd()):
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

                    camera_intrinsics = np.load(os.path.join(dataloader.get_cwd(), "camera_intrinsics.npy"), allow_pickle=True).item()
                    K = np.array([
                        [float(camera_intrinsics["fx"]), 0.0, float(camera_intrinsics["cx"])],
                        [0.0, float(camera_intrinsics["fy"]), float(camera_intrinsics["cy"])],
                        [0.0, 0.0, 1.0]
                    ], dtype=np.float64)

                    _, index_array = match_regular_to_regular(
                        (timestamps.reshape(-1, 1), 1000.0 / (config.get_rgb_fps() or 30.0)),
                        (eye_positions, 1000.0 / (config.get_mocap_freq() or 100.0)),
                        max_match_diff_ms=20.0
                    )

                    # Temporary lists to hold the experiment's data in RAM
                    exp_imgs = []
                    exp_gaze = []
                    exp_blink = []
                    
                    for i, frame in enumerate(rgb_frames):
                        if frame is None or frame.size == 0 or i >= len(index_array):
                            continue
                            
                        idx_mocap = index_array[i]
                        world_T_cam = np.linalg.inv(camera_poses[idx_mocap, 1:].reshape(4, 4))

                        eye_cam = (world_T_cam @ np.append(eye_positions[idx_mocap, 1:4], 1.0))[:3]
                        target_cam = (world_T_cam @ np.append(target_positions[idx_mocap, 1:4], 1.0))[:3]
                        head_T_cam = world_T_cam @ head_poses[idx_mocap, 1:].reshape(4, 4)

                        rvec_head, _ = cv2.Rodrigues(head_T_cam[:3, :3])
                        
                        center_i = convert_to_opencv_basis(eye_cam) * 1000.0
                        target_i = convert_to_opencv_basis(target_cam) * 1000.0
                        rvec_i = rotate_rvec_z_axis(flip_yaw_180(convert_to_opencv_basis(rvec_head.flatten())), -90)

                        try:
                            N_norm = dpc.norm(center=center_i, gazetarget=target_i, headrotvec=rvec_i, imsize=(224, 224), camparams=K)
                            norm_img = N_norm.GetImage(frame)
                            norm_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB) # Torch format
                            
                            gaze_yaw, gaze_pitch = dpc.GazeTo2d(N_norm.GetGaze(scale=True))
                            blink_val = 1 if int(blink_ann[i]) in [2, 3] else 0
                            
                            exp_imgs.append(norm_img)
                            exp_gaze.append([gaze_pitch, gaze_yaw])
                            exp_blink.append(blink_val)
                        except:
                            continue
                            
                    # Write the collected experiment data to HDF5
                    if len(exp_imgs) >= N:
                        current_len = img_ds.shape[0]
                        exp_len = len(exp_imgs)
                        
                        # Resize datasets
                        img_ds.resize(current_len + exp_len, axis=0)
                        gaze_ds.resize(current_len + exp_len, axis=0)
                        blink_ds.resize(current_len + exp_len, axis=0)
                        
                        # Append arrays
                        img_ds[current_len:] = np.array(exp_imgs, dtype='uint8')
                        gaze_ds[current_len:] = np.array(exp_gaze, dtype='float32')
                        blink_ds[current_len:] = np.array(exp_blink, dtype='uint8')
                        
                        # Record boundary
                        boundaries.append([current_len, current_len + exp_len - 1])
                        
                except Exception as e:
                    print(f"Error in {exp_dir_rel}: {e}")
            
            # Save the experiment boundaries array
            h5.create_dataset("exp_boundaries", data=np.array(boundaries, dtype='int32'))
            print(f"Saved {img_ds.shape[0]} frames to {h5_path}")

if __name__ == "__main__":
    create_h5_dataset(N=20)
