import os
import pickle
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm

import config
from data_loader import GazeDataLoader

def extract_160d_features(item, side):
    features = item[f'annotation_{side}_eye_features']
    headpose = np.array(item['tddfa-retinaface_headpose']).flatten()
    landmarks = np.array(features['landmarks']).flatten()
    iris_landmarks = np.array(features['iris_landmarks']).flatten()
    iris_diameters = np.array(features['iris_diameters']).flatten()
    eyelid_dist = np.array(features['eyelid_pupil_distances']).flatten()
    ear = np.array([features['ear']]).flatten()
    
    return np.concatenate([
        headpose, landmarks, iris_landmarks, iris_diameters, eyelid_dist, ear
    ]).astype(np.float32)

def create_h5_blink_dataset(included_exp_types: list, min_length: int = 15):
    base_dir = config.get_dataset_base_directory()
    mapping_path = os.path.join(base_dir, "subject_id_mapping.csv")
    output_dir = os.path.join(base_dir, "blink4hri_torch_dataset")
    os.makedirs(output_dir, exist_ok=True)
    
    mapping_df = pd.read_csv(mapping_path)
    print(f"DEBUG: Found {len(mapping_df)} subjects in mapping file.")
    
    for _, row in mapping_df.iterrows():
        subject_id = row["subject_id"]
        subject_dir_rel = row["subject_dir"]
        h5_path = os.path.join(output_dir, f"{subject_id}.h5")

        # Track total frames written to decide if we keep the file
        total_frames = 0
        
        exp_dirs = config.get_exp_directories_under_a_subject_directory(subject_dir_rel, included_exp_types)
        print(f"DEBUG: {subject_id} has {len(exp_dirs)} potential experiment directories.")
        
        with h5py.File(h5_path, 'w') as h5:
            left_eye_ds = h5.create_dataset("left_eye", shape=(0, 64, 64, 3), maxshape=(None, 64, 64, 3), dtype='uint8', chunks=(1, 64, 64, 3))
            right_eye_ds = h5.create_dataset("right_eye", shape=(0, 64, 64, 3), maxshape=(None, 64, 64, 3), dtype='uint8', chunks=(1, 64, 64, 3))
            left_feat_ds = h5.create_dataset("left_features", shape=(0, 160), maxshape=(None, 160), dtype='float32')
            right_feat_ds = h5.create_dataset("right_features", shape=(0, 160), maxshape=(None, 160), dtype='float32')
            blink_ds = h5.create_dataset("is_blink", shape=(0,), maxshape=(None,), dtype='uint8')
            orig_idx_ds = h5.create_dataset("original_indices", shape=(0,), maxshape=(None,), dtype='int32')
            
            str_dt = h5py.string_dtype(encoding='utf-8')
            exp_dir_ds = h5.create_dataset("exp_dir", shape=(0,), maxshape=(None,), dtype=str_dt)
            exp_type_ds = h5.create_dataset("exp_type", shape=(0,), maxshape=(None,), dtype=str_dt)
            point_ds = h5.create_dataset("point", shape=(0,), maxshape=(None,), dtype=str_dt)
            
            boundaries = []
            
            for exp_dir_rel in tqdm(exp_dirs, desc=f"Packing {subject_id}", leave=False):
                exp_dir_abs = os.path.join(base_dir, exp_dir_rel)

                if config.is_experiment_directory_excluded_from_blink4hri(exp_dir_abs):
                    print(f"Excluded directory from blink4hri: {exp_dir_abs}")
                    continue
                
                dataloader = GazeDataLoader(root_dir=exp_dir_abs, target_period=config.get_target_period(), 
                                            camera_pose_period=config.get_camera_pose_period(), 
                                            time_diff_max=config.get_time_diff_max(), get_latest_subdirectory_by_name=True)
                
                exp_cwd = dataloader.get_cwd()
                # Ensure this path matches your folder structure exactly (landmark vs landmarks)
                pkl_path = os.path.join(exp_cwd, "exordium_landmarks2", "exordium_data_fixed.pkl")

                if not os.path.exists(pkl_path):
                    print(f"DEBUG: Skipped {exp_dir_rel} - Pickle not found at {pkl_path}")
                    continue
                    
                with open(pkl_path, 'rb') as f:
                    exordium_data = pickle.load(f)
                
                blink_ann = dataloader.get_blink_annotations()
                
                if blink_ann is None or len(blink_ann) == 0:
                    print(f"DEBUG: Skipped {exp_dir_rel} - No blink annotations found.")
                    continue

                c_left_eyes, c_right_eyes, c_left_feats, c_right_feats, c_blinks, c_orig_indices = [], [], [], [], [], []

                for i in range(5, len(blink_ann)):
                    item = exordium_data.get(i)
                    if item is None or item['face_bbox'][0] == -1:
                        continue
                    
                    c_left_eyes.append(item['annotation_left_eye_features']['eye'])
                    c_right_eyes.append(item['annotation_right_eye_features']['eye'])
                    c_left_feats.append(extract_160d_features(item, 'left'))
                    c_right_feats.append(extract_160d_features(item, 'right'))
                    c_blinks.append(1 if int(blink_ann[i]) in [2, 3] else 0)
                    c_orig_indices.append(i)
                
                if len(c_blinks) < min_length:
                    print(f"DEBUG: Skipped {exp_dir_rel} - Only {len(c_blinks)} valid frames (min is {min_length}).")
                    continue

                # Cleanly extract exp_type and point from the relative path
                parts = exp_dir_rel.strip("/").split("/")
                c_exp_type = parts[-2] if len(parts) >= 2 else ""
                c_point = parts[-1] if len(parts) >= 1 else ""

                curr = blink_ds.shape[0]
                L = len(c_blinks)
                total_frames += L
                
                datasets = [left_eye_ds, right_eye_ds, left_feat_ds, right_feat_ds, blink_ds, orig_idx_ds, exp_dir_ds, exp_type_ds, point_ds]
                data_lists = [c_left_eyes, c_right_eyes, c_left_feats, c_right_feats, c_blinks, c_orig_indices, 
                              [exp_dir_rel] * L, [c_exp_type] * L, [c_point] * L]
                
                for ds, data in zip(datasets, data_lists):
                    ds.resize(curr + L, axis=0)
                    ds[curr:] = np.array(data)
                
                boundaries.append([curr, curr + L - 1])
            
            h5.create_dataset("exp_boundaries", data=np.array(boundaries, dtype='int32'))
            print(f"Dataset created: {h5_path} ({blink_ds.shape[0]} frames)")

        # CLEANUP: If no frames were added, remove the empty file
        if total_frames == 0:
            if os.path.exists(h5_path):
                os.remove(h5_path)
            print(f"Skipped subject (0 frames): {subject_id}")
        else:
            print(f"Dataset created: {h5_path} ({total_frames} frames)")

if __name__ == "__main__":
    types = config.get_experiment_types()
    print(f"DEBUG: Experiment types requested: {types}")
    create_h5_blink_dataset(included_exp_types=types)
