import os
import pandas as pd
import numpy as np
import json
import h5py
from sklearn.metrics import confusion_matrix
import config
from data_loader import GazeDataLoader

BASE_DIR = config.get_dataset_base_directory()
H5_DIR = os.path.join(BASE_DIR, "blink4hri_torch_dataset")
CSV_PATH = os.path.join(BASE_DIR, "blink_evaluation_results.csv")
SUBJECT_METADATA_PATH = os.path.join(BASE_DIR, "subject_metadata.json")

def load_subject_metadata(path: str = SUBJECT_METADATA_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing subject metadata at {path}.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def update_blink_results_csv():
    metadata = load_subject_metadata()
    if not os.path.exists(H5_DIR):
        print(f"Error: H5 directory not found at {H5_DIR}")
        return

    h5_files = [f for f in os.listdir(H5_DIR) if f.endswith('.h5')]
    mapping_path = os.path.join(BASE_DIR, "subject_id_mapping.csv")
    
    if not os.path.exists(mapping_path):
        print(f"Error: Mapping file not found at {mapping_path}")
        return
        
    mapping_df = pd.read_csv(mapping_path)
    new_rows = []
    
    for h5_filename in h5_files:
        subject_id = h5_filename.replace('.h5', '')
        h5_path = os.path.join(H5_DIR, h5_filename)
        
        # Map subject ID to relative directory (e.g., 2025-08-05/subj_xxxx)
        subject_row = mapping_df[mapping_df['subject_id'] == subject_id]
        if subject_row.empty:
            continue
            
        subject_dir_rel = subject_row.iloc[0]["subject_dir"].replace('\\', '/')
        meta = metadata.get(subject_dir_rel)
        if not meta:
            continue
            
        exp_types = config.get_experiment_types()
        # This function often returns absolute paths depending on config setup
        all_exp_dirs = config.get_exp_directories_under_a_subject_directory(subject_dir_rel, exp_types)
        valid_exps = [e for e in all_exp_dirs if not config.is_experiment_directory_excluded_from_blink4hri(os.path.join(BASE_DIR, e))]

        with h5py.File(h5_path, 'r') as h5:
            if 'exp_boundaries' not in h5:
                continue
            
            boundaries = h5['exp_boundaries'][:]
            is_blink_ds = h5['is_blink'][:]
            
            for b_idx, boundary in enumerate(boundaries):
                if b_idx >= len(valid_exps):
                    break
                    
                exp_path_full = valid_exps[b_idx]
                exp_dir_abs = exp_path_full if os.path.isabs(exp_path_full) else os.path.join(BASE_DIR, exp_path_full)
                
                # CRITICAL FIX: Calculate path relative to BASE_DIR to extract exp_type and point
                rel_path = os.path.relpath(exp_dir_abs, BASE_DIR)
                parts = rel_path.replace('\\', '/').split('/')
                
                # Standard Structure: [Date] / [Subject] / [ExpType] / [Point]
                # So parts[2] is ExpType, parts[3] is Point
                exp_type = parts[2] if len(parts) > 2 else ""
                point = parts[3] if len(parts) > 3 else ""
                
                start, end = boundary
                
                # Locate the latest timestamp subdirectory using GazeDataLoader
                try:
                    dataloader = GazeDataLoader(
                        root_dir=exp_dir_abs, 
                        target_period=config.get_target_period(), 
                        camera_pose_period=config.get_camera_pose_period(), 
                        time_diff_max=config.get_time_diff_max(), 
                        get_latest_subdirectory_by_name=True
                    )
                    exp_cwd = dataloader.get_cwd()
                except Exception:
                    print(f"Skipping {exp_dir_abs}: Could not load timestamp directory.")
                    continue
                
                estimations_base = os.path.join(exp_cwd, "blink_estimations")
                if not os.path.exists(estimations_base):
                    continue
                
                evaluated_models = [d for d in os.listdir(estimations_base) if os.path.isdir(os.path.join(estimations_base, d))]
                
                for model_name in evaluated_models:
                    pred_path = os.path.join(estimations_base, model_name, "blink_estimations.npy")
                    if not os.path.exists(pred_path):
                        continue
                        
                    preds = np.load(pred_path)
                    targets = is_blink_ds[start:end+1]
                    
                    # Trim or pad to match if necessary (due to stride/padding differences)
                    if len(preds) != len(targets):
                        min_len = min(len(preds), len(targets))
                        preds = preds[:min_len]
                        targets = targets[:min_len]
                    
                    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
                    
                    new_rows.append({
                        "subject_dir": subject_dir_rel,
                        "exp_type": exp_type,
                        "point": point,
                        "subject_name": meta["name"],
                        "gender": meta["gender"],
                        "birthyear": meta["birthyear"],
                        "glasses": meta["glasses"],
                        "height_cm": meta["height_cm"],
                        "blink_model": model_name,
                        "tp": int(tp),
                        "fp": int(fp),
                        "tn": int(tn),
                        "fn": int(fn),
                        "num_frames": len(targets)
                    })

    if not new_rows:
        print("No blink estimation results found.")
        return

    df_new = pd.DataFrame(new_rows)
    df_new = df_new.sort_values(by=["subject_name", "exp_type", "point", "blink_model"])
    df_new.to_csv(CSV_PATH, index=False)
    print(f"Updated {CSV_PATH} with {len(df_new)} rows. Exp types: {df_new['exp_type'].unique()}")

if __name__ == "__main__":
    update_blink_results_csv()
