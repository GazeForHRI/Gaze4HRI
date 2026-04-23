import os
import json
import torch
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from torchvision import transforms
import argparse

import config
from blinklinmult.models import BlinkLinMulT
from data_loader import GazeDataLoader

def standardize_high_level(features, mean_std_dict):
    if mean_std_dict is None:
        return features
    mean = torch.tensor(mean_std_dict['mean'], dtype=torch.float32)
    std = torch.tensor(mean_std_dict['std'], dtype=torch.float32)
    return (features - mean) / (std + 1e-7)

def test_model(weights, model_name, split="val", n_frames=15, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BlinkLinMulT(input_dim=160, output_dim=1, weights=None)
    
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"Weights file not found: {weights}")
    
    print(f"Loading checkpoint from: {weights}")
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()

    base_dir = config.get_dataset_base_directory()
    dataset_dir = os.path.join(base_dir, "blink4hri_torch_dataset")
    split_path = os.path.join(dataset_dir, "split.json")
    stats_path = os.path.join(dataset_dir, "feature_stats.json")
    mapping_path = os.path.join(base_dir, "subject_id_mapping.csv")

    with open(split_path, "r") as f:
        splits = json.load(f)
    
    target_files = splits.get(split, [])
    if not target_files:
        print(f"No files found for split: {split}")
        return

    mean_std_dict = None
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            mean_std_dict = json.load(f)
    else:
        print("WARNING: feature_stats.json not found! High-level features will NOT be standardized.")

    mapping_df = pd.read_csv(mapping_path)

    img_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"Starting inference for {len(target_files)} subjects in '{split}' split...")

    with torch.no_grad():
        for h5_file_path in tqdm(target_files, desc="Processing Subjects"):
            subject_id = os.path.basename(h5_file_path).replace('.h5', '')
            
            subject_row = mapping_df[mapping_df['subject_id'] == subject_id]
            if subject_row.empty:
                print(f"Warning: No mapping found for {subject_id}. Skipping.")
                continue
            
            subject_dir_rel = subject_row.iloc[0]["subject_dir"].replace('\\', '/')
            
            exp_types = config.get_experiment_types()
            all_exp_dirs = config.get_exp_directories_under_a_subject_directory(subject_dir_rel, exp_types)
            
            valid_exps = []
            for exp in all_exp_dirs:
                exp_abs = exp if os.path.isabs(exp) else os.path.join(base_dir, exp)
                if not config.is_experiment_directory_excluded_from_blink4hri(exp_abs):
                    valid_exps.append(exp)

            with h5py.File(h5_file_path, 'r') as h5:
                if 'exp_boundaries' not in h5:
                    print(f"Warning: No exp_boundaries in {h5_file_path}. Skipping.")
                    continue
                
                boundaries = h5['exp_boundaries'][:]
                
                for boundary_idx, boundary in enumerate(boundaries):
                    if boundary_idx >= len(valid_exps):
                        break
                        
                    exp_dir_rel = valid_exps[boundary_idx]
                    exp_dir_abs = os.path.join(base_dir, exp_dir_rel)
                    
                    # Use GazeDataLoader to automatically access the latest timestamp subdirectory
                    dataloader = GazeDataLoader(
                        root_dir=exp_dir_abs, 
                        target_period=config.get_target_period(), 
                        camera_pose_period=config.get_camera_pose_period(), 
                        time_diff_max=config.get_time_diff_max(), 
                        get_latest_subdirectory_by_name=True
                    )
                    exp_cwd = dataloader.get_cwd()

                    start_idx, end_idx = boundary
                    exp_len = end_idx - start_idx + 1
                    
                    left_eye_raw = h5['left_eye'][start_idx:end_idx+1]
                    right_eye_raw = h5['right_eye'][start_idx:end_idx+1]
                    left_feat_raw = h5['left_features'][start_idx:end_idx+1]
                    right_feat_raw = h5['right_features'][start_idx:end_idx+1]

                    strides_to_run = [n_frames, 1]
                    
                    for current_stride in strides_to_run:
                        current_model_name = f"{model_name}_stride_{current_stride}"
                        predictions = np.zeros(exp_len, dtype=np.uint8)

                        windows = []
                        if current_stride == 1:
                            for i in range(exp_len):
                                w_start = i - n_frames + 1
                                if w_start < 0:
                                    idx_seq = [0] * abs(w_start) + list(range(0, i + 1))
                                else:
                                    idx_seq = list(range(w_start, i + 1))
                                windows.append(idx_seq)
                        else:
                            for w_start in range(0, exp_len - n_frames + 1, current_stride):
                                windows.append(list(range(w_start, w_start + n_frames)))

                        windows = np.array(windows)
                        if len(windows) == 0:
                            continue

                        for b_start in range(0, len(windows), batch_size):
                            b_indices = windows[b_start:b_start+batch_size]
                            
                            le_batch = left_eye_raw[b_indices]
                            re_batch = right_eye_raw[b_indices]
                            
                            le_tensor = torch.from_numpy(le_batch).permute(0, 1, 4, 2, 3).float() / 255.0
                            re_tensor = torch.from_numpy(re_batch).permute(0, 1, 4, 2, 3).float() / 255.0

                            B_curr, N, C, H, W = le_tensor.shape
                            
                            le_tensor = torch.stack([img_transform(img) for img in le_tensor.view(-1, C, H, W)]).view(B_curr, N, C, H, W).to(device)
                            re_tensor = torch.stack([img_transform(img) for img in re_tensor.view(-1, C, H, W)]).view(B_curr, N, C, H, W).to(device)

                            lf_batch = left_feat_raw[b_indices]
                            rf_batch = right_feat_raw[b_indices]
                            
                            lf_tensor = torch.from_numpy(lf_batch).float()
                            rf_tensor = torch.from_numpy(rf_batch).float()
                            
                            lf_tensor = standardize_high_level(lf_tensor, mean_std_dict).to(device)
                            rf_tensor = standardize_high_level(rf_tensor, mean_std_dict).to(device)

                            out_l = model([le_tensor, lf_tensor])
                            out_r = model([re_tensor, rf_tensor])
                            
                            y_preds_l = out_l[1] if isinstance(out_l, tuple) else out_l
                            y_preds_r = out_r[1] if isinstance(out_r, tuple) else out_r
                            
                            avg_logits = (y_preds_l.squeeze(-1) + y_preds_r.squeeze(-1)) / 2.0
                            probs = torch.sigmoid(avg_logits)
                            
                            if current_stride == 1:
                                preds = (probs[:, -1] > 0.5).cpu().numpy().astype(np.uint8)
                                predictions[b_start:b_start+len(preds)] = preds
                            else:
                                preds = (probs > 0.5).cpu().numpy().astype(np.uint8)
                                for b_idx in range(len(preds)):
                                    global_w_start = (b_start + b_idx) * current_stride
                                    predictions[global_w_start:global_w_start+n_frames] = preds[b_idx]

                        # Save into the latest timestamp directory instead of the root exp_dir
                        save_dir = os.path.join(exp_cwd, "blink_estimations", current_model_name)
                        os.makedirs(save_dir, exist_ok=True)
                        np.save(os.path.join(save_dir, "blink_estimations.npy"), predictions)

    print("Inference completed successfully for both strides.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test BlinkLinMulT and save estimations")
    parser.add_argument("--weights", type=str, required=True, 
                        help="Absolute path to the trained model checkpoint (.pt file).")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Base name of the model directory to save under exp_dir/blink_estimations/")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                        help="Which dataset split to run inference on.")
    parser.add_argument("--n_frames", type=int, default=15, 
                        help="Number of sequence frames the model expects.")
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Batch size for generating estimations.")
    
    args = parser.parse_args()

    if args.split == "test":
        print("*" * 60)
        print("WARNING: YOU ARE RUNNING INFERENCE ON THE TEST SPLIT!")
        print("The test split should ONLY be used when the model is fully finalized.")
        print("*" * 60)

    test_model(
        weights=args.weights,
        model_name=args.model_name,
        split=args.split,
        n_frames=args.n_frames,
        batch_size=args.batch_size
    )

    if args.split == "test":
        print("*" * 60)
        print("WARNING: YOU ARE RUNNING INFERENCE ON THE TEST SPLIT!")
        print("The test split should ONLY be used when the model is fully finalized.")
        print("*" * 60)

