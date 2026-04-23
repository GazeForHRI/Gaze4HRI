import os
import json
import h5py
import numpy as np
import config

def compute_train_statistics():
    base_dir = config.get_dataset_base_directory()
    split_path = os.path.join(base_dir, "blink4hri_torch_dataset", "split.json")
    output_path = os.path.join(base_dir, "blink4hri_torch_dataset", "feature_stats.json")
    
    with open(split_path, 'r') as f:
        splits = json.load(f)
    
    train_files = splits['train']
    all_features = []
    
    print(f"Extracting features from {len(train_files)} training files...")
    for f_path in train_files:
        with h5py.File(f_path, 'r') as h5:
            if 'left_features' in h5 and 'right_features' in h5:
                all_features.append(h5['left_features'][:])
                all_features.append(h5['right_features'][:])
                
    if not all_features:
        print("Error: No features found in training set.")
        return
        
    # Concatenate all features into a single array: Shape [Total_Frames, 160]
    concat_features = np.concatenate(all_features, axis=0)
    
    # Calculate mean and std for each of the 160 dimensions
    mean_vals = np.mean(concat_features, axis=0)
    std_vals = np.std(concat_features, axis=0)
    
    stats_dict = {
        "mean": mean_vals.tolist(),
        "std": std_vals.tolist()
    }
    
    with open(output_path, 'w') as f:
        json.dump(stats_dict, f, indent=4)
        
    print(f"Successfully computed statistics across {concat_features.shape[0]:,} frames.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    compute_train_statistics()
