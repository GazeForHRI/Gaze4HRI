import os
import pandas as pd
import json
import random
import h5py
import numpy as np
import config

def generate_split(val_count=6, test_count=6):
    base_dir = config.get_dataset_base_directory()
    mapping_path = os.path.join(base_dir, "subject_id_mapping.csv")
    metadata_path = os.path.join(base_dir, "subject_metadata.json")
    h5_output_dir = os.path.join(base_dir, "blink4hri_torch_dataset")
    
    # Load mapping and metadata
    mapping_df = pd.read_csv(mapping_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    pure_pool = []
    incomplete_subjects = []

    print("Analyzing subjects for exclusions, metadata, and blink rates...")

    for _, row in mapping_df.iterrows():
        subject_id = row["subject_id"]
        subject_dir_rel = row["subject_dir"].replace('\\', '/')
        
        h5_path = os.path.join(h5_output_dir, f"{subject_id}.h5")
        if not os.path.exists(h5_path):
            continue

        # Extract blink stats from H5
        with h5py.File(h5_path, 'r') as h5:
            if 'is_blink' in h5:
                blinks = h5['is_blink'][:]
                b_count = int(np.sum(blinks))
                f_count = len(blinks)
            else:
                b_count = 0
                f_count = 0

        # Check for exclusions to determine "Pure" status
        exp_types = config.get_experiment_types()
        all_exp_dirs = config.get_exp_directories_under_a_subject_directory(subject_dir_rel, exp_types)
        
        exclusions = []
        for exp in all_exp_dirs:
            exp_abs = exp if os.path.isabs(exp) else os.path.join(base_dir, exp)
            if config.is_experiment_directory_excluded_from_blink4hri(exp_abs):
                exclusions.append(exp)
        
        # Metadata lookup (Glasses and Gender)
        meta = metadata.get(subject_dir_rel, {"glasses": 0, "gender": "m"})
        
        subject_info = {
            "id": subject_id,
            "h5": h5_path,
            "glasses": meta["glasses"],
            "gender": meta["gender"],
            "blinks": b_count,
            "frames": f_count
        }

        if len(exclusions) == 0:
            pure_pool.append(subject_info)
        else:
            incomplete_subjects.append(subject_info)

    # --- Targeted Stratification & Balance Logic ---
    # Split the pure subjects into glasses/no-glasses strata
    with_glasses = [s for s in pure_pool if s["glasses"] == 1]
    no_glasses = [s for s in pure_pool if s["glasses"] == 0]

    def sample_stratum(pool, target_total, target_female):
        """Prioritizes females in a given pool to reach a target count."""
        females = [s for s in pool if s["gender"] == "f"]
        males = [s for s in pool if s["gender"] == "m"]
        random.shuffle(females)
        random.shuffle(males)
        
        selected = []
        
        f_take = min(target_female, len(females))
        selected.extend(females[:f_take])
        
        m_take = target_total - len(selected)
        selected.extend(males[:m_take])
        
        if len(selected) < target_total:
            extra_f = target_total - len(selected)
            selected.extend(females[f_take : f_take + extra_f])
            
        for s in selected:
            pool.remove(s)
            
        return selected

    # Target 3 per stratum for Val and Test (total 6 each)
    per_stratum = 3 
    # Aim for at least 1 female per stratum bucket (total 2 per set)
    female_target = 1 

    best_split = None
    best_diff = float('inf')

    # Search for the split that minimizes blink rate variance between Train/Val/Test
    for seed in range(5000):
        random.seed(seed)
        wg_pool = list(with_glasses)
        ng_pool = list(no_glasses)
        
        val_subjects = []
        test_subjects = []

        try:
            val_subjects.extend(sample_stratum(wg_pool, per_stratum, female_target))
            val_subjects.extend(sample_stratum(ng_pool, per_stratum, female_target))

            test_subjects.extend(sample_stratum(wg_pool, per_stratum, female_target))
            test_subjects.extend(sample_stratum(ng_pool, per_stratum, female_target))
            
            used_ids = {s["id"] for s in val_subjects + test_subjects}
            train_subjects = [s for s in pure_pool if s["id"] not in used_ids] + incomplete_subjects

            def get_blink_rate(subset):
                t_blinks = sum(s["blinks"] for s in subset)
                t_frames = sum(s["frames"] for s in subset)
                return t_blinks / t_frames if t_frames > 0 else 0.0

            t_rate = get_blink_rate(train_subjects)
            v_rate = get_blink_rate(val_subjects)
            te_rate = get_blink_rate(test_subjects)

            # Calculate max discrepancy between any two splits
            rates = [t_rate, v_rate, te_rate]
            diff = max(rates) - min(rates)

            if diff < best_diff:
                best_diff = diff
                best_split = (train_subjects, val_subjects, test_subjects)

        except (IndexError, ValueError):
            continue

    if best_split is None:
        print("Warning: Stratification constraints could not be met. Falling back to simple pure split.")
        random.seed(42)
        random.shuffle(pure_pool)
        val_subjects = pure_pool[:val_count]
        test_subjects = pure_pool[val_count : val_count + test_count]
        used_ids = {s["id"] for s in val_subjects + test_subjects}
        train_subjects = [s for s in pure_pool if s["id"] not in used_ids] + incomplete_subjects
    else:
        train_subjects, val_subjects, test_subjects = best_split

    # Stats Helper
    def get_stats(subset):
        g = sum(1 for s in subset if s["glasses"] == 1)
        f = sum(1 for s in subset if s["gender"] == "f")
        t_blinks = sum(s["blinks"] for s in subset)
        t_frames = sum(s["frames"] for s in subset)
        rate = (t_blinks / t_frames * 100) if t_frames > 0 else 0.0
        return f"Glasses: {g}/{len(subset)}, Female: {f}/{len(subset)}, Blink Rate: {rate:.2f}%"

    # Save to JSON
    split_data = {
        "train": [s["h5"] for s in train_subjects],
        "val": [s["h5"] for s in val_subjects],
        "test": [s["h5"] for s in test_subjects]
    }

    with open(os.path.join(h5_output_dir, "split.json"), "w") as f:
        json.dump(split_data, f, indent=4)

    print(f"\nSplit Results:")
    print(f"Train: {len(train_subjects):2} subjects | {get_stats(train_subjects)}")
    print(f"Val:   {len(val_subjects):2} subjects | {get_stats(val_subjects)} (Pure)")
    print(f"Test:  {len(test_subjects):2} subjects | {get_stats(test_subjects)} (Pure)")
    print(f"Split saved to: {os.path.join(h5_output_dir, 'split.json')}")

if __name__ == "__main__":
    generate_split()
