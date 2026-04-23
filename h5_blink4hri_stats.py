import os
import h5py
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import config

def calculate_stats(subject_stats):
    """Calculates distribution stats for a list of (frames, blinks) tuples."""
    if not subject_stats:
        return None
    
    counts = np.array([s[0] for s in subject_stats])
    blinks = np.array([s[1] for s in subject_stats])
    # Percentage of frames that are blinks per subject
    ratios = (blinks / counts) * 100 
    
    return {
        "total_frames": int(np.sum(counts)),
        "total_blinks": int(np.sum(blinks)),
        "subject_count": len(subject_stats),
        "frame_dist": {
            "mean": np.mean(counts), "std": np.std(counts), 
            "median": np.median(counts), "min": np.min(counts), "max": np.max(counts)
        },
        "blink_ratio_dist": {
            "mean": np.mean(ratios), "std": np.std(ratios), 
            "median": np.median(ratios), "min": np.min(ratios), "max": np.max(ratios)
        }
    }

def print_section(title, stats):
    if not stats:
        return
    print(f"\n--- {title} ---")
    print(f"Total Subjects:         {stats['subject_count']}")
    print(f"Total Frames:           {stats['total_frames']:,}")
    print(f"Total Blink Frames:     {stats['total_blinks']:,} ({(stats['total_blinks']/stats['total_frames'])*100:.2f}%)")
    
    d = stats['frame_dist']
    print(f"Per-Subject Frames:     Mean: {d['mean']:.1f} | Std: {d['std']:.1f} | Min: {d['min']:,} | Max: {d['max']:,}")
    
    r = stats['blink_ratio_dist']
    print(f"Per-Subject Blink %:    Mean: {r['mean']:.2f}% | Std: {r['std']:.2f}% | Median: {r['median']:.2f}%")
    print(f"                        Range: [{r['min']:.2f}% - {r['max']:.2f}%]")

def print_table(title, df, group_cols):
    print(f"\n--- {title} ---")
    if df.empty:
        print("No data available.")
        return

    agg = df.groupby(group_cols).agg(
        total_frames=('frames', 'sum'),
        total_blinks=('blinks', 'sum')
    ).reset_index()
    
    agg['blink_ratio'] = (agg['total_blinks'] / agg['total_frames']) * 100

    if isinstance(group_cols, list):
        agg['Group'] = agg[group_cols[0]].astype(str) + " / " + agg[group_cols[1]].astype(str)
    else:
        agg['Group'] = agg[group_cols].astype(str)

    agg = agg.sort_values('Group')

    print(f"{'Group Name':<45} | {'Total Frames':<15} | {'Total Blinks':<15} | {'Blink %':<10}")
    print("-" * 93)
    for _, row in agg.iterrows():
        name = str(row['Group'])[:43]
        frames = f"{int(row['total_frames']):,}"
        blinks = f"{int(row['total_blinks']):,}"
        ratio = f"{row['blink_ratio']:.2f}%"
        print(f"{name:<45} | {frames:<15} | {blinks:<15} | {ratio:<10}")

def get_dataset_stats():
    base_dir = config.get_dataset_base_directory()
    dataset_dir = os.path.join(base_dir, "blink4hri_torch_dataset")
    split_path = os.path.join(dataset_dir, "split.json")
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Directory {dataset_dir} does not exist.")
        return

    split_info = {}
    if os.path.exists(split_path):
        with open(split_path, 'r') as f:
            split_info = json.load(f)

    h5_files = [f for f in os.listdir(dataset_dir) if f.endswith('.h5')]
    if not h5_files:
        print("No H5 files found.")
        return

    path_to_split = {}
    for split_name, paths in split_info.items():
        for p in paths:
            path_to_split[os.path.abspath(p)] = split_name

    all_data = [] # For global subject stats
    all_exp_data = [] # For granular breakdowns
    structure_printed = False

    for filename in tqdm(h5_files, desc="Analyzing Files"):
        file_path = os.path.abspath(os.path.join(dataset_dir, filename))
        
        try:
            with h5py.File(file_path, 'r') as h5:
                if not structure_printed:
                    print("\n--- Dataset Structure ---")
                    h5.visititems(lambda n, o: print(f"  {n:20} | Shape: {str(o.shape):15} | Dtype: {o.dtype}") if isinstance(o, h5py.Dataset) else None)
                    print("Note: 'exp_dir', 'exp_type', and 'point' ARE natively stored in H5.")
                    structure_printed = True

                blinks = h5['is_blink'][:]
                split = path_to_split.get(file_path, "Unassigned")
                all_data.append((len(blinks), int(np.sum(blinks)), split))
                
                if 'exp_boundaries' in h5 and 'exp_type' in h5 and 'point' in h5:
                    boundaries = h5['exp_boundaries'][:]
                    
                    for boundary in boundaries:
                        start, end = boundary
                        
                        # Read directly from H5 and decode from bytes to string
                        exp_type = h5['exp_type'][start].decode('utf-8')
                        point = h5['point'][start].decode('utf-8')
                        
                        b_frames = end - start + 1
                        b_blinks = int(np.sum(blinks[start:end+1]))
                        
                        all_exp_data.append({
                            'frames': b_frames,
                            'blinks': b_blinks,
                            'exp_type': exp_type,
                            'point': point
                        })

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Process Global Stats
    global_stats = calculate_stats([(d[0], d[1]) for d in all_data])
    print_section("Global Statistics", global_stats)

    for split in ["train", "val", "test"]:
        split_subjects = [(d[0], d[1]) for d in all_data if d[2] == split]
        if split_subjects:
            print_section(f"{split.upper()} Split", calculate_stats(split_subjects))
            
    # Process Granular Breakdowns
    if all_exp_data:
        df_exp = pd.DataFrame(all_exp_data)
        print_table("Statistics by Experiment Type", df_exp, "exp_type")
        print_table("Statistics by Point", df_exp, "point")
        print_table("Statistics by Exp Type & Point combination", df_exp, ["exp_type", "point"])

if __name__ == "__main__":
    get_dataset_stats()
