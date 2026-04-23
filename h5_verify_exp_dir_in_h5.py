import os
import h5py
import config

def verify_h5_contents():
    base_dir = config.get_dataset_base_directory()
    dataset_dir = os.path.join(base_dir, "blink4hri_torch_dataset")
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return

    h5_files = [f for f in os.listdir(dataset_dir) if f.endswith('.h5')]
    if not h5_files:
        print("No H5 files found in the dataset directory.")
        return
        
    test_file = h5_files[0]
    h5_path = os.path.join(dataset_dir, test_file)

    print(f"--- Inspecting File: {test_file} ---")
    
    with h5py.File(h5_path, 'r') as h5:
        if 'exp_boundaries' not in h5 or 'exp_dir' not in h5:
            print("This H5 file does not have boundaries or exp_dir stored.")
            return
            
        boundaries = h5['exp_boundaries'][:]
        print(f"Found {len(boundaries)} experiment boundaries.")
        print("Directly extracting strings from H5 arrays for the first 5 experiments:\n")
        
        for i in range(min(5, len(boundaries))):
            start, end = boundaries[i]
            
            # Since the array holds strings for every frame, we just read the first frame of the boundary
            # H5 string objects return as bytes, so we .decode('utf-8') them
            exp_dir_str = h5['exp_dir'][start].decode('utf-8')
            exp_type_str = h5['exp_type'][start].decode('utf-8')
            point_str = h5['point'][start].decode('utf-8')
            
            print(f"Boundary {i} [Frames {start}:{end}]")
            print(f"  -> Extracted exp_dir:  {exp_dir_str}")
            print(f"  -> Extracted exp_type: {exp_type_str}")
            print(f"  -> Extracted point:    {point_str}\n")

if __name__ == "__main__":
    verify_h5_contents()
