import os
import pandas as pd
import config

def generate_subject_mapping(output_path):
    # Get all subject directories from the raw dataset
    base_dir = config.get_dataset_base_directory()
    subject_dirs = config.get_dataset_subject_directories(rnd=False)
    
    mapping_data = []
    for i, s_dir in enumerate(sorted(subject_dirs)):
        # Generate an anonymous ID (e.g., subj_0001)
        subject_id = f"subj_{i+1:04d}"
        mapping_data.append({
            "subject_id": subject_id,
            "subject_dir": os.path.relpath(s_dir, base_dir)
        })
    
    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.to_csv(output_path, index=False)
    print(f"Subject mapping created with {len(mapping_df)} entries at: {output_path}")

if __name__ == "__main__":
    output_path = os.path.join(config.get_dataset_base_directory(), "subject_id_mapping.csv")
    generate_subject_mapping(output_path=output_path)
