import os
import numpy as np
import config
from data_loader import GazeDataLoader
import csv

def estimate_neutral_eye_position_in_world_frame(subject_dir):
    """
    Neutral eye frame shall be obtained per subject by using the head_pose_middle experiments.
    Just use the median eye position (just position, no orientation) through head_pose_middle experiments.
    For orientation, use the ideal "neutral_head_orientation_in_world_frame" from config.
    """
    points = config.get_point_variations()["head_pose_middle"]
    arr = np.zeros((len(points), 3), dtype=np.float64)

    for i, point in enumerate(points):
        exp_dir = f"{subject_dir}/head_pose_middle/{point}"
        dataloader = GazeDataLoader(
            root_dir=exp_dir,
            target_period=config.get_target_period(),
            camera_pose_period=config.get_camera_pose_period(),
            time_diff_max=config.get_time_diff_max(),
            get_latest_subdirectory_by_name=True
        )
        eye_positions = dataloader.load_eye_positions()[:,1:] # (n, 3), where 3 is [x, y, z], in meters. load in the world frame
        arr[i] = np.mean(eye_positions, axis=0)

    mean, std = np.mean(arr, axis=0), np.std(arr, axis=0)
    print(f"neutral eye position in world frame - Mean, std: {mean}, {std}")
    return mean, std

def save_neutral_eye_positions(results, base_dir=config.get_neutral_eye_position_per_subject_csv_path()):
    """
    Save mean and std eye positions per subject to a CSV file.

    Args:
        base_dir (str): Directory where CSV will be saved.
        results (list[tuple]): Each entry is (subject_dir, mean, std).
    """
    csv_path = os.path.join(base_dir, "neutral_eye_position_per_subject.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_dir", "mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"])
        for subject_dir, mean, std in results:
            writer.writerow([subject_dir, *mean, *std])
    print(f"Saved results to {csv_path}")

def save_neutral_eye_positions_for_all_subjects(subject_dirs, base_dir):    
    results = []
    for subject_dir in subject_dirs:
        mean, std = estimate_neutral_eye_position_in_world_frame(subject_dir)
        results.append((subject_dir, mean, std))

    save_neutral_eye_positions(results, base_dir)

def _load_neutral_eye_position(subject_dir, csv_path):
    """
    Load the neutral eye position (mean and std) for a given subject from the CSV.

    Args:
        subject_dir (str): Subject directory string as saved in the CSV.
        csv_path (str): Path to the CSV file.

    Returns:
        tuple: (mean_position, std_position), each as np.ndarray of shape (3,)
    """
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["subject_dir"] == subject_dir:
                mean = np.array([float(row["mean_x"]), float(row["mean_y"]), float(row["mean_z"])])
                std = np.array([float(row["std_x"]), float(row["std_y"]), float(row["std_z"])])
                return mean, std

    raise ValueError(f"Subject {subject_dir} not found in {csv_path}")

def load_neutral_eye_pose_in_world_frame(subject_dir, csv_path):
    mean, std = _load_neutral_eye_position(subject_dir, csv_path)
    m = np.zeros((4,4))
    m[:3,:3] = config.get_neutral_eye_orientation_in_world_frame()
    m[:3,3] = mean.T
    m[3,3] = 1.0
    return m

if __name__ == "__main__":
    SUBJECT_DIRS = config.get_dataset_subject_directories()
    BASE_DIR = config.get_dataset_base_directory()

    # calculate and save to csv
    save_neutral_eye_positions_for_all_subjects(subject_dirs=SUBJECT_DIRS, base_dir=BASE_DIR)

    # Example usage on how to load for a subject
    csv_path = config.get_neutral_eye_position_per_subject_csv_path()
    subject_dir = SUBJECT_DIRS[0]
    mean_pos, std_pos = _load_neutral_eye_position(subject_dir, csv_path)
    print("Mean:", mean_pos, "Std:", std_pos)

