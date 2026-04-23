import os
import numpy as np
import config
from data_loader import GazeDataLoader

def estimate_neutral_camera_position_in_world_frame():
    """
    This has been written to make sure the intersection points are very similar (almost identical) for all subjects.
    """
    SUBJECT_DIRS = config.get_dataset_subject_directories()

    print("Number of subjects: ", len(SUBJECT_DIRS))


    arr = np.zeros((len(SUBJECT_DIRS), 3), dtype=np.float64)

    for i, subject_dir in enumerate(SUBJECT_DIRS):
        exp_dir = f"{subject_dir}/lighting_10/p1"
        dataloader = GazeDataLoader(
            root_dir=exp_dir,
            target_period=config.get_target_period(),
            camera_pose_period=config.get_camera_pose_period(),
            time_diff_max=config.get_time_diff_max(),
            get_latest_subdirectory_by_name=True
        )
        camera_poses = dataloader.load_camera_poses(frame="world") # (n, 17), where 17 is [timestamp_in_ms, *flattened_homogeneous_transformation_matrix], in meters. load in the world frame
        camera_positions = camera_poses[:, (4,8,12)] # (n, 3), where 3 is [x, y, z]
        subject_mean = np.mean(camera_positions, axis=0)
        arr[i] = subject_mean

    mean, std = np.mean(arr, axis=0), np.std(arr, axis=0)
    print(f"neutral camera position in world frame - Mean, std: {mean}, {std}")
    # RESULT:
    # Number of subjects:  55
    # neutral camera position in world frame - Mean, std: [1.31863931 0.06311004 1.11644309], [0.00139032 0.0031499  0.00128769]

if __name__ == "__main__":
    # 1-estimate_neutral_camera_position_in_world_frame by processing all subjects (only the relevant experiment types)
    estimate_neutral_camera_position_in_world_frame()