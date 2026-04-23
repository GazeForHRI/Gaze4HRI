### SHARED CODE TO TEST PARTICULAR HYPOTHESIS AND DISCOVER NEW PATTERNS:
import os
import numpy as np

def get_latest_subdirectory_by_name(parent_directory):
    try:
        subdirs = [d for d in os.listdir(parent_directory)
                   if os.path.isdir(os.path.join(parent_directory, d))]
        if not subdirs:
            raise Exception(f"No subdirectories in {parent_directory}")
        return max(subdirs)
    except Exception as e:
        raise FileNotFoundError(f"{parent_directory}: {e}")

def print_array(arr):
    arr = np.asarray(arr)

    if arr.ndim == 1:
        print("[" + " ".join(f"{x:.2f}" for x in arr) + "]")
    elif arr.ndim == 2:
        for row in arr:
            print("[" + " ".join(f"{x:.2f}" for x in row) + "]")
    else:
        raise ValueError("Only 1D or 2D arrays are supported.")

# ------------- Utility functions ------------- #

def extract_camera_yaw_from_head_in_cam(head_R: np.ndarray, neutral_R: np.ndarray) -> float:
    cam_R = neutral_R @ head_R.T
    yaw_rad = np.arctan2(cam_R[1, 0], cam_R[0, 0])
    return np.degrees(yaw_rad)

def extract_head_yaw_deviation(head_poses_cam: np.ndarray, neutral_R: np.ndarray) -> np.ndarray:
    return np.array([
        extract_camera_yaw_from_head_in_cam(row[1:].reshape(4, 4)[:3, :3], neutral_R)
        for row in head_poses_cam
    ])

