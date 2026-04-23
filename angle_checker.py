

try:
    from config import get_neutral_cam_orientation_in_world_frame, get_neutral_head_orientation_in_cam_frame
except ModuleNotFoundError:
    from robot_controller.gaze.config import (
        get_neutral_cam_orientation_in_world_frame,
        get_neutral_head_orientation_in_cam_frame,
    )

from scipy.spatial.transform import Rotation as R
import numpy as np

def extract_roll_pitch_yaw_wrt_neutral_orientation(_R, neutral_R):
    """
    WARNING: Because we calculate the roll, pitch, and yaw angles with respect to a neutral orientation, whether return[0] is roll, pitch, or yaw depends on the neutral orientation. So, beware of the frame neutral orientation is in.
    Extracts the yaw angle from a given orientation with respect to a neutral orientation.

    Args:
        orientation (np.ndarray): 3x3 rotation matrix representing the current orientation.
        neutral_orientation (np.ndarray): 3x3 rotation matrix representing the neutral orientation.

    Returns:
        float: Yaw angle in degrees with respect to the neutral orientation.
    """
    if _R.shape != (3, 3) or neutral_R.shape != (3, 3):
        raise ValueError("Both orientation and neutral_orientation must be 3x3 rotation matrices.")

    # Compute the relative orientation
    R_rel = np.linalg.inv(neutral_R) @ _R
    r = R.from_matrix(R_rel)
    return r.as_euler('zyx', degrees=True)

def _extract_head_yaws(head_poses: np.ndarray, camera_poses: np.ndarray, neutral_R: np.ndarray) -> np.ndarray:
    """
    Returns the signed head yaw in degrees (not absolute) relative to the neutral orientation.
    
    Args:
        head_poses (np.ndarray): shape (N, 17), flattened 4x4 head pose matrices in world frame.
        camera_poses (np.ndarray): shape (N, 17), corresponding camera poses.
        neutral_R (np.ndarray): (3,3) matrix representing the neutral head orientation in camera frame.
    
    Returns:
        np.ndarray: shape (N,) of signed yaw anglcd robes in degrees.
    """
    if head_poses.shape[1] != 17:
        raise ValueError("Invalid shape for head_poses or camera_poses")

    # Transform head poses to camera frame
    transformed = []
    for i in range(min(head_poses.shape[0], camera_poses.shape[0])):
        head_mat = head_poses[i, 1:].reshape(4, 4)
        cam_mat = camera_poses[i, 1:].reshape(4, 4)
        cam_inv = np.linalg.inv(cam_mat)
        head_in_cam = cam_inv @ head_mat
        transformed.append(head_in_cam)

    head_yaws = []
    for i in range(len(transformed)):
        R_head = transformed[i][:3, :3]
        R_rel = R_head @ neutral_R.T
        yaw = R.from_matrix(R_rel).as_euler("zyx", degrees=True)[0]  # Extract yaw (around Z) in world frame
        head_yaws.append(yaw)

    return np.array(head_yaws)

def get_x_axis(rotation_matrix: np.ndarray) -> np.ndarray:
    return rotation_matrix[:, 0]

def print_gaze_unit_for_symmetry(gazes: np.ndarray, neutral_R: np.ndarray) -> np.ndarray:
    """
    Extracts the yaw angle between the gaze vector and the neutral x-axis direction in the camera frame.
    Yaw is computed in the XZ plane.
    
    Args:
        gazes (np.ndarray): (N, 4) array where each row is [timestamp, x, y, z].
        neutral_R (np.ndarray): 3x3 rotation matrix representing the neutral orientation (camera frame).

    Returns:
        np.ndarray: Array of yaw angles in radians (positive = left of head's x-axis, negative = right).
    """
    if gazes.shape[1] != 4:
        raise ValueError("Invalid shape for gazes")

    head_x_axis = get_x_axis(neutral_R)  # (3,)
    print("head_x_axis: ", head_x_axis)

    head_yaws = []
    for i in range(len(gazes)):
        gaze_vec = gazes[i, 1:].copy()
        gaze_vec /= np.linalg.norm(gaze_vec)  # Normalize
        print("gaze_vec: ", gaze_vec)
        exit()
        # Project both vectors onto XZ plane
        gaze_proj = np.array([gaze_vec[0], gaze_vec[1], 0.0])
        head_proj = np.array([head_x_axis[0], head_x_axis[1], 0.0])

        if np.linalg.norm(gaze_proj) < 1e-6 or np.linalg.norm(head_proj) < 1e-6:
            head_yaws.append(0.0)
            continue

        gaze_proj /= np.linalg.norm(gaze_proj)
        head_proj /= np.linalg.norm(head_proj)

        # Compute signed yaw angle using atan2 of the cross product
        cross_y = np.cross(head_proj, gaze_proj)[1]  # Y component of cross product
        dot = np.dot(head_proj, gaze_proj)
        yaw_angle = np.arctan2(cross_y, dot)  # signed angle in radians

        head_yaws.append(yaw_angle)

    return np.array(head_yaws)


def extract_head_yaws(dataloader) -> np.ndarray:
    """
    Extracts the head yaw angles from the data loader.
    
    Args:
        dataloader (GazeDataLoader): Data loader instance with head poses loaded.
    
    Returns:
        np.ndarray: Array of head yaw angles in degrees.
    """
    head_poses = dataloader.load_head_poses()
    camera_poses = dataloader.load_camera_poses()
    neutral_R = get_neutral_head_orientation_in_cam_frame()
    return _extract_head_yaws(head_poses, camera_poses, neutral_R)

def extract_camera_yaw(camera_pose, neutral_R):
    """
    Extracts the yaw angle of a single camera pose relative to the neutral camera orientation.

    Args:
        camera_pose (np.ndarray): 4x4 matrix representing the camera pose in world frame.
        neutral_R (np.ndarray): 3x3 matrix representing the neutral camera orientation in world frame.

    Returns:
        float: Yaw angle in degrees.
    """
    R_cam = camera_pose[:3, :3]
    R_rel = np.linalg.inv(neutral_R) @ R_cam
    return R.from_matrix(R_rel).as_euler("zyx", degrees=True)[0]  # Extract yaw (around Z)

def extract_camera_yaws(camera_poses, neutral_R):
    """
    Extracts the yaw angle of camera poses relative to the neutral camera orientation.
    
    Args:
        camera_poses (np.ndarray): shape (N, 17), flattened 4x4 camera pose matrices in world frame.
        neutral_R (np.ndarray): (3,3) matrix representing the neutral camera orientation in world frame.
    
    Returns:
        np.ndarray: shape (N,) of yaw angles in degrees.
    """
    if camera_poses.shape[1] != 17:
        raise ValueError("Invalid shape for camera_poses")

    camera_yaws = []
    for i in range(camera_poses.shape[0]):
        cam_mat = camera_poses[i, 1:].reshape(4, 4)
        camera_yaws.append(extract_camera_yaw(cam_mat, neutral_R))

    return np.array(camera_yaws)


def print_camera_yaws(data_loader):
    """
    Prints the camera yaw angles relative to the neutral orientation.
    
    Args:
        data_loader (GazeDataLoader): Data loader instance with camera poses loaded.
    """
    camera_poses = data_loader.load_camera_poses()
    neutral_R = get_neutral_cam_orientation_in_world_frame()
    camera_yaws = extract_camera_yaws(camera_poses, neutral_R)
    
    print("Camera Yaws (degrees):")
    for i, yaw in enumerate(camera_yaws):
        print(f"Frame {i}: {yaw:.2f}°")


if __name__ == "__main__":
    from data_loader import GazeDataLoader
    import config
    exp_dir = ""

    dataloader = GazeDataLoader(
        root_dir=exp_dir,
        target_period=config.get_target_period(),
        camera_pose_period=config.get_camera_pose_period(),
        time_diff_max=config.get_time_diff_max(),
        get_latest_subdirectory_by_name=True
    )
    # head_yaws = extract_head_yaws(dataloader)
    # print("Median Head Yaw: ", np.median(head_yaws))
    # exit()
    camera_poses = dataloader.load_camera_poses()
    print_camera_yaws(dataloader)
    exit()
    # gaze ground truth left-right symmetry check
    gazes_in_cam_frame = dataloader.load_gaze_ground_truths(frame="camera")
    print_gaze_unit_for_symmetry(gazes_in_cam_frame, neutral_R)

