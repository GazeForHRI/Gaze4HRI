import os
import numpy as np
import cv2 as cv

try:
    # Try relative import if running as part of a package (i.e.: run through ROS)
    import robot_controller.gaze.config as config
except ImportError:
    # Fallback for standalone usage (run as a normal script, not through ROS)
    import config

try:
    # Try relative import if running as part of a package (i.e.: run through ROS)
    from .data_matcher import match_regular_to_regular, match_irregular_to_regular
except ImportError:
    # Fallback for standalone usage (run as a normal script, not through ROS)
    from data_matcher import match_regular_to_regular, match_irregular_to_regular

class GazeDataLoader:
    def __init__(self, root_dir, target_period, camera_pose_period, time_diff_max, get_latest_subdirectory_by_name=True):
        """
        Initializes the GazeDataLoader.

        Args:
            root_dir (str): Root directory containing experiment data.
            target_period (float): Sampling period (in milliseconds) of the target_positions data.
            camera_pose_period (float): Sampling period (in milliseconds) of the camera_poses data.
            time_diff_max (float): Maximum allowable time difference (in milliseconds) for data matching.
            get_latest_subdirectory_by_name (bool): If True, selects the latest subdirectory under root_dir.
        """
        self.root_dir = root_dir
        self.cwd = os.path.join(root_dir, self.get_latest_subdirectory_by_name(root_dir) if get_latest_subdirectory_by_name else "")
        self.target_period = target_period
        self.camera_pose_period = camera_pose_period
        self.head_pose_period = camera_pose_period
        self.time_diff_max = time_diff_max

    def get_latest_subdirectory_by_name(self, directory):
        subdirs = [os.path.join(directory, d) for d in os.listdir(directory)
                   if os.path.isdir(os.path.join(directory, d))]
        if not subdirs:
            raise FileNotFoundError(f"No subdirectories found in {directory}")
        return os.path.basename(max(subdirs, key=os.path.getmtime))

    def get_subject_dir(self):
        return '/'.join(self.cwd.split('/')[:-3])  # Assumes structure "subject_dir/exp_type/point/timestamp"

    def load(self, name):
        path = os.path.join(self.cwd, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        return np.load(path)

    def load_eye_positions(self):
        return self.load("eye_positions")

    def load_target_positions(self, frame="world", neutral_eye_pose_in_world_frame = None):
        """Loads target positions in the specified frame.

        Args:
            frame (str, optional): The frame to return the target positions in. Defaults to "world".
            neutral_eye_pose_in_world_frame (np.ndarray, optional): The neutral eye pose in the world frame. load the subject's neutral eye pose by calling neutral_eye_position_calculation.py's load_neutral_eye_position function.

        Raises:
            ValueError: If frame is "neutral_eye" and neutral_eye_pose_in_world_frame is None.

        Returns:
            np.ndarray: The target positions in the specified frame.
        """
        target_positions_in_world = self.load("target_positions") # (n,4), where 4 is [timestamp_in_ms, x, y, z]
        if frame == "world":
            return target_positions_in_world
        elif frame == "neutral_eye":
            if neutral_eye_pose_in_world_frame is None:
                raise ValueError("neutral_eye_pose_in_world_frame must be provided when frame is 'neutral_eye'")

            neutral_eye_mat_inv = np.linalg.inv(neutral_eye_pose_in_world_frame)
            for i in range(target_positions_in_world.shape[0]):
                position = np.array([*target_positions_in_world[i, 1:], 1.0]) # add 1 to the end to homogenize.
                pos_in_neutral_eye = neutral_eye_mat_inv @ position
                target_positions_in_world[i, 1:] = pos_in_neutral_eye[:-1]
            return target_positions_in_world # now all in neutral eye frame, not world
        elif frame == "neutral_camera":
            neutral_cam_mat_inv = np.linalg.inv(config.get_neutral_cam_pose_in_world_frame())
            for i in range(target_positions_in_world.shape[0]):
                position = np.array([*target_positions_in_world[i, 1:], 1.0]) # add 1 to the end to homogenize.
                pos_in_neutral_cam = neutral_cam_mat_inv @ position
                target_positions_in_world[i, 1:] = pos_in_neutral_cam[:-1]
            return target_positions_in_world # now all in neutral eye frame, not world
        else:
            raise ValueError(f"Invalid frame type: {frame}. Use 'world', 'neutral_eye', or 'neutral_camera'.")

    def load_camera_poses(self, frame="world"):
        poses_in_world = self.load("camera_poses")
        if frame == "world":
            return poses_in_world
        elif frame == "neutral_camera":
            neutral_cam_mat_inv = np.linalg.inv(config.get_neutral_cam_pose_in_world_frame())
            for i in range(poses_in_world.shape[0]):
                pose_mat = self.flattened_to_homogeneous_matrix(poses_in_world[i, 1:])
                pose_in_neutral_cam = neutral_cam_mat_inv @ pose_mat
                poses_in_world[i, 1:] = pose_in_neutral_cam.flatten()

            return poses_in_world # now all in neutral camera frame, not world
        else:
            raise ValueError(f"Invalid frame type: {frame}. Use 'world' or 'neutral_camera'.")

    def load_head_poses(self, frame="world"):
        head_poses = self.load("head_poses")
        if frame == "world":
            return head_poses
        elif "camera":
            return self.transform_head_poses_to_camera_frame(
                head_poses,
                self.load_camera_poses(),
            )
        else:
            raise ValueError(f"Invalid frame type: {frame}. Use 'camera' or 'world'.")
    
    def load_table_pose(self):
        return self.load("table_pose")

    def load_rgb_video(self, as_numpy=False):
        """
        Loads rgb_video.mp4 as either a list of frames or returns path to the video file.

        Args:
            as_numpy (bool): If True, returns list of frames as numpy arrays.
                            If False, returns the file path string.

        Returns:
            List[np.ndarray] or str: RGB frames or file path, depending on as_numpy.
        """
        video_path = os.path.join(self.get_cwd(), "rgb_video.mp4")

        if not os.path.exists(video_path):
            print(f"[load_rgb_video] Warning: {video_path} not found.")
            return [] if as_numpy else None

        if not as_numpy:
            return video_path

        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[load_rgb_video] Error: Unable to open video file {video_path}")
            return []

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def load_rgb_timestamps(self):
        return self.load("rgb_timestamps")

    def load_head_bboxes(self):
        return self.load("head_bboxes")

    def get_gaze_estimations_dir(self, model):
        if model is None:
            raise ValueError("Model name cannot be None. None is no longer supported")
        return os.path.join(self.cwd, f"gaze_estimations/{model}")
    
    def load_gaze_estimation_valid_indices(self, model):
        path = os.path.join(self.get_gaze_estimations_dir(model=model), "gaze_directions_indices.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Gaze estimation indices file not found: {path}")
        return np.load(path)

    def load_gaze_estimations(self, model, frame="camera"):
        path = os.path.join(self.get_gaze_estimations_dir(model=model), "gaze_directions.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Gaze estimation file not found: {path}")
        est = np.load(path)
        
        if frame == "camera":
            return est
        elif "world":
            return self.transform_gaze_estimations_between_camera_and_world_frame(
                self.load_camera_poses(),
                est,
                cam_to_world=True
            )
        else:
            raise ValueError(f"Invalid frame type: {frame}. Use 'camera' or 'world'.")

    def load_gaze_estimations_full_tensor(self, model, frame="camera"):
        """
        Load N×M×3 full-tensor gaze estimates saved by MCGaze.

        Mirrors load_gaze_estimations() behavior, including the exact same
        camera<->world transformation path (irregular=True pipeline).

        Args:
            model (str): subdir under .../gaze_estimations/ (e.g., 'mcgaze_clip_size_7')
            frame (str): 'camera' (as saved) or 'world'

        Returns:
            np.ndarray: (N, M, 3) float64
        """
        base_dir = self.get_gaze_estimations_dir(model=model)
        full_path = os.path.join(base_dir, "gaze_directions_full.npy")
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Gaze full-tensor file not found: {full_path}")

        full = np.load(full_path)  # (N, M, 3), right-aligned per row, left zero-pad
        if frame == "camera":
            return full
        if frame != "world":
            raise ValueError(f"Invalid frame type: {frame}. Use 'camera' or 'world'.")

        # ---- EXACT same transform path as load_gaze_estimations(..., frame='world') ----
        # Build an "irregular" Nx4 list from valid slots (non-zero vectors),
        # run transform_gaze_estimations_between_camera_and_world_frame,
        # then reshape back to (N, M, 3) with zeros preserved for padding.

        rgb_ts = self.load_rgb_timestamps()  # (N,)
        N, M, _ = full.shape
        if rgb_ts.shape[0] < N:
            N = rgb_ts.shape[0]
            full = full[:N]

        # Flatten valid slots into irregular list [timestamp, x, y, z]
        # Slot index s in [0..M-1] maps to frame index f via: f = t - (M-1 - s)
        rows = []
        idx_map = []  # to put transformed vectors back: list of (t, s)
        for t in range(N):
            for s in range(M):
                v = full[t, s]
                if not np.any(v):  # zero-padding => skip
                    continue
                f = t - (M - 1 - s)
                if f < 0 or f >= N:
                    continue
                rows.append([float(rgb_ts[f]), float(v[0]), float(v[1]), float(v[2])])
                idx_map.append((t, s))

        if not rows:
            # Nothing valid; return original (all zeros)
            return full

        irregular = np.asarray(rows, dtype=np.float64)  # shape (K, 4)

        # Use the same transformation route as existing loader for 'world'
        cam_poses = self.load_camera_poses(frame="world")  # (Kc, 17)
        irregular_world = self.transform_gaze_estimations_between_camera_and_world_frame(
            cam_poses,
            irregular,
            cam_to_world=True
        )

        # Rebuild (N, M, 3) with zeros preserved where slots were padding
        out = np.copy(full)
        for (t, s), row in zip(idx_map, irregular_world):
            out[t, s, :] = row[1:]  # [ts, x, y, z] -> take vector part

        return out

    def calculate_gaze_ground_truths(self, target_positions, eye_positions):
        res = target_positions - eye_positions
        res[:,0] = target_positions[:,0]
        res[:,1:] = res[:,1:] / np.linalg.norm(res[:,1:], axis=1, keepdims=True)
        return res

    def load_gaze_ground_truths(self, frame="camera"):
        targets = self.load_target_positions()
        eyes = self.load_eye_positions()
        gt_world = self.calculate_gaze_ground_truths(targets, eyes)
        if frame == "world":
            return gt_world
        elif frame == "camera":
            cam_poses = self.load_camera_poses()
            return self.transform_gaze_ground_truths_between_camera_and_world_frame(cam_poses, gt_world, cam_to_world=False)

    def transform_gaze_estimations_between_camera_and_world_frame(self, camera_poses, gaze_vectors, cam_to_world=True):
        return self.transform_between_camera_and_world_frame(camera_poses, gaze_vectors, cam_to_world, irregular=True)

    def transform_gaze_ground_truths_between_camera_and_world_frame(self, camera_poses, gaze_vectors, cam_to_world=True):
        return self.transform_between_camera_and_world_frame(camera_poses, gaze_vectors, cam_to_world, irregular=False)

    def transform_between_camera_and_world_frame(self, camera_poses, gaze_vectors, cam_to_world=True, irregular=True):
        gaze_vectors = np.copy(gaze_vectors)
        if irregular:
            gaze_vectors, matched_cam = match_irregular_to_regular(
                irregular_data=gaze_vectors,
                regular_data=camera_poses,
                regular_period_ms=self.camera_pose_period
            )
            for i in range(gaze_vectors.shape[0]):
                pose4x4 = self.flattened_to_homogeneous_matrix(matched_cam[i])
                if not cam_to_world:
                    pose4x4 = np.linalg.inv(pose4x4)
                R = pose4x4[:3, :3]
                gv = gaze_vectors[i, 1:]
                rotated = R @ gv
                rotated /= np.linalg.norm(rotated)
                gaze_vectors[i, 1:] = rotated
        else:
            idx_cam, idx_gv = match_regular_to_regular(
                (camera_poses, self.camera_pose_period),
                (gaze_vectors, self.target_period),
                max_match_diff_ms=5.0
            )
            for ic, ig in zip(idx_cam, idx_gv):
                pose4x4 = self.flattened_to_homogeneous_matrix(camera_poses[ic, 1:])
                if not cam_to_world:
                    pose4x4 = np.linalg.inv(pose4x4)
                R = pose4x4[:3, :3]
                gv = gaze_vectors[ig, 1:]
                rotated = R @ gv
                rotated /= np.linalg.norm(rotated)
                gaze_vectors[ig, 1:] = rotated

        return gaze_vectors

    def transform_head_poses_to_camera_frame(self, head_poses: np.ndarray, camera_poses: np.ndarray) -> np.ndarray:
        """
        Transforms head poses from world frame to camera frame using corresponding camera poses.

        Args:
            head_poses (np.ndarray): Array of shape (N, 17).
            camera_poses (np.ndarray): Array of shape (N, 17).

        Returns:
            np.ndarray: Transformed head poses of shape (N, 17).
        """
        if head_poses.shape[1] != 17 or camera_poses.shape[1] != 17:
            raise ValueError("Input arrays must have 17 columns (flattened 4x4 matrices + timestamp).")

        n_head, n_cam = head_poses.shape[0], camera_poses.shape[0]
        if n_head != n_cam:
            if abs(n_head - n_cam) > 1:
                raise ValueError("Input arrays must be of shape (N, 17) and match in size (difference > 1).")
            # Trim longer array to match the shorter
            min_len = min(n_head, n_cam)
            head_poses = head_poses[:min_len]
            camera_poses = camera_poses[:min_len]

        transformed = []
        for i in range(head_poses.shape[0]):
            ts = head_poses[i, 0]
            head_mat = head_poses[i, 1:].reshape(4, 4)
            cam_mat = camera_poses[i, 1:].reshape(4, 4)
            cam_inv = np.linalg.inv(cam_mat)
            head_in_cam = cam_inv @ head_mat
            transformed_row = np.concatenate(([ts], head_in_cam.flatten()))
            transformed.append(transformed_row)

        return np.array(transformed)

    def flattened_to_homogeneous_matrix(self, array):
        return array.reshape(4, 4)

    def get_cwd(self):
        return self.cwd
    
    def get_blink_annotations(self, annotator=None):
        """
        Load blink annotations. If annotator is None, auto-detect the first file named
        'blink_annotations_by_*.npy' in the current working directory (self.get_cwd()).

        Returns
        -------
        np.ndarray (int16)
        """
        dirpath = self.get_cwd()

        if annotator is None:
            # Find first matching file in cwd
            candidates = [
                f for f in os.listdir(dirpath)
                if f.startswith("blink_annotations_by_") and f.endswith(".npy")
            ]
            if not candidates:
                raise FileNotFoundError(
                    f"No blink_annotations_by_*.npy found in {dirpath}"
                )
            candidates.sort()  # make selection deterministic
            path = os.path.join(dirpath, candidates[0])
        else:
            # Use explicit annotator
            path = os.path.join(dirpath, f"blink_annotations_by_{annotator}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

        arr = np.load(path)

        # Normalize dtype to int16 (some files were saved with wrong dtype)
        if arr.dtype != np.int16:
            arr = arr.astype(np.int16)

        return arr

    def load_ur5_joint_states(self):
        return self.load("ur5_joint_states")
    
    def load_ur5_base_pose(self):
        return self.load("ur5_base_pose")