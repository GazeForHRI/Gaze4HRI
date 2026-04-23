import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from data_matcher import match_irregular_to_regular
from scipy.spatial.transform import Rotation as R
from frame_db import append_frame_results

_LABEL_UNCERTAIN = 2
_LABEL_BLINK = 3

def pretty_print_full(full, n_rows=5, precision=3):
    """
    Pretty print the (N, M, 3) full tensor row by row.
    
    Args:
        full (np.ndarray): Shape (N, M, 3)
        n_rows (int): How many rows to print
        precision (int): Decimal places for floats
    """
    np.set_printoptions(precision=precision, suppress=True)
    N, M, _ = full.shape
    print(f"Full tensor shape: {full.shape}")
    print("-" * 80)
    for i in range(min(n_rows, N)):
        row = full[i]  # shape (M, 3)
        print(f"Row {i}:")
        for j, vec in enumerate(row):
            print(f"  [{j:02d}] {vec}")
        print("-" * 80)
    np.set_printoptions()  # reset to default


def pretty_print_array(arr, precision=4, width=15):
    # Make sure it's a NumPy array
    arr = np.array(arr)

    # Build format string like "{:15.4f}"
    fmt = f"{{:{width}.{precision}f}}"

    # Convert each element to formatted string
    formatted = np.vectorize(lambda x: fmt.format(x))(arr)

    # Join row-wise to mimic NumPy's print
    for row in formatted:
        print(" ".join(row) if isinstance(row, (np.ndarray, list)) else row)


def direction_error(est, label):
    est = np.array(est)
    label = np.array(label)
    est_norm = est / np.linalg.norm(est)
    label_norm = label / np.linalg.norm(label)
    dot_product = np.clip(np.dot(est_norm, label_norm), -1.0, 1.0)
    angular_error = np.degrees(np.arccos(dot_product))
    euclidean_error = np.linalg.norm(est - label)
    return angular_error, euclidean_error


# --- add these near the top with your other imports/constants if you want explicit labels ---
_LABEL_UNCERTAIN = 2
_LABEL_BLINK = 3

def process_direction_errors(dataloader, model, gaze_target_bias=None, save_to_file=True,
                             use_full_tensor=False, full_tensor_frame_to_select=None):
    if use_full_tensor:
        if type(full_tensor_frame_to_select) is not int:
            raise ValueError("When use_full_tensor is True, full_tensor_frame_to_select must be an integer indicating which frame to select from the full tensor. set -1 to select the latest frame, 0 to select the oldest frame.")
        skip_zero_padding = True
        full = dataloader.load_gaze_estimations_full_tensor(model=model, frame="world")  # (N, M, 3)
        N, M, _ = full.shape
        col = full_tensor_frame_to_select  # negative indexing okay
        selected_vecs = full[:, col, :]                                                # (N, 3)

        # 2) Build irregular [ts, x, y, z]
        timestamps = dataloader.load_rgb_timestamps()[:N].astype(np.float64)           # (N,)
        gaze_estimations = np.column_stack([timestamps, selected_vecs])                # (N, 4)

        # 3) Optionally drop padded zeros
        if skip_zero_padding:
            valid_mask = np.linalg.norm(gaze_estimations[:, 1:], axis=1) > 0.0
            if not np.any(valid_mask):
                print("No valid (non-zero) estimates to evaluate.")
                return np.empty((0, 4))
            gaze_estimations = gaze_estimations[valid_mask]
    else:
        gaze_estimations = dataloader.load_gaze_estimations(model=model, frame="world") # (N, 4)

    gaze_ground_truth = dataloader.load_gaze_ground_truths(frame="world")

    # Align to regular stream (keeps only rows at/after stable start of GT)
    gaze_estimations_aligned, matched_gt_vals = match_irregular_to_regular(
        irregular_data=gaze_estimations,
        regular_data=gaze_ground_truth,
        regular_period_ms=dataloader.target_period
    )

    if gaze_estimations_aligned.shape[0] == 0:
        # nothing survives the stable-start filter
        return np.empty((0, 4))

    # --- new: build validity by looking up each aligned timestamp in RGB timestamps ---
    # Assumption per your note: rgb_ts, gaze estimations, and annotations are 1:1 by timestamp.
    error_results = []  # rows: [ts, ang_err, euc_err, is_valid]
    try:
        rgb_ts = dataloader.load_rgb_timestamps().astype(np.float64).reshape(-1)  # (T,)
        blink_ann = dataloader.get_blink_annotations()  # shape (T,), ints {0,1,2,3}
        # map exact timestamp -> index (fast O(1) lookups)
        ts_to_idx = {float(t): i for i, t in enumerate(rgb_ts)} if rgb_ts.size else {}
    except Exception as e:
        rgb_ts, blink_ann, ts_to_idx = np.array([]), None, {}
        print(f"[WARN] Could not prepare blink lookup; treating all frames as valid. {e}")

    # we mistakenly caused a 3 frame shift while saving results for gaze3d_clip_len_8, so apply a reverse shift to offset that.
    if model.startswith("gaze3d_clip_len_8"):
        i_est_max = gaze_estimations_aligned.shape[0]-3 # add -3 offset for gaze3d_clip_len_8
    else:
        i_est_max = gaze_estimations_aligned.shape[0]
    for i_est in range(i_est_max):
        ts = float(gaze_estimations_aligned[i_est, 0])
        est_vector = gaze_estimations_aligned[i_est-(3 if model.startswith("gaze3d_clip_len_8") else 0), 1:].copy() # add -3 offset for gaze3d_clip_len_8
        label_vector = matched_gt_vals[i_est]

        if gaze_target_bias is not None:
            est_vector = est_vector + gaze_target_bias

        if np.linalg.norm(est_vector) < 1e-6:
            ang_err = 0.0
            euc_err = 0.0
            is_valid = 0.0
        else:
            ang_err, euc_err = direction_error(est_vector, label_vector)

            # default: valid=1.0
            is_valid = 1.0
            if blink_ann is not None and ts_to_idx:
                idx = ts_to_idx.get(ts, None)
                if idx is not None:
                    lab = int(blink_ann[idx])
                    # INVALID if {UNCERTAIN=2, BLINK=3}; VALID otherwise (0 or 1)
                    if lab == _LABEL_UNCERTAIN or lab == _LABEL_BLINK:
                        is_valid = 0.0
                # if model.startswith("gaze3d_clip_len_8"): #@TODO: temoporary for debug, remove later. this is for gaze3d_clip_len_8. we will consider invalid even if any of the last 8 frames ([t-7, t]) is a blink
                #     idx = ts_to_idx.get(ts, None)
                #     if idx is not None:
                #         for _idx in range(max(idx-9, 0), min(idx+9, blink_ann.shape[0]-1)):
                #             lab = int(blink_ann[_idx])
                #             # INVALID if {UNCERTAIN=2, BLINK=3}; VALID otherwise (0 or 1)
                #             if lab == _LABEL_UNCERTAIN or lab == _LABEL_BLINK:
                #                 is_valid = 0.0
                #                 break
                # else:
                #     idx = ts_to_idx.get(ts, None)
                #     if idx is not None:
                #         lab = int(blink_ann[idx])
                #         # INVALID if {UNCERTAIN=2, BLINK=3}; VALID otherwise (0 or 1)
                #         if lab == _LABEL_UNCERTAIN or lab == _LABEL_BLINK:
                #             is_valid = 0.0

        error_results.append([ts, ang_err, euc_err, is_valid])

    if not error_results:
        print("No estimations found.")
        return np.array(error_results, dtype=np.float64)

    if save_to_file:
        save_dir = os.path.join(dataloader.get_gaze_estimations_dir(model=model), "direction_errors")
        os.makedirs(save_dir, exist_ok=True)
        suffix = "" if gaze_target_bias is None else \
                f"_bias=({gaze_target_bias[0]:.2f},{gaze_target_bias[1]:.2f},{gaze_target_bias[2]:.2f})"
        filename = os.path.join(save_dir, suffix)

        save_direction_error_results(filename, error_results)

        try:
            error_results_sorted = error_results
            append_frame_results(
                exp_dir_abs=dataloader.get_cwd(),
                gaze_model=model,
                error_results_sorted=error_results_sorted
            )
        except Exception as e:
            print(f"[WARN] Failed to append frame results for {dataloader.get_cwd()} ({model}): {e}")

    return np.array(error_results, dtype=np.float64)


def save_direction_error_results(filename, error_results):
    """
    Accepts rows [ts, ang_err, euc_err, is_valid].
    Writes num_frames & num_valid_frames.
    Computes stats from valid-only frames.
    Timeline plot shows ALL frames (valid + invalid).
    """
    error_results = np.array(error_results, dtype=np.float64)

    num_frames = int(error_results.shape[0])
    valid_mask = (error_results[:, 3] > 0.5) if num_frames > 0 else np.array([], dtype=bool)
    num_valid_frames = int(np.count_nonzero(valid_mask))

    # --- stats from VALID frames only ---
    if num_valid_frames > 0:
        valid = error_results[valid_mask]
        angular_errors = valid[:, 1].flatten()
        euclidean_errors = valid[:, 2].flatten()

        mean_ang = float(np.mean(angular_errors))
        std_ang  = float(np.std(angular_errors))
        med_ang  = float(np.median(angular_errors))

        mean_euc = float(np.mean(euclidean_errors))
        std_euc  = float(np.std(euclidean_errors))
        med_euc  = float(np.median(euclidean_errors))
    else:
        angular_errors = np.array([], dtype=np.float64)
        euclidean_errors = np.array([], dtype=np.float64)
        mean_ang = std_ang = med_ang = np.nan
        mean_euc = std_euc = med_euc = np.nan

    stats = (
        f"num_frames={num_valid_frames}\n" # for backward compatibility with weighted aggregations and stuff.
        f"num_all_frames={num_frames}\n"
        f"angular_error(valid-only): mean={mean_ang:.4f}, median={med_ang:.4f}, std={std_ang:.4f}\n"
        f"euclidean_error(valid-only): mean={mean_euc:.4f}, median={med_euc:.4f}, std={std_euc:.4f}"
    )

    with open(filename + "errors.txt", "w") as f:
        f.write(stats)

    print("-" * 20)
    print("Angular Error Statistics (valid-only):")
    print(stats)
    print("-" * 20)

    # --- PLOT 1: Timeline with ALL frames (valid + invalid) ---
    if num_frames > 0:
        plt.figure()
        ts_all = error_results[:, 0] - error_results[0, 0]
        ang_all = error_results[:, 1].flatten()
        plt.plot(ts_all, ang_all, label="All frames")
        plt.title("Angular Error Over Time (all frames)")
        plt.xlabel("Time (ms)")
        plt.ylabel("Angular Error (degrees)")
        plt.grid()
        plt.savefig(filename + "timeline_plot.png")
        plt.close()

    # --- PLOT 2: Histogram from valid-only frames ---
    if num_valid_frames > 0:
        plt.figure()
        weights = np.ones_like(angular_errors) / len(angular_errors) * 100
        plt.hist(angular_errors, bins=50, weights=weights)
        plt.axvline(np.nanmean(angular_errors), color="red", linestyle="--", label="Mean")
        plt.axvline(np.nanmedian(angular_errors), color="green", linestyle="--", label="Median")
        plt.legend()
        plt.title("Angular Error Distribution (valid-only)")
        plt.xlabel("Angular Error (degrees)")
        plt.ylabel("Frequency (%)")
        plt.grid()
        plt.savefig(filename + "freq_dist_plot.png")
        plt.close()


def parse_direction_errors_file(directory_path):
    """
    Parse the errors.txt file saved by save_direction_error_results and extract angular error stats.
    
    Args:
        directory_path (str): Path to the directory containing the errors.txt file.

    Returns:
        dict: Dictionary with 'mean', 'median', and 'std' for angular error.
    """
    error_txt_path = os.path.join(directory_path, "direction_errors", "errors.txt")
    if not os.path.exists(error_txt_path):
        raise FileNotFoundError(f"{error_txt_path} not found")

    with open(error_txt_path, "r") as f:
        lines = f.readlines()

    stats = {}
    for line in lines:
        if "angular_error" in line:
            parts = line.strip().split(", ")
            for part in parts:
                if "mean=" in part:
                    stats["mean"] = float(part.split("mean=")[-1])
                elif "median=" in part:
                    stats["median"] = float(part.split("median=")[-1])
                elif "std=" in part:
                    stats["std"] = float(part.split("std=")[-1])
            break

    if not stats:
        raise ValueError(f"Could not parse angular error stats from {error_txt_path}")
    return stats


def visualize_gaze_multi(cur_img, gazes, head_bbox, colors=None):
    if colors is None:
        colors = [(0, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0)]

    x1, y1, x2, y2, _tracked_obj_id, _index = head_bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    max_dx = min(cx - x1, x2 - cx)
    max_dy = min(cy - y1, y2 - cy)
    max_len = min(max_dx, max_dy)

    thick = max(5, int(max_len * 0.05)) // 2

    for i, gaze in enumerate(gazes):
        color = colors[i % len(colors)]
        if np.linalg.norm(gaze) < 1e-5:
            continue
        gaze_dir = gaze / np.linalg.norm(gaze)
        gaze_dir = np.array([gaze_dir[1], gaze_dir[2], gaze_dir[0]])
        dx = gaze_dir[0]
        dy = gaze_dir[1]

        scale_x = max_dx / abs(dx) if abs(dx) > 1e-5 else float('inf')
        scale_y = max_dy / abs(dy) if abs(dy) > 1e-5 else float('inf')
        arrow_len = min(scale_x, scale_y, max_len)

        end_x = int(cx - dx * arrow_len)
        end_y = int(cy - dy * arrow_len)

        cv.arrowedLine(cur_img, (cx, cy), (end_x, end_y), color, thickness=thick, tipLength=0.2)

    return cur_img


def visualize_gaze_video_from_arrays(rgb_frames, gaze_data_list, head_bboxes=None, output_video_path=None, fps=30, colors=None, should_show_visualization=True):
    if not should_show_visualization and output_video_path is None:
        raise ValueError("At least one of should_show_visualization or output_video_path is not None must be True. After all, why even call this function if you are neither going to watch the visualization nor save it to a file?")
    if not rgb_frames:
        print("Error: No frames provided.")
        return
    if gaze_data_list is None or len(gaze_data_list) == 0 or not all(isinstance(gd, np.ndarray) for gd in gaze_data_list):
        print("Error: Invalid gaze data list.")
        return

    height, width = rgb_frames[0].shape[:2]
    if head_bboxes is None:
        head_bboxes = np.tile([0, 0, width, height, -1, -1], (len(rgb_frames), 1))

    writer = None
    if output_video_path is not None:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        writer = cv.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for idx, frame in enumerate(rgb_frames):
        gaze_vectors = []
        for gaze_data in gaze_data_list:
            if idx < len(gaze_data):
                gaze_vectors.append(gaze_data[idx, 1:])
            else:
                gaze_vectors.append(gaze_data[-1, 1:])

        annotated_frame = visualize_gaze_multi(frame.copy(), gaze_vectors, head_bboxes[idx], colors=colors)

        if writer:
            writer.write(annotated_frame)

        if should_show_visualization:
            cv.imshow("Gaze Visualization", annotated_frame)
            if cv.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break

    if writer:
        writer.release()

    if should_show_visualization:
        cv.destroyAllWindows()


def load_and_visualize_gaze_video(dataloader, model, gaze, head_bboxes=None, output_video_path=None, fps=30, should_show_visualization=True):
    frames = dataloader.load_rgb_video(as_numpy=True)
    head_bboxes_data = dataloader.load_head_bboxes()

    if gaze == "estimation":
        gaze_data_list = [dataloader.load_gaze_estimations(model=model, frame="camera")]
        colors = [(0, 0, 255)]
    elif gaze == "ground_truth":
        gaze_data_list = [dataloader.load_gaze_ground_truths(frame="camera")]
        colors = [(0, 255, 0)]
    elif gaze == "estimation_and_ground_truth":
        est = dataloader.load_gaze_estimations(model=model, frame="camera")
        gt = dataloader.load_gaze_ground_truths(frame="camera")
        gaze_data_list = [est, gt]
        colors = [(0, 0, 255), (0, 255, 0)]
        
    elif gaze == "ground_truth_gaze_and_head_direction":
        head_poses, cam_poses = dataloader.load_head_poses(), dataloader.load_camera_poses()
        head_cam = dataloader.transform_head_poses_to_camera_frame(
            head_poses,
            cam_poses
        )
        head_dirs = []
        for i in range(head_cam.shape[0]):
            mat = head_cam[i, 1:].reshape(4, 4)
            rot = R.from_matrix(mat[:3, :3])
            x_axis = rot.apply([1.0, 0.0, 0.0])
            x_axis /= np.linalg.norm(x_axis)
            head_dirs.append([head_cam[i, 0]] + x_axis.tolist())
        head_dirs = np.array(head_dirs)
        gt = dataloader.load_gaze_ground_truths(frame="camera")
        gaze_data_list = [head_dirs, gt]
        colors = [(255, 0, 0), (0, 255, 0)]
    else:
        raise ValueError("Invalid gaze type. Use 'estimation', 'ground_truth', 'estimation_and_ground_truth', or 'ground_truth_gaze_and_head_direction'.")

    if head_bboxes_data is not None and len(head_bboxes_data.shape) == 2:
        head_bboxes = head_bboxes_data
    else:
        height, width = frames[0].shape[:2]
        head_bboxes = np.tile([0, 0, width, height], (len(frames), 1))

    visualize_gaze_video_from_arrays(
        rgb_frames=frames,
        gaze_data_list=gaze_data_list,
        head_bboxes=head_bboxes,
        output_video_path=output_video_path,
        fps=fps,
        colors=colors,
        should_show_visualization=should_show_visualization,
    )

    if output_video_path:
        print(f"Output video saved to {output_video_path}")
    
def visualize_gaze_video_on_head_crops(rgb_frames, gaze_data_list, head_bboxes,
                                       output_video_path=None, fps=30, colors=None,
                                       should_show_visualization=True):
    """
    Visualize gaze on cropped head images only.

    Args:
        rgb_frames (list[np.ndarray]): Full-size RGB frames.
        gaze_data_list (list[np.ndarray]): Each array is Nx4 [timestamp, gx, gy, gz].
        head_bboxes (np.ndarray): Nx6 bounding boxes [x1,y1,x2,y2,tracked_id,index].
        output_video_path (str): Path to save the visualization video.
        fps (int): Frames per second for visualization.
        colors (list[tuple]): List of BGR colors for each gaze vector set.
        should_show_visualization (bool): Whether to display in a window.
    """
    if not should_show_visualization and output_video_path is None:
        raise ValueError("Either visualization or saving must be enabled.")

    if len(rgb_frames) == 0:
        print("Error: No frames provided.")
        return
    if head_bboxes is None or len(head_bboxes) == 0:
        print("Error: No head bounding boxes provided.")
        return
    
    index_list = []
    rgb_timestamps = dataloader.load_rgb_timestamps()
    from data_matcher import match_regular_to_regular
    
    for gaze_data in gaze_data_list:
        _, index_array = match_regular_to_regular(
                tuple1=(rgb_timestamps.reshape(-1,1), 1000.0 / config.get_rgb_fps()),
                tuple2=(gaze_data, 1000.0 / config.get_mocap_freq()),
                max_match_diff_ms=config.get_time_diff_max(),
                stability_tolerance_ms=2.0,
                stability_window_size=5
            )
        index_list.append(index_array)

    writer = None
    for idx, (frame, bbox) in enumerate(zip(rgb_frames, head_bboxes)):
        # --- Crop head region ---
        x1, y1, x2, y2, *_ = map(int, bbox)
        crop = frame[y1:y2, x1:x2].copy()

        # --- Get gaze vectors for this frame ---
        gaze_vectors = []
        for gaze_data, index in zip(gaze_data_list, index_list):
            if idx < len(index):
                gaze_vectors.append(gaze_data[index[idx], 1:])
            else:
                gaze_vectors.append(gaze_data[-1, 1:])

        # --- Draw gaze in cropped image coordinates ---
        head_box_local = [0, 0, crop.shape[1], crop.shape[0], -1, -1]
        annotated_crop = visualize_gaze_multi(crop, gaze_vectors, head_box_local, colors=colors)

        # --- Init video writer if needed ---
        if writer is None and output_video_path is not None:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            h, w = annotated_crop.shape[:2]
            writer = cv.VideoWriter(output_video_path, fourcc, fps, (w, h))

        if writer:
            writer.write(annotated_crop)

        if should_show_visualization:
            cv.imshow("Head Crop Gaze Visualization", annotated_crop)
            if cv.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break

    if writer:
        writer.release()

    if should_show_visualization:
        cv.destroyAllWindows()


def load_and_visualize_gaze_video_on_head_crops(dataloader, model, gaze, head_bboxes=None,
                                                output_video_path=None, fps=30,
                                                should_show_visualization=True):
    """
    Load video & gaze data, then visualize on head cropped frames.
    """
    frames = dataloader.load_rgb_video(as_numpy=True)
    head_bboxes_data = dataloader.load_head_bboxes()

    # --- Select gaze data ---
    if gaze == "estimation":
        gaze_data_list = [dataloader.load_gaze_estimations(model=model, frame="camera")]
        colors = [(0, 0, 255)]
    elif gaze == "ground_truth":
        gaze_data_list = [dataloader.load_gaze_ground_truths(frame="camera")]
        colors = [(0, 255, 0)]
    elif gaze == "estimation_and_ground_truth":
        est = dataloader.load_gaze_estimations(model=model, frame="camera")
        gt = dataloader.load_gaze_ground_truths(frame="camera")
        gaze_data_list = [est, gt]
        colors = [(0, 0, 255), (0, 255, 0)]
    elif gaze == "ground_truth_gaze_and_head_direction":
        head_poses, cam_poses = dataloader.load_head_poses(), dataloader.load_camera_poses()
        head_cam = dataloader.transform_head_poses_to_camera_frame(head_poses, cam_poses)
        head_dirs = []
        from scipy.spatial.transform import Rotation as R
        for i in range(head_cam.shape[0]):
            mat = head_cam[i, 1:].reshape(4, 4)
            rot = R.from_matrix(mat[:3, :3])
            x_axis = rot.apply([1.0, 0.0, 0.0])
            x_axis /= np.linalg.norm(x_axis)
            head_dirs.append([head_cam[i, 0]] + x_axis.tolist())
        head_dirs = np.array(head_dirs)
        gt = dataloader.load_gaze_ground_truths(frame="camera")
        gaze_data_list = [head_dirs, gt]
        colors = [(255, 0, 0), (0, 255, 0)]
    else:
        raise ValueError("Invalid gaze type.")

    # --- Use provided head_bboxes or default ---
    if head_bboxes_data is not None and len(head_bboxes_data.shape) == 2:
        head_bboxes = head_bboxes_data
    else:
        height, width = frames[0].shape[:2]
        head_bboxes = np.tile([0, 0, width, height, -1, -1], (len(frames), 1))

    visualize_gaze_video_on_head_crops(
        rgb_frames=frames,
        gaze_data_list=gaze_data_list,
        head_bboxes=head_bboxes,
        output_video_path=output_video_path,
        fps=fps,
        colors=colors,
        should_show_visualization=should_show_visualization,
    )

    if output_video_path:
        print(f"Output video saved to {output_video_path}")


if __name__ == "__main__":
    import config
    from data_loader import GazeDataLoader
    import os

    base_dir = config.get_dataset_base_directory()
    exp_dir = f"{base_dir}/2025-08-06/subj_xxxx/line_movement_fast/horizontal"
    dataloader = GazeDataLoader(
        root_dir=exp_dir,
        target_period=config.get_target_period(),
        camera_pose_period=config.get_camera_pose_period(),
        time_diff_max=config.get_time_diff_max(),
        get_latest_subdirectory_by_name=True
    )
    
    process_direction_errors(
        dataloader,
        model="puregaze_rectification_unrectified",
        gaze_target_bias=None,
        save_to_file=True
    )

    # load_and_visualize_gaze_video_on_head_crops(
    #     dataloader=dataloader,
    #     model="puregaze_rectification",
    #     gaze="estimation_and_ground_truth",
    #     head_bboxes=dataloader.load_head_bboxes(),
    #     output_video_path=None,
    # )
