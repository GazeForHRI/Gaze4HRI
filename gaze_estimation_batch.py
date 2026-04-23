import os
import numpy as np
import config
from gaze_estimation import GazeEstimationStatus, GazeEstimationValidationException, GazeModel
import cv2 as cv
import time

def get_latest_subdirectory_by_name(parent_directory):
    """
    Returns the name of the latest subdirectory based on lexicographical ordering (e.g., timestamp-named dirs).

    Args:
        parent_directory (str): Path to the directory containing timestamp-named subdirectories.

    Raises:
        Exception: If no subdirectories are found.

    Returns:
        str: Name of the latest subdirectory.
    """
    try:
        # List all entries and filter only directories
        subdirs = [d for d in os.listdir(parent_directory)
                   if os.path.isdir(os.path.join(parent_directory, d))]

        if not subdirs:
            raise Exception(f"No subdirectories found in '{parent_directory}'")

        # Sort subdirectories by name (assumes timestamp-naming)
        latest_subdir = max(subdirs)
        return latest_subdir

    except FileNotFoundError:
        raise Exception(f"The directory '{parent_directory}' does not exist.")


def load_video_frames_from_mp4(video_path):
    """Load frames from an MP4 file into a list of RGB NumPy arrays."""
    cap = cv.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frames.append(rgb_frame)
    cap.release()
    return frames

def get_all_experiment_paths(subject_dirs, exp_types):
    paths = []
    for subj_path in subject_dirs:
        for exp in exp_types:
            if exp == "horizontal_movement":
                parent_dir = os.path.join(subj_path, exp)
                try:
                    latest_subdir = get_latest_subdirectory_by_name(parent_dir)
                    exp_dir = os.path.join(parent_dir, latest_subdir)
                    paths.append(exp_dir)
                except Exception as e:
                    print(f"Warning: failed to resolve path under {parent_dir}: {e}")
                continue
            
            if exp == "line_movement_slow" or exp == "line_movement_fast": # Fix: Added to handle line movement experiment types
                for pt in config.get_line_movement_types():
                    parent_dir = os.path.join(subj_path, exp, pt)
                    try:
                        latest_subdir = get_latest_subdirectory_by_name(parent_dir)
                        exp_dir = os.path.join(parent_dir, latest_subdir)
                        paths.append(exp_dir)
                    except Exception as e:
                        print(f"Warning: failed to resolve path under {parent_dir}: {e}")
                continue

            for pt in config.get_point_variations().get(exp, []):
                parent_dir = os.path.join(subj_path, exp, pt)
                try:
                    latest_subdir = get_latest_subdirectory_by_name(parent_dir)
                    exp_dir = os.path.join(parent_dir, latest_subdir)
                    paths.append(exp_dir)
                except Exception as e:
                    print(f"Warning: failed to resolve path under {parent_dir}: {e}")
                    continue
    return paths

def process_one_directory(exp_dir, gaze_model: GazeModel):
    try:
        rgb_video_path = os.path.join(exp_dir, "rgb_video.mp4")
        timestamps_path = os.path.join(exp_dir, "rgb_timestamps.npy")
        bboxes_path = os.path.join(exp_dir, "head_bboxes.npy")

        if not os.path.exists(rgb_video_path) or not os.path.exists(timestamps_path) or not os.path.exists(bboxes_path):
            print("os.path.exists(rgb_video_path):", os.path.exists(rgb_video_path))
            print("os.path.exists(timestamps_path):", os.path.exists(timestamps_path))
            print("os.path.exists(bboxes_path):", os.path.exists(bboxes_path))
            raise GazeEstimationValidationException("Missing input files")

        frames = load_video_frames_from_mp4(rgb_video_path)
        timestamps = np.load(timestamps_path)
        bboxes = np.load(bboxes_path)

        if len(frames) != len(timestamps):
            raise GazeEstimationValidationException("Mismatch in frame and timestamp count")

        gaze_with_ts, gaze_valid_indices = gaze_model.estimate(frames, timestamps, bboxes, exp_dir= exp_dir)
        gaze_model.save_estimation(gaze_directions=gaze_with_ts, valid_indices=gaze_valid_indices, output_dir=os.path.join(exp_dir, "gaze_estimations"), append_model_name_to_output_path=True) # append model name to output path so that we can distingush estimations made by different models.
        return GazeEstimationStatus.SUCCESSFUL

    except GazeEstimationValidationException as ve:
        print(f"Validation failed: {ve}")
        return GazeEstimationStatus.VALIDATION_FAILED
    except Exception as e:
        print(f"Unexpected error: {e}")
        return GazeEstimationStatus.UNEXPECTED_ERROR

def run_for_model(gaze_model: GazeModel, subject_dirs, exp_types, socket_path: str,
                  shard_idx: int, num_shards: int):
    all_dirs = get_all_experiment_paths(subject_dirs=subject_dirs, exp_types=exp_types)

    # Deterministic sharding by exp_dir index — one whole exp_dir per process
    all_dirs = sorted(all_dirs)
    if num_shards > 1:
        all_dirs = [d for i, d in enumerate(all_dirs) if i % num_shards == shard_idx]

    # Per-socket log to avoid contention
    socket_tag = os.path.basename(socket_path).replace(".sock", "")
    log_path = os.path.join(os.getcwd(),
        f"gaze_estimation_batch_log_{socket_tag}_for_{gaze_model.get_model_name()}.txt")

    processed_status = {}
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            for line in f:
                path, status = line.strip().split("||")
                processed_status[path] = status

    # >>> ADDED: tracking for throughput
    start_t = time.monotonic()
    processed_count = 0
    def _print_stats():
        elapsed_min = max((time.monotonic() - start_t) / 60.0, 1e-9)
        rate = processed_count / elapsed_min
        print(f"[STATS] {processed_count} exp_dirs processed | {rate:.2f} exp_dirs/min | {elapsed_min:.2f} min elapsed",
              flush=True)

    with open(log_path, "a") as log_file:
        for exp_dir in all_dirs:
            video_path = os.path.join(exp_dir, "rgb_video.mp4")
            if video_path in processed_status:
                print(f"Skipping already processed: {video_path}", flush=True)
                continue

            print(f"Processing: {video_path}", flush=True)
            status = process_one_directory(exp_dir, gaze_model)
            log_file.write(f"{video_path}||{status.name}\n")
            log_file.flush()

            # >>> ADDED: increment after each exp_dir completes (success or fail) and print stats
            if status == GazeEstimationStatus.SUCCESSFUL:
                processed_count += 1
                _print_stats()

    # >>> ADDED: final summary for this model/run
    print("[STATS][FINAL]", end=" ", flush=True)
    _print_stats()



def main():
    from gaze_model_mcgaze import MCGaze
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default="/tmp/mcgaze_server.sock",
                    help="UNIX socket path of the MCGaze server to use")
    ap.add_argument("--shard_idx", type=int, default=0,
                    help="This process's shard index (0..num_shards-1)")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="Total number of shards/processes")
    args = ap.parse_args()

    SUBJECT_DIRS = config.get_dataset_subject_directories()
    EXP_TYPES = ["lighting_10", "lighting_25", "lighting_50", "lighting_100", "circular_movement", "head_pose_middle", "head_pose_left", "head_pose_right", "line_movement_slow", "line_movement_fast"]
    MODELS = [
        MCGaze(clip_size=7, rectification=False, socket_path=args.socket),
        MCGaze(clip_size=7, rectification=True,  socket_path=args.socket),
    ]
    
    for model in MODELS:
        print(f"Running gaze estimation with model: {model.get_model_name()} on socket {args.socket} "
              f"(shard {args.shard_idx if hasattr(args,'shard_idx') else args.shard_idx}/{args.num_shards})")
        run_for_model(model,
                      subject_dirs=SUBJECT_DIRS,
                      exp_types=EXP_TYPES,
                      socket_path=args.socket,
                      shard_idx=args.shard_idx,
                      num_shards=args.num_shards)
        print(f"Completed gaze estimation with model: {model.get_model_name()} on socket {args.socket}\n")

if __name__ == "__main__":
    main()