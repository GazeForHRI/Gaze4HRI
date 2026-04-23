import os
from head_detection import track_and_save_one_face_always_visible
import config
from head_detection import get_latest_subdirectory_by_name

SUBJECT_DIRS = config.get_dataset_subject_directories()

EXP_TYPES = ["lighting_10", "lighting_25", "lighting_50", "lighting_100", "circular_movement", "head_pose_middle", "head_pose_left", "head_pose_right", "line_movement_slow", "line_movement_fast"]

def get_all_experiment_paths():

    paths = []

    for subj_path in SUBJECT_DIRS:
        for exp in EXP_TYPES:
            if exp == "horizontal_movement":
                parent_dir = os.path.join(subj_path, exp)
                try:
                    latest_subdir = get_latest_subdirectory_by_name(parent_dir)
                    video_path = os.path.join(parent_dir, latest_subdir, "rgb_video.mp4")
                    bboxes_out = os.path.join(parent_dir, latest_subdir, "head_bboxes.npy")
                    paths.append((video_path, bboxes_out))
                except Exception as e:
                    print(f"Warning: failed to resolve path under {parent_dir}: {e}")
                continue  # skip to next experiment
            elif exp.startswith("line_movement_"):
                # Default structure with point variations
                for _type in config.get_line_movement_types():
                    parent_dir = os.path.join(subj_path, exp, _type)
                    print("parent_dir:", parent_dir)
                    try:
                        latest_subdir = get_latest_subdirectory_by_name(parent_dir)
                        video_path = os.path.join(parent_dir, latest_subdir, "rgb_video.mp4")
                        bboxes_out = os.path.join(parent_dir, latest_subdir, "head_bboxes.npy")
                        paths.append((video_path, bboxes_out))
                    except Exception as e:
                        print(f"Warning: failed to resolve path under {parent_dir}: {e}")
                        continue
            # Default structure with point variations
            for pt in config.get_point_variations().get(exp, []):
                parent_dir = os.path.join(subj_path, exp, pt)
                try:
                    latest_subdir = get_latest_subdirectory_by_name(parent_dir)
                    video_path = os.path.join(parent_dir, latest_subdir, "rgb_video.mp4")
                    bboxes_out = os.path.join(parent_dir, latest_subdir, "head_bboxes.npy")
                    paths.append((video_path, bboxes_out))
                except Exception as e:
                    print(f"Warning: failed to resolve path under {parent_dir}: {e}")
                    continue
    return paths



def main():
    all_paths = get_all_experiment_paths()
    processed_log_path = os.path.join(os.getcwd(), "head_detection_batch_log.txt")
    processed_status = {}

    # Load existing log to avoid reprocessing
    if os.path.exists(processed_log_path):
        with open(processed_log_path, "r") as f:
            for line in f:
                video_path, status = line.strip().split("||")
                processed_status[video_path] = status

    with open(processed_log_path, "a") as log_file:
        for video_path, out_path in all_paths:
            if not os.path.exists(video_path):
                print(f"Warning: video not found: {video_path}")
                continue
            if video_path in processed_status:
                print(f"Skipping already processed: {video_path}")
                continue

            print(f"Processing: {video_path}")
            status, bboxes = track_and_save_one_face_always_visible(video_path, out_path)
            log_file.write(f"{video_path}||{status.name}\n")
            log_file.flush()  # Ensure log is saved immediately


if __name__ == "__main__":
    main()
