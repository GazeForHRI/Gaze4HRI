import os
import sys
import traceback
from tqdm import tqdm
import config
from data_loader import GazeDataLoader

# Import models to initialize them once globally
import mediapipe as mp
from exordium.video.tddfa_v2 import TDDFA_V2
import exordium.video.iris as ex_iris
from exordium_landmarks import extract_and_save_exordium_features, fix_exordium

def get_latest_subdirectory_by_name(parent_directory):
    try:
        subdirs = [d for d in os.listdir(parent_directory)
                   if os.path.isdir(os.path.join(parent_directory, d))]
        if not subdirs:
            raise Exception(f"No subdirectories found in '{parent_directory}'")
        return max(subdirs)
    except FileNotFoundError:
        raise Exception(f"The directory '{parent_directory}' does not exist.")

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
                    pass
                continue

            if exp in ["line_movement_slow", "line_movement_fast"]:
                for pt in config.get_line_movement_types():
                    parent_dir = os.path.join(subj_path, exp, pt)
                    try:
                        latest_subdir = get_latest_subdirectory_by_name(parent_dir)
                        exp_dir = os.path.join(parent_dir, latest_subdir)
                        paths.append(exp_dir)
                    except Exception as e:
                        pass
                continue            

            for pt in config.get_point_variations().get(exp, []):
                parent_dir = os.path.join(subj_path, exp, pt)
                try:
                    latest_subdir = get_latest_subdirectory_by_name(parent_dir)
                    exp_dir = os.path.join(parent_dir, latest_subdir)
                    paths.append(exp_dir)
                except Exception as e:
                    continue
    return paths

def run_exordium_batch(subject_dirs, exp_types):
    all_dirs = get_all_experiment_paths(subject_dirs, exp_types)
    
    # Use v2 naming to guarantee isolation from any previous runs
    status_log_path = os.path.join(os.getcwd(), "exordium_batch_status_v2.txt")
    error_log_path = os.path.join(os.getcwd(), "exordium_batch_errors_v2.log")
    
    processed_status = {}

    if os.path.exists(status_log_path):
        with open(status_log_path, "r") as f:
            for line in f:
                if "||" in line:
                    parts = line.strip().split("||")
                    if len(parts) == 2:
                        status = parts[0].strip()
                        path = parts[1].strip()
                        processed_status[path] = status

    print("\n[INIT] Loading Deep Learning Models into memory (This happens only once)...")
    sys.stdout.flush()
    global_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    global_tddfa = TDDFA_V2()
    global_iris = ex_iris.IrisWrapper()
    print("[INIT] Models loaded successfully.\n")
    sys.stdout.flush()

    with open(status_log_path, "a") as status_log, open(error_log_path, "a") as error_log:
        with tqdm(total=len(all_dirs), desc="Processing Directories", unit="dir") as pbar:
            for exp_dir in all_dirs:
                # Skip if already processed successfully (whether ALL, SOME, or NONE). 
                # We only retry if it FAILED (crashed).
                if exp_dir in processed_status and processed_status[exp_dir] != "FAILED":
                    pbar.update(1)
                    continue

                try:
                    print(f"\nProcessing: {exp_dir}")
                    sys.stdout.flush()
                    
                    dataloader = GazeDataLoader(
                        root_dir=exp_dir,
                        target_period=config.get_target_period(),
                        camera_pose_period=config.get_camera_pose_period(),
                        time_diff_max=config.get_time_diff_max(),
                        get_latest_subdirectory_by_name=False
                    )

                    processed_frames, total_frames = extract_and_save_exordium_features(
                        dataloader, 
                        face_mesh_detector=global_face_mesh, 
                        face_model=global_tddfa, 
                        iris_model=global_iris
                    )
                    
                    # Determine string status mapping
                    if processed_frames == total_frames:
                        log_status = "ALL"
                    elif processed_frames == 0:
                        log_status = "NONE"
                    else:
                        log_status = f"SOME ({processed_frames}/{total_frames})"

                    status_log.write(f"{log_status} || {exp_dir}\n")
                    status_log.flush()

                except Exception as e:
                    status_log.write(f"FAILED || {exp_dir}\n")
                    status_log.flush()
                    
                    error_log.write(f"{'='*80}\n")
                    error_log.write(f"FAILED DIRECTORY: {exp_dir}\n")
                    error_log.write(f"ERROR: {str(e)}\n")
                    error_log.write("TRACEBACK:\n")
                    traceback.print_exc(file=error_log)
                    error_log.write(f"{'='*80}\n\n")
                    error_log.flush()
                
                finally:
                    pbar.update(1)

def fix_exordium_batch(subject_dirs, exp_types):
    all_dirs = get_all_experiment_paths(subject_dirs, exp_types)

    status_log_path = os.path.join(os.getcwd(), "fix_exordium_batch_status_v2.txt")
    error_log_path = os.path.join(os.getcwd(), "fix_exordium_batch_errors_v2.log")

    processed_status = {}

    if os.path.exists(status_log_path):
        with open(status_log_path, "r") as f:
            for line in f:
                if "||" in line:
                    parts = line.strip().split("||")
                    if len(parts) == 2:
                        status = parts[0].strip()
                        path = parts[1].strip()
                        processed_status[path] = status

    total_frames_all = 0
    total_processed_before_all = 0
    total_processed_after_all = 0

    with open(status_log_path, "a") as status_log, open(error_log_path, "a") as error_log:
        with tqdm(total=len(all_dirs), desc="Fixing Exordium Directories", unit="dir") as pbar:
            for exp_dir in all_dirs:
                if exp_dir in processed_status and processed_status[exp_dir] != "FAILED":
                    pbar.update(1)
                    continue

                try:
                    # 1. Initialize the DataLoader for the specific experiment
                    # This is required for the new fix_exordium to access video frames
                    dataloader = GazeDataLoader(
                        root_dir=exp_dir,
                        target_period=config.get_target_period(),
                        camera_pose_period=config.get_camera_pose_period(),
                        time_diff_max=config.get_time_diff_max(),
                        get_latest_subdirectory_by_name=False
                    )

                    if not os.path.exists(dataloader.get_cwd()):
                        status_log.write(f"SKIP (No CWD) || {exp_dir}\n")
                        pbar.update(1)
                        continue

                    # 2. Call the updated fix_exordium with the dataloader object
                    # Using cheap_limit=5 as discussed for continuous movement
                    processed_frames, total_frames, previous_processed_frames = fix_exordium(
                        dataloader, 
                        cheap_limit=5, 
                        hermite_coefficent=0.5
                    )

                    total_frames_all += total_frames
                    total_processed_before_all += previous_processed_frames
                    total_processed_after_all += processed_frames

                    if processed_frames == total_frames:
                        log_status = "ALL"
                    elif processed_frames == 0:
                        log_status = "NONE"
                    else:
                        log_status = f"SOME ({processed_frames}/{total_frames})"

                    status_log.write(f"{log_status} || {exp_dir}\n")
                    status_log.flush()

                except Exception as e:
                    status_log.write(f"FAILED || {exp_dir}\n")
                    status_log.flush()

                    error_log.write(f"{'='*80}\n")
                    error_log.write(f"FAILED DIRECTORY: {exp_dir}\n")
                    error_log.write(f"ERROR: {str(e)}\n")
                    error_log.write("TRACEBACK:\n")
                    traceback.print_exc(file=error_log)
                    error_log.write(f"{'='*80}\n\n")
                    error_log.flush()

                finally:
                    pbar.update(1)

    print("\n--- FIX EXORDIUM BATCH SUMMARY ---")
    if total_frames_all > 0:
        total_missing_before_all = total_frames_all - total_processed_before_all
        total_missing_after_all = total_frames_all - total_processed_after_all

        missing_before_pct = 100.0 * total_missing_before_all / total_frames_all
        missing_after_pct = 100.0 * total_missing_after_all / total_frames_all

        print(f"Total frames checked : {total_frames_all}")
        print(f"Missing before fix   : {total_missing_before_all} ({missing_before_pct:.2f}%)")
        print(f"Missing after fix    : {total_missing_after_all} ({missing_after_pct:.2f}%)")
    else:
        print("No frames were processed in this batch.")               

def main():
    SUBJECT_DIRS = config.get_dataset_subject_directories()
    EXP_TYPES = config.get_experiment_types()
    
    # print("Starting Exordium batch feature extraction (v2)...")
    # run_exordium_batch(SUBJECT_DIRS, EXP_TYPES)
    # print("\nBatch extraction completed. Check exordium_batch_errors_v2.log for any failures.")

    print("Starting Exordium feature fixing...")
    fix_exordium_batch(SUBJECT_DIRS, EXP_TYPES)
    print("\nBatch feature fixing completed. Check fix_exordium_batch_errors_v2.log for any failures.")

if __name__ == "__main__":
    main()

