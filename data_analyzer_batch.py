import os
import config
from data_analyer import process_direction_errors
from data_loader import GazeDataLoader
from tqdm import tqdm  # NEW

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

def run_data_analysis_for_model(subject_dirs, exp_types, model_name):
    all_dirs = get_all_experiment_paths(subject_dirs, exp_types)
    log_path = os.path.join(os.getcwd(), f"data_analyzer_batch_log_for_{model_name}.txt")
    processed_status = {}

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            for line in f:
                path, status = line.strip().split("||")
                processed_status[path] = status

    with open(log_path, "a") as log_file, tqdm(total=len(all_dirs), desc=f"{model_name}", unit="dir") as pbar:  # NEW
        for exp_dir in all_dirs:
            if exp_dir in processed_status:
                print(f"Skipping already processed: {exp_dir}")
                pbar.update(1)  # NEW
                continue

            try:
                print(f"Processing: {model_name}, {exp_dir}")
                dataloader = GazeDataLoader(
                    root_dir=exp_dir,
                    target_period=config.get_target_period(),
                    camera_pose_period=config.get_camera_pose_period(),
                    time_diff_max=config.get_time_diff_max(),
                    get_latest_subdirectory_by_name=False
                )

                model_estimation_dir = os.path.join(dataloader.get_gaze_estimations_dir(model=model_name))
                if not os.path.exists(model_estimation_dir):
                    raise FileNotFoundError(f"Missing gaze_estimations for model '{model_name}' in directory: {model_estimation_dir}")

                process_direction_errors(dataloader, model=model_name)
                log_file.write(f"{exp_dir}||SUCCESSFUL\n")
                log_file.flush()

            except Exception as e:
                print(f"Error processing {exp_dir}: {e}")
                log_file.write(f"{exp_dir}||FAILED\n")
                log_file.flush()
            finally:
                pbar.update(1)  # NEW

def run_data_analysis(subject_dirs, exp_types, models):
    for model_name in models:
        print(f"Running data analysis with model: {model_name}")
        run_data_analysis_for_model(subject_dirs, exp_types, model_name)
        print(f"Completed data analysis with model: {model_name}\n")

def main():
    SUBJECT_DIRS = config.get_dataset_subject_directories()
    EXP_TYPES = ["lighting_10", "lighting_25", "lighting_50", "lighting_100", "circular_movement", "head_pose_middle", "head_pose_left", "head_pose_right", "line_movement_slow", "line_movement_fast"]
    MODELS = config.get_currently_analyzed_models()
    run_data_analysis(SUBJECT_DIRS, EXP_TYPES, MODELS)
    print("Data analysis completed for all models.")

if __name__ == "__main__":
    main()
