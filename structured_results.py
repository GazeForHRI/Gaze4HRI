### DOCUMENTATION
# Core Functions

#     read_existing_results() — loads the current CSV if it exists.

#     write_results_df(df) — writes a DataFrame to CSV.

#     parse_error_stats(path) — extracts mean, median, std from errors.txt.

#     build_result_row(base_path, exp_dir, model_name) — constructs a row from folder structure + metadata + parsed stats.

#     update_results_csv() — traverses all evaluated directories and updates gaze_evaluation_results.csv with new entries for all known models.
import os
import pandas as pd
import json
import re
import config

BASE_DIR = config.get_dataset_base_directory()
SUBJECT_DIRS = config.get_dataset_subject_directories()
CSV_PATH = os.path.join(BASE_DIR, "gaze_evaluation_results.csv")
MODELS = config.get_currently_analyzed_models()
SUBJECT_METADATA_PATH = os.path.join(BASE_DIR, "subject_metadata.json")

def load_subject_metadata(path: str = SUBJECT_METADATA_PATH) -> dict:
    """
    Load subject metadata saved by subject_stats.py.
    Raises if the file is missing or malformed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing subject metadata at {path}. "
            f"subject_stats.py needs to be run."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(
            f"subject_metadata.json is empty or malformed at {path}. "
            f"subject_stats.py needs to be run."
        )
    return data

SUBJECT_METADATA = load_subject_metadata()

def read_existing_results(csv_path=CSV_PATH):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame(columns=[
        "subject_dir", "exp_type", "point", "subject_name", "gender", "birthyear", "glasses", "height_cm",
        "gaze_model", "error_mean", "error_median", "error_std"
    ])

def write_results_df(df, csv_path=CSV_PATH):
    df.to_csv(csv_path, index=False)

def parse_error_stats(error_file_path):
    with open(error_file_path, "r") as f:
        content = f.read()
    
    match_mean = re.search(r"mean=([\d.]+)", content)
    match_median = re.search(r"median=([\d.]+)", content)
    match_std = re.search(r"std=([\d.]+)", content)
    match_num = re.search(r"num_frames=(\d+)", content)

    if not all([match_mean, match_median, match_std, match_num]):
        print(f"[WARNING] Malformed or empty error file found: {error_file_path}")
        return None

    mean = float(match_mean.group(1))
    median = float(match_median.group(1))
    std = float(match_std.group(1))
    num_samples = int(match_num.group(1))
    
    return round(mean, 4), round(median, 4), round(std, 4), num_samples

def build_result_row(base_path, exp_dir, model_name):
    parts = exp_dir[len(base_path):].strip("/").split("/")
    if len(parts) < 3:
        raise ValueError(f"Invalid path structure: {exp_dir}")

    date_str, subject_name, exp_type = parts[:3]
    point = parts[3] if len(parts) > 3 else None
    subject_dir = f"{date_str}/{subject_name}"
    error_file = os.path.join(exp_dir, "gaze_estimations", model_name, "direction_errors", "errors.txt")

    if not os.path.exists(error_file):
        return None

    stats = parse_error_stats(error_file)
    if stats is None:
        return None

    meta = SUBJECT_METADATA.get(date_str+"/"+subject_name)
    if meta is None:
        raise ValueError(f"No metadata found for subject: {subject_name}")

    return {
        "subject_dir": subject_dir,
        "exp_type": exp_type,
        "point": point if exp_type != "horizontal_movement" else "",
        "subject_name": meta["name"],
        "gender": meta["gender"],
        "birthyear": meta["birthyear"],
        "glasses": meta["glasses"],
        "height_cm": meta["height_cm"],
        "gaze_model": model_name,
        "error_mean": stats[0],
        "error_median": stats[1],
        "error_std": stats[2],
        "num_samples": stats[3],
    }

def update_results_csv(base_path, subject_dirs, csv_path, models):
    new_rows = []
    visited = set()
    for subject_dir in subject_dirs:
        for root, dirs, files in os.walk(subject_dir):
            for model in models:
                model_dir = f"gaze_estimations/{model}/direction_errors"
                if model_dir in root and root not in visited:
                    visited.add(root)
                    exp_dir = root.split(f"/gaze_estimations/{model}/direction_errors")[0]
                    if not os.path.exists(exp_dir):
                        raise FileNotFoundError(f"Experiment directory does not exist: {exp_dir}")
                    if config.is_experiment_directory_excluded_from_eval(exp_dir): # skip excluded exp_dir
                        print(f"Skipping excluded experiment: {exp_dir}")
                        continue
                    row = build_result_row(base_path, exp_dir, model)
                    if row:
                        new_rows.append(row)

    if not new_rows:
        print("No new results found.")
        return

    df_existing = read_existing_results(csv_path)
    df_new = pd.DataFrame(new_rows)

    if df_existing.empty:
        combined_df = df_new
    else:
        combined_df = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates(
            subset=["subject_dir", "exp_type", "point", "subject_name", "gaze_model"], keep="last"
        )

    combined_df = combined_df.sort_values(by=["subject_name", "exp_type", "point", "gaze_model"])
    write_results_df(combined_df, csv_path)
    print(f"Updated results database with {len(df_new)} new entries.")

def main():
    print(f"Gathered {len(SUBJECT_METADATA)} subjects' metadata.")
    print("Updating gaze evaluation results CSV...")
    update_results_csv(base_path=BASE_DIR, subject_dirs=SUBJECT_DIRS, csv_path=CSV_PATH, models=MODELS)
    print("Update complete.")

if __name__ == "__main__":
    main()
