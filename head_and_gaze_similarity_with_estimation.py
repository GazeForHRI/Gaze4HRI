import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from data_loader import GazeDataLoader
import config

# experiment types that can be included in this analysis.
INCLUDED_EXPERIMENT_TYPES = ["circular_movement", "lighting_10", "lighting_25", "lighting_50", "lighting_100", "head_pose_left", "head_pose_middle", "head_pose_right"]

def vector_from_pose_matrix(flat_matrix):
    mat = flat_matrix.reshape(4, 4)
    rot = R.from_matrix(mat[:3, :3])
    vec = rot.apply([1.0, 0.0, 0.0])
    return vec / np.linalg.norm(vec)

def compute_median_vector(vectors):
    med = np.median(np.stack(vectors), axis=0)
    return med / np.linalg.norm(med)

def build_gaze_cosine_similarity_csv(subject_dirs, exp_types, models, output_path):
    records = []

    for subj_dir in subject_dirs:
        for exp in exp_types:
            for pt in config.get_point_variations().get(exp, []):
                exp_path = os.path.join(subj_dir, exp, pt)
                try:
                    dataloader = GazeDataLoader(
                        root_dir=exp_path,
                        target_period=config.get_target_period(),
                        camera_pose_period=config.get_camera_pose_period(),
                        time_diff_max=config.get_time_diff_max(),
                        get_latest_subdirectory_by_name=True
                    )

                    head_pose_world = dataloader.load_head_poses()
                    gt_gaze_world = dataloader.load_gaze_ground_truths(frame="world")

                    head_dirs = [vector_from_pose_matrix(pose[1:]) for pose in head_pose_world]
                    gt_gaze_dirs = [vec[1:4] / np.linalg.norm(vec[1:4]) for vec in gt_gaze_world]

                    for model in models:
                        try:
                            est_gaze_world = dataloader.load_gaze_estimations(model=model, frame="world")
                            est_gaze_dirs = [vec[1:4] / np.linalg.norm(vec[1:4]) for vec in est_gaze_world]

                            head_med = compute_median_vector(head_dirs)
                            gt_gaze_med = compute_median_vector(gt_gaze_dirs)
                            est_gaze_med = compute_median_vector(est_gaze_dirs)

                            records.append({
                                "subject_dir": os.path.join(*exp_path.split(os.sep)[-4:-2]) + "/",
                                "experiment_type": exp,
                                "point": pt,
                                "model": model,
                                "head_direction_gt": " ".join(f"{v:.4f}" for v in head_med),
                                "gaze_gt": " ".join(f"{v:.4f}" for v in gt_gaze_med),
                                "gaze_est": " ".join(f"{v:.4f}" for v in est_gaze_med),
                            })
                        except Exception as e:
                            print(f"Skipping model {model} for {exp_path}: {e}")
                except Exception as e:
                    print(f"Skipping session {exp_path}: {e}")

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Saved CSV to {output_path}")
    return df

def load_head_and_gaze_directions_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No data found in CSV: {csv_path}")

    def str_to_vec(s):
        return np.array([float(x) for x in s.strip().split()])

    df["head_direction_gt"] = df["head_direction_gt"].apply(str_to_vec)
    df["gaze_gt"] = df["gaze_gt"].apply(str_to_vec)
    df["gaze_est"] = df["gaze_est"].apply(str_to_vec)

    return df

def compute_cosine_similarities(df, output_json_path):
    grouped = {}
    models = df["model"].unique()
    exp_types = INCLUDED_EXPERIMENT_TYPES

    def cosine_stats(a_list, b_list):
        sims = [float(np.dot(a, b)) for a, b in zip(a_list, b_list)]
        return {
            "mean": round(float(np.mean(sims)), 4),
            "median": round(float(np.median(sims)), 4),
            "std": round(float(np.std(sims)), 4),
        }

    for model in models:
        grouped[model] = {}
        df_model = df[df["model"] == model]

        for exp in exp_types:
            grouped[model][exp] = {}
            for pt in config.get_point_variations().get(exp, []):
                subdf = df_model[(df_model["experiment_type"] == exp) & (df_model["point"] == pt)]
                if not subdf.empty:
                    gaze_est = np.stack(subdf["gaze_est"])
                    gaze_gt = np.stack(subdf["gaze_gt"])
                    head_dir = np.stack(subdf["head_direction_gt"])
                    grouped[model][exp][pt] = {
                        "gaze_est_and_gaze_gt": cosine_stats(gaze_est, gaze_gt),
                        "gaze_est_and_head_gt": cosine_stats(gaze_est, head_dir),
                        "gaze_gt_and_head_gt": cosine_stats(gaze_gt, head_dir),
                    }

            subdf_all = df_model[df_model["experiment_type"] == exp]
            if not subdf_all.empty:
                gaze_est = np.stack(subdf_all["gaze_est"])
                gaze_gt = np.stack(subdf_all["gaze_gt"])
                head_dir = np.stack(subdf_all["head_direction_gt"])
                grouped[model][exp]["all"] = {
                    "gaze_est_and_gaze_gt": cosine_stats(gaze_est, gaze_gt),
                    "gaze_est_and_head_gt": cosine_stats(gaze_est, head_dir),
                    "gaze_gt_and_head_gt": cosine_stats(gaze_gt, head_dir),
                }

        combined = df_model[df_model["experiment_type"].isin(["head_pose_left", "head_pose_middle", "head_pose_right"])]
        if not combined.empty:
            gaze_est = np.stack(combined["gaze_est"])
            gaze_gt = np.stack(combined["gaze_gt"])
            head_dir = np.stack(combined["head_direction_gt"])
            grouped[model]["head_pose_all"] = {
                "gaze_est_and_gaze_gt": cosine_stats(gaze_est, gaze_gt),
                "gaze_est_and_head_gt": cosine_stats(gaze_est, head_dir),
                "gaze_gt_and_head_gt": cosine_stats(gaze_gt, head_dir),
            }

        all_df = df_model[df_model["experiment_type"].isin(exp_types)]
        if not all_df.empty:
            gaze_est = np.stack(all_df["gaze_est"])
            gaze_gt = np.stack(all_df["gaze_gt"])
            head_dir = np.stack(all_df["head_direction_gt"])
            grouped[model]["all"] = {
                "gaze_est_and_gaze_gt": cosine_stats(gaze_est, gaze_gt),
                "gaze_est_and_head_gt": cosine_stats(gaze_est, head_dir),
                "gaze_gt_and_head_gt": cosine_stats(gaze_gt, head_dir),
            }

    grouped["all"] = {}
    df_all = df[df["experiment_type"].isin(exp_types)]
    for exp in exp_types:
        grouped["all"][exp] = {}
        for pt in config.get_point_variations().get(exp, []):
            subdf = df_all[(df_all["experiment_type"] == exp) & (df_all["point"] == pt)]
            if not subdf.empty:
                gaze_est = np.stack(subdf["gaze_est"])
                gaze_gt = np.stack(subdf["gaze_gt"])
                head_dir = np.stack(subdf["head_direction_gt"])
                grouped["all"][exp][pt] = {
                    "gaze_est_and_gaze_gt": cosine_stats(gaze_est, gaze_gt),
                    "gaze_est_and_head_gt": cosine_stats(gaze_est, head_dir),
                    "gaze_gt_and_head_gt": cosine_stats(gaze_gt, head_dir),
                }

        subdf_all = df_all[df_all["experiment_type"] == exp]
        if not subdf_all.empty:
            gaze_est = np.stack(subdf_all["gaze_est"])
            gaze_gt = np.stack(subdf_all["gaze_gt"])
            head_dir = np.stack(subdf_all["head_direction_gt"])
            grouped["all"][exp]["all"] = {
                "gaze_est_and_gaze_gt": cosine_stats(gaze_est, gaze_gt),
                "gaze_est_and_head_gt": cosine_stats(gaze_est, head_dir),
                "gaze_gt_and_head_gt": cosine_stats(gaze_gt, head_dir),
            }

    combined = df_all[df_all["experiment_type"].isin(["head_pose_left", "head_pose_middle", "head_pose_right"])]
    if not combined.empty:
        gaze_est = np.stack(combined["gaze_est"])
        gaze_gt = np.stack(combined["gaze_gt"])
        head_dir = np.stack(combined["head_direction_gt"])
        grouped["all"]["head_pose_all"] = {
            "gaze_est_and_gaze_gt": cosine_stats(gaze_est, gaze_gt),
            "gaze_est_and_head_gt": cosine_stats(gaze_est, head_dir),
            "gaze_gt_and_head_gt": cosine_stats(gaze_gt, head_dir),
        }

    if not df_all.empty:
        gaze_est = np.stack(df_all["gaze_est"])
        gaze_gt = np.stack(df_all["gaze_gt"])
        head_dir = np.stack(df_all["head_direction_gt"])
        grouped["all"]["all"] = {
            "gaze_est_and_gaze_gt": cosine_stats(gaze_est, gaze_gt),
            "gaze_est_and_head_gt": cosine_stats(gaze_est, head_dir),
            "gaze_gt_and_head_gt": cosine_stats(gaze_gt, head_dir),
        }

    with open(output_json_path, "w") as f:
        json.dump(grouped, f, indent=2)
    print(f"Saved cosine similarities JSON to {output_json_path}")

    for model in grouped:
        if model == "all":
            continue
        for exp in grouped[model]:
            if not isinstance(grouped[model][exp], dict):
                continue
            gaze_vals, head_vals, labels = [], [], []
            for pt in config.get_point_variations().get(exp, []):
                if pt in grouped[model][exp]:
                    pt_data = grouped[model][exp][pt]
                    if "gaze_est_and_gaze_gt" in pt_data and "gaze_est_and_head_gt" in pt_data:
                        labels.append(pt)
                        gaze_vals.append(pt_data["gaze_est_and_gaze_gt"]["mean"])
                        head_vals.append(pt_data["gaze_est_and_head_gt"]["mean"])

            if labels:
                x = np.arange(len(labels))
                width = 0.35
                fig, ax = plt.subplots()
                ax.bar(x - width/2, gaze_vals, width, label='Gaze GT')
                ax.bar(x + width/2, head_vals, width, label='Head Dir')
                ax.set_ylabel('Mean Cosine Similarity')
                ax.set_title(f'{model} - {exp}')
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45)
                ax.legend()
                fig.tight_layout()
                plt.savefig(os.path.join(os.path.dirname(output_json_path), f"plot_{model}_{exp}.png"))
                plt.close()

def run():
    BASE_DIR = config.get_dataset_base_directory()
    SUBJECT_DIRS = config.get_dataset_subject_directories()
    EXP_TYPES = INCLUDED_EXPERIMENT_TYPES
    MODELS = ["puregaze", "l2cs", "gazetr", "mcgaze_clip_size_3", "mcgaze_clip_size_7", "l2cs_rectification", "puregaze_rectification", "gazetr_rectification"]
    OUTPUT_DIR = os.path.join(BASE_DIR, "head_and_gaze_similarity_with_estimation_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, "head_and_gaze_directions.csv")
    OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, "cosine_similarity_statistics.json")

    df = build_gaze_cosine_similarity_csv(SUBJECT_DIRS, EXP_TYPES, MODELS, output_path=OUTPUT_CSV_PATH)
    df = load_head_and_gaze_directions_from_csv(OUTPUT_CSV_PATH)
    compute_cosine_similarities(df, output_json_path=OUTPUT_JSON_PATH)

if __name__ == "__main__":
    run()
