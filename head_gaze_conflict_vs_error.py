# head_gaze_conflict_vs_error.py

import os
import json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.stats import spearmanr
from data_loader import GazeDataLoader
import config
from hypotheses_and_patterns import get_latest_subdirectory_by_name
from data_analyer import parse_direction_errors_file
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from scipy import stats
import pingouin as pg

# experiment types that can be included in this analysis
INCLUDED_EXPERIMENT_TYPES = ["head_pose_left", "head_pose_middle", "head_pose_right"]

REQUIRED_HEADPOSE_EXPS = set(INCLUDED_EXPERIMENT_TYPES)

def _filter_complete_headpose_subjects(df_vid: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only subjects (subject_dir) that have at least one row for *each*
    of the REQUIRED_HEADPOSE_EXPS. Returns (df_filtered, df_dropped_subjects).
    """
    if df_vid.empty:
        return df_vid, pd.DataFrame(columns=["subject_dir","present_exps"])

    # For each subject_dir, collect which head_pose experiment types are present
    present = (
        df_vid.groupby("subject_dir")["experiment_type"]
              .apply(lambda s: set(x for x in s.unique() if x in REQUIRED_HEADPOSE_EXPS))
              .reset_index(name="present_exps")
    )
    present["is_complete"] = present["present_exps"].apply(lambda S: REQUIRED_HEADPOSE_EXPS.issubset(S))

    complete_subjects = present.loc[present["is_complete"], "subject_dir"]
    df_filtered = df_vid[df_vid["subject_dir"].isin(complete_subjects)].copy()

    dropped = present.loc[~present["is_complete"], ["subject_dir","present_exps"]].copy()
    return df_filtered, dropped


# -------------------- utils --------------------

def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))

def _extract_head_dirs_in_cam(head_cam_poses: np.ndarray) -> np.ndarray:
    """head_cam_poses: [T, 1+16] with timestamp in col0, 4x4 row-major in cols 1:17."""
    dirs = []
    for i in range(head_cam_poses.shape[0]):
        mat = head_cam_poses[i, 1:].reshape(4, 4)
        Rw = R.from_matrix(mat[:3, :3])
        x_axis = Rw.apply([1.0, 0.0, 0.0])
        x_axis = x_axis / np.linalg.norm(x_axis)
        dirs.append(x_axis)
    return np.asarray(dirs, dtype=float)

def _subject_dir_from_any_path(p: str) -> str:
    """Return canonical 'YYYY-MM-DD/Subject/' from a subject root, exp_dir, or run dir path."""
    parts = os.path.normpath(p).split(os.sep)
    # run path: .../<DATE>/<SUBJ>/<exp>/<pt>/<run>
    if len(parts) >= 5:
        return os.path.join(parts[-5], parts[-4]) + "/"
    # exp path: .../<DATE>/<SUBJ>/<exp>/<pt>
    if len(parts) >= 4:
        return os.path.join(parts[-4], parts[-3]) + "/"
    # subject root: .../<DATE>/<SUBJ>
    if len(parts) >= 2:
        return os.path.join(parts[-2], parts[-1]) + "/"
    return parts[-1] + "/"

def _holm_adjust(pvals: pd.Series) -> pd.Series:
    """Holm step-down adjustment across a family (monotone, FWER control)."""
    s = pd.Series(pvals, copy=True)
    order = s.dropna().sort_values().index.tolist()
    m = len(order)
    out = pd.Series(index=s.index, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order, start=1):
        factor = m - rank + 1
        padj = min(1.0, s.loc[idx] * factor)
        running_max = max(running_max, padj)
        out.loc[idx] = running_max
    return out

# -------------------- core measures --------------------

def _video_conflict_angle_mean(dataloader: GazeDataLoader) -> float:
    """
    Mean angle between head-forward and GT gaze over frames in a video, in camera frame.
    (mean, not median)
    """
    head_poses = dataloader.load_head_poses()
    cam_poses  = dataloader.load_camera_poses()
    head_cam   = dataloader.transform_head_poses_to_camera_frame(head_poses, cam_poses)
    head_dirs  = _extract_head_dirs_in_cam(head_cam)                  # [T,3]
    gt_gaze    = dataloader.load_gaze_ground_truths(frame="camera")   # [T,1+3+...]
    gaze_dirs  = gt_gaze[:, 1:4]                                      # [T,3]

    n = min(head_dirs.shape[0], gaze_dirs.shape[0])
    if n <= 0:
        raise ValueError("No overlapping frames for conflict angle computation.")
    ang = np.array([_angle_between(head_dirs[i], gaze_dirs[i]) for i in range(n)], dtype=float)
    return float(np.mean(ang))

def _weighted_simple_linreg(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    """Weighted least squares slope/intercept for y ~ a + b*x (analytical). Returns (b, a)."""
    x = x.astype(float); y = y.astype(float); w = w.astype(float)
    if not np.isfinite(w).all() or np.sum(w > 0) == 0:
        w = np.ones_like(x)
    W  = float(np.sum(w))
    xw = float(np.sum(w * x) / W)
    yw = float(np.sum(w * y) / W)
    cov_w = float(np.sum(w * (x - xw) * (y - yw)))
    var_w = float(np.sum(w * (x - xw) ** 2))
    if var_w <= 0:
        return np.nan, np.nan
    b = cov_w / var_w
    a = yw - b * xw
    return float(b), float(a)

def _read_num_valid_frames(gaze_dir: str) -> int:
    """Read num_frames (VALID frames) from direction_errors/errors.txt."""
    p = os.path.join(gaze_dir, "direction_errors", "errors.txt")
    if not os.path.exists(p):
        return 0
    num_valid = 0
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("num_frames="):
                try:
                    num_valid = int(line.split("=", 1)[1])
                except Exception:
                    num_valid = 0
                break
    return num_valid

# -------------------- table build --------------------

def build_subject_video_table(subject_dirs: List[str], models: List[str]) -> pd.DataFrame:
    """
    One row per (subject_dir, exp_type, point, model) video:
      subject_dir, subject (short), experiment_type, point, gaze_model, num_frames, video_error_mean, conflict_angle_median
    Only head_pose_*; skip directories excluded by config.
    """
    rows = []
    main_keys = set(config.get_main_models_included_in_the_paper().keys())

    for subject_root in subject_dirs:
        for exp in INCLUDED_EXPERIMENT_TYPES:
            points = config.get_point_variations().get(exp, []) + config.get_line_movement_types() # + config.get_line_movement_types() in case you want to analyze for line_movement exp types.
            for pt in points:
                exp_dir = os.path.join(subject_root, exp, pt)
                try:
                    if config.is_experiment_directory_excluded_from_eval(exp_dir):
                        continue
                except Exception:
                    pass

                try:
                    latest = get_latest_subdirectory_by_name(exp_dir)
                except Exception:
                    latest = None
                if not latest:
                    continue
                video_dir = os.path.join(exp_dir, latest)

                subject_dir_short = _subject_dir_from_any_path(video_dir)
                subject_label = os.path.basename(subject_dir_short.rstrip("/"))  # short name

                # conflict angle mean (per video)
                try:
                    dataloader = GazeDataLoader(
                        root_dir=exp_dir,
                        target_period=config.get_target_period(),
                        camera_pose_period=config.get_camera_pose_period(),
                        time_diff_max=config.get_time_diff_max(),
                        get_latest_subdirectory_by_name=True
                    )
                    conflict_mean = _video_conflict_angle_mean(dataloader)
                except Exception as e:
                    print(f"[conflict] skip {video_dir}: {e}")
                    continue

                for m in models:
                    if m not in main_keys:
                        continue
                    try:
                        gaze_dir = os.path.join(video_dir, "gaze_estimations", m)
                        stats_dict = parse_direction_errors_file(gaze_dir)  # {'mean','median','std'}
                        num_valid_frames = _read_num_valid_frames(gaze_dir)  # valid-only
                        rows.append({
                            "subject_dir": subject_dir_short,
                            "subject": subject_label,
                            "experiment_type": exp,
                            "point": pt,
                            "gaze_model": m,
                            "num_frames": int(num_valid_frames),
                            "video_error_mean": float(stats_dict["mean"]),
                            # keep column name for backward compat (value is mean)
                            "conflict_angle_median": float(conflict_mean),
                        })
                    except Exception:
                        pass

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["gaze_model"].isin(main_keys)].copy()
    return df

# -------------------- subject/model stats --------------------

def fit_subject_level_slopes(df_vid: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each (subject_dir, gaze_model): WLS slope of error ~ conflict (weights=num_frames).
    Then per model: one-sample t-test of subject slopes > 0 (one-sided), and apply Holm
    correction ACROSS models. Report slope per 10°.
    """
    if df_vid.empty:
        return pd.DataFrame(), pd.DataFrame()

    # per-(subject, model) slopes
    slopes = []
    for (subj_dir, m), dsm in df_vid.groupby(["subject_dir", "gaze_model"]):
        dsm = dsm[dsm["num_frames"] > 0]
        if dsm.shape[0] < 2:
            continue
        b, a = _weighted_simple_linreg(
            dsm["conflict_angle_median"].to_numpy(),   # mean conflict angle per video
            dsm["video_error_mean"].to_numpy(),
            dsm["num_frames"].to_numpy()
        )
        if np.isfinite(b):
            slopes.append({"subject_dir": subj_dir, "gaze_model": m, "slope_deg_per_deg": float(b)})

    per_subject = pd.DataFrame(slopes)
    if per_subject.empty:
        return pd.DataFrame(), per_subject

    # model-level tests (one-sided > 0), collect raw p's then Holm-adjust across models
    rows = []
    raw_pvals = {}

    for m, dm in per_subject.groupby("gaze_model"):
        vals = dm["slope_deg_per_deg"].to_numpy(dtype=float)
        n = vals.size
        mean_b = float(np.mean(vals))
        sd_b   = float(np.std(vals, ddof=1)) if n > 1 else np.nan
        se_b   = float(sd_b / np.sqrt(n)) if (n > 1 and np.isfinite(sd_b)) else np.nan

        pval = np.nan
        if n >= 2 and np.isfinite(sd_b) and sd_b > 0:
            # two-sided t-test, convert to one-sided for H1: mean > 0
            t_stat, p_two = stats.ttest_1samp(vals, popmean=0.0)
            pval = (p_two / 2.0) if mean_b >= 0 else (1.0 - p_two / 2.0)

        mean_val = mean_b
        t_crit = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
        ci_lo = mean_val - t_crit * se_b if np.isfinite(se_b) else np.nan
        ci_hi = mean_val + t_crit * se_b if np.isfinite(se_b) else np.nan

        rows.append({
            "gaze_model": m,
            "n_subjects": int(n),
            "slope": float(mean_val),
            "ci95_low": float(ci_lo),
            "ci95_high": float(ci_hi),
            "pval_greater_0": float(pval) if np.isfinite(pval) else np.nan,
            "prop_pos_slope": float(np.mean(vals > 0.0)),
        })
        raw_pvals[m] = pval

    per_model = pd.DataFrame(rows).sort_values("gaze_model").reset_index(drop=True)

    # Holm adjust across models
    per_model["pval_greater_0_holm"] = _holm_adjust(per_model["pval_greater_0"])

    # map to display names
    disp_map = config.get_main_models_included_in_the_paper()
    per_model["gaze_model"] = per_model["gaze_model"].map(disp_map)
    per_subject["gaze_model"] = per_subject["gaze_model"].map(disp_map)
    return per_model, per_subject

# -------------------- extras: angle range + histogram --------------------

def export_conflict_angle_range(outdir: str):
    """
    Reads the video-level table and outputs the min/max of conflict_angle_median.
    """
    fp = os.path.join(outdir, "head_gaze_conflict__video_level_table.csv")
    if not os.path.isfile(fp):
        print(f"Video-level table not found: {fp}")
        return
    df = pd.read_csv(fp)
    if df.empty:
        print("Video-level table is empty.")
        return
    lo = float(df["conflict_angle_median"].min())
    hi = float(df["conflict_angle_median"].max())
    rng = pd.DataFrame([{"conflict_angle_min": lo, "conflict_angle_max": hi}])
    out_csv = os.path.join(outdir, "head_gaze_conflict__angle_range.csv")
    rng.to_csv(out_csv, index=False)
    print(f"Saved conflict angle range to {out_csv}")

def export_conflict_angle_histogram_1deg(outdir: str):
    """
    Creates a 1°-binned frequency histogram (0..ceil(max)) of conflict angles
    and saves both PNG and CSV (bin edges & counts).
    This is the distribution of video-level mean head–gaze conflict angles across the entire dataset of head_pose experiments.
    """
    fp = os.path.join(outdir, "head_gaze_conflict__video_level_table.csv")
    if not os.path.isfile(fp):
        print(f"Video-level table not found: {fp}")
        return
    df = pd.read_csv(fp)
    if df.empty:
        print("Video-level table is empty.")
        return

    vals = df["conflict_angle_median"].to_numpy(dtype=float)
    if vals.size == 0:
        print("No conflict angles to plot.")
        return

    lo = max(0.0, float(np.floor(np.nanmin(vals))))  # start at 0 for readability
    hi = float(np.ceil(np.nanmax(vals)))
    # Ensure at least one bin
    if hi <= lo:
        hi = lo + 1.0
    edges = np.arange(lo, hi + 1.0, 1.0)  # 1° bins

    counts, _ = np.histogram(vals, bins=edges)

    # Save CSV of histogram
    hist_df = pd.DataFrame({
        "bin_left_deg": edges[:-1],
        "bin_right_deg": edges[1:],
        "count": counts
    })
    out_csv = os.path.join(outdir, "head_gaze_conflict__angle_hist_1deg.csv")
    hist_df.to_csv(out_csv, index=False)

    # Plot PNG
    plt.figure(figsize=(7.2, 4.2))
    plt.hist(vals, bins=edges)
    plt.title("Head–Gaze Conflict Angle (1° bins)")
    plt.xlabel("Conflict angle (deg)")
    plt.ylabel("Frequency (videos)")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.6)
    out_png = os.path.join(outdir, "head_gaze_conflict__angle_hist_1deg.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"Saved histogram PNG to {out_png} and CSV to {out_csv}")

# -------------------- legacy correlation helpers (unchanged) --------------------

def angle_between_vectors(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def compute_gaze_vs_head_angle_stats(subject_dirs, exp_types, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    records = []

    for subject_root in subject_dirs:
        for exp in exp_types:
            for pt in config.get_point_variations().get(exp, []):
                exp_path = os.path.join(subject_root, exp, pt)
                try:
                    if config.is_experiment_directory_excluded_from_eval(exp_path):
                        continue
                except Exception:
                    pass
                try:
                    dataloader = GazeDataLoader(
                        root_dir=exp_path,
                        target_period=config.get_target_period(),
                        camera_pose_period=config.get_camera_pose_period(),
                        time_diff_max=config.get_time_diff_max(),
                        get_latest_subdirectory_by_name=True
                    )
                    head_poses, cam_poses = dataloader.load_head_poses(), dataloader.load_camera_poses()
                    head_cam = dataloader.transform_head_poses_to_camera_frame(head_poses, cam_poses)
                    gt_gaze_cam = dataloader.load_gaze_ground_truths(frame="camera")
                    head_dirs = []
                    for i in range(head_cam.shape[0]):
                        mat = head_cam[i, 1:].reshape(4, 4)
                        rot = R.from_matrix(mat[:3, :3])
                        x_axis = rot.apply([1.0, 0.0, 0.0])
                        x_axis /= np.linalg.norm(x_axis)
                        head_dirs.append(x_axis)
                    angles = [
                        angle_between_vectors(head_dir, gaze[1:4])
                        for head_dir, gaze in zip(head_dirs, gt_gaze_cam)
                    ]
                    angles = np.array(angles)
                    records.append({
                        "subject_dir": _subject_dir_from_any_path(exp_path),
                        "experiment_type": exp,
                        "point": pt,
                        "angle_mean": float(np.mean(angles)),
                        "angle_median": float(np.median(angles)),
                        "angle_std": float(np.std(angles))
                    })
                except Exception as e:
                    print(f"Skipping {exp_path}: {e}")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "head_gaze_conflict_vs_error_angles_between_head_orientation_and_gaze.csv"), index=False)
    print("Saved angles CSV")
    return df

def compute_correlation_with_errors(df_angles, subject_dirs, exp_types, models, output_dir):
    pearson_grouped = {model: {} for model in models}
    pearson_grouped["all"] = {}
    spearman_grouped = {model: {} for model in models}
    spearman_grouped["all"] = {}
    container = {model: {} for model in models}
    container["all"] = {}

    def append_entry(container, model, exp_type, angle_val, err_val):
        container[model].setdefault(exp_type, []).append((angle_val, err_val))
        container["all"].setdefault(exp_type, []).append((angle_val, err_val))

    for subject_root in subject_dirs:
        for exp in exp_types:
            for pt in config.get_point_variations().get(exp, []):
                exp_path = os.path.join(subject_root, exp, pt)
                try:
                    if config.is_experiment_directory_excluded_from_eval(exp_path):
                        continue
                except Exception:
                    pass
                for model in models:
                    try:
                        full_exp_path = os.path.join(exp_path, get_latest_subdirectory_by_name(exp_path))
                        gaze_dir = os.path.join(full_exp_path, "gaze_estimations", model)
                        err_stats = parse_direction_errors_file(gaze_dir)
                        mean_ang_err = err_stats["mean"]

                        row = df_angles[
                            (df_angles["subject_dir"] == _subject_dir_from_any_path(full_exp_path)) &
                            (df_angles["experiment_type"] == exp) &
                            (df_angles["point"] == pt)
                        ]
                        if row.empty:
                            continue
                        median_angle = row.iloc[0]["angle_median"]

                        append_entry(container, model, exp, median_angle, mean_ang_err)
                    except Exception as e:
                        print(f"Skipping {exp_path}, {model}: {e}")

    for grouped, corr_fn in [(pearson_grouped, lambda a, b: np.corrcoef(a, b)[0, 1]),
                              (spearman_grouped, lambda a, b: spearmanr(a, b).correlation)]:
        for model in container:
            for exp in INCLUDED_EXPERIMENT_TYPES:
                entries = container[model].get(exp, [])
                if entries:
                    a, b = zip(*entries)
                    grouped[model][exp] = round(float(corr_fn(a, b)), 4)

            combined = []
            for subexp in INCLUDED_EXPERIMENT_TYPES:
                combined.extend(container[model].get(subexp, []))
            if combined:
                a, b = zip(*combined)
                grouped[model]["head_pose_all"] = round(float(corr_fn(a, b)), 4)

            if "head_pose_all" in grouped[model]:
                grouped[model]["all"] = grouped[model]["head_pose_all"]

    with open(os.path.join(output_dir, "head_gaze_conflict_vs_error_correlation_coefficients_pearson.json"), "w") as f:
        json.dump(pearson_grouped, f, indent=2)
    print("Saved Pearson correlation coefficients JSON")

    with open(os.path.join(output_dir, "head_gaze_conflict_vs_error_correlation_coefficients_spearman.json"), "w") as f:
        json.dump(spearman_grouped, f, indent=2)
    print("Saved Spearman correlation coefficients JSON")

def load_gaze_vs_head_angle_stats_from_file(csv_path):
    return pd.read_csv(csv_path)

# -------------------- entry --------------------

def run_conflict_regression(subject_dirs: List[str], models: List[str], outdir: str):
    os.makedirs(outdir, exist_ok=True)

    # Build full video-level table
    df_vid = build_subject_video_table(subject_dirs, models)

    # NEW: restrict to complete-case subjects for head_pose analysis (n=50)
    df_vid, dropped = _filter_complete_headpose_subjects(df_vid)
    if not dropped.empty:
        dropped.to_csv(os.path.join(outdir, "head_pose__dropped_incomplete_subjects.csv"), index=False)
        print(f"Dropped {dropped.shape[0]} incomplete head-pose subjects; list saved to head_pose__dropped_incomplete_subjects.csv")

    # Now everything below uses the filtered df_vid
    per_model, per_subject = fit_subject_level_slopes(df_vid)

    df_vid.to_csv(os.path.join(outdir, "head_gaze_conflict__video_level_table.csv"), index=False)
    per_subject.to_csv(os.path.join(outdir, "head_gaze_conflict__subject_slopes.csv"), index=False)
    per_model.to_csv(os.path.join(outdir, "head_gaze_conflict__per_model_summary.csv"), index=False)

    # Head Pose subject-level descriptives (ALL only) and single aggregated boxplot
    export_headpose_subject_level_descriptives(df_vid, outdir=outdir)
    plot_headpose_boxplot_weighted_rows(df_vid, outdir=outdir, ymax=60.0)
    run_pairwise_models_headpose(df_vid, outdir=outdir)

    print("Wrote:",
          "head_gaze_conflict__video_level_table.csv,",
          "head_gaze_conflict__subject_slopes.csv,",
          "head_gaze_conflict__per_model_summary.csv")
    return per_model, per_subject


# -------------------- Head Pose: subject-level descriptives + boxplot --------------------

def _summ_subject_level(vals: np.ndarray) -> Dict[str, float]:
    """
    Given a 1D array of subject-level mean errors for a single model,
    return n, mean, sd, se, and 95% CI bounds.
    """
    vals = np.asarray(vals, dtype=float)
    n = int(vals.size)
    mean = float(np.mean(vals)) if n > 0 else np.nan
    sd = float(np.std(vals, ddof=1)) if n > 1 else np.nan
    se = float(sd / np.sqrt(n)) if (n > 1 and np.isfinite(sd)) else np.nan
    t_crit = stats.t.ppf(0.975, n - 1) if n > 0 else np.nan
    ci_lo = float(mean - t_crit * se)
    ci_hi = float(mean + t_crit * se)
    return {
        "n_subjects": n,
        "mean_subject": mean,
        "sd_subject": sd,
        "se_subject": se,
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
    }


def _subject_level_video_means_headpose(df_vid: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to subject-level, frame-weighted means per (gaze_model) across
    ALL head-pose videos (left/middle/right combined).
    Uses video_error_mean weighted by num_frames (valid frames).
    Returns:
      ['gaze_model','gaze_model_display','subject_dir','mean_error']
    """
    need = {"gaze_model","subject_dir","experiment_type","video_error_mean","num_frames"}
    miss = need - set(df_vid.columns)
    if miss:
        raise ValueError(f"Missing required columns in df_vid: {sorted(miss)}")

    # Keep only head-pose experiments and copy
    d = df_vid[df_vid["experiment_type"].isin(INCLUDED_EXPERIMENT_TYPES)].copy()
    if d.empty:
        return pd.DataFrame(columns=[
            "gaze_model","gaze_model_display","subject_dir","mean_error"
        ])

    # Map display names
    disp_map = config.get_main_models_included_in_the_paper()
    d["gaze_model_display"] = d["gaze_model"].map(disp_map)

    # Frame-weighted per (subject, model) mean across ALL head-pose videos
    d["_wx"] = d["video_error_mean"] * d["num_frames"]
    agg = (
        d.groupby(["gaze_model","gaze_model_display","subject_dir"], as_index=False)
         .agg(total_w=("num_frames","sum"), sum_wx=("_wx","sum"))
    )
    agg = agg[agg["total_w"] > 0]
    agg["mean_error"] = agg["sum_wx"] / agg["total_w"]

    return agg.loc[:, ["gaze_model","gaze_model_display","subject_dir","mean_error"]]


def export_headpose_subject_level_descriptives(df_vid: pd.DataFrame, outdir: str) -> pd.DataFrame:
    """
    Subject-level descriptives across ALL head-pose experiments combined (head_pose_all only).
    Output columns:
      ['experiment_type','gaze_model','n_subjects','mean_subject','sd_subject',
       'se_subject','ci95_low','ci95_high']
    Saves a single CSV: head_pose_subject_level_descriptives__all.csv
    """
    import os
    os.makedirs(outdir, exist_ok=True)

    subj = _subject_level_video_means_headpose(df_vid)  # one row per (subject, model)
    if subj.empty:
        print("No Head Pose subject-level means to summarize.")
        return subj

    rows_all = []
    for gm_disp, dgm in subj.groupby("gaze_model_display", as_index=False):
        vals = dgm["mean_error"].to_numpy(dtype=float)
        stats_dict = _summ_subject_level(vals)  # n, mean, sd, se, ci95_low, ci95_high
        rows_all.append({
            "experiment_type": "head_pose_all",
            "gaze_model": gm_disp,
            **stats_dict,
        })

    all_df = pd.DataFrame(rows_all).sort_values(["gaze_model"]).reset_index(drop=True)

    # Save a SINGLE file with head_pose_all rows only
    combined_csv = os.path.join(outdir, "head_pose_subject_level_descriptives__all.csv")
    all_df.to_csv(combined_csv, index=False)
    print(f"Saved Head Pose subject-level descriptives (ALL ONLY) to {combined_csv}")

    return all_df



def _weighted_quantile(values: np.ndarray, weights: np.ndarray, qs) -> np.ndarray:
    """Weighted quantiles (helper duplicated locally for standalone use)."""
    if values.size == 0:
        return np.array([np.nan] * len(list(qs)))
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return np.array([np.nan] * len(list(qs)))
    cdf = cw / cw[-1]
    return np.interp(qs, cdf, v)


def _draw_weighted_box(ax, x_center: float, width: float,
                       q1: float, med: float, q3: float,
                       wl: float, wu: float, mean_val: float,
                       x_min: float = 0.0, x_max: float = 100.0):
    """
    Draw a single weighted box at x_center with given stats.
    All strokes are black; box has a light neutral fill.
    Assumes x_center has already been clamped to keep the box in-bounds.
    """
    if width <= 0:
        return

    left = x_center - width / 2.0
    # box (light neutral fill)
    rect = plt.Rectangle((left, q1), width, q3 - q1,
                         facecolor="#cfd8e3", edgecolor="black", linewidth=1.25, alpha=0.6)
    ax.add_patch(rect)

    # median (black)
    ax.plot([left, left + width], [med, med], color="black", linewidth=1.5)

    # whiskers + caps (black)
    ax.plot([x_center, x_center], [wl, q1], color="black", linewidth=1.25)
    ax.plot([x_center, x_center], [q3, wu], color="black", linewidth=1.25)
    cap = width * 0.25
    ax.plot([x_center - cap, x_center + cap], [wl, wl], color="black", linewidth=1.25)
    ax.plot([x_center - cap, x_center + cap], [wu, wu], color="black", linewidth=1.25)

    # mean marker (black triangle)
    ax.plot(x_center, mean_val, marker="^", markersize=6, color="black")


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denom = weights.sum()
    return float((values * weights).sum() / denom) if denom > 0 else np.nan


def _weighted_boxplot_data(vals: np.ndarray, wts: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    Compute weighted (q1, median, q3, lower whisker, upper whisker, mean).
    Whiskers use Tukey rule with weighted IQR: lower = max(min, q1 - 1.5*IQR) clipped to data,
    upper = min(max, q3 + 1.5*IQR) clipped to data.
    """
    q1, med, q3 = _weighted_quantile(vals, wts, [0.25, 0.50, 0.75])
    iqr = q3 - q1
    low_bound = q1 - 1.5 * iqr
    up_bound = q3 + 1.5 * iqr
    vmin = vals.min() if vals.size else np.nan
    vmax = vals.max() if vals.size else np.nan
    wl = max(vmin, low_bound)
    wu = min(vmax, up_bound)
    mean_val = _weighted_mean(vals, wts)
    return float(q1), float(med), float(q3), float(wl), float(wu), float(mean_val)

def plot_headpose_boxplot_weighted_rows(
    df_vid: pd.DataFrame,
    outdir: str,
    ymax: Optional[float] = 60.0,
    show: bool = False,
):
    """
    Single boxplot for all head-pose experiments combined (head_pose_left/middle/right).
    - X-axis is categorical (models spaced evenly).
    - Models are ordered left→right by *subject-level mean error* (best first),
      where each subject's per-model mean is weighted by num_frames across all head-pose videos.
    - Row weights for the box statistics are num_frames; the values are video_error_mean.
    - Model labels are cleaned to drop any ' w/ ...' suffix (e.g., 'GazeTR').

    Requires helpers:
      _weighted_boxplot_data(vals, wts) and _draw_weighted_box(ax, x_center, width, ...)
    """
    os.makedirs(outdir, exist_ok=True)

    # Keep only head-pose videos
    df = df_vid[df_vid["experiment_type"].isin(INCLUDED_EXPERIMENT_TYPES)].copy()
    if df.empty:
        print("No head-pose rows to plot.")
        return

    # Display name mapping (fallback if display_model_name is not present)
    if "gaze_model_display" not in df.columns:
        try:
            df["gaze_model_display"] = df["gaze_model"].map(config.display_model_name)
        except Exception:
            disp_map = config.get_main_models_included_in_the_paper()
            df["gaze_model_display"] = df["gaze_model"].map(disp_map)

    # -------- Order models by subject-level mean error (best first) --------
    d_tmp = df.loc[:, ["gaze_model_display", "subject_dir", "video_error_mean", "num_frames"]].copy()
    d_tmp["_wx"] = d_tmp["video_error_mean"] * d_tmp["num_frames"]

    # subject-level (frame-weighted across all head_pose videos)
    subj_means = (
        d_tmp.groupby(["gaze_model_display", "subject_dir"], as_index=False)
             .agg(total_w=("num_frames","sum"), sum_wx=("_wx","sum"))
    )
    subj_means["mean_error"] = subj_means["sum_wx"] / subj_means["total_w"]

    # model mean of subject means → sort ascending
    model_order = (
        subj_means.groupby("gaze_model_display")["mean_error"]
                  .mean()
                  .sort_values()
                  .index
                  .tolist()
    )

    # Clean labels (drop any " w/ ..." suffix)
    model_order_clean = [m.split(" w/")[0].strip() for m in model_order]

    # Evenly spaced categorical x positions (0,1,2,...)
    x_positions = {m: i for i, m in enumerate(model_order)}

    # -------- Plot weighted-rows boxplots (values = video_error_mean, weights = num_frames) --------
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    width = 0.6

    for m_disp in model_order:
        dM = df[df["gaze_model_display"] == m_disp]
        if dM.empty:
            continue

        vals = dM["video_error_mean"].to_numpy(dtype=float)
        wts  = dM["num_frames"].to_numpy(dtype=float)
        if vals.size == 0 or wts.sum() == 0:
            continue

        q1, med, q3, wl, wu, mean_val = _weighted_boxplot_data(vals, wts)
        xc = float(x_positions[m_disp])

        # Uniform style: neutral box fill, black lines/markers
        _draw_weighted_box(ax, xc, width, q1, med, q3, wl, wu, mean_val)

        # Annotate if whisker exceeds y cap
        if ymax is not None and wu > ymax:
            ax.text(xc - 0.15, ymax - 0.6, f"{wu:.1f}",
                    ha="center", va="top", fontsize=9)

    # Axes formatting
    ax.set_title("Angular Error — Head Pose (all)")
    ax.set_ylabel("Angular Error (deg)")
    ax.set_xlabel("")
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(model_order_clean, rotation=0, ha="center")
    if ymax is not None:
        ax.set_ylim(0.0, ymax)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4)

    fig.tight_layout()
    out_png = os.path.join(outdir, "head_pose__boxplot_weighted_rows__all.png")
    fig.savefig(out_png, dpi=220)
    if show:
        plt.show()
    plt.close(fig)

def run_pairwise_models_headpose(df_vid: pd.DataFrame, outdir: str, padjust: str = "holm") -> pd.DataFrame:
    """
    Perform pairwise model comparisons across all head-pose experiments combined.
    Follows subject-level, frame-weighted aggregation principles.
    """
    # 1. Get subject-level means per model (already frame-weighted in this helper)
    subj_df = _subject_level_video_means_headpose(df_vid)
    
    if subj_df.empty or subj_df["gaze_model_display"].nunique() < 2:
        print("[conflict] Not enough models for pairwise comparison.")
        return pd.DataFrame()

    # Ensure types are consistent for pingouin
    subj_df["subject_dir"] = subj_df["subject_dir"].astype(str)
    subj_df["gaze_model_display"] = subj_df["gaze_model_display"].astype(str)

    # 2. Run pairwise parametric tests (paired/within-subjects)
    res = pg.pairwise_tests(
        data=subj_df,
        dv="mean_error",
        within="gaze_model_display",
        subject="subject_dir",
        parametric=True,
        padjust=padjust,
        effsize="cohen",
        return_desc=True
    )

    # Sort for consistent output
    if {"A", "B"}.issubset(res.columns):
        res.sort_values(["A", "B"], inplace=True)

    out_csv = os.path.join(outdir, "head_pose_pairwise_model_comparisons.csv")
    res.to_csv(out_csv, index=False)
    print(f"[conflict] Saved Head Pose pairwise model comparisons -> {out_csv}")
    
    return res

def run():
    BASE_DIR = config.get_dataset_base_directory()
    SUBJECT_DIRS = config.get_dataset_subject_directories()
    OUTPUT_DIR = os.path.join(BASE_DIR, "head_gaze_conflict_vs_error_results")

    MODELS = list(config.get_main_models_included_in_the_paper().keys())
    _ = run_conflict_regression(SUBJECT_DIRS, MODELS, OUTPUT_DIR)

    # new exports
    export_conflict_angle_range(OUTPUT_DIR)
    export_conflict_angle_histogram_1deg(OUTPUT_DIR)

if __name__ == "__main__":
    run()
