# camera_viewpoint.py
"""
Compare camera viewpoint conditions: lighting_100 vs circular_movement.

We run subject-fixed (within-subjects) pairwise tests for each model with Pingouin,
Holm-corrected, and report subject-level means/stds.

Pipeline:
  raw CSV rows
    -> filter to exp_type in {'lighting_100', 'circular_movement'}
    -> filter to main paper models (by key)
    -> subject-level, frame-weighted mean error per (model, subject, condition)
    -> keep only subjects who have BOTH conditions for a given model
    -> pg.pairwise_tests(dv='mean_error', within='condition', subject='subject',
                         parametric=True, padjust='holm', effsize='cohen',
                         return_desc=True)

Outputs (to --outdir):
  - camera_viewpoint_subject_level_descriptives.csv
      columns: gaze_model_display, condition, n_subjects, mean_subject, sd_subject, se_subject, ci95_low, ci95_high
  - pairwise_camera_viewpoint_within__<ModelDisplay>.csv (one per model)
  - pairwise_camera_viewpoint_within__all_models.csv (stacked)
  - camera_viewpoint_subject_level_matrix.csv
      wide table with per-subject mean_error for both conditions (inner-joined subjects)

Usage:
  python camera_viewpoint.py --csv /path/to/gaze_evaluation_results.csv --outdir /path/to/out
If --csv omitted, defaults to <config.get_dataset_base_directory()>/gaze_evaluation_results.csv
If --outdir omitted, defaults to <config.get_dataset_base_directory()>/camera_viewpoint
"""
import os
import re
import argparse
from typing import Optional, Iterable, Dict, List, Tuple

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

TARGET_EXPS = {"lighting_100", "circular_movement"}
CONDITION_ORDER = ["lighting_100", "circular_movement"]  # for consistent ordering in outputs


# --------------------------- Utilities ---------------------------

def _require_cols(df: pd.DataFrame, cols: Iterable[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def load_and_prepare(csv_path: Optional[str]) -> pd.DataFrame:
    """
    Load CSV and standardize columns to:
      ['gaze_model','gaze_model_display','subject','condition','angular_error','num_samples']

    - Filters to exp_type in TARGET_EXPS
    - Filters to main models via config.get_main_models_included_in_the_paper().keys()
    - Uses error_mean as angular_error; num_samples=1 if missing
    """
    import config  # local project config

    if csv_path is None:
        base_dir = config.get_dataset_base_directory()
        csv_path = os.path.join(base_dir, "gaze_evaluation_results.csv")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # expected columns
    _require_cols(df, ["gaze_model", "exp_type", "error_mean"])
    if "subject" not in df.columns:
        if "subject_dir" in df.columns:
            df["subject"] = df["subject_dir"]
        else:
            raise ValueError("CSV must include 'subject' or 'subject_dir'.")

    # filter to target experiment types
    df = df[df["exp_type"].isin(TARGET_EXPS)].copy()
    if df.empty:
        raise ValueError(f"No rows with exp_type in {TARGET_EXPS} found.")

    # filter to main paper models by key (CSV stores keys)
    main_model_keys = set(config.get_main_models_included_in_the_paper().keys())
    df = df[df["gaze_model"].isin(main_model_keys)].copy()
    if df.empty:
        raise ValueError("After filtering to main models, no rows remain. Check model keys / CSV.")

    # normalize
    df["angular_error"] = df["error_mean"].astype(float)
    df["condition"] = df["exp_type"].astype(str)
    if "num_samples" not in df.columns:
        df["num_samples"] = 1

    # attach display names
    df["gaze_model_display"] = df["gaze_model"].map(config.display_model_name)

    cols = ["gaze_model", "gaze_model_display", "subject", "condition", "angular_error", "num_samples"]
    return df.loc[:, cols].copy()


def _subject_level_weighted_means(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to subject-level, frame-weighted mean per (model, subject, condition).
    Returns: ['gaze_model','gaze_model_display','subject','condition','mean_error']
    """
    need = {"gaze_model", "gaze_model_display", "subject", "condition", "angular_error", "num_samples"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing required cols: {missing}")

    d = df.copy()
    d["_wx"] = d["angular_error"] * d["num_samples"]
    agg = (
        d.groupby(["gaze_model", "gaze_model_display", "subject", "condition"], as_index=False)
         .agg(total_w=("num_samples", "sum"), sum_wx=("_wx", "sum"))
    )
    agg["mean_error"] = agg["sum_wx"] / agg["total_w"]
    return agg.loc[:, ["gaze_model", "gaze_model_display", "subject", "condition", "mean_error"]]


def _inner_subjects_with_both_conditions(dmodel: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only subjects that have BOTH conditions present for this model.
    """
    cond_counts = dmodel.groupby("subject")["condition"].nunique()
    keep_subjects = cond_counts[cond_counts >= 2].index
    return dmodel[dmodel["subject"].isin(keep_subjects)].copy()


def compute_subject_level_descriptives(csv_path: Optional[str], outdir: str) -> pd.DataFrame:
    """
    Subject-level descriptives per (model, condition), using subject-level mean_error:
      n_subjects, mean_subject, sd_subject, se_subject, ci95_low, ci95_high
    """
    os.makedirs(outdir, exist_ok=True)
    df = load_and_prepare(csv_path)
    subj = _subject_level_weighted_means(df)

    rows = []
    for (gm, gmd, cond), g in subj.groupby(["gaze_model", "gaze_model_display", "condition"]):
        vals = g["mean_error"].to_numpy(dtype=float)
        n = vals.size
        mean = float(np.mean(vals)) if n else np.nan
        sd = float(np.std(vals, ddof=1)) if n > 1 else np.nan
        se = float(sd / np.sqrt(n)) if (n > 1 and np.isfinite(sd)) else np.nan
        t_crit = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
        ci_lo = float(mean - t_crit * se) if np.isfinite(se) else np.nan
        ci_hi = float(mean + t_crit * se) if np.isfinite(se) else np.nan
        rows.append({
            "gaze_model_display": gmd,
            "condition": cond,
            "n_subjects": int(n),
            "mean_subject": mean,
            "sd_subject": sd,
            "se_subject": se,
            "ci95_low": ci_lo,
            "ci95_high": ci_hi,
        })

    out_df = pd.DataFrame(rows).sort_values(["gaze_model_display", "condition"]).reset_index(drop=True)
    out_csv = os.path.join(outdir, "camera_viewpoint_subject_level_descriptives.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"[camera_viewpoint] Saved subject-level descriptives -> {out_csv}")
    return out_df


def run_pairwise_camera_viewpoint_within_models(
    csv_path: Optional[str],
    outdir: str,
    padjust: str = "holm"
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Within-subject pairwise tests comparing lighting_100 vs circular_movement, per model.
    Uses subject-level mean_error points and retains only subjects with BOTH conditions.
    Returns combined DataFrame and dict of per-model results (keyed by display name).
    """
    os.makedirs(outdir, exist_ok=True)
    df = load_and_prepare(csv_path)
    subj = _subject_level_weighted_means(df)

    # Make sure categorical ordering is stable (optional)
    subj["condition"] = pd.Categorical(subj["condition"], categories=CONDITION_ORDER, ordered=True)
    subj["subject"] = subj["subject"].astype(str)

    combined_list: List[pd.DataFrame] = []
    per_model: Dict[str, pd.DataFrame] = {}

    for gm, dmodel in subj.groupby("gaze_model"):
        display = dmodel["gaze_model_display"].iloc[0]

        # Keep only subjects appearing in BOTH conditions for this model
        dmodel_both = _inner_subjects_with_both_conditions(dmodel)
        if dmodel_both["subject"].nunique() < 2:
            # Need at least two paired points to run a meaningful paired test
            print(f"[camera_viewpoint] Skipping {display}: fewer than 2 paired subjects.")
            continue

        res = pg.pairwise_tests(
            data=dmodel_both,
            dv="mean_error",
            within="condition",
            subject="subject",
            parametric=True,
            padjust=padjust,
            effsize="cohen",
            return_desc=True
        )

        # Annotate with model display name and tidy
        res.insert(0, "gaze_model", display)

        # Optional: sort by A,B for readability if present
        if {"A", "B"}.issubset(res.columns):
            res = res.sort_values(["A", "B"]).reset_index(drop=True)

        per_model[display] = res
        combined_list.append(res)

        safe = _sanitize_filename(display)
        out_csv = os.path.join(outdir, f"pairwise_camera_viewpoint_within__{safe}.csv")
        res.to_csv(out_csv, index=False)

    if not combined_list:
        return pd.DataFrame(), per_model

    combined = pd.concat(combined_list, axis=0, ignore_index=True)
    combined = combined.sort_values(["gaze_model"] + ([c for c in ["A", "B"] if c in combined.columns]))
    all_csv = os.path.join(outdir, "pairwise_camera_viewpoint_within__all_models.csv")
    combined.to_csv(all_csv, index=False)
    print(f"[camera_viewpoint] Wrote per-model & combined pairwise results -> {outdir}")
    return combined, per_model


def export_subject_level_matrix(csv_path: Optional[str], outdir: str) -> pd.DataFrame:
    """
    Export a wide matrix: rows = (model_display, subject), columns = conditions,
    values = subject-level mean_error; *inner-joined* on subjects with both conditions.
    Helpful to quickly inspect per-subject paired values.
    """
    os.makedirs(outdir, exist_ok=True)
    df = load_and_prepare(csv_path)
    subj = _subject_level_weighted_means(df)

    # Pivot then drop rows with any NaN across the two conditions (ensures pairs)
    wide = (
        subj.pivot_table(
            index=["gaze_model_display", "subject"],
            columns="condition",
            values="mean_error",
            aggfunc="mean"
        )
        .reset_index()
    )

    # Keep only rows with both conditions present
    for cond in TARGET_EXPS:
        if cond not in wide.columns:
            wide[cond] = np.nan
    wide = wide.dropna(subset=list(TARGET_EXPS), how="any").copy()

    out_csv = os.path.join(outdir, "camera_viewpoint_subject_level_matrix.csv")
    wide.to_csv(out_csv, index=False)
    print(f"[camera_viewpoint] Saved subject-level matrix -> {out_csv}")
    return wide


def analyze_camera_viewpoint(
    csv_path: Optional[str],
    outdir: str,
    padjust: str = "holm"
):
    """
    Top-level entry:
      1) Subject-level descriptives per condition (subject means & stds)
      2) Paired, subject-fixed pairwise tests (Holm) per model
      3) Subject-level wide matrix for quick inspection
    """
    os.makedirs(outdir, exist_ok=True)
    compute_subject_level_descriptives(csv_path, outdir)
    run_pairwise_camera_viewpoint_within_models(csv_path, outdir, padjust=padjust)
    export_subject_level_matrix(csv_path, outdir)
    print("[camera_viewpoint] Complete.")


# --------------------------- CLI ---------------------------

if __name__ == "__main__":
    import config  # type: ignore

    default_csv = os.path.join(config.get_dataset_base_directory(), "gaze_evaluation_results.csv")
    default_out = os.path.join(config.get_dataset_base_directory(), "camera_viewpoint")

    ap = argparse.ArgumentParser(description="Compare lighting_100 vs circular_movement (subject-fixed, Holm-corrected).")
    ap.add_argument("--csv", type=str, default=default_csv, help="Path to gaze_evaluation_results.csv")
    ap.add_argument("--outdir", type=str, default=default_out, help="Directory for CSV outputs")
    ap.add_argument("--padjust", type=str, default="holm", choices=["holm", "fdr_bh", "bonf", "sidak", "none"],
                    help="Multiple-comparison correction method (pingouin)")
    args = ap.parse_args()

    analyze_camera_viewpoint(csv_path=args.csv, outdir=args.outdir, padjust=args.padjust)
