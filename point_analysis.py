# point_analysis.py
"""
Point Analysis (within-model) for the 'lighting_100' experiment.

Goal
----
For each gaze model, compare gaze-target points (e.g., p1..p9) using paired
within-subject tests, analogous to the lighting analysis (but now the factor
is 'point' instead of 'lighting').

What this script does
---------------------
1) Loads the raw CSV (same format as lighting analysis).
2) Filters to experiment_type == 'lighting_100'.
3) Extracts the point label for each video (expects patterns like 'p1'..'p9'
   in exp_dir/video names, or uses a dedicated 'point' column if present).
4) Collapses to subject-level, frame-weighted mean error per (model, point).
5) Runs pairwise, within-subject comparisons across points within each model:
   - pingouin.pairwise_tests (parametric t-tests), two-sided, paired.
   - Holm p-adjust.
   - Cohen's d effect size.
   Saves:
     outdir/pairwise_point_within__<MODEL_DISPLAY>__lighting_100.csv
     outdir/pairwise_point_within__all_models__lighting_100.csv

Assumptions about the CSV
-------------------------
Columns include at least:
- exp_type (experiment type; we use rows where exp_type == 'lighting_100')
- gaze_model
- subject OR subject_dir
- error_mean  (per-row mean angular error in degrees)
- num_samples (row weight = #frames in that row/video)
Optional:
- error_std
- exp_dir / video / point column to infer target point (see _extract_point)

Usage
-----
python point_analysis.py --csv /path/to/gaze_evaluation_results.csv --outdir /path/to/out

Notes
-----
- Model names in outputs use config.display_model_name(model) so they are
  paper-ready and consistent with your figures/tables.
- If the script cannot infer the point label, it raises a helpful error.
"""

import os
import re
import argparse
from typing import Iterable, Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import pingouin as pg

import config  # must provide display_model_name(...) and dataset base if you want defaults

from frame_db import load_frame_dataset
from pitch_yaw_stats import load_pitch_yaw_db
from scipy import stats

# --------------------------- Helpers ---------------------------

POINT_PATTERN = re.compile(r'(?:^|[^A-Za-z0-9])(p[1-9])(?=[^A-Za-z0-9]|$)', re.IGNORECASE)

def _require_cols(df: pd.DataFrame, cols: Iterable[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def _standardize_point_label(raw: str) -> str:
    """
    Map anything like 'p1', 'p1' to canonical 'p1'. Leaves 'p10+' alone if ever present.
    """
    m = POINT_PATTERN.search(str(raw))
    if not m:
        return ""
    return m.group(1).upper()  # e.g., p1..p9

def _extract_point_from_row(row: pd.Series) -> str:
    """
    Try several columns to infer the point label. Priority order:
      1) 'point' (if present; numeric -> 'P#', string with p# ok)
      2) any of ['exp_dir','video','trial','trial_name'] via regex p[1-9]
    Returns canonical 'P#' or '' if not found.
    """
    # direct point column
    if "point" in row:
        val = row["point"]
        if pd.notna(val):
            s = str(val).strip()
            if s.isdigit():
                return f"P{int(s)}"
            std = _standardize_point_label(s)
            if std:
                return std

    # search other common name columns
    for c in ["exp_dir", "video", "trial", "trial_name"]:
        if c in row and pd.notna(row[c]):
            std = _standardize_point_label(row[c])
            if std:
                return std

    return ""


# --------------------------- Data prep ---------------------------

def load_and_prepare(csv_path: Optional[str], experiment_type: str = "lighting_100") -> pd.DataFrame:
    """
    Load CSV, filter to experiment_type, derive:
      - subject (from 'subject' or 'subject_dir')
      - angular_error (from 'error_mean')
      - point (canonical 'p1'..'p9' via _extract_point_from_row)
      - gaze_model_display (via config.display_model_name)
    Returns columns:
      ['gaze_model','gaze_model_display','subject','point','angular_error','num_samples']
    """
    if csv_path is None:
        base_dir = config.get_dataset_base_directory()
        csv_path = os.path.join(base_dir, "gaze_evaluation_results.csv")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # normalize expected columns
    _require_cols(df, ["exp_type", "gaze_model", "error_mean"])
    df["angular_error"] = df["error_mean"]
    df["experiment_type"] = df["exp_type"]

    main_model_keys = set(config.get_main_models_included_in_the_paper().keys())
    df = df[df["gaze_model"].isin(main_model_keys)].copy()

    # subject
    if "subject" not in df.columns:
        if "subject_dir" in df.columns:
            df["subject"] = df["subject_dir"]
        else:
            raise ValueError("CSV must have 'subject' or 'subject_dir'.")

    if "num_samples" not in df.columns:
        df["num_samples"] = 1

    # keep only target experiment_type(s); also create a friendly label for outputs
    if isinstance(experiment_type, (list, tuple, set)):
        wanted = list(experiment_type)
        df = df[df["experiment_type"].isin(wanted)].copy()
        exp_label = "_".join(sorted(set(wanted)))
    else:
        if experiment_type == "all_lighting":
            wanted = [f"lighting_{x}" for x in (10, 25, 50, 100)]
            df = df[df["experiment_type"].isin(wanted)].copy()
            exp_label = "all_lighting"
        elif experiment_type == "all_lighting_and_circular":
            wanted = [f"lighting_{x}" for x in (10, 25, 50, 100)] + ["circular_movement"]
            df = df[df["experiment_type"].isin(wanted)].copy()
            exp_label = "all_lighting_and_circular"
        else:
            df = df[df["experiment_type"] == experiment_type].copy()
            exp_label = str(experiment_type)

    if df.empty:
        raise ValueError(f"No rows found for experiment_type == '{experiment_type}'.")

    # display name for paper
    df["gaze_model_display"] = df["gaze_model"].map(config.display_model_name)

    keep = ["gaze_model", "gaze_model_display", "subject", "point", "angular_error", "num_samples"]
    out = df.loc[:, keep].copy()
    out["experiment_label"] = exp_label  # carry a clean name into outputs
    return out


def subject_level_point_means(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to subject-level, frame-weighted mean error per (model, point).
    Returns:
      ['gaze_model','gaze_model_display','subject','point','mean_error']
    """
    need = {"gaze_model","gaze_model_display","subject","point","angular_error","num_samples"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    d = df.copy()
    d["_wx"] = d["angular_error"] * d["num_samples"]

    agg = (
        d.groupby(["gaze_model","gaze_model_display","subject","point"], as_index=False)
         .agg(total_w=("num_samples","sum"), sum_wx=("_wx","sum"))
    )
    agg["mean_error"] = agg["sum_wx"] / agg["total_w"]
    return agg.loc[:, ["gaze_model","gaze_model_display","subject","point","mean_error"]]

def compute_point_subject_level_descriptives(
    csv_path: Optional[str],
    outdir: str,
    experiment_type: str = "lighting_100",
) -> pd.DataFrame:
    """
    Subject-level descriptives per (MODEL × POINT).
    Each subject contributes a single frame-weighted mean error to each (model, point).

    Writes ONE CSV that includes ALL models for the given experiment_type:
      outdir/point_subject_level_descriptives__<experiment_label>.csv

    Columns:
      gaze_model, gaze_model_display, point, n_subjects, mean_subject, sd_subject, se_subject, ci95_low, ci95_high
    """
    import numpy as np
    import os

    os.makedirs(outdir, exist_ok=True)
    df = load_and_prepare(csv_path, experiment_type=experiment_type)
    exp_label = df["experiment_label"].iloc[0]

    subj = subject_level_point_means(df)  # ['gaze_model','gaze_model_display','subject','point','mean_error']

    def _summ(g):
        vals = g["mean_error"].to_numpy(dtype=float)
        n = vals.size
        mean = float(np.mean(vals)) if n else np.nan
        sd = float(np.std(vals, ddof=1)) if n > 1 else np.nan
        se = float(sd / np.sqrt(n)) if (n > 1 and np.isfinite(sd)) else np.nan
        t_crit = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
        ci_lo = float(mean - t_crit * se) if np.isfinite(se) else np.nan
        ci_hi = float(mean + t_crit * se) if np.isfinite(se) else np.nan
        return pd.Series({
            "n_subjects": int(n),
            "mean_subject": mean,
            "sd_subject": sd,
            "se_subject": se,
            "ci95_low": ci_lo,
            "ci95_high": ci_hi,
        })

    desc = (
        subj.groupby(["gaze_model", "gaze_model_display", "point"], as_index=False)
            .apply(_summ, include_groups=False)
            .reset_index(drop=True)
            .sort_values(["gaze_model_display", "point"])
    )

    out_csv = os.path.join(outdir, f"point_subject_level_descriptives__{exp_label}.csv")
    desc.to_csv(out_csv, index=False)
    print(f"Saved subject-level descriptives to {out_csv}")
    return desc


# --------------------------- Pairwise tests ---------------------------

def run_pairwise_points_within_models(
    csv_path: Optional[str],
    outdir: str,
    experiment_type: str = "lighting_100",
    padjust: str = "holm",
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    For each model, paired comparisons across points within the specified experiment_type.
    Uses pingouin.pairwise_tests with:
      dv='mean_error', within='point', subject='subject', parametric=True,
      two-sided, Holm p-adjust, Cohen's d, and descriptive stats.

    Writes:
      - outdir/pairwise_point_within__<MODEL_DISPLAY>__lighting_100.csv
      - outdir/pairwise_point_within__all_models__lighting_100.csv

    Returns:
      (combined_df, per_model_dict_by_display_name)
    """
    os.makedirs(outdir, exist_ok=True)

    df = load_and_prepare(csv_path, experiment_type=experiment_type)
    subj_df = subject_level_point_means(df)
    exp_label = df["experiment_label"].iloc[0]

    # Ensure strings for pingouin
    subj_df["subject"] = subj_df["subject"].astype(str)
    subj_df["point"] = subj_df["point"].astype(str)

    combined_list: List[pd.DataFrame] = []
    per_model: Dict[str, pd.DataFrame] = {}

    for model, dmodel in subj_df.groupby("gaze_model"):
        if dmodel.empty:
            continue
        display = dmodel["gaze_model_display"].iloc[0]

        res = pg.pairwise_tests(
            data=dmodel,
            dv="mean_error",
            within="point",
            subject="subject",
            parametric=True,
            padjust=padjust,
            effsize="cohen",
            return_desc=True,
        )

        # annotate and tidy
        res.insert(0, "gaze_model", display)
        res.insert(1, "experiment_label", exp_label)

        # order by pair names if present
        if {"A","B"}.issubset(res.columns):
            res.sort_values(["A","B"], inplace=True)

        per_model[display] = res
        combined_list.append(res)

        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", display)
        os.makedirs(os.path.join(outdir, safe), exist_ok=True)
        out_csv = os.path.join(outdir, safe, f"pairwise_point_within_{exp_label}.csv")
        res.to_csv(out_csv, index=False)

    if not combined_list:
        return pd.DataFrame(), per_model

    combined = pd.concat(combined_list, axis=0, ignore_index=True)
    combined.sort_values(["gaze_model","A","B"], inplace=True)
    os.makedirs(os.path.join(outdir, "all_models"), exist_ok=True)
    combined_csv = os.path.join(outdir, "all_models", f"pairwise_point_within_{exp_label}.csv")
    combined.to_csv(combined_csv, index=False)
    return combined, per_model


def subject_level_row_means(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to subject-level, frame-weighted mean error per (model, row).
    Rows are p123, p456, p789.
    """
    row_map = {"p1": "R1", "p2": "R1", "p3": "R1",
               "p4": "R2", "p5": "R2", "p6": "R2",
               "p7": "R3", "p8": "R3", "p9": "R3"}
    d = df.copy()
    d["row"] = d["point"].map(row_map)
    d["_wx"] = d["angular_error"] * d["num_samples"]
    agg = (
        d.groupby(["gaze_model","gaze_model_display","subject","row"], as_index=False)
         .agg(total_w=("num_samples","sum"), sum_wx=("_wx","sum"))
    )
    agg["mean_error"] = agg["sum_wx"] / agg["total_w"]
    return agg.loc[:, ["gaze_model","gaze_model_display","subject","row","mean_error"]]


def subject_level_col_means(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to subject-level, frame-weighted mean error per (model, col).
    Columns are p147, p258, p369.
    """
    col_map = {"p1": "C1", "p4": "C1", "p7": "C1",
               "p2": "C2", "p5": "C2", "p8": "C2",
               "p3": "C3", "p6": "C3", "p9": "C3"}
    d = df.copy()
    d["col"] = d["point"].map(col_map)
    d["_wx"] = d["angular_error"] * d["num_samples"]
    agg = (
        d.groupby(["gaze_model","gaze_model_display","subject","col"], as_index=False)
         .agg(total_w=("num_samples","sum"), sum_wx=("_wx","sum"))
    )
    agg["mean_error"] = agg["sum_wx"] / agg["total_w"]
    return agg.loc[:, ["gaze_model","gaze_model_display","subject","col","mean_error"]]

def run_pairwise_rows_within_models(csv_path, outdir, experiment_type="lighting_100", padjust="holm"):
    os.makedirs(outdir, exist_ok=True)
    df = load_and_prepare(csv_path, experiment_type=experiment_type)
    subj_df = subject_level_row_means(df)
    exp_label = df["experiment_label"].iloc[0]

    subj_df["subject"] = subj_df["subject"].astype(str)
    subj_df["row"] = subj_df["row"].astype(str)

    combined_list, per_model = [], {}
    for model, dmodel in subj_df.groupby("gaze_model"):
        display = dmodel["gaze_model_display"].iloc[0]
        res = pg.pairwise_tests(
            data=dmodel, dv="mean_error", within="row", subject="subject",
            parametric=True, padjust=padjust, effsize="cohen", return_desc=True
        )
        res.insert(0, "gaze_model", display)
        res.insert(1, "experiment_label", exp_label)
        res.sort_values(["A","B"], inplace=True)
        per_model[display] = res
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", display)
        os.makedirs(os.path.join(outdir, safe), exist_ok=True)
        res.to_csv(os.path.join(outdir, safe, f"pairwise_row_within_{exp_label}.csv"), index=False)
        combined_list.append(res)
    if combined_list:
        combined = pd.concat(combined_list, axis=0, ignore_index=True)
        os.makedirs(os.path.join(outdir, "all_models"), exist_ok=True)
        combined.to_csv(os.path.join(outdir, "all_models", f"pairwise_row_within_{exp_label}.csv"), index=False)
        return combined, per_model
    return pd.DataFrame(), per_model


def run_pairwise_cols_within_models(csv_path, outdir, experiment_type="lighting_100", padjust="holm"):
    os.makedirs(outdir, exist_ok=True)
    df = load_and_prepare(csv_path, experiment_type=experiment_type)
    subj_df = subject_level_col_means(df)
    exp_label = df["experiment_label"].iloc[0]

    subj_df["subject"] = subj_df["subject"].astype(str)
    subj_df["col"] = subj_df["col"].astype(str)

    combined_list, per_model = [], {}
    for model, dmodel in subj_df.groupby("gaze_model"):
        display = dmodel["gaze_model_display"].iloc[0]
        res = pg.pairwise_tests(
            data=dmodel, dv="mean_error", within="col", subject="subject",
            parametric=True, padjust=padjust, effsize="cohen", return_desc=True
        )
        res.insert(0, "gaze_model", display)
        res.insert(1, "experiment_label", exp_label)
        res.sort_values(["A","B"], inplace=True)
        per_model[display] = res
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", display)
        os.makedirs(os.path.join(outdir, safe), exist_ok=True)
        res.to_csv(os.path.join(outdir, safe, f"pairwise_col_within_{exp_label}.csv"), index=False)
        combined_list.append(res)
    if combined_list:
        combined = pd.concat(combined_list, axis=0, ignore_index=True)
        os.makedirs(os.path.join(outdir, "all_models"), exist_ok=True)
        combined.to_csv(os.path.join(outdir, "all_models", f"pairwise_col_within_{exp_label}.csv"), index=False)
        return combined, per_model
    return pd.DataFrame(), per_model


def run_pairwise_models_within_points(
    csv_path: Optional[str],
    outdir: str,
    experiment_type: str = "lighting_100",
    padjust: str = "holm",
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Subject-fixed pairwise comparisons BETWEEN MODELS, run separately for each point (p1..p9).
    Pipeline:
      raw -> load_and_prepare(experiment_type) -> subject_level_point_means
      -> for each point, pingouin.pairwise_tests with within='gaze_model', subject='subject'
         (paired, parametric, Holm, Cohen's d).
    Writes:
      - outdir/pairwise_model_within_point_p#.csv  (nine files)
      - outdir/pairwise_model_within_point__all.csv (stacked)
    Returns:
      (combined_df, per_point_dict)
    """
    os.makedirs(outdir, exist_ok=True)
    df = load_and_prepare(csv_path, experiment_type=experiment_type)
    subj = subject_level_point_means(df)  # ['gaze_model','gaze_model_display','subject','point','mean_error']

    # Ensure strings for pingouin
    subj["subject"] = subj["subject"].astype(str)
    subj["point"] = subj["point"].astype(str)

    per_point: Dict[str, pd.DataFrame] = {}
    combined: List[pd.DataFrame] = []

    points = sorted(subj["point"].unique().tolist(), key=lambda s: (len(s), s))  # p1..p9
    for pt in points:
        dP = subj[subj["point"] == pt].copy()
        if dP["gaze_model"].nunique() < 2:
            continue

        res = pg.pairwise_tests(
            data=dP,
            dv="mean_error",
            within="gaze_model",
            subject="subject",
            parametric=True,
            padjust=padjust,
            effsize="cohen",
            return_desc=True,
        )

        # annotate and tidy
        res.insert(0, "point", pt)
        # map A/B to DISPLAY names (paper-ready)
        if {"A","B"}.issubset(res.columns):
            res["A"] = res["A"].map(config.display_model_name)
            res["B"] = res["B"].map(config.display_model_name)
            if "Contrast" in res.columns:
                res["Contrast"] = res.apply(lambda r: f"{r['A']} - {r['B']}", axis=1)
            res.sort_values(["A","B"], inplace=True)

        per_point[pt] = res
        out_csv = os.path.join(outdir, f"pairwise_model_within_point_{pt}.csv")
        res.to_csv(out_csv, index=False)
        combined.append(res)

    if not combined:
        return pd.DataFrame(), per_point

    all_df = pd.concat(combined, axis=0, ignore_index=True)
    all_df.sort_values(["point", "A", "B"], inplace=True)
    all_csv = os.path.join(outdir, "pairwise_model_within_point__all.csv")
    all_df.to_csv(all_csv, index=False)
    return all_df, per_point


def summarize_pointwise_champions(
    csv_path: Optional[str],
    outdir: str,
    experiment_type: str = "lighting_100",
    alpha: float = 0.05,
    padjust: str = "holm",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Uses subject-level means and the per-point pairwise model tests to decide a 'champion' model at each point.
    Decision rule (per point):
      - Build a directed graph where A → B if p-corr < alpha AND mean(A) < mean(B).
      - Any model with outdegree == (K-1) (beats all others significantly) is a 'champion'.
      - If no such model exists, mark 'no decisive winner'; we also report the model(s) with the lowest mean.

    Writes:
      - outdir/pointwise_champions__lighting_100.csv (one row per point)
      - outdir/pointwise_wins_tally__lighting_100.csv (one row per model: n_points_won, points_won, also tied_top)
    Returns:
      (champions_df, tally_df)
    """
    os.makedirs(outdir, exist_ok=True)
    df = load_and_prepare(csv_path, experiment_type=experiment_type)
    exp_label = df["experiment_label"].iloc[0]
    subj = subject_level_point_means(df)

    # Mean by (point, display_name) for tie-breaking and ordering
    mean_by_model_point = (
        subj.groupby(["point", "gaze_model_display"])["mean_error"]
            .mean()
            .rename("mean_subject")
            .reset_index()
    )

    # Get pairwise per-point results (ensure we have them)
    _, per_point = run_pairwise_models_within_points(
        csv_path=csv_path, outdir=outdir, experiment_type=experiment_type, padjust=padjust
    )

    champion_rows = []
    wins_map: Dict[str, List[str]] = {}          # model_display -> [points won]
    ties_top_map: Dict[str, List[str]] = {}      # model_display -> [points tied at top mean (non-sig)]

    for pt, res in per_point.items():
        # Collect model set and means
        m_means = mean_by_model_point[mean_by_model_point["point"] == pt] \
                    .set_index("gaze_model_display")["mean_subject"].to_dict()
        models = sorted(m_means.keys())
        K = len(models)
        if K < 2:
            continue

        # Directed wins graph from pairwise tests
        wins = {m: set() for m in models}
        if not res.empty and {"A","B","p-corr","mean(A)","mean(B)"}.issubset(res.columns):
            for _, r in res.iterrows():
                A, B = str(r["A"]), str(r["B"])
                p_corr = float(r["p-corr"]) if pd.notna(r["p-corr"]) else np.nan
                ma = float(r["mean(A)"]) if pd.notna(r["mean(A)"]) else np.nan
                mb = float(r["mean(B)"]) if pd.notna(r["mean(B)"]) else np.nan
                if np.isfinite(p_corr) and p_corr < alpha and np.isfinite(ma) and np.isfinite(mb):
                    if ma < mb:
                        wins[A].add(B)
                    elif mb < ma:
                        wins[B].add(A)

        # Champions: beat everyone else
        champions = [m for m in models if len(wins[m]) == (K - 1)]

        # If no decisive champion, note the lowest-mean models (possible ties)
        sorted_by_mean = sorted(models, key=lambda m: m_means[m])
        best_mean = m_means[sorted_by_mean[0]]
        tied_best = [m for m in sorted_by_mean if np.isclose(m_means[m], best_mean, rtol=0.0, atol=1e-9)]

        if champions:
            decisive = True
            champ_display = "; ".join(champions)
            for c in champions:
                wins_map.setdefault(c, []).append(pt)
        else:
            decisive = False
            champ_display = ""  # none
            # record top-mean ties as "tied_top"
            for m in tied_best:
                ties_top_map.setdefault(m, []).append(pt)

        champion_rows.append({
            "experiment_label": exp_label,
            "point": pt,
            "models_considered": K,
            "decisive": decisive,
            "champion_models": champ_display,
            "order_by_mean": " < ".join(sorted_by_mean),            # lower is better
            "means_deg": "; ".join([f"{m}:{m_means[m]:.2f}" for m in sorted_by_mean]),
        })

    champions_df = pd.DataFrame(champion_rows).sort_values("point").reset_index(drop=True)
    champions_csv = os.path.join(outdir, f"pointwise_champions__{exp_label}.csv")
    champions_df.to_csv(champions_csv, index=False)

    # Wins tally
    all_models = sorted(mean_by_model_point["gaze_model_display"].unique().tolist())
    rows = []
    for m in all_models:
        pwon = wins_map.get(m, [])
        ptie = ties_top_map.get(m, [])
        rows.append({
            "experiment_label": exp_label,
            "gaze_model": m,
            "n_points_won": len(pwon),
            "points_won": ",".join(sorted(pwon)),
            "n_points_tied_topmean": len(ptie),
            "points_tied_topmean": ",".join(sorted(ptie)),
        })
    tally_df = pd.DataFrame(rows).sort_values(["n_points_won","n_points_tied_topmean","gaze_model"], ascending=[False, False, True])
    tally_csv = os.path.join(outdir, f"pointwise_wins_tally__{exp_label}.csv")
    tally_df.to_csv(tally_csv, index=False)

    print(f"Wrote per-point champions to {champions_csv}")
    print(f"Wrote wins tally to {tally_csv}")
    return champions_df, tally_df


# Two-variable regression for abs. pitch and yaw

def _ols_2pred(x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    OLS with intercept for y ~ b0 + b1*x1 + b2*x2.
    Returns dict with b0,b1,b2,se0,se1,se2,t0,t1,t2,p0,p1,p2,R2,n,df.
    NaN-safe; returns NaNs if degenerate.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    y  = np.asarray(y,  dtype=float)
    m = min(x1.size, x2.size, y.size)
    if m < 5:
        return {k: np.nan for k in ["b0","b1","b2","se0","se1","se2","t0","t1","t2","p0","p1","p2","R2","n","df"]}
    x1 = x1[:m]; x2 = x2[:m]; y = y[:m]
    msk = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
    x1 = x1[msk]; x2 = x2[msk]; y = y[msk]
    n = y.size
    p = 3  # intercept + 2 predictors
    if n <= p:
        return {k: np.nan for k in ["b0","b1","b2","se0","se1","se2","t0","t1","t2","p0","p1","p2","R2","n","df"]}

    X = np.column_stack([np.ones(n, dtype=float), x1, x2])  # [n,3]
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return {k: np.nan for k in ["b0","b1","b2","se0","se1","se2","t0","t1","t2","p0","p1","p2","R2","n","df"]}

    beta = XtX_inv @ (X.T @ y)
    yhat = X @ beta
    resid = y - yhat
    df = n - p
    if df <= 0:
        return {k: np.nan for k in ["b0","b1","b2","se0","se1","se2","t0","t1","t2","p0","p1","p2","R2","n","df"]}

    rss = float(np.sum(resid**2))
    tss = float(np.sum((y - float(np.mean(y)))**2))
    s2  = rss / df
    var_beta = s2 * XtX_inv
    se = np.sqrt(np.maximum(0.0, np.diag(var_beta)))

    from scipy.stats import t as _t
    tvals = beta / se
    pvals = 2.0 * _t.sf(np.abs(tvals), df=df)

    R2 = 1.0 - (rss / tss) if tss > 0 else np.nan

    return {
        "b0": float(beta[0]), "b1": float(beta[1]), "b2": float(beta[2]),
        "se0": float(se[0]),  "se1": float(se[1]),  "se2": float(se[2]),
        "t0": float(tvals[0]), "t1": float(tvals[1]), "t2": float(tvals[2]),
        "p0": float(pvals[0]), "p1": float(pvals[1]), "p2": float(pvals[2]),
        "R2": float(R2), "n": int(n), "df": int(df),
    }



def run_twovar_abs_pitch_yaw_lighting100(
    csv_path: Optional[str],
    outdir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    lighting_100 (static gaze): subject-level OLS per model using *video means*.
    We DO NOT rely on exp_dir. We join by (subject_dir, point).

        mean_error_video = a + b_pitch * |pitch_mean_video| + b_yaw * |yaw_mean_video|

    Pipeline
    --------
    1) From gaze_evaluation_results.csv (lighting_100):
         - normalize subject_dir
         - ensure point labels (p1..p9) are canonical via _standardize_point_label
         - filter to main-paper models
         - compute per-(subject_dir, point, gaze_model) mean error (frame-weighted if num_samples present)
    2) From pitch_yaw_stats DB (lighting_100):
         - normalize subject_dir column name
         - compute per-(subject_dir, point) video means of |gaze_pitch| and |gaze_yaw|
    3) Inner-merge (1) and (2) on (subject_dir, point)
    4) For each (subject_dir × gaze_model), regress across that subject's point videos
    5) Save:
         - outdir/point__twovar_abs_pitch_yaw__per_subject.csv
         - outdir/point__twovar_abs_pitch_yaw__agg_by_model.csv
    """
    os.makedirs(outdir, exist_ok=True)

    # ---------- Load CSV ----------
    if csv_path is None:
        base_dir = config.get_dataset_base_directory()
        csv_path = os.path.join(base_dir, "gaze_evaluation_results.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    _require_cols(df, ["exp_type", "gaze_model", "error_mean"])
    df = df[df["exp_type"] == "lighting_100"].copy()

    # subject_dir normalization
    if "subject_dir" not in df.columns:
        if "subject" in df.columns:
            df["subject_dir"] = df["subject"]
        else:
            raise ValueError("CSV must include 'subject_dir' or 'subject'.")

    # num_samples default (used for weighted averaging if duplicates exist)
    if "num_samples" not in df.columns:
        df["num_samples"] = 1

    # point label — ensure canonical 'p1'..'p9'
    if "point" not in df.columns:
        raise ValueError("CSV must include 'point' for lighting_100 (p1..p9).")
    df = df.copy()
    df["point"] = df["point"].astype(str).apply(_standardize_point_label)
    if (df["point"] == "").any():
        raise ValueError("Some rows lack a recognizable point label (expected p1..p9).")

    # filter to main models (by keys present in CSV)
    main_keys = set(config.get_main_models_included_in_the_paper().keys())
    df = df[df["gaze_model"].isin(main_keys)].copy()
    if df.empty:
        print("[lighting100/2var] No rows for main-paper models after filtering.")
        return pd.DataFrame(), pd.DataFrame()

    # frame-weighted mean error per (subject_dir, point, gaze_model)
    d_err = df.loc[:, ["subject_dir", "point", "gaze_model", "error_mean", "num_samples"]].copy()
    d_err["_wx"] = pd.to_numeric(d_err["error_mean"], errors="coerce") * pd.to_numeric(d_err["num_samples"], errors="coerce")
    d_err["_w"]  = pd.to_numeric(d_err["num_samples"], errors="coerce")
    d_err = (
        d_err.groupby(["subject_dir", "point", "gaze_model"], as_index=False)
             .agg(sum_wx=("_wx", "sum"), sum_w=("_w", "sum"))
    )
    d_err["error_mean"] = d_err["sum_wx"] / d_err["sum_w"]
    d_err = d_err.loc[:, ["subject_dir", "point", "gaze_model", "error_mean"]].copy()

    # ---------- Pitch/Yaw DB: per-(subject_dir, point) |pitch|/|yaw| video means ----------
    pdb = load_pitch_yaw_db()
    if pdb.empty:
        print("[lighting100/2var] pitch_yaw_stats_db empty.")
        return pd.DataFrame(), pd.DataFrame()
    # keep lighting_100 only
    if "experiment_type" not in pdb.columns:
        raise ValueError("pitch_yaw_stats DB must include 'experiment_type'.")
    pdb = pdb[pdb["experiment_type"] == "lighting_100"].copy()

    # subject_dir column normalization
    subj_col = "subject"
    if "subject" not in pdb.columns:
        # try common alternates used in your other scripts
        if "subject_dir_py" in pdb.columns:
            subj_col = "subject_dir_py"
        elif "subject_dir" in pdb.columns:
            subj_col = "subject_dir"
        else:
            raise ValueError("pitch_yaw_stats DB must include a subject column ('subject' / 'subject_dir' / 'subject_dir_py').")

    # point normalization to 'p1'..'p9'
    if "point" not in pdb.columns:
        raise ValueError("pitch_yaw_stats DB must include 'point' (p1..p9) for lighting_100.")
    pdb = pdb.copy()
    pdb["point"] = pdb["point"].astype(str).apply(_standardize_point_label)

    for c in ("gaze_pitch", "gaze_yaw"):
        if c not in pdb.columns:
            raise ValueError(f"pitch_yaw_stats DB missing '{c}'.")
        pdb[c] = pd.to_numeric(pdb[c], errors="coerce")

    # per-(subject_dir, point) means of absolute pitch/yaw
    d_py = (
        pdb.groupby([subj_col, "point"], as_index=False)
           .agg(pitch_mean=("gaze_pitch", lambda s: float(np.nanmean(np.abs(s)))),
                yaw_mean=("gaze_yaw",     lambda s: float(np.nanmean(np.abs(s)))))
           .rename(columns={subj_col: "subject_dir"})
    )

    # ---------- Join error means with pitch/yaw means on (subject_dir, point) ----------
    model_vid = d_err.merge(d_py, on=["subject_dir", "point"], how="inner", validate="m:1")
    if model_vid.empty:
        print("[lighting100/2var] No overlap between CSV and pitch/yaw DB on (subject_dir, point).")
        return pd.DataFrame(), pd.DataFrame()

    # display names
    model_vid["gaze_model_display"] = model_vid["gaze_model"].map(config.display_model_name)

    # ---------- Subject-level OLS per model ----------
    rows = []
    grp_cols = ["gaze_model", "gaze_model_display", "subject_dir"]
    for (gmk, gmd, subj), d in model_vid.groupby(grp_cols, sort=False):
        d = d.dropna(subset=["pitch_mean", "yaw_mean", "error_mean"]).copy()
        # match line_movement_analysis.py: require at least 5 observations (points)
        if d.shape[0] < 5:
            continue

        # Use the SAME predictor order as line_movement_analysis.py:
        #   x1 = |yaw|, x2 = |pitch|
        x_yaw   = d["yaw_mean"].to_numpy(float)
        x_pitch = d["pitch_mean"].to_numpy(float)
        y_err   = d["error_mean"].to_numpy(float)

        res = _ols_2pred(x1=x_yaw, x2=x_pitch, y=y_err)

        # Map back to named coefficients:
        #   res["b1"] corresponds to yaw  → b_yaw
        #   res["b2"] corresponds to pitch → b_pitch
        rows.append({
            "subject": str(subj),
            "gaze_model": gmk,
            "gaze_model_display": gmd,
            "n_videos": int(d.shape[0]),
            "b_pitch": res["b2"], "se_pitch": res["se2"], "t_pitch": res["t2"], "p_pitch": res["p2"],
            "b_yaw":   res["b1"], "se_yaw":  res["se1"], "t_yaw":  res["t1"], "p_yaw":  res["p1"],
            "intercept": res["b0"],
            "R2": res["R2"],
        })


    per_subject = pd.DataFrame(rows).sort_values(["gaze_model_display", "subject"]).reset_index(drop=True)

    out_rows = os.path.join(outdir, "point__twovar_abs_pitch_yaw__per_subject.csv")
    per_subject.to_csv(out_rows, index=False)
    print(f"[lighting100/2var] Wrote subject-level slopes: {out_rows}")

    if per_subject.empty:
        # still write an empty agg for consistency
        out_agg = os.path.join(outdir, "point__twovar_abs_pitch_yaw__agg_by_model.csv")
        pd.DataFrame().to_csv(out_agg, index=False)
        return per_subject, pd.DataFrame()

    # ---------- Aggregate across subjects ----------
    def _summ_model(g: pd.DataFrame) -> pd.Series:
        def _m(x):
            x = pd.to_numeric(x, errors="coerce")
            return float(np.nanmean(x)) if x.notna().any() else np.nan
        def _s(x):
            x = pd.to_numeric(x, errors="coerce").dropna()
            return float(np.std(x, ddof=1)) if len(x) > 1 else np.nan
        return pd.Series({
            "n_subjects": int(g["subject"].nunique()),
            "b_pitch_mean": _m(g["b_pitch"]),
            "b_pitch_std":  _s(g["b_pitch"]),
            "b_yaw_mean":   _m(g["b_yaw"]),
            "b_yaw_std":    _s(g["b_yaw"]),
            "R2_mean":      _m(g["R2"]),
            "R2_std":       _s(g["R2"]),
        })

    agg = (per_subject.groupby(["gaze_model", "gaze_model_display"], as_index=False)
                      .apply(_summ_model, include_groups=False)
                      .reset_index(drop=True)
                      .sort_values("gaze_model_display"))
    out_agg = os.path.join(outdir, "point__twovar_abs_pitch_yaw__agg_by_model.csv")
    agg.to_csv(out_agg, index=False)
    print(f"[lighting100/2var] Wrote aggregated slopes: {out_agg}")

    return per_subject, agg

def compute_aggregate_descriptives(
    df_subj: pd.DataFrame, 
    agg_col: str, 
    outdir: str, 
    exp_label: str
) -> pd.DataFrame:
    """Generic descriptive calculator for points, rows, or columns."""
    def _summ(g):
        vals = g["mean_error"].to_numpy(dtype=float)
        n = vals.size
        mean = float(np.mean(vals)) if n else np.nan
        sd = float(np.std(vals, ddof=1)) if n > 1 else np.nan
        se = float(sd / np.sqrt(n)) if (n > 1 and np.isfinite(sd)) else np.nan
        t_crit = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
        ci_lo = float(mean - t_crit * se) if np.isfinite(se) else np.nan
        ci_hi = float(mean + t_crit * se) if np.isfinite(se) else np.nan
        return pd.Series({
            "n_subjects": int(n),
            "mean_subject": mean,
            "sd_subject": sd,
            "se_subject": se,
            "ci95_low": ci_lo,
            "ci95_high": ci_hi,
        })

    desc = (
        df_subj.groupby(["gaze_model", "gaze_model_display", agg_col], as_index=False)
            .apply(_summ, include_groups=False)
            .reset_index(drop=True)
            .sort_values(["gaze_model_display", agg_col])
    )
    out_csv = os.path.join(outdir, f"{agg_col}_subject_level_descriptives__{exp_label}.csv")
    desc.to_csv(out_csv, index=False)
    print(f"Saved {agg_col} descriptives to {out_csv}")
    return desc

# --------------------------- CLI ---------------------------

def main():
    import config
    base = config.get_dataset_base_directory()
    csv_path = os.path.join(base, "gaze_evaluation_results.csv")
    outdir = os.path.join(base, "point_analysis")
    os.makedirs(outdir, exist_ok=True)

    experiments = ["lighting_100"]

    for exp in experiments:
        print(f"\n=== Running Comprehensive Spatial Analysis for: {exp} ===")
        df_raw = load_and_prepare(csv_path, experiment_type=exp)
        exp_label = df_raw["experiment_label"].iloc[0]

        # 1. POINT ANALYSIS
        print("Processing Points...")
        subj_points = subject_level_point_means(df_raw)
        compute_aggregate_descriptives(subj_points, "point", outdir, exp_label)
        run_pairwise_points_within_models(csv_path, outdir, exp, padjust="holm")

        # 2. ROW ANALYSIS (Vertical Bias)
        print("Processing Rows...")
        subj_rows = subject_level_row_means(df_raw)
        compute_aggregate_descriptives(subj_rows, "row", outdir, exp_label)
        run_pairwise_rows_within_models(csv_path, outdir, exp, padjust="holm")

        # 3. COLUMN ANALYSIS (Horizontal Bias)
        print("Processing Columns...")
        subj_cols = subject_level_col_means(df_raw)
        compute_aggregate_descriptives(subj_cols, "col", outdir, exp_label)
        run_pairwise_cols_within_models(csv_path, outdir, exp, padjust="holm")

        # 4. CROSS-MODEL RANKINGS
        print("Processing Cross-Model Comparisons...")
        run_pairwise_models_within_points(csv_path, outdir, exp, padjust="holm")
        summarize_pointwise_champions(csv_path, outdir, exp, alpha=0.05, padjust="holm")

        # 5. ECCENTRICITY REGRESSION
        if exp == "lighting_100":
            run_twovar_abs_pitch_yaw_lighting100(csv_path, outdir)

    print("\nAll subject-level spatial analysis results saved to:", outdir)

if __name__ == "__main__":
    main()
