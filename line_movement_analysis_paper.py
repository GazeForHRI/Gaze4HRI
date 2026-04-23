# line_movement_analysis_paper.py
"""
Target Motion (line_movement) analysis from raw gaze_evaluation_results.csv.

We treat "Target Motion" as 4 fixed combinations:
  - slow-horizontal, slow-vertical, fast-horizontal, fast-vertical

Pipeline (DESCRIPTIVES ONLY for now):
  raw rows
    -> per-video rows (already in CSV)
    -> per-subject_dir × (model, combo) frame-weighted mean
    -> per (model, combo) subject-level descriptives:
         n_subjects, mean_of_subject_means, sd_of_subject_means,
         se, 95% CI

Key requirements:
  - Use 'subject_dir' as the unique subject identifier (NOT 'subject').
  - Weight by num_samples when aggregating videos.
  - Filter to the main models included in the paper (via config.get_main_models_included_in_the_paper()).
  - Use display names in outputs (via config.display_model_name).

Outputs (under --outdir):
  - line_movement__subject_level_descriptives.csv

Usage:
  python line_movement_analysis.py \
    --csv /path/to/gaze_evaluation_results.csv \
    --outdir /path/to/outdir
"""

import os
import re
import argparse
from typing import Optional, Iterable, Dict, List, Tuple
import math
import numpy as np
import pandas as pd
import pingouin as pg  # not used yet (kept for future pairwise), safe to leave
from scipy import stats  # not used yet (kept for parity with lighting)

import config

# DBs for per-frame data
from frame_db import load_frame_dataset
from pitch_yaw_stats import load_pitch_yaw_db

# We will reuse your intersection finder from line_movement_analysis.py
from line_movement_analysis import extract_intersection_points
from data_loader import GazeDataLoader


# --------------------------- Patterns & Canonical Labels ---------------------------

# Expected exp_type examples:
#   line_movement_slow_horizontal
#   line_movement_slow_vertical
#   line_movement_fast_horizontal
#   line_movement_fast_vertical
_LINE_MOVEMENT_RE = re.compile(
    r"^line_movement_(slow|fast)_(horizontal|vertical)$"
)

_SPEED_FROM_EXPTYPE = {"line_movement_fast": "fast", "line_movement_slow": "slow"}

REQUIRED_TM_COMBOS = {"slow-horizontal", "slow-vertical", "fast-horizontal", "fast-vertical"}

def _filter_complete_tm_subjects(df: pd.DataFrame, combo_col: str = "tm_combo") -> pd.DataFrame:
    """
    Ensures a consistent N across all models and combinations by keeping only 
    subjects who have all 4 combinations for every model in the analysis family.
    """
    if df.empty:
        return df
    
    # 1. Count unique combinations per (subject, model)
    counts = df.groupby(["subject_dir", "gaze_model"])[combo_col].nunique()
    
    # 2. Create a completeness matrix (subjects as rows, models as columns)
    # unstack() will place NaNs where a subject is missing a model entirely
    completeness_matrix = (counts.unstack() == 4)
    
    # 3. A subject is 'globally complete' only if they have 4 combos for ALL models
    is_globally_complete = completeness_matrix.fillna(False).all(axis=1)
    
    complete_subjects = is_globally_complete[is_globally_complete].index
    return df[df["subject_dir"].isin(complete_subjects)].copy()

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
    # beta = (X'X)^-1 X'y
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

    # t-stats & p-values (two-sided, df=n-p)
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

def _standardize(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    m = np.nanmean(z)
    s = np.nanstd(z)
    if not np.isfinite(s) or s <= 0:
        return np.full_like(z, np.nan)
    return (z - m) / s

def _exp_speed_from_exptype(s: str) -> str:
    s = str(s).strip().lower()
    if s == "line_movement_fast": return "fast"
    if s == "line_movement_slow": return "slow"
    return ""

def _ols_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Unweighted OLS slope/intercept for y ~ a + b*x (NaN-safe)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = min(x.size, y.size)
    if m < 3:
        return np.nan, np.nan
    x = x[:m]; y = y[:m]
    msk = np.isfinite(x) & np.isfinite(y)
    x = x[msk]; y = y[msk]
    if x.size < 3:
        return np.nan, np.nan
    xbar = float(np.mean(x)); ybar = float(np.mean(y))
    sxx = float(np.sum((x - xbar)**2))
    if sxx <= 0:
        return np.nan, np.nan
    b = float(np.sum((x - xbar)*(y - ybar)) / sxx)
    a = ybar - b * xbar
    return b, a

def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from scipy.stats import pearsonr
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = min(x.size, y.size)
    if m < 3:
        return np.nan, np.nan
    x = x[:m]; y = y[:m]
    msk = np.isfinite(x) & np.isfinite(y)
    x = x[msk]; y = y[msk]
    if x.size < 3:
        return np.nan, np.nan
    try:
        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan

def run_abs_angle_regressions_from_dbs(outdir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    EXACT algorithm per your 7 steps:

    1) Join frame_db and pitch_yaw_stats_db on (exp_dir, frame_idx, exp_type, point)
       -> per-frame table with: error, is_valid, gaze_pitch, gaze_yaw, timestamp_ms, etc.
    2) For each (video), use extract_intersection_points(dataloader, speed, movement_type)
       to get intersection indices in the camera sequence; map those to timestamps via
       camera_poses[:,0] (we take timestamp at index vel_idx+1).
    3) Use these timestamps to partition frames (by their timestamp_ms) into 5 segments.
    4) Keep main segments: indices 0, 2, 4.
    5) Filter out invalid frames (is_valid == 1).
    6) Regress:
         - horizontal main segments:  error ~ |gaze_yaw|
         - vertical   main segments:  error ~ |gaze_pitch|
    7) Save per-row and aggregated CSVs.

    Returns: (rows_df, agg_df)
    """
    os.makedirs(outdir, exist_ok=True)

    # ---- 1) load & join DBs ----
    fdf = load_frame_dataset()
    if fdf.empty:
        print("[abs-angle/DB] frame_db is empty.")
        return pd.DataFrame(), pd.DataFrame()

    pdf = load_pitch_yaw_db()
    if pdf.empty:
        print("[abs-angle/DB] pitch_yaw_stats_db is empty.")
        return pd.DataFrame(), pd.DataFrame()

    # Restrict to line_movement + points we care about
    fdf = fdf[fdf["exp_type"].isin(["line_movement_fast","line_movement_slow"])].copy()
    fdf = fdf[fdf["point"].isin(["horizontal","vertical"])].copy()
    if fdf.empty:
        print("[abs-angle/DB] No line_movement rows in frame_db.")
        return pd.DataFrame(), pd.DataFrame()

    pdf = pdf[pdf["experiment_type"].isin(["line_movement_fast","line_movement_slow"])].copy()
    pdf = pdf[pdf["point"].isin(["horizontal","vertical"])].copy()
    if pdf.empty:
        print("[abs-angle/DB] No line_movement rows in pitch_yaw_stats_db.")
        return pd.DataFrame(), pd.DataFrame()

    # Harmonize column names for join
    pdf_ = pdf.rename(columns={
        "subject": "subject_dir_py",
        "experiment_type": "exp_type",
        "exp_dir_rel": "exp_dir",
    })

    # Keep only needed columns from pitch_yaw
    pdf_ = pdf_[["exp_dir","frame_idx","point","exp_type","gaze_pitch","gaze_yaw","timestamp_ms","subject_dir_py"]].copy()

    # Inner join per frame (video+frame_idx+type+point)
    merged = fdf.merge(
        pdf_,
        on=["exp_dir","frame_idx","point","exp_type"],
        how="inner",
        validate="m:m"
    )
    if merged.empty:
        print("[abs-angle/DB] Join produced no rows.")
        return pd.DataFrame(), pd.DataFrame()

    # derive speed & movement_type for convenience
    merged["speed"] = merged["exp_type"].map(_exp_speed_from_exptype)
    merged["movement_type"] = merged["point"]

    merged["_tmp_combo"] = merged["speed"] + "-" + merged["movement_type"]
    merged = _filter_complete_tm_subjects(merged, combo_col="_tmp_combo")
    merged.drop(columns=["_tmp_combo"], inplace=True)
    # subject_dir: use frame_db’s (YYYY-MM-DD/name)
    merged["subject_dir"] = merged["subject_dir"].astype(str)

    # Ensure numeric
    for c in ("error","is_valid","gaze_pitch","gaze_yaw","timestamp_ms"):
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    # Right after merged is built and before display names:
    main_model_keys = set(config.get_main_models_included_in_the_paper().keys())
    merged = merged[merged["gaze_model"].isin(main_model_keys)].copy()
    if merged.empty:
        print("[abs-angle/DB] No rows for main-paper models after filtering.")
        return pd.DataFrame(), pd.DataFrame()

    # Display names strictly from main-paper set
    disp_map_main = config.get_main_models_included_in_the_paper()
    merged["gaze_model_display"] = merged["gaze_model"].map(disp_map_main)


    # ---- 2) build segment boundaries per (video) using intersection timestamps ----
    # We’ll process per (exp_dir, speed, movement_type, gaze_model, subject_dir)
    rows = []

    base_dir = config.get_dataset_base_directory()

    groups = merged.groupby(["exp_dir","speed","movement_type","gaze_model","gaze_model_display","subject_dir"], sort=False)
    for (exp_dir_rel, speed, mv_type, gmk, gmd, subj), d in groups:
        try:
            # Create dataloader for THIS video (get_latest_subdirectory_by_name=False is safer because exp_dir is already the run folder)
            exp_dir_abs = os.path.join(base_dir, exp_dir_rel)
            try:
                if config.is_experiment_directory_excluded_from_eval(exp_dir_abs):
                    continue
            except Exception:
                # If checker throws for any odd path, just skip safely
                continue
            dl = GazeDataLoader(
                root_dir=exp_dir_abs,
                target_period=config.get_target_period(),
                camera_pose_period=config.get_camera_pose_period(),
                time_diff_max=config.get_time_diff_max(),
                get_latest_subdirectory_by_name=False
            )

            # 2) intersection indices (in velocity array index space)
            inter_idxs = extract_intersection_points(
                dataloader=dl,
                line_movement_speed=speed,
                line_movement_type=mv_type,
                visualize_for_debug=False
            )
            if len(inter_idxs) != 4:
                # If we didn't get 4, skip this video gracefully
                # (you can log/print if you want visibility)
                continue

            # Map those velocity indices to camera timestamp at position index (vel_idx+1)
            cam = dl.load_camera_poses()  # shape (N,17), col0 = timestamp_ms
            cam_ts = cam[:, 0].astype(float)
            # guard lengths
            ts_points = []
            for vidx in inter_idxs:
                idx = int(max(0, min(len(cam_ts)-1, vidx+1)))
                ts_points.append(float(cam_ts[idx]))
            ts_points = sorted(ts_points)

            # 3) partition this video's frames by timestamp into 5 segments
            # timestamps in joined df come from pitch_yaw_stats_db.timestamp_ms
            t_min = float(np.nanmin(d["timestamp_ms"].values))
            t_max = float(np.nanmax(d["timestamp_ms"].values))
            if not np.isfinite(t_min) or not np.isfinite(t_max):
                continue

            # segment windows: [t0, t1], (t1, t2], (t2, t3], (t3, t4], (t4, t5]
            bounds = [t_min] + ts_points + [t_max]
            # 5 segments: 0..4
            seg_masks = []
            for s in range(5):
                lo, hi = bounds[s], bounds[s+1]
                if s == 0:
                    mask = (d["timestamp_ms"] >= lo) & (d["timestamp_ms"] <= hi)
                else:
                    mask = (d["timestamp_ms"] > lo) & (d["timestamp_ms"] <= hi)
                seg_masks.append(mask)

            # 4) keep main segments (0,2,4); 5) then filter invalid frames
            main_segs = [0, 2, 4]
            for seg_id in main_segs:
                dd = d[seg_masks[seg_id]].copy()
                if dd.empty:
                    continue
                dd = dd[dd["is_valid"] == 1]   # valid frames only
                if dd.shape[0] < 3:
                    continue

                # 6) regress per main segment
                if mv_type == "horizontal":
                    # error ~ |gaze_yaw|
                    x = np.abs(dd["gaze_yaw"].to_numpy(dtype=float))
                else:
                    # vertical: error ~ |gaze_pitch|
                    x = np.abs(dd["gaze_pitch"].to_numpy(dtype=float))
                y = dd["error"].to_numpy(dtype=float)

                b, a = _ols_fit(x, y)
                r, p = _pearson(x, y)

                rows.append({
                    "subject_dir": subj,                       # YYYY-MM-DD/name
                    "speed": speed,
                    "movement_type": mv_type,
                    "segment_id": int(seg_id),                 # 0,2,4 only
                    "gaze_model": gmk,
                    "gaze_model_display": gmd,
                    "n_frames": int(dd.shape[0]),
                    "slope": float(b),
                    "intercept": float(a),
                    "pearson_r": float(r),
                    "pearson_p": float(p),
                    "exp_dir": exp_dir_rel,
                })

        except Exception as e:
            # Skip this video on any loader/segmentation problem (non-fatal)
            # print(f"[abs-angle/DB] Skip video {exp_dir_rel}: {e}")
            continue

    rows_df = pd.DataFrame(rows).sort_values(
        ["movement_type","speed","segment_id","gaze_model_display","subject_dir"]
    ).reset_index(drop=True)

    out_rows = os.path.join(outdir, "line_movement__abs_angle_regressions_rows_DB.csv")
    rows_df.to_csv(out_rows, index=False)
    print(f"[abs-angle/DB] Wrote per-row regressions: {out_rows}")

    if rows_df.empty:
        return rows_df, pd.DataFrame()

    # 7) aggregate across subjects per (movement_type, speed, segment_id) with Fisher z + Holm
    def _holm_stepdown(pvals: pd.Series) -> pd.Series:
        s = pd.Series(pvals, copy=True)
        order = s.dropna().sort_values().index.tolist()
        m = len(order)
        out = pd.Series(index=s.index, dtype=float)
        running = 0.0
        for rank, idx in enumerate(order, start=1):
            factor = m - rank + 1
            padj = min(1.0, float(s.loc[idx]) * factor)
            running = max(running, padj)
            out.loc[idx] = running
        return out

    def _agg_one_family(df_family: pd.DataFrame) -> pd.DataFrame:
        """
        df_family contains rows for ONE (movement_type, speed, segment_id) but many models.
        For each model:
        - keep rows with n_frames >= 4 (so Fisher variance 1/(n-3) is defined)
        - compute inverse-variance weighted mean z̄ and se(z̄)
        - z-test for H0: z̄ = 0  (two-sided)
        - back-transform r̄ = tanh(z̄)
        - also report simple (unweighted) mean ± sd of slope and r for reference
        Then apply Holm over the per-model p-values within this family.
        """
        from math import atanh, tanh, sqrt
        import numpy as _np
        rows_out = []
        for model, dm in df_family.groupby("gaze_model_display", sort=False):
            # Fisher z inputs
            dm = dm.copy()
            dm = dm[_np.isfinite(dm["pearson_r"]) & _np.isfinite(dm["n_frames"])]
            dm = dm[dm["n_frames"] >= 4]  # ensure (n-3) > 0
            if dm.empty:
                rows_out.append({
                    "gaze_model_display": model,
                    "subjects": int(df_family[df_family["gaze_model_display"]==model]["subject_dir"].nunique()),
                    "videos":   int(df_family[df_family["gaze_model_display"]==model]["exp_dir"].nunique()),
                    "rows":     0,
                    "fisher_z_mean": _np.nan,
                    "fisher_z_se": _np.nan,
                    "fisher_z_p_two_sided": _np.nan,
                    "fisher_r_mean_bktr": _np.nan,  # tanh(z̄)
                    "slope_mean": _np.nan,
                    "slope_std": _np.nan,
                    "r_mean_unweighted": _np.nan,
                    "r_std_unweighted": _np.nan,
                })
                continue

            # Inverse-variance weights wi = (n_i - 3)
            ri = _np.clip(dm["pearson_r"].to_numpy(float), -0.999999, 0.999999)
            ni = dm["n_frames"].to_numpy(int)
            zi = _np.arctanh(ri)
            wi = (ni - 3).astype(float)
            wi[wi < 1.0] = _np.nan  # guard, though we filtered n>=4

            msk = _np.isfinite(zi) & _np.isfinite(wi)
            zi = zi[msk]; wi = wi[msk]
            if zi.size == 0:
                zbar = _np.nan
                se_zbar = _np.nan
                p_two = _np.nan
            else:
                W = _np.nansum(wi)
                zbar = float(_np.nansum(wi * zi) / W)
                se_zbar = float(1.0 / _np.sqrt(W))  # se of weighted average under fixed-effect
                # z-test vs 0 (H0: zbar=0)
                if _np.isfinite(zbar) and _np.isfinite(se_zbar) and se_zbar > 0:
                    zstat = zbar / se_zbar
                    # two-sided p from standard normal
                    from scipy.stats import norm
                    p_two = float(2.0 * norm.sf(abs(zstat)))
                else:
                    p_two = _np.nan

            # reference (unweighted) stats
            slope_vals = df_family.loc[df_family["gaze_model_display"]==model,"slope"].to_numpy(float)
            r_vals     = df_family.loc[df_family["gaze_model_display"]==model,"pearson_r"].to_numpy(float)

            rows_out.append({
                "gaze_model_display": model,
                "subjects": int(df_family[df_family["gaze_model_display"]==model]["subject_dir"].nunique()),
                "videos":   int(df_family[df_family["gaze_model_display"]==model]["exp_dir"].nunique()),
                "rows":     int(df_family[df_family["gaze_model_display"]==model].shape[0]),
                "fisher_z_mean": zbar,
                "fisher_z_se": se_zbar,
                "fisher_z_p_two_sided": p_two,
                "fisher_r_mean_bktr": (float(_np.tanh(zbar)) if _np.isfinite(zbar) else _np.nan),
                "slope_mean": float(_np.nanmean(slope_vals)) if slope_vals.size else _np.nan,
                "slope_std":  float(_np.nanstd(slope_vals, ddof=1)) if slope_vals.size > 1 else _np.nan,
                "r_mean_unweighted": float(_np.nanmean(r_vals)) if r_vals.size else _np.nan,
                "r_std_unweighted":  float(_np.nanstd(r_vals, ddof=1)) if r_vals.size > 1 else _np.nan,
            })

        fam = pd.DataFrame(rows_out)

        # Holm across models in THIS family, based on Fisher z p-values
        if not fam.empty and "fisher_z_p_two_sided" in fam.columns:
            fam["p_holm"] = _holm_stepdown(fam["fisher_z_p_two_sided"])
        else:
            fam["p_holm"] = np.nan
        return fam

    agg_blocks = []
    for (mv, spd, seg), df_family in rows_df.groupby(["movement_type","speed","segment_id"], sort=False):
        fam = _agg_one_family(df_family)
        fam.insert(0, "movement_type", mv)
        fam.insert(1, "speed", spd)
        fam.insert(2, "segment_id", seg)
        agg_blocks.append(fam)

    agg_df = pd.concat(agg_blocks, ignore_index=True) if agg_blocks else pd.DataFrame()
    agg_df = agg_df.sort_values(["movement_type","speed","segment_id","gaze_model_display"]).reset_index(drop=True)

    out_agg = os.path.join(outdir, "line_movement__abs_angle_regressions_agg_DB.csv")
    agg_df.to_csv(out_agg, index=False)
    print(f"[abs-angle/DB] Wrote aggregated regressions (Fisher z + Holm): {out_agg}")

    return rows_df, agg_df


def _parse_speed_from_exp_type(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    return _SPEED_FROM_EXPTYPE.get(s.strip(), None)


def _parse_line_combo(exp_type: str) -> Optional[str]:
    """
    Parse exp_type -> canonical combo label:
      slow-horizontal, slow-vertical, fast-horizontal, fast-vertical
    Return None if not line_movement.
    """
    if not isinstance(exp_type, str):
        return None
    m = _LINE_MOVEMENT_RE.match(exp_type.strip())
    if not m:
        return None
    speed, axis = m.group(1), m.group(2)
    return f"{speed}-{axis}"


# --------------------------- Utilities ---------------------------

def _require_cols(df: pd.DataFrame, cols: Iterable[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_and_prepare(csv_path: Optional[str]) -> pd.DataFrame:
    """
    Load CSV and keep only line_movement rows. Ensure columns:
      ['gaze_model','subject_dir','experiment_type','tm_combo',
       'angular_error','num_samples','gaze_model_display']

    - angular_error comes from 'error_mean'
    - experiment_type comes from 'exp_type'
    - tm_combo is one of {'slow-horizontal','slow-vertical','fast-horizontal','fast-vertical'}
    - filter models to main models (keys) per config.get_main_models_included_in_the_paper()
    """
    if csv_path is None:
        base_dir = config.get_dataset_base_directory()
        csv_path = os.path.join(base_dir, "gaze_evaluation_results.csv")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Basic schema checks / mapping
    if "error_mean" not in df.columns:
        raise ValueError("Expected column 'error_mean' was not found.")
    df["angular_error"] = df["error_mean"]

    if "exp_type" not in df.columns:
        raise ValueError("Expected column 'exp_type' was not found.")
    df["experiment_type"] = df["exp_type"]

    # subject_dir is REQUIRED (unique key); if missing, try to derive
    if "subject_dir" not in df.columns:
        # Last resort: fall back to 'subject' but this is NOT preferred (can duplicate)
        if "subject" in df.columns:
            df["subject_dir"] = df["subject"]
        else:
            raise ValueError("CSV must include 'subject_dir' (unique subject key).")

    # Required columns
    req = ["gaze_model", "subject_dir", "experiment_type", "angular_error"]
    _require_cols(df, req)

    # Filter to main models (by keys used in CSV)
    main_keys = set(config.get_main_models_included_in_the_paper().keys())
    df = df[df["gaze_model"].isin(main_keys)].copy()

    # num_samples default
    if "num_samples" not in df.columns:
        df["num_samples"] = 1

    df["tm_speed"] = df["experiment_type"].apply(_parse_speed_from_exp_type)
    df = df[~df["tm_speed"].isna()].copy()   # only Target Motion rows

    if "point" not in df.columns:
        raise ValueError("CSV must include the 'point' column with values 'horizontal'/'vertical' for line_movement.")
    df["tm_axis"] = df["point"].astype(str).str.strip().str.lower()
    df = df[df["tm_axis"].isin(["horizontal", "vertical"])].copy()

    df["tm_combo"] = (df["tm_speed"] + "-" + df["tm_axis"]).astype(str)

    df = _filter_complete_tm_subjects(df)

    # Attach display names for paper
    df["gaze_model_display"] = df["gaze_model"].map(config.display_model_name)

    # Keep the minimal set we need
    cols = ["gaze_model", "gaze_model_display", "subject_dir",
            "experiment_type", "tm_speed", "tm_axis", "tm_combo",
            "angular_error", "num_samples"]
    if "error_std" in df.columns:
        cols.append("error_std")  # not used here but handy to keep
    df = df.loc[:, cols].copy()

    return df


def _subject_level_video_means(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to subject-level (per subject_dir) weighted means for each (model, tm_combo).
    Weights = num_samples (frames).

    Returns columns:
      ['gaze_model','gaze_model_display','subject_dir','tm_combo','mean_error']
    """
    need = {"gaze_model","gaze_model_display","subject_dir","tm_combo","angular_error","num_samples"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing required cols: {missing}")

    d = df.loc[:, ["gaze_model","gaze_model_display","subject_dir","tm_combo","angular_error","num_samples"]].copy()
    d["_wx"] = d["angular_error"] * d["num_samples"]

    agg = (
        d.groupby(["gaze_model","gaze_model_display","subject_dir","tm_combo"], as_index=False)
         .agg(total_w=("num_samples","sum"), sum_wx=("_wx","sum"))
    )
    agg["mean_error"] = agg["sum_wx"] / agg["total_w"]
    return agg.loc[:, ["gaze_model","gaze_model_display","subject_dir","tm_combo","mean_error"]]


def compute_subject_level_descriptives(
    csv_path: Optional[str],
    outdir: str
) -> pd.DataFrame:
    """
    Subject-level descriptives per (model × tm_combo), using subject_dir as the key.
    Steps:
      - frame-weighted video means per (gaze_model, tm_combo, subject_dir)
      - aggregate across subjects: n, mean, sd, se, 95% CI
    """
    os.makedirs(outdir, exist_ok=True)

    df = load_and_prepare(csv_path)
    subj_df = _subject_level_video_means(df)  # ['gaze_model','gaze_model_display','subject_dir','tm_combo','mean_error']
    df_all = df.copy()
    df_all["tm_combo"] = "all"
    subj_df_all = _subject_level_video_means(df_all)
    subj_df = pd.concat([subj_df, subj_df_all], ignore_index=True)
    # Ensure display names exist
    if "gaze_model_display" not in subj_df.columns or subj_df["gaze_model_display"].isna().any():
        subj_df = subj_df.copy()
        subj_df["gaze_model_display"] = subj_df["gaze_model"].map(config.display_model_name)

    def _summ(g: pd.DataFrame) -> pd.Series:
        vals = g["mean_error"].to_numpy(dtype=float)
        n = vals.size
        mean = float(np.mean(vals)) if n > 0 else np.nan
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

    gb_cols = ["gaze_model", "gaze_model_display", "tm_combo"]
    gb = subj_df.groupby(gb_cols, as_index=False)

    # Newer pandas: keep group keys out of the apply-ed frame (no FutureWarning)
    try:
        desc = gb.apply(_summ, include_groups=False)
    except TypeError:
        # Fallback for older pandas without include_groups:
        # apply returns an index with group keys -> bring them back to columns.
        tmp = gb.apply(_summ)
        desc = tmp.reset_index()  # moves group keys from index to columns

    # Now desc already has the group columns present as regular columns.
    desc = (
        desc.sort_values(["gaze_model_display", "tm_combo"], kind="stable")
            .reset_index(drop=True)
    )

    out_csv = os.path.join(outdir, "line_movement__subject_level_descriptives.csv")
    desc.to_csv(out_csv, index=False)
    print(f"Saved subject-level descriptives to {out_csv}")
    return desc

def _subject_level_video_means_with_speed_axis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Frame-weighted collapse per (gaze_model, subject_dir, tm_combo, tm_speed, tm_axis).
    Returns one row per subject×model×combo with columns:
      ['gaze_model','gaze_model_display','subject_dir','tm_combo','tm_speed','tm_axis',
       'mean_error','total_w']
    where 'total_w' is the total frame count across videos for that (subject_dir, model, combo).
    """
    need = {"gaze_model","gaze_model_display","subject_dir","tm_combo","tm_speed","tm_axis","angular_error","num_samples"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing required cols: {missing}")

    d = df.loc[:, ["gaze_model","gaze_model_display","subject_dir","tm_combo","tm_speed","tm_axis","angular_error","num_samples"]].copy()
    d["_wx"] = d["angular_error"] * d["num_samples"]
    agg = (
        d.groupby(["gaze_model","gaze_model_display","subject_dir","tm_combo","tm_speed","tm_axis"], as_index=False)
         .agg(total_w=("num_samples","sum"), sum_wx=("_wx","sum"))
    )
    agg["mean_error"] = agg["sum_wx"] / agg["total_w"]
    return agg.loc[:, ["gaze_model","gaze_model_display","subject_dir","tm_combo","tm_speed","tm_axis","mean_error","total_w"]]



def _make_model_ranking(res_df: pd.DataFrame,
                        d_subset: pd.DataFrame,
                        alpha: float = 0.05) -> str:
    """
    Produce a tiered ranking using SUBJECT-LEVEL means + Holm p-values:
      Mi ≺ Mj  <=>  mean_i > mean_j  AND  p_holm(Mi,Mj) < alpha
    Returns a ready-to-write multiline string.
    """
    if res_df is None or res_df.empty:
        return "No pairwise results (not enough models/subjects)."

    # subject-level descriptives from the exact subset used for testing
    subj_stats = _subjectlevel_mean_std_by_model_subset(d_subset)

    # tiers
    tiers = _induce_preference_tiers(res_df, subj_stats, alpha=alpha)

    lines = []
    lines.append("Tiers (best to worse; ties share a tier):")
    for k, tier in enumerate(tiers, 1):
        lines.append(f"Tier {k}: " + " ~ ".join(tier))
    lines.append("")
    lines.append("Subject-level means (deg) used to order within/among tiers:")
    for _, r in subj_stats.iterrows():
        lines.append(f"  - {r['gaze_model_display']}: {r['mean']:.4f} ± {r['std'] if np.isfinite(r['std']) else float('nan'):.4f}  (subjects={int(r['count_subjects'])})")
    lines.append(f"\nalpha = {alpha} (Holm-adjusted)")
    return "\n".join(lines)


def _subjectlevel_mean_std_by_model_subset(d_subj: pd.DataFrame) -> pd.DataFrame:
    """
    Subject-level descriptives for a *subset* dataframe that already has
    one row per (subject_dir × model) with column 'mean_error'.
    Returns:
      ['gaze_model','gaze_model_display','mean','std','count_subjects']
    """
    need = {"gaze_model","gaze_model_display","subject_dir","mean_error"}
    missing = need - set(d_subj.columns)
    if missing:
        raise ValueError(f"Missing cols for subjectlevel stats: {missing}")

    g = (
        d_subj.groupby(["gaze_model","gaze_model_display"], as_index=False)
              .agg(mean=("mean_error", "mean"),
                   std=("mean_error", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else np.nan),
                   count_subjects=("subject_dir","nunique"))
    )
    g.sort_values("mean", inplace=True)
    return g


def _induce_preference_tiers(pairwise_df: pd.DataFrame,
                             subj_stats_df: pd.DataFrame,
                             alpha: float = 0.05) -> List[List[str]]:
    """
    Build tiers using SUBJECT-LEVEL means (lower better) + Holm p-values:
      Mi ≺ Mj  iff mean_i > mean_j AND Holm(Mi,Mj) < alpha
    Models are walked in ascending subject-level mean; neighbors join the same
    tier if neither strictly dominates the other by the rule above.
    """
    # display -> subject-level mean
    mean_map = dict(zip(subj_stats_df["gaze_model_display"], subj_stats_df["mean"]))
    models_sorted = list(subj_stats_df.sort_values("mean")["gaze_model_display"])

    # find adjusted p column from pingouin
    pcol = None
    for cand in ("p-corr", "p_corr", "p-adjust", "p_adjust"):
        if cand in pairwise_df.columns:
            pcol = cand
            break
    if pcol is None:
        # fall back to uncorrected if needed (shouldn't happen with padjust="holm")
        pcol = "p-unc" if "p-unc" in pairwise_df.columns else None
    if pcol is None:
        raise ValueError("Adjusted (or unadjusted) p-value column not found in pairwise results.")

    # lookup Holm p-values, order-agnostic on (A,B)
    p_holm = {}
    for _, r in pairwise_df.iterrows():
        a, b = str(r["A"]), str(r["B"])
        p_holm[(a, b)] = float(r[pcol])

    def holm_p(a: str, b: str):
        if (a, b) in p_holm: return p_holm[(a, b)]
        if (b, a) in p_holm: return p_holm[(b, a)]
        return np.nan

    def prefers(i: str, j: str) -> bool:
        # "j preferred to i" means mean_j < mean_i AND significant
        mi, mj = mean_map.get(i, np.nan), mean_map.get(j, np.nan)
        p = holm_p(i, j)
        if not (np.isfinite(mi) and np.isfinite(mj) and np.isfinite(p)):
            return False
        return (mi > mj) and (p < alpha)

    tiers: List[List[str]] = []
    current: List[str] = []
    for m in models_sorted:
        if not current:
            current = [m]
            continue
        prev = current[-1]
        if not prefers(prev, m) and not prefers(m, prev):
            current.append(m)
        else:
            tiers.append(current)
            current = [m]
    if current:
        tiers.append(current)
    return tiers


def run_pairwise_combos_within_models(csv_path: Optional[str], outdir: str, padjust: str = "holm") -> pd.DataFrame:
    """
    Within-subject pairwise tests comparing the 4 motion combos (Slow-H, Slow-V, 
    Fast-H, Fast-V) for each model.
    """
    os.makedirs(outdir, exist_ok=True)
    df = load_and_prepare(csv_path)
    # Get subject-level, frame-weighted means for the 4 distinct combos
    subj_df = _subject_level_video_means(df)

    combined_list = []
    for m, dmodel in subj_df.groupby("gaze_model"):
        display = dmodel["gaze_model_display"].iloc[0]
        
        # Paired t-tests across the 4 combos for this specific model
        res = pg.pairwise_tests(
            data=dmodel,
            dv="mean_error",
            within="tm_combo",
            subject="subject_dir",
            parametric=True,
            padjust=padjust,
            effsize="cohen",
            return_desc=True
        )
        res.insert(0, "gaze_model", display)
        combined_list.append(res)
        
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(display))
        res.to_csv(os.path.join(outdir, f"pairwise_combos_within__{safe}.csv"), index=False)

    combined = pd.concat(combined_list, axis=0, ignore_index=True)
    combined.to_csv(os.path.join(outdir, "pairwise_combos_within__all_models.csv"), index=False)
    print(f"[tm-pairwise] Saved within-model combo comparisons to {outdir}")
    return combined

def run_pairwise_models_for_line_movement(
    csv_path: Optional[str],
    outdir: str,
    padjust: str = "holm",
    alpha: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    """
    Do subject-fixed pairwise model tests (Holm) for:
      1) slow_horizontal   (tm_combo = slow-horizontal)
      2) slow_vertical     (tm_combo = slow-vertical)
      3) fast_horizontal   (tm_combo = fast-horizontal)
      4) fast_vertical     (tm_combo = fast-vertical)
      5) slow              (tm_speed = slow)
      6) fast              (tm_speed = fast)
      7) all               (both speeds)
    Writes:
      - pairwise_models__<label>.csv
      - model_ranking__<label>.txt
    Returns dict: {label: DataFrame}
    """
    os.makedirs(outdir, exist_ok=True)
    df = load_and_prepare(csv_path)
    subj_df = _subject_level_video_means_with_speed_axis(df)

    # Ensure clean dtypes for Pingouin
    subj_df["gaze_model_display"] = subj_df["gaze_model_display"].astype(str)
    subj_df["subject_dir"] = subj_df["subject_dir"].astype(str)
    subj_df["tm_combo"] = subj_df["tm_combo"].astype(str)
    subj_df["tm_speed"] = subj_df["tm_speed"].astype(str)

    aggs = {
        "slow_horizontal":  (subj_df["tm_combo"] == "slow-horizontal"),
        "slow_vertical":    (subj_df["tm_combo"] == "slow-vertical"),
        "fast_horizontal":  (subj_df["tm_combo"] == "fast-horizontal"),
        "fast_vertical":    (subj_df["tm_combo"] == "fast-vertical"),
        "slow":             (subj_df["tm_speed"] == "slow"),
        "fast":             (subj_df["tm_speed"] == "fast"),
        "all":              (subj_df["tm_speed"].isin(["slow","fast"])),
    }

    results: Dict[str, pd.DataFrame] = {}

    for label, mask in aggs.items():
        d0 = subj_df[mask].copy()
        out_csv = os.path.join(outdir, f"pairwise_models__{label}.csv")
        out_txt = os.path.join(outdir, f"model_ranking__{label}.txt")

        if d0.empty:
            pd.DataFrame().to_csv(out_csv, index=False)
            with open(out_txt, "w") as f:
                f.write("Not enough data for this aggregation.\n")
            results[label] = pd.DataFrame()
            continue

        # For aggregations that pool multiple combos (slow/fast/all), we need exactly
        # one row per (subject_dir × gaze_model_display). Collapse with frame weights.
        if label in ("slow", "fast", "all"):
            d0 = d0.copy()
            d0["_wx2"] = d0["mean_error"] * d0["total_w"]
            d = (
                d0.groupby(["gaze_model","gaze_model_display","subject_dir"], as_index=False)
                  .agg(total_w=("total_w","sum"), sum_wx=("_wx2","sum"))
            )
            d["mean_error"] = d["sum_wx"] / d["total_w"]
            d = d.loc[:, ["gaze_model","gaze_model_display","subject_dir","mean_error"]]
        else:
            # single combo → already one row per subject×model
            d = d0.loc[:, ["gaze_model","gaze_model_display","subject_dir","mean_error"]].copy()

        # Need at least two models
        if d["gaze_model_display"].nunique() < 2:
            pd.DataFrame().to_csv(out_csv, index=False)
            with open(out_txt, "w") as f:
                f.write("Not enough models to compare.\n")
            results[label] = pd.DataFrame()
            continue

        # Run Pingouin pairwise tests (within-subjects: model; subject: subject_dir)
        res = pg.pairwise_tests(
            data=d,
            dv="mean_error",
            within="gaze_model_display",
            subject="subject_dir",
            parametric=True,
            padjust=padjust,
            effsize="cohen",
            return_desc=True,
        )

        # Tidy & save
        if {"A","B"}.issubset(res.columns):
            res = res.sort_values(["A","B"], kind="stable").reset_index(drop=True)

        res.to_csv(out_csv, index=False)

        # Ranking (use the exact d we tested on)
        ranking_text = _make_model_ranking(res_df=res, d_subset=d, alpha=alpha)
        with open(out_txt, "w") as f:
            f.write(ranking_text + "\n")

        results[label] = res

    return results


def run_twovar_abs_pitch_yaw_regressions_from_dbs(outdir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Two-predictor OLS per-video (no segmentation):
      y = error,  x1 = |gaze_yaw|,  x2 = |gaze_pitch|
    Valid frames only. Main-paper models only. Skip excluded exp_dirs.
    Aggregation: fixed-effect inverse-variance meta-analysis of b_yaw and b_pitch
    across videos per (movement_type, speed, model). Holm across models per family.

    Returns (per_video_rows_df, aggregated_df)
    """
    os.makedirs(outdir, exist_ok=True)

    # ---- Load DBs & join (identical schema handling as your other runner) ----
    fdf = load_frame_dataset()
    if fdf.empty:
        print("[2var/DB] frame_db is empty.")
        return pd.DataFrame(), pd.DataFrame()
    pdf = load_pitch_yaw_db()
    if pdf.empty:
        print("[2var/DB] pitch_yaw_stats_db is empty.")
        return pd.DataFrame(), pd.DataFrame()

    # Keep only line_movement + horizontal/vertical
    fdf = fdf[fdf["exp_type"].isin(["line_movement_fast","line_movement_slow"])].copy()
    fdf = fdf[fdf["point"].isin(["horizontal","vertical"])].copy()
    pdf = pdf[pdf["experiment_type"].isin(["line_movement_fast","line_movement_slow"])].copy()
    pdf = pdf[pdf["point"].isin(["horizontal","vertical"])].copy()

    if fdf.empty or pdf.empty:
        print("[2var/DB] No line_movement rows after initial filtering.")
        return pd.DataFrame(), pd.DataFrame__()

    pdf_ = pdf.rename(columns={
        "subject": "subject_dir_py",
        "experiment_type": "exp_type",
        "exp_dir_rel": "exp_dir",
    })[["exp_dir","frame_idx","point","exp_type","gaze_pitch","gaze_yaw","timestamp_ms","subject_dir_py"]]

    merged = fdf.merge(
        pdf_, on=["exp_dir","frame_idx","point","exp_type"], how="inner", validate="m:m"
    )
    if merged.empty:
        print("[2var/DB] Join produced no rows.")
        return pd.DataFrame(), pd.DataFrame()

    # speed & axis
    merged["speed"] = merged["exp_type"].map(_exp_speed_from_exptype)
    merged["movement_type"] = merged["point"]

    merged["_tmp_combo"] = merged["speed"] + "-" + merged["movement_type"]
    merged = _filter_complete_tm_subjects(merged, combo_col="_tmp_combo")
    merged.drop(columns=["_tmp_combo"], inplace=True)
    # numeric and validity
    for c in ("error","is_valid","gaze_pitch","gaze_yaw","timestamp_ms"):
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    # Keep main models only
    main_keys = set(config.get_main_models_included_in_the_paper().keys())
    merged = merged[merged["gaze_model"].isin(main_keys)].copy()
    if merged.empty:
        print("[2var/DB] No rows for main-paper models after filtering.")
        return pd.DataFrame(), pd.DataFrame()

    disp_map_main = config.get_main_models_included_in_the_paper()
    merged["gaze_model_display"] = merged["gaze_model"].map(disp_map_main)

    # ---- Per-video OLS (valid frames only), absolute yaw/pitch ----
    base_dir = config.get_dataset_base_directory()
    rows = []

    grp_cols = ["exp_dir","speed","movement_type","gaze_model","gaze_model_display","subject_dir"]
    for (exp_dir_rel, speed, mv_type, gmk, gmd, subj), d in merged.groupby(grp_cols, sort=False):
        exp_dir_abs = os.path.join(base_dir, exp_dir_rel)
        # Skip excluded directories
        try:
            if config.is_experiment_directory_excluded_from_eval(exp_dir_abs):
                continue
        except Exception:
            continue

        dd = d[d["is_valid"] == 1].copy()
        if dd.shape[0] < 8:
            continue

        x_yaw   = np.abs(dd["gaze_yaw"].to_numpy(float))
        x_pitch = np.abs(dd["gaze_pitch"].to_numpy(float))
        y_err   = dd["error"].to_numpy(float)

        res = _ols_2pred(x1=x_yaw, x2=x_pitch, y=y_err)

        # standardized (beta) version (z-scored per video)
        zy = _standardize(y_err)
        z1 = _standardize(x_yaw)
        z2 = _standardize(x_pitch)
        res_std = _ols_2pred(z1, z2, zy)

        rows.append({
            "subject_dir": subj,
            "speed": speed,
            "movement_type": mv_type,
            "gaze_model": gmk,
            "gaze_model_display": gmd,
            "exp_dir": exp_dir_rel,
            "n_frames": int(res["n"]) if math.isfinite(res["n"]) else int(dd.shape[0]),
            # unstandardized coefficients (deg per deg)
            "b_yaw":   res["b1"], "se_yaw": res["se1"], "p_yaw": res["p1"],
            "b_pitch": res["b2"], "se_pitch": res["se2"], "p_pitch": res["p2"],
            "intercept": res["b0"], "R2": res["R2"],
            # standardized beta coefficients
            "beta_yaw":   res_std["b1"], "se_beta_yaw": res_std["se1"], "p_beta_yaw": res_std["p1"],
            "beta_pitch": res_std["b2"], "se_beta_pitch": res_std["se2"], "p_beta_pitch": res_std["p2"],
            "R2_std": res_std["R2"],
        })

    per_video = pd.DataFrame(rows).sort_values(
        ["movement_type","speed","gaze_model_display","subject_dir"]
    ).reset_index(drop=True)

    out_rows = os.path.join(outdir, "line_movement__twovar_abs_pitch_yaw__per_video.csv")
    per_video.to_csv(out_rows, index=False)
    print(f"[2var/DB] Wrote per-video two-variable regressions: {out_rows}")

    if per_video.empty:
        return per_video, pd.DataFrame()

    # ---- Aggregate across videos (fixed-effect inverse-variance meta-analysis) ----
    def _holm_stepdown(pvals: pd.Series) -> pd.Series:
        s = pd.Series(pvals, copy=True)
        order = s.dropna().sort_values().index.tolist()
        m = len(order)
        out = pd.Series(index=s.index, dtype=float)
        running = 0.0
        for rank, idx in enumerate(order, start=1):
            factor = m - rank + 1
            padj = min(1.0, float(s.loc[idx]) * factor)
            running = max(running, padj)
            out.loc[idx] = running
        return out

    def _meta_one_coef(df_family: pd.DataFrame, coef_col: str, se_col: str) -> pd.Series:
        """
        Fixed-effect meta for one coefficient across videos:
          theta_hat = sum(w*theta)/sum(w),  SE = 1/sqrt(sum(w)),
          z = theta_hat/SE, p(two-sided), proportion positive, simple mean/std too.
        """
        d = df_family.copy()
        theta = pd.to_numeric(d[coef_col], errors="coerce").to_numpy(float)
        se    = pd.to_numeric(d[se_col],   errors="coerce").to_numpy(float)
        n     = pd.to_numeric(d["n_frames"], errors="coerce").to_numpy(float)

        # valid rows: finite theta and se>0
        msk = np.isfinite(theta) & np.isfinite(se) & (se > 0)
        theta = theta[msk]; se = se[msk]; n = n[msk]
        if theta.size == 0:
            return pd.Series({
                f"{coef_col}_meta": np.nan, f"{coef_col}_se_meta": np.nan,
                f"{coef_col}_z": np.nan, f"{coef_col}_p_two": np.nan,
                f"{coef_col}_mean": np.nan, f"{coef_col}_std": np.nan,
                f"{coef_col}_prop_pos": np.nan, f"{coef_col}_videos": 0,
                f"{coef_col}_sum_n": 0,
            })

        w = 1.0 / (se ** 2)
        W = float(np.nansum(w))
        theta_hat = float(np.nansum(w * theta) / W)
        se_hat = float(1.0 / np.sqrt(W))
        from scipy.stats import norm
        z = theta_hat / se_hat if se_hat > 0 else np.nan
        p_two = float(2.0 * norm.sf(abs(z))) if np.isfinite(z) else np.nan

        return pd.Series({
            f"{coef_col}_meta": theta_hat, f"{coef_col}_se_meta": se_hat,
            f"{coef_col}_z": z, f"{coef_col}_p_two": p_two,
            f"{coef_col}_mean": float(np.nanmean(theta)),
            f"{coef_col}_std":  float(np.nanstd(theta, ddof=1)) if theta.size > 1 else np.nan,
            f"{coef_col}_prop_pos": float(np.mean(theta > 0.0)),
            f"{coef_col}_videos": int(theta.size),
            f"{coef_col}_sum_n": int(np.nansum(n)),
        })

    blocks = []
    for (mv, spd, model_disp), dfm in per_video.groupby(["movement_type","speed","gaze_model_display"], sort=False):
        # unstandardized
        meta_yaw   = _meta_one_coef(dfm, "b_yaw",   "se_yaw")
        meta_pitch = _meta_one_coef(dfm, "b_pitch", "se_pitch")
        # standardized betas
        meta_beta_yaw   = _meta_one_coef(dfm, "beta_yaw",   "se_beta_yaw")
        meta_beta_pitch = _meta_one_coef(dfm, "beta_pitch", "se_beta_pitch")

        row = pd.Series({"movement_type": mv, "speed": spd, "gaze_model_display": model_disp})
        blocks.append(pd.concat([row, meta_yaw, meta_pitch, meta_beta_yaw, meta_beta_pitch]))

    agg = pd.DataFrame(blocks)

    # Holm within each (movement_type, speed) family, separately for yaw and pitch, raw and standardized
    agg["p_holm_yaw"]        = np.nan
    agg["p_holm_pitch"]      = np.nan
    agg["p_holm_beta_yaw"]   = np.nan
    agg["p_holm_beta_pitch"] = np.nan
    for (mv, spd), fam in agg.groupby(["movement_type","speed"], sort=False):
        idx = fam.index
        agg.loc[idx, "p_holm_yaw"]        = _holm_stepdown(agg.loc[idx, "b_yaw_p_two"])
        agg.loc[idx, "p_holm_pitch"]      = _holm_stepdown(agg.loc[idx, "b_pitch_p_two"])
        agg.loc[idx, "p_holm_beta_yaw"]   = _holm_stepdown(agg.loc[idx, "beta_yaw_p_two"])
        agg.loc[idx, "p_holm_beta_pitch"] = _holm_stepdown(agg.loc[idx, "beta_pitch_p_two"])

    out_agg = os.path.join(outdir, "line_movement__twovar_abs_pitch_yaw__agg.csv")
    agg.to_csv(out_agg, index=False)
    print(f"[2var/DB] Wrote aggregated two-variable regressions (meta + Holm): {out_agg}")

    return per_video, agg


import pandas as pd
import numpy as np

def summarize_twovar_subject_level(per_video_csv_path: str,
                                   out_csv_path: str = "twovar_subject_level_4cond_summary.csv"
                                   ) -> pd.DataFrame:
    """
    Subject-level summary of two-variable slopes (pitch & yaw) 
    saving BOTH SD and 95% CI for publication.
    """
    df = pd.read_csv(per_video_csv_path)

    # 1) Per-(subject, model, condition) mean over videos
    grp_cond = df.groupby(
        ["gaze_model_display","subject_dir","movement_type","speed"], as_index=False
    ).agg(b_pitch=("b_pitch","mean"), b_yaw=("b_yaw","mean"))

    # 2) Per-subject, 4-condition average for each model
    subj_four = grp_cond.groupby(["gaze_model_display","subject_dir"], as_index=False).agg(
        b_pitch_mean4=("b_pitch","mean"),
        b_yaw_mean4=("b_yaw","mean")
    )

    # 3) Summarize across subjects (Mean, SD, CI)
    def _calc_stats(x):
        vals = x.to_numpy(dtype=float)
        n = len(vals)
        mean = np.mean(vals)
        sd = np.std(vals, ddof=1) if n > 1 else 0.0
        se = sd / np.sqrt(n) if n > 1 else 0.0
        t_crit = stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
        
        return pd.Series({
            "mean": mean,
            "sd": sd,
            "ci_low": mean - t_crit * se,
            "ci_high": mean + t_crit * se
        })

    pitch_stats = subj_four.groupby("gaze_model_display")["b_pitch_mean4"].apply(_calc_stats).unstack().add_prefix("b_pitch_")
    yaw_stats = subj_four.groupby("gaze_model_display")["b_yaw_mean4"].apply(_calc_stats).unstack().add_prefix("b_yaw_")
    counts = subj_four.groupby("gaze_model_display")["subject_dir"].nunique().rename("n_subjects")

    summ = pd.concat([counts, pitch_stats, yaw_stats], axis=1).reset_index()
    
    # Save all columns to CSV
    summ.to_csv(out_csv_path, index=False)
    return summ

# Example usage:
# per_video_csv = "/path/to/line_movement__twovar_abs_pitch_yaw__per_video.csv"
# out_csv = "/path/to/twovar_subject_level_4cond_summary.csv"
# table_subject_level = summarize_twovar_subject_level(per_video_csv, out_csv)
# print(table_subject_level)



# --------------------------- CLI ---------------------------

def analyze_line_movement(
    csv_path: Optional[str],
    outdir: str,
):
    """
    Entry point for Target Motion (line_movement) DESCRIPTIVES.
    """
    os.makedirs(outdir, exist_ok=True)
    _ = compute_subject_level_descriptives(csv_path=csv_path, outdir=outdir)
    # run pairwise model comparisons + rankings for all requested aggregations
    run_pairwise_combos_within_models(csv_path=csv_path, outdir=outdir)
    run_pairwise_models_for_line_movement(csv_path=csv_path, outdir=outdir, padjust="holm", alpha=0.05)
    # DB-driven, timestamp-segmented main-segment regressions (valid frames only)
    _rows_db, _agg_db = run_abs_angle_regressions_from_dbs(outdir=outdir)
    # Two-variable (|yaw|, |pitch|) → error, whole trajectory per video
    _per_video, _agg = run_twovar_abs_pitch_yaw_regressions_from_dbs(outdir=outdir)

    # run_twovar_abs_pitch_yaw_regressions_from_dbs summary:
    summ_input_csv = os.path.join(default_out, "line_movement__twovar_abs_pitch_yaw__per_video.csv")
    summ_out_csv =  os.path.join(default_out, "line_movement__twovar_abs_pitch_yaw__agg_4cond_summary.csv")
    summarize_twovar_subject_level(per_video_csv_path=summ_input_csv, out_csv_path=summ_out_csv)

    print("Line movement (Target Motion) subject-level descriptives complete.")


if __name__ == "__main__":
    default_csv = os.path.join(config.get_dataset_base_directory(), "gaze_evaluation_results.csv")
    default_out = os.path.join(config.get_dataset_base_directory(), "line_movement_analysis")

    ap = argparse.ArgumentParser(description="Target Motion (line_movement) subject-level descriptives.")
    ap.add_argument("--csv", type=str, default=default_csv,
                    help="Path to gaze_evaluation_results.csv (defaults to config base).")
    ap.add_argument("--outdir", type=str, default=default_out,
                    help="Directory to write CSV summaries.")
    args = ap.parse_args()

    analyze_line_movement(csv_path=args.csv, outdir=args.outdir)
