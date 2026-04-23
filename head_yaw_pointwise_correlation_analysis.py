# pointwise_headyaw_correlation_analysis.py
import os
import math
import numpy as np
import pandas as pd
from scipy import stats

import config  # assuming your config.py defines BASE_DIR

# ---------------------- config (point/side selection) ----------------------

C1_POINTS = {"p1", "p4", "p7"}  # right-only
C2_POINTS = {"p2", "p5", "p8"}  # both
C3_POINTS = {"p3", "p6", "p9"}  # left-only

# explicit 12 targets in desired order
POINTSIDE_TARGETS = (
    [("p1", "right"), ("p4", "right"), ("p7", "right")] +
    [("p2", "left"), ("p2", "right"),
     ("p5", "left"), ("p5", "right"),
     ("p8", "left"), ("p8", "right")] +
    [("p3", "left"), ("p6", "left"), ("p9", "left")]
)

# ---------------------- fisher helpers ----------------------

def fisher_z(r: np.ndarray) -> np.ndarray:
    r = np.clip(r.astype(float), -0.999999, 0.999999)  # avoid inf
    return 0.5 * np.log((1.0 + r) / (1.0 - r))

def inv_fisher_z(z: np.ndarray) -> np.ndarray:
    return (np.exp(2.0 * z) - 1.0) / (np.exp(2.0 * z) + 1.0)

# ---------------------- stats helpers ----------------------

def mean_ci_t(zvals: np.ndarray, alpha: float = 0.05):
    zvals = np.asarray(zvals, dtype=float)
    zvals = zvals[~np.isnan(zvals)]
    n = zvals.size
    if n == 0:
        return np.nan, (np.nan, np.nan), np.nan, np.nan, 0
    z_mean = float(np.mean(zvals))
    if n == 1:
        return z_mean, (np.nan, np.nan), np.nan, np.nan, 1
    s = float(np.std(zvals, ddof=1))
    se = s / math.sqrt(n)
    t_stat = z_mean / se if se > 0 else np.inf
    df = n - 1
    p_val = 2.0 * stats.t.sf(abs(t_stat), df)
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    ci_lo = z_mean - t_crit * se
    ci_hi = z_mean + t_crit * se
    return z_mean, (ci_lo, ci_hi), t_stat, p_val, n

def holm_adjust(pvals: pd.Series) -> pd.Series:
    s = pd.Series(pvals, copy=True)
    idx = s.dropna().sort_values().index.tolist()
    m = len(idx)
    adj = pd.Series(index=s.index, dtype=float)
    running_max = 0.0
    for rank, i in enumerate(idx, start=1):
        factor = m - rank + 1
        padj_i = min(1.0, s.loc[i] * factor)
        running_max = max(running_max, padj_i)
        adj.loc[i] = running_max
    return adj

# ---------------------- core analysis ----------------------

def _load_corr_csv(corr_csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(corr_csv_path):
        raise FileNotFoundError(corr_csv_path)
    df = pd.read_csv(corr_csv_path)
    need = {"subject_dir", "point", "model", "side", "corr"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")
    df["point"] = df["point"].astype(str)
    df["side"] = df["side"].astype(str).str.lower()
    df["model"] = df["model"].astype(str)
    df["subject"] = df["subject_dir"].astype(str)
    df["corr"] = pd.to_numeric(df["corr"], errors="coerce")
    df = df.dropna(subset=["corr"])
    return df

def _subject_level_zs_for_point_side(df_model: pd.DataFrame, point: str, side: str) -> pd.DataFrame:
    d = df_model[(df_model["point"] == point) & (df_model["side"] == side)].copy()
    if d.empty:
        return pd.DataFrame(columns=["subject", "z_mean_subj"])
    d["z"] = fisher_z(d["corr"].to_numpy())
    subj = (
        d.groupby("subject", as_index=False)["z"]
         .mean()
         .rename(columns={"z": "z_mean_subj"})
    )
    return subj

def _paper_summary_from_pointside_df(tmp_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Build a 1-row paper summary from a per-model point/side table `tmp_df`
    (columns: ['model','point','side','n_subjects','z_mean','z_ci_low','z_ci_high',
               't_stat','p_raw','r_mean','r_ci_low','r_ci_high','p_holm'])

    Overall correlation is computed by treating the 12 (point,side) cells as the
    units: average Fisher-z across available cells, t-CI across cells, then invert to r.
    Also reports min/max r across cells and how many cells are significant after Holm.

    Returns a DataFrame with one row:
      ['model','n_conditions','z_overall','z_ci_low','z_ci_high','t_overall','df',
       'p_overall','r_overall','r_ci_low','r_ci_high',
       'n_sig_holm',
       'min_cell','min_r','min_n_subjects','min_p_holm',
       'max_cell','max_r','max_n_subjects','max_p_holm']
    """
    d = tmp_df.copy()
    d = d[np.isfinite(d["z_mean"])].copy()
    k = int(d.shape[0])

    if k == 0:
        return pd.DataFrame([{
            "model": tmp_df["model"].iloc[0] if "model" in tmp_df.columns and not tmp_df.empty else None,
            "n_conditions": 0,
            "z_overall": np.nan, "z_ci_low": np.nan, "z_ci_high": np.nan,
            "t_overall": np.nan, "df": np.nan, "p_overall": np.nan,
            "r_overall": np.nan, "r_ci_low": np.nan, "r_ci_high": np.nan,
            "n_sig_holm": 0,
            "min_cell": None, "min_r": np.nan, "min_n_subjects": np.nan, "min_p_holm": np.nan,
            "max_cell": None, "max_r": np.nan, "max_n_subjects": np.nan, "max_p_holm": np.nan,
        }])

    # Overall Fisher-z across cells (equal weight per condition)
    z_vals = d["z_mean"].to_numpy(dtype=float)
    z_overall = float(np.mean(z_vals))
    if k == 1:
        z_ci_low = z_ci_high = np.nan
        t_overall = np.nan
        p_overall = np.nan
        df = 0
    else:
        s = float(np.std(z_vals, ddof=1))
        se = s / np.sqrt(k)
        df = k - 1
        t_overall = z_overall / se if se > 0 else np.inf
        p_overall = 2.0 * stats.t.sf(abs(t_overall), df)
        tcrit = stats.t.ppf(1.0 - 0.05 / 2.0, df)
        z_ci_low = z_overall - tcrit * se
        z_ci_high = z_overall + tcrit * se

    # Back-transform
    r_overall = float(inv_fisher_z(np.array([z_overall]))[0])
    r_ci_low = float(inv_fisher_z(np.array([z_ci_low]))[0]) if np.isfinite(z_ci_low) else np.nan
    r_ci_high = float(inv_fisher_z(np.array([z_ci_high]))[0]) if np.isfinite(z_ci_high) else np.nan

    # Significance count across cells (Holm)
    n_sig_holm = int((d["p_holm"] < alpha).sum())

    # Min/max r cells for one-line variability mention
    d_nonan = d[np.isfinite(d["r_mean"])].copy()
    if d_nonan.empty:
        min_cell = max_cell = None
        min_row = max_row = None
    else:
        i_min = int(d_nonan["r_mean"].idxmin())
        i_max = int(d_nonan["r_mean"].idxmax())
        min_row = d.loc[i_min]
        max_row = d.loc[i_max]
        min_cell = f"{min_row['point']}-{min_row['side']}"
        max_cell = f"{max_row['point']}-{max_row['side']}"

    row = {
        "model": d["model"].iloc[0],
        "n_conditions": k,
        "z_overall": z_overall,
        "z_ci_low": z_ci_low if k > 1 else np.nan,
        "z_ci_high": z_ci_high if k > 1 else np.nan,
        "t_overall": t_overall,
        "df": df,
        "p_overall": p_overall,
        "r_overall": r_overall,
        "r_ci_low": r_ci_low,
        "r_ci_high": r_ci_high,
        "n_sig_holm": n_sig_holm,
        "min_cell": min_cell,
        "min_r": float(min_row["r_mean"]) if d_nonan.shape[0] else np.nan,
        "min_n_subjects": int(min_row["n_subjects"]) if d_nonan.shape[0] else np.nan,
        "min_p_holm": float(min_row["p_holm"]) if d_nonan.shape[0] else np.nan,
        "max_cell": max_cell,
        "max_r": float(max_row["r_mean"]) if d_nonan.shape[0] else np.nan,
        "max_n_subjects": int(max_row["n_subjects"]) if d_nonan.shape[0] else np.nan,
        "max_p_holm": float(max_row["p_holm"]) if d_nonan.shape[0] else np.nan,
    }
    return pd.DataFrame([row])


def analyze_pointwise_correlations(
    corr_csv_path: str,
    corr_type: str,
    outdir: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    os.makedirs(outdir, exist_ok=True)
    df = _load_corr_csv(corr_csv_path)

    results = []
    paper_summaries = []  # NEW: collect paper-ready 1-row summaries

    for model, dmodel in df.groupby("model"):
        tmp_rows = []
        for (pt, sd) in POINTSIDE_TARGETS:
            subj_z = _subject_level_zs_for_point_side(dmodel, pt, sd)
            z_mean, (z_lo, z_hi), t_stat, p_raw, n_subj = mean_ci_t(subj_z["z_mean_subj"].to_numpy(), alpha=alpha)
            r_mean = float(inv_fisher_z(np.array([z_mean]))[0]) if not np.isnan(z_mean) else np.nan
            r_lo   = float(inv_fisher_z(np.array([z_lo]))[0])   if np.isfinite(z_lo) else np.nan
            r_hi   = float(inv_fisher_z(np.array([z_hi]))[0])   if np.isfinite(z_hi) else np.nan
            tmp_rows.append({
                "model": model,
                "point": pt,
                "side": sd,
                "n_subjects": int(n_subj),
                "z_mean": z_mean,
                "z_ci_low": z_lo,
                "z_ci_high": z_hi,
                "t_stat": t_stat,
                "p_raw": p_raw,
                "r_mean": r_mean,
                "r_ci_low": r_lo,
                "r_ci_high": r_hi,
            })
        tmp_df = pd.DataFrame(tmp_rows)
        tmp_df["p_holm"] = holm_adjust(tmp_df["p_raw"])

        # Write per-model detailed table
        model_csv = os.path.join(outdir, f"pointwise_{corr_type}_corr_summary__{model}.csv")
        tmp_df.to_csv(model_csv, index=False)

        # NEW: build per-model 1-row paper summary
        paper_1row = _paper_summary_from_pointside_df(tmp_df, alpha=alpha)
        paper_summaries.append(paper_1row)

        results.append(tmp_df)

    # Combined detailed table (also fix the filename typo)
    combined = pd.concat(results, axis=0, ignore_index=True)
    combined_csv = os.path.join(outdir, f"pointwise_{corr_type}_corr_summary__combined.csv")
    combined.to_csv(combined_csv, index=False)
    print(f"Wrote combined summary to: {combined_csv}")

    # NEW: Combined paper summaries (one row per model)
    combined_paper = pd.concat(paper_summaries, axis=0, ignore_index=True)
    combined_paper_csv = os.path.join(outdir, f"pointwise_{corr_type}_corr_paper_summary__combined.csv")
    combined_paper.to_csv(combined_paper_csv, index=False)
    print(f"Wrote paper-ready combined summary to: {combined_paper_csv}")

    return combined


# ---------------------- main entry ----------------------

if __name__ == "__main__":
    BASE_DIR = config.get_dataset_base_directory()
    OUTPUT_DIR_FOR_TEXT_FILES = os.path.join(BASE_DIR, "head_yaw_vs_error_results")
    CORR_TYPE = "pearson"  # "pearson" or "spearman"
    CORR_CSV_PATH = os.path.join(
        OUTPUT_DIR_FOR_TEXT_FILES,
        f"head_yaw_vs_error_{CORR_TYPE}_correlation_coefficients.csv"
    )

    analyze_pointwise_correlations(
        corr_csv_path=CORR_CSV_PATH,
        corr_type=CORR_TYPE,
        outdir=OUTPUT_DIR_FOR_TEXT_FILES,
        alpha=0.05,
    )
