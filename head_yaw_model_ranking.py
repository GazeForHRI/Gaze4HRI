# head_yaw_model_ranking.py
import os
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pingouin as pg

import config  # expects display_model_name(), get_dataset_base_directory()

# --------------------------- Data prep ---------------------------

def _subjectlevel_mean_std_by_model(df_subj: pd.DataFrame) -> pd.DataFrame:
    """
    Subject-level descriptives (primary for the paper):
      - mean = average of subject means
      - std  = std of subject means (ddof=1)
      - count_subjects = number of subjects contributing
    Returns: ['gaze_model','gaze_model_display','mean','std','count_subjects']
    """
    g = (
        df_subj.groupby(["gaze_model","gaze_model_display"])
               .agg(mean=("mean_error","mean"),
                    std=("mean_error", lambda x: float(np.std(x, ddof=1))),
                    count_subjects=("subject","nunique"))
               .reset_index()
    )
    # sort by lower (better) mean
    g.sort_values("mean", inplace=True)
    return g


def _require_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def load_and_prepare(csv_path: Optional[str]) -> pd.DataFrame:
    """
    Load raw gaze_evaluation_results.csv and filter to circular_movement.
    Ensures minimal columns:
      ['gaze_model','subject','experiment_type','angular_error','num_samples']
    Adds 'gaze_model_display'.
    """
    if csv_path is None:
        base_dir = config.get_dataset_base_directory()
        csv_path = os.path.join(base_dir, "gaze_evaluation_results.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    if "error_mean" not in df.columns:
        raise ValueError("Expected column 'error_mean' not found.")
    df["angular_error"] = df["error_mean"]
    if "error_std" not in df.columns:
        raise ValueError("Expected column 'error_std' not found.")
    if "exp_type" not in df.columns:
        raise ValueError("Expected column 'exp_type' not found.")
    df["experiment_type"] = df["exp_type"]

    # subject id
    if "subject" not in df.columns:
        if "subject_dir" in df.columns:
            df["subject"] = df["subject_dir"]
        else:
            raise ValueError("CSV must have 'subject' or 'subject_dir'.")

    req = ["gaze_model", "experiment_type", "subject", "angular_error"]
    _require_cols(df, req)
    if "num_samples" not in df.columns:
        df["num_samples"] = 1

    # keep circular only
    df = df[df["experiment_type"] == "circular_movement"].copy()
    if df.empty:
        raise ValueError("No rows with experiment_type == 'circular_movement'.")

    cols = ["gaze_model", "subject", "experiment_type", "angular_error", "error_std", "num_samples"]
    df = df.loc[:, cols].copy()
    df["gaze_model_display"] = df["gaze_model"].map(config.display_model_name)
    return df

def _subject_level_video_means(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to subject-level weighted mean error per (model).
    Weight = num_samples per video row.
    Returns: ['gaze_model','gaze_model_display','subject','mean_error'].
    """
    need = {"gaze_model","gaze_model_display","subject","angular_error","num_samples"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing cols: {missing}")
    d = df.loc[:, ["gaze_model","gaze_model_display","subject","angular_error","num_samples"]].copy()
    d["_wx"] = d["angular_error"] * d["num_samples"]

    agg = (
        d.groupby(["gaze_model","gaze_model_display","subject"], as_index=False)
         .agg(total_w=("num_samples","sum"), sum_wx=("_wx","sum"))
    )
    agg["mean_error"] = agg["sum_wx"] / agg["total_w"]
    return agg.loc[:, ["gaze_model","gaze_model_display","subject","mean_error"]]

# --------------------------- Pooled stats (for table) ---------------------------

def _weighted_pooled_mean_std(df_subj: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Pooled (frame-weighted) mean ± std per model for circular_movement,
    matching analyze_by_group.py:
      mean = (Σ n_i m_i) / (Σ n_i)
      var  = [ Σ n_i s_i^2 + Σ n_i m_i^2 − (Σ n_i) * mean^2 ] / (Σ n_i)
    Returns: ['gaze_model','gaze_model_display','mean','std','count_frames']
    """
    d = df_raw.copy()
    # guards
    if "error_std" not in d.columns:
        raise ValueError("Expected column 'error_std' in raw CSV for pooled variance calculation.")
    if "num_samples" not in d.columns:
        d["num_samples"] = 1

    # per-row terms
    d["_wx"]  = d["angular_error"] * d["num_samples"]             # n_i * m_i
    d["_ws2"] = (d["error_std"] ** 2) * d["num_samples"]          # n_i * s_i^2
    d["_wm2"] = (d["angular_error"] ** 2) * d["num_samples"]      # n_i * m_i^2

    agg = d.groupby("gaze_model").agg(
        total_n=("num_samples","sum"),
        sum_wx=("_wx","sum"),
        sum_ws2=("_ws2","sum"),
        sum_wm2=("_wm2","sum"),
    )

    mean = agg["sum_wx"] / agg["total_n"]
    var  = (agg["sum_ws2"] + agg["sum_wm2"] - agg["total_n"] * (mean ** 2)) / agg["total_n"]
    std  = np.sqrt(var.clip(lower=0))

    out = pd.DataFrame({
        "gaze_model": mean.index,
        "mean": mean.values,
        "std": std.values,
        "count_frames": agg["total_n"].values,
    })

    disp = df_subj.groupby("gaze_model")["gaze_model_display"].first().reset_index()
    out = out.merge(disp, on="gaze_model", how="left")
    out = out.loc[:, ["gaze_model","gaze_model_display","mean","std","count_frames"]]
    out.sort_values("mean", inplace=True)
    return out


# --------------------------- Pairwise tests & ranking ---------------------------

def run_pairwise_models_circular(
    csv_path: Optional[str],
    outdir: str,
    padjust: str = "holm",
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[List[str]]]:
    """
    - Builds subject-level mean errors per model.
    - Paired parametric pairwise tests across models (within-subject).
    - Holm adjust p-values across all pairs.
    - Creates a partial order (tiers) using decision rule:
         Mi ≺ Mj  <=>  mean_i > mean_j  AND  p_holm(Mi,Mj) < alpha
      Lower mean error = better.
    Outputs:
      - pairwise CSV
      - pooled mean±std CSV
      - returns (pairwise_df, pooled_df, tiers)
    """
    os.makedirs(outdir, exist_ok=True)
    df_raw = load_and_prepare(csv_path)
    df_subj = _subject_level_video_means(df_raw)
    # Subject-level descriptives (primary for paper)
    subj_stats = _subjectlevel_mean_std_by_model(df_subj)
    subj_stats.to_csv(os.path.join(outdir, "circular_subjectlevel_mean_std_by_model.csv"), index=False)
    # Pairwise within-subject tests across models
    res = pg.pairwise_tests(
        data=df_subj,
        dv="mean_error",
        within="gaze_model_display",
        subject="subject",
        parametric=True,
        padjust=padjust,
        effsize="cohen",
        return_desc=True,
    )
    # clean & order
    res.sort_values(["A","B"], inplace=True)
    res.to_csv(os.path.join(outdir, "pairwise_models_within_circular.csv"), index=False)

    # Pooled (weighted by frames) mean±std per model
    pooled = _weighted_pooled_mean_std(df_subj, df_raw)
    pooled.to_csv(os.path.join(outdir, "circular_pooled_mean_std_by_model.csv"), index=False)

    # Build preference relation using display names
    pref = _induce_preference_tiers(res, subj_stats, alpha=alpha)

    # Save tiers as a small txt
    tiers_txt = os.path.join(outdir, "circular_model_tiers.txt")
    with open(tiers_txt, "w") as f:
        f.write("Tiers (best to worse; ties share a tier):\n")
        for k, tier in enumerate(pref, 1):
            f.write(f"Tier {k}: " + " ~ ".join(tier) + "\n")

    return res, pooled, pref

def _induce_preference_tiers(pairwise_df: pd.DataFrame, subj_stats_df: pd.DataFrame, alpha: float = 0.05) -> List[List[str]]:
    """
    Construct tiers from the preference relation using SUBJECT-LEVEL means:
      Mi ≺ Mj  iff mean_i > mean_j AND Holm-adjusted p(Mi vs Mj) < alpha.
    Tiers are built by walking the models sorted by subject-level mean (ascending)
    and merging neighbors that are not strictly ordered by the rule above.
    """
    # map display -> subject-level mean
    mean_map = dict(zip(subj_stats_df["gaze_model_display"], subj_stats_df["mean"]))
    models_sorted = list(subj_stats_df.sort_values("mean")["gaze_model_display"])

    # Find adjusted p column from pingouin
    pcol = None
    for cand in ["p-corr","p_corr","p-adjust","p_adjust"]:
        if cand in pairwise_df.columns:
            pcol = cand
            break
    if pcol is None:
        raise ValueError("Could not find adjusted p-value column in pairwise results.")

    # lookup for Holm p-values (order-agnostic on (A,B))
    p_holm = {}
    for _, r in pairwise_df.iterrows():
        a, b = str(r["A"]), str(r["B"])
        p_holm[(a, b)] = float(r[pcol])

    def holm_p(a: str, b: str) -> Optional[float]:
        if (a, b) in p_holm: return p_holm[(a, b)]
        if (b, a) in p_holm: return p_holm[(b, a)]
        return None

    # Preference predicate using SUBJECT-LEVEL means + Holm p
    def prefers(i: str, j: str) -> bool:
        m_i, m_j = mean_map[i], mean_map[j]
        p = holm_p(i, j)
        if p is None:
            return False
        # "j preferred to i" means mean_j < mean_i and significant
        return (m_i > m_j) and (p < alpha)

    # Build tiers greedily
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


# --------------------------- Main entry ---------------------------

if __name__ == "__main__":
    BASE_DIR = config.get_dataset_base_directory()
    DEFAULT_CSV = os.path.join(BASE_DIR, "gaze_evaluation_results.csv")
    OUTDIR = os.path.join(BASE_DIR, "head_yaw_vs_error_results", "head_yaw_model_ranking")

    os.makedirs(OUTDIR, exist_ok=True)
    pairwise_df, pooled_df, tiers = run_pairwise_models_circular(
        csv_path=DEFAULT_CSV,
        outdir=OUTDIR,
        padjust="holm",
        alpha=0.05,
    )

    # Pretty print to console
    subj_csv = os.path.join(OUTDIR, "circular_subjectlevel_mean_std_by_model.csv")
    if os.path.isfile(subj_csv):
        subj_df = pd.read_csv(subj_csv)
        print("\n== Subject-level mean ± std (deg) ==")
        for _, r in subj_df.iterrows():
            print(f"{r['gaze_model_display']}: {r['mean']:.2f} ± {r['std']:.2f}  (subjects={int(r['count_subjects'])})")

    print("\n== Tiers (best→worse) ==")
    for k, tier in enumerate(tiers, 1):
        print(f"Tier {k}: " + " ~ ".join(tier))

    print(f"\nSaved outputs to: {OUTDIR}")
