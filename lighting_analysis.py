# lighting_analysis.py
"""
Create lighting-condition analyses directly from the raw gaze_evaluation_results.csv.

Outputs (saved as PNG under --outdir):
1) Line + error bars (mean ± pooled std) across lighting levels for all models,
   plus one per-model figure.
2) Weighted-rows boxplots (per-model & all-models) showing the distribution of
   angular errors at each lighting level (weights = num_samples). No subject-level
   collapsing is performed.

Assumptions about the CSV (as produced by your pipeline):
- Columns include at least:
    - exp_type (experiment type; we'll use lighting_10/25/50/100)
    - gaze_model
    - subject_dir (or subject)  [kept but NOT used to aggregate]
    - error_mean (per-row mean angular error in degrees)
    - error_std  (per-row std of angular error in degrees)  [optional but preferred]
    - num_samples (row weight / count)
- If `subject` isn't present, we derive it from `subject_dir`.

Usage:
    python lighting_analysis.py --csv /path/to/gaze_evaluation_results.csv --outdir /path/to/out

If you have your `config` module, you may omit --csv and it will default to:
    os.path.join(config.get_dataset_base_directory(), "gaze_evaluation_results.csv")
"""
import config
import os
import re
import argparse
from typing import Tuple, Optional, Iterable, Dict, List
import pingouin as pg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# --------------------------- Utilities ---------------------------

LIGHTING_PATTERN = re.compile(r"^lighting_(\d+)$")

def compute_subject_level_descriptives(
    csv_path: Optional[str],
    outdir: str
) -> pd.DataFrame:
    """
    Subject-level descriptives per (model, lighting), where each subject contributes
    a single frame-weighted mean error for that (model, lighting):
        - n_subjects
        - mean_of_subject_means
        - sd_of_subject_means
        - se (sd / sqrt(n))
        - 95% CI (mean ± 1.96*se)
    Output uses DISPLAY names (paper-ready).
    """
    os.makedirs(outdir, exist_ok=True)
    df = load_and_prepare(csv_path)                 # raw rows
    subj_df = _subject_level_video_means(df)        # ['gaze_model','gaze_model_display','subject','lighting','mean_error']

    def _summ(g):
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

    desc = (
        subj_df
        .groupby(["gaze_model", "gaze_model_display", "lighting"], as_index=False)
        .apply(_summ, include_groups=False)   # <- key fix
        .reset_index(drop=True)
        .sort_values(["gaze_model_display", "lighting"])
    )

    out_csv = os.path.join(outdir, "lighting_subject_level_descriptives.csv")
    desc.to_csv(out_csv, index=False)
    print(f"Saved subject-level descriptives to {out_csv}")
    return desc


def _draw_weighted_box_colored(
    ax,
    x_center: float,
    width: float,
    q1: float, med: float, q3: float,
    wl: float, wu: float, mean_val: float,
    facecolor: str = "#cfd8e3",
    edgecolor: str = "black",
):
    """
    Draw a single weighted box at categorical x_center with given stats.
    Styled with a provided facecolor (model-specific) and black edges.
    """
    if width <= 0:
        return

    left = x_center - width / 2.0

    # box
    rect = plt.Rectangle((left, q1), width, q3 - q1,
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=1.25, alpha=0.85)
    ax.add_patch(rect)

    # median
    ax.plot([left, left + width], [med, med], color=edgecolor, linewidth=1.5)

    # whiskers + caps
    ax.plot([x_center, x_center], [wl, q1], color=edgecolor, linewidth=1.25)
    ax.plot([x_center, x_center], [q3, wu], color=edgecolor, linewidth=1.25)
    cap = width * 0.25
    ax.plot([x_center - cap, x_center + cap], [wl, wl], color=edgecolor, linewidth=1.25)
    ax.plot([x_center - cap, x_center + cap], [wu, wu], color=edgecolor, linewidth=1.25)

    # mean marker (triangle)
    ax.plot(x_center, mean_val, marker="^", markersize=6, color=edgecolor)


def plot_weighted_boxplots_by_lighting(
    df: pd.DataFrame,
    outdir: str,
    ymax: Optional[float] = 60.0,
    show: bool = False,
):
    """
    Create one boxplot figure per lighting level.
    - X-axis is categorical (models spaced evenly).
    - Models are ordered left→right by *subject-level mean error* (best first) computed within that lighting.
    - Styling is uniform: light gray boxes, black lines/markers (no per-model colors).
    - Uses row weights = num_samples (no subject aggregation here).
    - Saves 4 PNGs: boxplot_weighted_rows__by_lighting_<L>.png
    """
    raise Exception("# NOTE for FG submission: These boxplots currently represent the distribution of frames/videos. For statistical consistency with our T-tests (which are subject-level), these should be modified to represent the distribution of the $N=52$ subject-level means.")
    os.makedirs(outdir, exist_ok=True)

    # Ensure display names are available
    if "gaze_model_display" not in df.columns:
        df = df.copy()
        df["gaze_model_display"] = df["gaze_model"].map(config.display_model_name)

    # Fixed categorical box width
    width = 0.6

    for L in sorted(df["lighting"].unique().tolist()):
        dL = df[df["lighting"] == L]
        if dL.empty:
            continue

        # ---- Order models by *subject-level* mean error within this lighting
        # Step 1: compute subject-level mean per model (frame-weighted by num_samples)
        d_tmp = dL.loc[:, ["gaze_model_display", "subject", "angular_error", "num_samples"]].copy()
        d_tmp["_wx"] = d_tmp["angular_error"] * d_tmp["num_samples"]
        subj_means = (
            d_tmp.groupby(["gaze_model_display", "subject"], as_index=False)
                 .agg(total_w=("num_samples","sum"), sum_wx=("_wx","sum"))
        )
        subj_means["mean_error"] = subj_means["sum_wx"] / subj_means["total_w"]

        # Step 2: model mean of subject means → sort ascending (best first)
        model_order = (
            subj_means.groupby("gaze_model_display")["mean_error"]
            .mean()
            .sort_values()
            .index
            .tolist()
        )

        # --- NEW: labels without " w/ ..." suffix for display
        model_order_clean = [m.split(" w/")[0].strip() for m in model_order]

        # Evenly spaced categorical x positions for the *sorted* models
        x_positions = {m: i for i, m in enumerate(model_order)}
        # Evenly spaced categorical x positions for the *sorted* models
        x_positions = {m: i for i, m in enumerate(model_order)}

        # ---- Plot
        fig, ax = plt.subplots(figsize=(7.2, 4.6))

        for m_disp in model_order:
            dM = dL[dL["gaze_model_display"] == m_disp]
            if dM.empty:
                continue

            vals = dM["angular_error"].to_numpy(dtype=float)
            wts  = dM["num_samples"].to_numpy(dtype=float)
            if vals.size == 0 or wts.sum() == 0:
                continue

            q1, med, q3, wl, wu, mean_val = _weighted_boxplot_data(vals, wts)
            xc = float(x_positions[m_disp])

            # Uniform style: light gray box, black strokes/markers
            _draw_weighted_box(
                ax, xc, width,
                q1, med, q3, wl, wu, mean_val,
            )

            # If the boxplot whisker exceeds the y cap, annotate it
            if ymax is not None and wu > ymax:
                ax.text(
                    xc-0.2, ymax - 0.5, f"{wu:.1f}",
                    ha="center", va="top", fontsize=9
                )

        # Axes formatting
        ax.set_title(f"Angular Error at Lighting {L}")
        ax.set_ylabel("Angular Error (deg)")
        ax.set_xlabel("")  # or "Model" if you prefer a simple axis label
        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels(model_order_clean, rotation=0, ha="center")
        if ymax is not None:
            ax.set_ylim(0.0, ymax)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4)

        fig.tight_layout()
        out_png = os.path.join(outdir, f"boxplot_weighted_rows__by_lighting_{int(L)}.png")
        fig.savefig(out_png, dpi=220)
        if show:
            plt.show()
        plt.close(fig)


def plot_weighted_boxplots_combined_single_axis(
    df: pd.DataFrame,
    outdir: str,
    ymax: Optional[float] = 60.0,
    title: str = "Angular Error by Illumination (single axes)",
    show: bool = False,
):
    """
    Single-axes figure: 4 lighting blocks (10,25,50,100) laid out left→right.
    Within each block, draw weighted boxplots for all models. X has one set of
    rotated model labels, and a second (minor) tick row with the illumination labels.
    Output: boxplot_weighted_rows__combined_single_axis.png
    """
    os.makedirs(outdir, exist_ok=True)

    # Ensure display names available
    if "gaze_model_display" not in df.columns:
        df = df.copy()
        df["gaze_model_display"] = df["gaze_model"].map(config.display_model_name)

    lightings = sorted(df["lighting"].unique().tolist())
    if not lightings:
        return

    # Use consistent model set across lightings (intersection), keep display names
    models_all = [
        m for m in df["gaze_model_display"].unique().tolist()
        if not df[df["gaze_model_display"] == m].empty
    ]

    # Determine ordering *within each lighting* by subject-level mean error (best→worst)
    # (If you prefer a fixed global order, compute once and reuse instead.)
    perL_order: Dict[int, List[str]] = {}
    for L in lightings:
        dL = df[df["lighting"] == L]
        d_tmp = dL.loc[:, ["gaze_model_display", "subject", "angular_error", "num_samples"]].copy()
        d_tmp["_wx"] = d_tmp["angular_error"] * d_tmp["num_samples"]
        subj_means = (
            d_tmp.groupby(["gaze_model_display", "subject"], as_index=False)
                 .agg(total_w=("num_samples","sum"), sum_wx=("_wx","sum"))
        )
        subj_means["mean_error"] = subj_means["sum_wx"] / subj_means["total_w"]
        order = (
            subj_means.groupby("gaze_model_display")["mean_error"]
            .mean()
            .sort_values()
            .index
            .tolist()
        )
        perL_order[L] = order

    # Layout positions
    width = 0.6
    gap_between_blocks = 1.2   # blank slots between lighting blocks

    # Compute absolute x positions for every (L, model) tuple
    positions = []
    labels_models = []
    block_centers = []
    x_cursor = 0.0
    for i, L in enumerate(lightings):
        order = perL_order[L]
        n = len(order)
        # left edge of this block at current x_cursor
        for j, m_disp in enumerate(order):
            pos = x_cursor + j
            positions.append((L, m_disp, pos))
            labels_models.append(m_disp.split(" w/")[0])  # shorter label
        # center for the lighting label
        block_centers.append(x_cursor + (n - 1) / 2.0)
        # advance cursor for next block
        x_cursor += n + gap_between_blocks

    # Plot
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    # Draw every box
    for (L, m_disp, pos) in positions:
        dLM = df[(df["lighting"] == L) & (df["gaze_model_display"] == m_disp)]
        if dLM.empty:
            continue
        vals = dLM["angular_error"].to_numpy(dtype=float)
        wts  = dLM["num_samples"].to_numpy(dtype=float)
        if vals.size == 0 or wts.sum() == 0:
            continue
        q1, med, q3, wl, wu, mean_val = _weighted_boxplot_data(vals, wts)
        _draw_weighted_box(ax, float(pos), width, q1, med, q3, wl, wu, mean_val)

        # optional: annotate whisker overflow
        if ymax is not None and wu > ymax:
            ax.text(float(pos) - 0.15, ymax - 0.6, f"{wu:.1f}",
                    ha="center", va="top", fontsize=11, fontweight="bold")

    # Axes style
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_ylabel("Angular Error (deg)", fontsize=14, fontweight="bold")
    ax.set_xlabel("")  # no global x label

    # Major ticks: one per model position (rotated names)
    x_positions_only = [p for (_, _, p) in positions]
    ax.set_xticks(x_positions_only)
    ax.set_xticklabels(labels_models, rotation=45, ha="right")
    for lab in ax.get_xticklabels():
        lab.set_fontweight("bold")

    # Illumination labels centered under each block
    ax.set_xticks(block_centers, minor=True)
    ax.set_xticklabels([str(L) for L in lightings], minor=True, fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", which="minor", length=0, pad=26)  # extra space for the 10/25/50/100 row

    # Make y-axis text big & bold
    ax.yaxis.label.set_size(14)
    ax.yaxis.label.set_fontweight("bold")
    for lab in ax.get_yticklabels():
        lab.set_fontsize(12)
        lab.set_fontweight("bold")


    if ymax is not None:
        ax.set_ylim(0.0, ymax)

    # Vertical separators between blocks (cosmetic)
    # Draw at the gaps between blocks
    x_cursor = 0.0
    for i, L in enumerate(lightings[:-1]):
        n = len(perL_order[L])
        split_at = x_cursor + n - 0.4 + 0.6 * 0  # near block end
        ax.axvline(split_at + gap_between_blocks - 0.6, color="0.7", linestyle=":", linewidth=1.0)
        x_cursor += n + gap_between_blocks

    # Ticks/spines/grid
    ax.tick_params(axis="both", which="both", width=1.6, length=6)
    for sp in ax.spines.values():
        sp.set_linewidth(1.6)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8)

    fig.tight_layout()
    out_png = os.path.join(outdir, "boxplot_weighted_rows__combined_single_axis.png")
    fig.savefig(out_png, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def _extract_lighting_level(exp_type: str) -> Optional[int]:
    """
    Convert 'lighting_25' -> 25 (int). Return None if not a lighting experiment.
    """
    if not isinstance(exp_type, str):
        return None
    m = LIGHTING_PATTERN.match(exp_type)
    return int(m.group(1)) if m else None


def _require_cols(df: pd.DataFrame, cols: Iterable[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _weighted_pooled_mean_std(
    df: pd.DataFrame,
    mean_col: str = "angular_error",
    std_col: str = "error_std",
    n_col: str = "num_samples",
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute pooled (population) mean and std using row-level means/stds and weights n_i.

    For a group:
      mean = (Σ n_i * m_i) / (Σ n_i)
      var  = [ Σ n_i s_i^2 + Σ n_i m_i^2 − (Σ n_i) * mean^2 ] / (Σ n_i)
      std  = sqrt(max(var, 0))

    If std_col is missing, we fallback to unweighted sample std of mean_col within the group,
    then treat it as s_i for all rows (which is a crude approximation). Best results occur
    when std_col is present in the CSV.
    """
    # If std_col isn't available, create a fallback column filled with per-group std of means
    if std_col not in df.columns:
        tmp = df.groupby(["gaze_model", "lighting"])[mean_col].std(ddof=0).rename("_group_std")
        df = df.merge(tmp, left_on=["gaze_model", "lighting"], right_index=True, how="left")
        df["_s_fallback"] = df["_group_std"].fillna(0.0)
        s_col = "_s_fallback"
    else:
        s_col = std_col

    df = df.copy()
    df["_wx"]  = df[mean_col] * df[n_col]
    df["_ws2"] = (df[s_col] ** 2) * df[n_col]
    df["_wm2"] = (df[mean_col] ** 2) * df[n_col]

    agg = df.groupby(["gaze_model", "lighting"]).agg(
        total_n=("num_samples", "sum"),
        sum_wx=("_wx", "sum"),
        sum_ws2=("_ws2", "sum"),
        sum_wm2=("_wm2", "sum"),
    )

    mean = agg["sum_wx"] / agg["total_n"]
    var = (agg["sum_ws2"] + agg["sum_wm2"] - agg["total_n"] * (mean ** 2)) / agg["total_n"]
    std = np.sqrt(var.clip(lower=0))
    return mean, std, agg["total_n"]


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, qs: Iterable[float]) -> np.ndarray:
    """
    Weighted quantiles of `values` for probabilities in qs (between 0 and 1).
    Uses the "cdf" method: sort by value, take cumulative normalized weight,
    interpolate at the desired quantile probabilities.
    """
    if values.size == 0:
        return np.array([np.nan] * len(list(qs)))
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cum_w = np.cumsum(w)
    if cum_w[-1] <= 0:
        return np.array([np.nan] * len(list(qs)))
    cdf = cum_w / cum_w[-1]
    return np.interp(qs, cdf, v)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denom = weights.sum()
    return float((values * weights).sum() / denom) if denom > 0 else np.nan


# --------------------------- Plotters ---------------------------

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


def plot_weighted_boxplots_per_model(
    df: pd.DataFrame,
    outdir: str,
    ymin: float = 0.0,
    ymax: Optional[float] = 60.0,
    show: bool = False,
):
    """
    Weighted-rows boxplots per model (no subject aggregation).
    Boxes are placed at TRUE numeric x-positions (10,25,50,100). Shows weighted mean.
    All strokes are black; boxes have a neutral fill. Narrower figure.
    """
    raise Exception("# NOTE for FG submission: These boxplots currently represent the distribution of frames/videos. For statistical consistency with our T-tests (which are subject-level), these should be modified to represent the distribution of the $N=52$ subject-level means.")
    lightings = sorted(df["lighting"].unique().tolist())
    gaps = np.diff(lightings)
    min_gap = gaps.min() if len(gaps) > 0 else 10.0
    width = 0.22 * min_gap

    for model, dmodel in df.groupby("gaze_model"):
        fig, ax = plt.subplots(figsize=(6.6, 4.4))

        for L in lightings:
            d = dmodel[dmodel["lighting"] == L]
            vals = d["angular_error"].to_numpy(dtype=float)
            wts  = d["num_samples"].to_numpy(dtype=float)
            if vals.size == 0 or wts.sum() == 0:
                continue
            q1, med, q3, wl, wu, mean_val = _weighted_boxplot_data(vals, wts)

            # --- HARD CLAMP center so width fits inside [0,100]
            half = width / 2.0
            x_center = float(L)
            x_center = max(0.0 + half, min(100.0 - half, x_center))

            _draw_weighted_box(ax, x_center, width, q1, med, q3, wl, wu, mean_val, x_min=0.0, x_max=100.0)

        ax.set_xlabel("Lighting Intensity (%)")
        ax.set_ylabel("Angular Error (deg)")
        ax.set_title(f"Angular Error vs Lighting — {config.display_model_name(model)} (Weighted Boxplot)")
        if ymax is not None:
            ax.set_ylim(ymin, ymax)
        ax.set_xlim(0, 100)                  # strict 0–100
        ax.set_xticks(lightings)
        ax.set_xticklabels([str(x) for x in lightings])
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4)
        fig.tight_layout()

        fname = os.path.join(outdir, f"boxplot_weighted_rows_vs_lighting__{model}.png")
        fig.savefig(fname, dpi=220)
        if show:
            plt.show()
        plt.close(fig)

# --------------------------- Data Prep ---------------------------

def load_and_prepare(csv_path: Optional[str]) -> pd.DataFrame:
    """
    Load the raw CSV and produce a DataFrame with the columns we need:
        ['gaze_model','subject','experiment_type','lighting',
         'angular_error','error_std','num_samples']
    Filter to lighting experiments only and attach numeric 'lighting' column.
    """
    if csv_path is None:
        # Try to pull a default from config, if available
        try:
            base_dir = config.get_dataset_base_directory()
            csv_path = os.path.join(base_dir, "gaze_evaluation_results.csv")
        except Exception:
            raise ValueError("CSV path not provided and default via `config` could not be resolved.")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalize expected columns
    if "error_mean" not in df.columns:
        raise ValueError("Expected column 'error_mean' was not found.")
    df["angular_error"] = df["error_mean"]

    if "exp_type" not in df.columns:
        raise ValueError("Expected column 'exp_type' was not found.")
    df["experiment_type"] = df["exp_type"]

    # subject column (kept for completeness; not used to aggregate the boxplots)
    if "subject" not in df.columns:
        if "subject_dir" in df.columns:
            df["subject"] = df["subject_dir"]
        else:
            raise ValueError("CSV must have 'subject' or 'subject_dir'.")

    # required columns check
    req = ["gaze_model", "experiment_type", "subject", "angular_error"]
    _require_cols(df, req)
    main_model_keys = set(config.get_main_models_included_in_the_paper().keys())
    df = df[df["gaze_model"].isin(main_model_keys)].copy()
    if "num_samples" not in df.columns:
        df["num_samples"] = 1
    # error_std may be missing; handled in _weighted_pooled_mean_std

    # Keep lighting experiments only and add numeric lighting level
    df["lighting"] = df["experiment_type"].apply(_extract_lighting_level)
    df = df[~df["lighting"].isna()].copy()
    df["lighting"] = df["lighting"].astype(int)

    cols = ["gaze_model", "subject", "experiment_type", "lighting",
            "angular_error", "num_samples"]
    if "error_std" in df.columns:
        cols.append("error_std")
    df = df.loc[:, cols].copy()
    df["gaze_model_display"] = df["gaze_model"].map(config.display_model_name)

    return df


# --------------------------- Main analysis ---------------------------

def analyze_lighting(
    csv_path: Optional[str],
    outdir: str,
    show: bool = False,              # kept for CLI compatibility (unused)
    ymax: Optional[float] = 60.0,    # kept for CLI compatibility (unused)
    models_for_multi_model_plot: List[str] = [],  # kept for CLI compatibility (unused)
):
    """
    SUBJECT-LEVEL LIGHTING ANALYSIS ONLY.

    Pipeline:
      raw rows -> frame-weighted video means -> SUBJECT-LEVEL means (per model × lighting)

    Outputs written to `outdir`:
      - lighting_subject_level_descriptives.csv
      - pairwise_lighting_within__<ModelDisplay>.csv  (one per model)
      - pairwise_lighting_within__all_models.csv
      - lighting_sensitivity_index.csv               (10–100; includes Δ10→100)
      - lighting_sensitivity_index_10_25_50.csv      (10–50; includes Δ10→50)

    Notes:
      * No pooled/frame-level descriptives are computed.
      * No weighted-rows boxplots or multi-model plots are produced.
    """
    os.makedirs(outdir, exist_ok=True)

    # Validate CSV loads & schema (re-uses existing loader but we do not keep its outputs here)
    df = load_and_prepare(csv_path)

    # 1) Subject-level descriptives per (model × lighting)
    compute_subject_level_descriptives(csv_path=csv_path, outdir=outdir)

    # plot_weighted_boxplots_by_lighting(
    #     df,
    #     outdir=outdir,
    #     ymax=ymax,
    #     show=show,
    # )

    # # single-axes combined figure (four lighting blocks side-by-side)
    # plot_weighted_boxplots_combined_single_axis(
    #     df=df,
    #     outdir=outdir,
    #     ymax=ymax,
    #     title="Angular Error by Illumination Level",
    #     show=show,
    # )
    # 2) Within-subject pairwise tests across lighting (per model; Holm-adjusted)
    run_pairwise_lighting_within_models(csv_path=csv_path, outdir=outdir)

    # 3) Single-number sensitivities (subject-level points)
    compute_lighting_sensitivity(csv_path=csv_path, outdir=outdir)           # 10–100 with Δ10→100
    compute_lighting_sensitivity_3levels(csv_path=csv_path, outdir=outdir)   # 10–50 with Δ10→50

    # 4) Pairwise *model* comparisons within each lighting (subject-level)
    _ = run_pairwise_models_within_lighting(csv_path=csv_path, outdir=outdir, padjust="holm")

    print("Subject-level lighting analysis complete (descriptives, pairwise tests, sensitivities).")

# --------------------------- CLI ---------------------------

def plot_weighted_boxplots_multi_models(
    df: pd.DataFrame,
    models: List[str],
    out_png: str,
    title: str = "Angular Error vs Lighting Intensity",
    ymin: float = 0.0,
    ymax: Optional[float] = 60.0,
    show: bool = False,
    ncols: Optional[int] = None,
):
    """
    Render several per-model weighted boxplots side-by-side in a single figure.

    - Each subplot shows a single model: x = lighting (10/25/50/100), y = angular_error.
    - Distribution is computed with row weights = num_samples (no subject aggregation).
    - Styling matches the per-model function: black whiskers/medians/means, neutral box fill.
    - Boxes are clamped inward so lighting=100 never clips.

    Args:
        df: DataFrame from load_and_prepare(...)
        models: list of gaze_model names to plot (order respected; missing ones are skipped)
        out_png: output path for the combined PNG
        title: figure-wide title
        ymin, ymax: y-axis limits (use ymax=None for auto)
        show: whether to plt.show()
        ncols: number of columns; default = len(valid_models) (single row).
               Set to e.g. 3 or 4 to wrap into multiple rows if many models.
    """
    raise Exception("# NOTE for FG submission: These boxplots currently represent the distribution of frames/videos. For statistical consistency with our T-tests (which are subject-level), these should be modified to represent the distribution of the $N=52$ subject-level means.")
    # Keep only requested models that actually exist:
    avail = set(df["gaze_model"].unique().tolist())
    models = [m for m in models if m in avail]
    if not models:
        raise ValueError("None of the requested models exist in the DataFrame.")

    # Lighting grid (shared for all subplots)
    lightings = sorted(df["lighting"].unique().tolist())
    if not lightings:
        raise ValueError("No lighting levels found in DataFrame.")

    gaps = np.diff(lightings)
    min_gap = gaps.min() if len(gaps) > 0 else 10.0
    width = 0.22 * min_gap

    # Layout
    M = len(models)
    if ncols is None or ncols <= 0:
        ncols = M
    nrows = int(np.ceil(M / ncols))

    # Figure size: aim for compact axes (~6.0in width per subplot)
    per_w, per_h = 5.8, 4.4
    fig_w = max(6.0, per_w * ncols)
    fig_h = max(4.0, per_h * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False, sharey=True)

    ax_list = axes.ravel()

    for idx, model in enumerate(models):
        ax = ax_list[idx]
        dmodel = df[df["gaze_model"] == model]

        # Draw weighted boxes at true x positions, with inward clamping
        for L in lightings:
            d = dmodel[dmodel["lighting"] == L]
            vals = d["angular_error"].to_numpy(dtype=float)
            wts  = d["num_samples"].to_numpy(dtype=float)
            if vals.size == 0 or wts.sum() == 0:
                continue

            q1, med, q3, wl, wu, mean_val = _weighted_boxplot_data(vals, wts)

            half = width / 2.0
            x_center = float(L)
            # hard clamp so the whole box sits inside [0,100]
            x_center = max(0.0 + half, min(100.0 - half, x_center))

            _draw_weighted_box(ax, x_center, width, q1, med, q3, wl, wu, mean_val, x_min=0.0, x_max=100.0)

        # Axes formatting
        if idx % ncols == 0:
            ax.set_ylabel("Angular Error (deg)")
        else:
            ax.set_ylabel("")

        ax.set_xlabel("Lighting Intensity (%)")
        ax.set_title(config.display_model_name(model))
        if ymax is not None:
            ax.set_ylim(ymin, ymax)
        ax.set_xlim(0, 100)
        ax.set_xticks(lightings)
        ax.set_xticklabels([str(x) for x in lightings])
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4)

    # Hide any unused axes
    for j in range(M, nrows * ncols):
        fig.delaxes(ax_list[j])

    fig.suptitle(title, y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=220)
    if show:
        plt.show()
    plt.close(fig)


### TESTING CODE FOR PAPER.

def _subject_level_video_means(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to subject-level weighted means per (model, lighting).
    Weight each video's error by its frame count (num_samples).
    Returns: ['gaze_model','gaze_model_display','subject','lighting','mean_error'].
    """
    need = {"gaze_model","gaze_model_display","subject","lighting","angular_error","num_samples"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing required cols: {missing}")

    d = df.loc[:, ["gaze_model","gaze_model_display","subject","lighting","angular_error","num_samples"]].copy()
    d["_wx"] = d["angular_error"] * d["num_samples"]

    agg = (
        d.groupby(["gaze_model","gaze_model_display","subject","lighting"], as_index=False)
         .agg(total_w=("num_samples","sum"), sum_wx=("_wx","sum"))
    )
    agg["mean_error"] = agg["sum_wx"] / agg["total_w"]
    return agg.loc[:, ["gaze_model","gaze_model_display","subject","lighting","mean_error"]]


def run_pairwise_lighting_within_models(
    csv_path: Optional[str],
    outdir: str,
    padjust: str = "holm"
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Within-subject pairwise tests across lighting levels for each model.
    Pipeline:
      raw rows -> frame-weighted video means -> subject-level means (per model × lighting)
      -> paired two-sided t-tests within subjects (Pingouin pairwise_tests, Holm).
    Outputs use DISPLAY names.
    """
    os.makedirs(outdir, exist_ok=True)
    df = load_and_prepare(csv_path)
    subj_df = _subject_level_video_means(df)

    models = sorted(subj_df["gaze_model"].unique().tolist())
    combined_list: List[pd.DataFrame] = []
    per_model: Dict[str, pd.DataFrame] = {}

    for m in models:
        dmodel = subj_df[subj_df["gaze_model"] == m].copy()
        if dmodel.empty:
            continue

        # Pingouin pairwise parametric tests, within-subject
        res = pg.pairwise_tests(
            data=dmodel,
            dv="mean_error",
            within="lighting",
            subject="subject",
            parametric=True,
            padjust=padjust,
            effsize="cohen",
            return_desc=True
        )

        # Add display name column for outputs
        display = dmodel["gaze_model_display"].iloc[0]
        res.insert(0, "gaze_model", display)

        per_model[display] = res
        combined_list.append(res)

        # File name uses display name (sanitized)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", display)
        res.to_csv(os.path.join(outdir, f"pairwise_lighting_within__{safe}.csv"), index=False)

    if not combined_list:
        return pd.DataFrame(), per_model

    combined = pd.concat(combined_list, axis=0, ignore_index=True)
    combined.sort_values(["gaze_model", "A", "B"], inplace=True)
    combined.to_csv(os.path.join(outdir, "pairwise_lighting_within__all_models.csv"), index=False)
    return combined, per_model


def compute_lighting_sensitivity(
    csv_path: Optional[str],
    outdir: str
) -> pd.DataFrame:
    """
    Lighting sensitivity per model using subject-level points:
      For each (subject, lighting) we use the subject's frame-weighted mean error.
      Fit OLS (error ~ lighting) across all subject×lighting points of a model.
      Report slope (deg per 1 lighting point), Pearson r, p, n_subjects, Δ(10→100).
    """
    os.makedirs(outdir, exist_ok=True)
    df = load_and_prepare(csv_path)
    subj_df = _subject_level_video_means(df)

    rows = []
    for m, dmodel in subj_df.groupby("gaze_model"):
        if dmodel.empty:
            continue

        slope, intercept, r_val, p_val, stderr = stats.linregress(
            dmodel["lighting"].astype(float), dmodel["mean_error"].astype(float)
        )
        display = dmodel["gaze_model_display"].iloc[0]
        rows.append({
            "gaze_model": display,     # DISPLAY name in outputs
            "slope": float(slope),
            "r": float(r_val),
            "pval": float(p_val),
            "n_subjects": int(dmodel["subject"].nunique()),
            "delta_10_to_100": float(slope * (100 - 10))
        })

    out_df = pd.DataFrame(rows).sort_values("gaze_model").reset_index(drop=True)
    out_csv = os.path.join(outdir, "lighting_sensitivity_index.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Saved lighting sensitivity index to {out_csv}")
    return out_df


def compute_lighting_sensitivity_3levels(
    csv_path: Optional[str],
    outdir: str
) -> pd.DataFrame:
    """
    Same as compute_lighting_sensitivity but restricted to lighting ∈ {10,25,50}.
    Report Δ(10→50). Uses subject-level points.
    """
    os.makedirs(outdir, exist_ok=True)
    df = load_and_prepare(csv_path)
    df = df[df["lighting"].isin([10, 25, 50])].copy()

    subj_df = _subject_level_video_means(df)

    rows = []
    for m, dmodel in subj_df.groupby("gaze_model"):
        if dmodel.empty:
            continue

        slope, intercept, r_val, p_val, stderr = stats.linregress(
            dmodel["lighting"].astype(float), dmodel["mean_error"].astype(float)
        )
        display = dmodel["gaze_model_display"].iloc[0]
        rows.append({
            "gaze_model": display,     # DISPLAY name in outputs
            "slope": float(slope),
            "r": float(r_val),
            "pval": float(p_val),
            "n_subjects": int(dmodel["subject"].nunique()),
            "delta_10_to_50": float(slope * (50 - 10))
        })

    out_df = pd.DataFrame(rows).sort_values("gaze_model").reset_index(drop=True)
    out_csv = os.path.join(outdir, "lighting_sensitivity_index_10_25_50.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Saved 10/25/50 lighting sensitivity index to {out_csv}")
    return out_df

def run_pairwise_models_within_each_lighting(
    csv_path: Optional[str],
    outdir: str,
    padjust: str = "holm",
) -> Dict[int, pd.DataFrame]:
    """
    Pairwise model comparisons within each lighting level (10/25/50/100).
    - Collapses to subject-level (weighted by num_samples) via _subject_level_video_means.
    - For each lighting L, runs paired parametric tests across models:
        dv='mean_error', within='gaze_model_display', subject='subject'
      using pingouin.pairwise_tests with Holm p-adjust and Cohen's d.
    - Writes one CSV per lighting:
        pairwise_model_within_lighting_{L}.csv
      and a combined CSV:
        pairwise_model_within_lighting__all.csv
    - Returns a dict {lighting_level: DataFrame}.
    """
    os.makedirs(outdir, exist_ok=True)
    df = load_and_prepare(csv_path)
    subj_df = _subject_level_video_means(df)

    # Ensure types are clean for pingouin
    subj_df["gaze_model_display"] = subj_df["gaze_model_display"].astype(str)
    subj_df["subject"] = subj_df["subject"].astype(str)

    results_by_L: Dict[int, pd.DataFrame] = {}
    combined = []

    for L in sorted(subj_df["lighting"].unique()):
        dL = subj_df[subj_df["lighting"] == L].copy()
        if dL.empty:
            continue

        # Require at least two models present
        if dL["gaze_model_display"].nunique() < 2:
            continue

        # Paired parametric pairwise tests across models, within-subject
        res = pg.pairwise_tests(
            data=dL,
            dv="mean_error",
            within="gaze_model_display",
            subject="subject",
            parametric=True,
            padjust=padjust,
            effsize="cohen",
            return_desc=True,
        )

        # Add lighting column and tidy
        res.insert(0, "lighting", int(L))

        # Sort for readability
        if {"A", "B"}.issubset(res.columns):
            res.sort_values(["A", "B"], inplace=True)

        results_by_L[int(L)] = res
        combined.append(res)

        # Save per-lighting CSV
        per_csv = os.path.join(outdir, f"pairwise_model_within_lighting_{int(L)}.csv")
        res.to_csv(per_csv, index=False)

    if combined:
        all_df = pd.concat(combined, axis=0, ignore_index=True)
        all_df.sort_values(["lighting", "A", "B"], inplace=True)
        all_csv = os.path.join(outdir, "pairwise_model_within_lighting__all.csv")
        all_df.to_csv(all_csv, index=False)

    return results_by_L

def _pairwise_models_one_lighting_pingouin_schema(
    dL: pd.DataFrame,
    padjust: str = "holm",
) -> pd.DataFrame:
    """
    Pairwise *model* comparisons at a single lighting level (subject-fixed), in the
    exact Pingouin schema used elsewhere.

    Input dL columns (subject-level): 
      ['gaze_model','gaze_model_display','subject','lighting','mean_error']

    Returns Pingouin-style columns:
      Contrast, A, B, mean(A), std(A), mean(B), std(B), Paired, Parametric,
      T, dof, alternative, p-unc, p-corr, p-adjust, BF10, cohen
    plus two convenience columns up front:
      lighting, lighting_label
    """
    # Run pairwise tests across models (within-subjects factor = model)
    res = pg.pairwise_tests(
        data=dL,
        dv="mean_error",
        within="gaze_model",
        subject="subject",
        parametric=True,
        padjust=padjust,
        effsize="cohen",
        return_desc=True,
    )

    # Attach lighting info
    L = int(dL["lighting"].iloc[0])
    res.insert(0, "lighting", L)
    res.insert(1, "lighting_label", f"lighting_{L}")

    # Map A/B and Contrast to DISPLAY names for paper-readiness,
    # while keeping Pingouin's numeric/stat columns intact.
    def _to_disp(x: str) -> str:
        return config.display_model_name(x)

    res["A"] = res["A"].map(_to_disp)
    res["B"] = res["B"].map(_to_disp)
    # Rebuild 'Contrast' as "A - B" with display names
    if "Contrast" in res.columns:
        res["Contrast"] = res.apply(lambda r: f"{r['A']} - {r['B']}", axis=1)

    # Ensure canonical Pingouin columns exist (older versions naming fallback)
    rename_map = {
        "p-unc": "p-unc",
        "p-corr": "p-corr",
        "p-adjust": "p-adjust",
        "BF10": "BF10",
        "cohen": "cohen",
        "T": "T",
        "dof": "dof",
        "alternative": "alternative",
        "Paired": "Paired",
        "Parametric": "Parametric",
        "mean(A)": "mean(A)",
        "std(A)": "std(A)",
        "mean(B)": "mean(B)",
        "std(B)": "std(B)",
    }
    for c in rename_map:
        if c not in res.columns:
            res[c] = np.nan  # backstop if a column is missing on some versions

    # Order columns to match your within-model CSVs (with lighting fields first)
    ordered = [
        "lighting", "lighting_label",
        "Contrast", "A", "B", "mean(A)", "std(A)", "mean(B)", "std(B)",
        "Paired", "Parametric", "T", "dof", "alternative",
        "p-unc", "p-corr", "p-adjust", "BF10", "cohen",
    ]
    # Keep only available columns in that order
    ordered = [c for c in ordered if c in res.columns]
    res = res.loc[:, ordered].copy()

    # Sort for readability
    res.sort_values(by=["A", "B"], inplace=True, ignore_index=True)
    return res


def run_pairwise_models_within_lighting(
    csv_path: Optional[str],
    outdir: str,
    padjust: str = "holm",
) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """
    Produce Pingouin-style pairwise *model* comparisons at each fixed lighting level.

    Writes:
      - pairwise_model_within_lighting_10.csv
      - pairwise_model_within_lighting_25.csv
      - pairwise_model_within_lighting_50.csv
      - pairwise_model_within_lighting_100.csv
      - pairwise_model_within_lighting__all.csv  (stacked)

    All with columns:
      Contrast, A, B, mean(A), std(A), mean(B), std(B), Paired, Parametric,
      T, dof, alternative, p-unc, p-corr, p-adjust, BF10, cohen
    plus (lighting, lighting_label) up front.
    """
    os.makedirs(outdir, exist_ok=True)

    df = load_and_prepare(csv_path)
    subj_df = _subject_level_video_means(df)  # subject-level, frame-weighted

    combined_list: List[pd.DataFrame] = []
    per_L: Dict[int, pd.DataFrame] = {}

    for L, dL in subj_df.groupby("lighting"):
        if dL.empty:
            continue
        resL = _pairwise_models_one_lighting_pingouin_schema(dL, padjust=padjust)
        per_L[int(L)] = resL
        combined_list.append(resL)

        out_csv = os.path.join(outdir, f"pairwise_model_within_lighting_{int(L)}.csv")
        resL.to_csv(out_csv, index=False)

    if not combined_list:
        return pd.DataFrame(), per_L

    combined = pd.concat(combined_list, axis=0, ignore_index=True)
    combined.sort_values(["lighting", "A", "B"], inplace=True)
    combined.to_csv(os.path.join(outdir, "pairwise_model_within_lighting__all.csv"), index=False)

    print(f"Wrote model-within-lighting pairwise CSVs to {outdir}")
    return combined, per_L


if __name__ == "__main__":
    import config  # type: ignore
    default_csv = os.path.join(config.get_dataset_base_directory(), "gaze_evaluation_results.csv")
    default_out = os.path.join(config.get_dataset_base_directory(), "lighting_analysis")

    ap = argparse.ArgumentParser(description="Lighting analysis: errorbar and weighted boxplot charts from raw CSV.")
    ap.add_argument("--csv", type=str, default=default_csv,
                    help="Path to gaze_evaluation_results.csv. Defaults to config base if available.")
    ap.add_argument("--outdir", type=str, default=default_out,
                    help="Directory to write PNGs and CSV summaries.")
    ap.add_argument("--show", action="store_true", help="Show figures interactively.")
    ap.add_argument("--ymax", type=float, default=60.0,
                    help="Y-axis maximum for charts (use e.g. 60). Set to -1 for auto.")
    args = ap.parse_args()

    ymax = None if (args.ymax is not None and args.ymax < 0) else args.ymax
    models_for_multi_model_plot = config.get_currently_analyzed_models()
    analyze_lighting(csv_path=args.csv, outdir=args.outdir, show=args.show, ymax=ymax, models_for_multi_model_plot=models_for_multi_model_plot)