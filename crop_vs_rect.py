# crop_vs_rect.py
"""
Crop vs. Rectified (per-model) — subject-level paired tests over the full dataset.

- Uses only gaze_model KEYS in the CSV to infer:
    variant ∈ {"crop","rectified"} and base_key (key with 'rectification' removed).
- Collapses to SUBJECT-LEVEL frame-weighted means per (base_key, variant).
- Runs paired two-sided t-test (crop vs rectified) and reports Cohen's d (paired).
- Writes ONE CSV: BASE_DIR/crop_vs_rect/full_dataset_results.csv

Output columns
--------------
base_key                : canonical key-style base id (from CSV keys)
base_model_display      : pretty base display name (derived at the END from your mapping)
n_subjects              : subjects with BOTH variants
mean_crop, sd_crop
mean_rectified, sd_rectified
mean_diff, sd_diff      : diff = crop - rectified (negative => crop better)
t, dof, p_val
ci95%_low, ci95%_high
cohen_d_paired
"""

import os
import re
import pandas as pd
import numpy as np
import pingouin as pg
import config
from scipy import stats

# --------------------------- Helpers ---------------------------

def _load_results_csv() -> pd.DataFrame:
    base = config.get_dataset_base_directory()
    csv_path = os.path.join(base, "gaze_evaluation_results.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    # Required columns
    if "gaze_model" not in df.columns or "error_mean" not in df.columns:
        raise ValueError("CSV must contain 'gaze_model' and 'error_mean'.")
    if "subject" not in df.columns:
        if "subject_dir" in df.columns:
            df["subject"] = df["subject_dir"]
        else:
            raise ValueError("CSV must have 'subject' or 'subject_dir'.")

    if "num_samples" not in df.columns:
        df["num_samples"] = 1

    df["angular_error"] = df["error_mean"]
    return df


_rect_token = re.compile(r"(?i)rectification")

def _infer_variant_and_base_key_from_key(key: str) -> tuple[str, str]:
    """
    Groups models by removing both 'rectification' and 'unrectified' tags.
    """
    s = str(key)
    variant = "rectified" if "rectification" in s.lower() else "crop"

    # Remove both tokens and tidy underscores to get the base architecture ID
    base = re.sub(r'[_-]*(rectification|unrectified)[_-]*', '_', s, flags=re.IGNORECASE)
    base = re.sub(r'__+', '_', base).strip('_-')
    
    return variant, base

def _build_base_display_map_from_config() -> dict:
    """
    Build a mapping: base_key -> base_model_display (pretty),
    derived ONLY at the end from config.get_model_display_names().
    """
    disp_map = config.get_model_display_names()  # {csv_key: "Base w/ Crop"/"Base w/ Rect."}

    base_display = {}
    for csv_key, pretty in disp_map.items():
        # Derive base_key from csv_key the same way we do for the CSV rows
        _, base_key = _infer_variant_and_base_key_from_key(csv_key)

        # Extract pretty base name by stripping ' w/ ...'
        pretty_str = str(pretty)
        if " w/ " in pretty_str:
            base_pretty = pretty_str.split(" w/ ")[0].strip()
        else:
            # Fallback if someone added a different format
            base_pretty = pretty_str.strip()

        # Prefer first occurrence; keep consistent
        if base_key not in base_display:
            base_display[base_key] = base_pretty

    return base_display


def _subject_level_weighted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to subject-level frame-weighted mean per (base_key, variant, subject).
    """
    d = df.loc[:, ["base_key", "variant", "subject", "angular_error", "num_samples"]].copy()
    d["_wx"] = d["angular_error"] * d["num_samples"]
    g = (
        d.groupby(["base_key", "variant", "subject"], as_index=False)
         .agg(total_w=("num_samples", "sum"), sum_wx=("_wx", "sum"))
    )
    g["mean_error"] = g["sum_wx"] / g["total_w"]
    return g.loc[:, ["base_key", "variant", "subject", "mean_error"]]


def _paired_one_model(dsub: pd.DataFrame) -> pd.DataFrame:
    """
    Paired test for one base_key. Returns a single-row DataFrame.
    Includes Mean, SD, and 95% CI for both Crop and Rectified variants.
    """
    piv = dsub.pivot_table(index="subject", columns="variant", values="mean_error", aggfunc="mean")

    # Require both variants
    if not {"crop", "rectified"}.issubset(piv.columns):
        return pd.DataFrame()

    piv = piv.dropna(subset=["crop", "rectified"], how="any")
    n = len(piv)
    base_key = dsub["base_key"].iloc[0]

    # Case for insufficient data
    if n < 2:
        return pd.DataFrame([{
            "base_key": base_key,
            "n_subjects": n,
            "mean_crop": piv["crop"].mean() if "crop" in piv else np.nan,
            "sd_crop": piv["crop"].std(ddof=1) if n > 1 else np.nan,
            "ci95_low_crop": np.nan,
            "ci95_high_crop": np.nan,
            "mean_rectified": piv["rectified"].mean() if "rectified" in piv else np.nan,
            "sd_rectified": piv["rectified"].std(ddof=1) if n > 1 else np.nan,
            "ci95_low_rect": np.nan,
            "ci95_high_rect": np.nan,
            "mean_diff": np.nan,
            "sd_diff": np.nan,
            "t": np.nan,
            "dof": np.nan,
            "p_val": np.nan,
            "ci95%_low": np.nan,   # This is the CI of the difference
            "ci95%_high": np.nan,  # This is the CI of the difference
            "cohen_d_paired": np.nan,
        }])

    # Extract arrays
    crop = piv["crop"].to_numpy()
    rect = piv["rectified"].to_numpy()
    diff = crop - rect  # negative => crop better (lower error)

    # 1. Standard Descriptive Stats
    m_crop, sd_crop = float(np.mean(crop)), float(np.std(crop, ddof=1))
    m_rect, sd_rect = float(np.mean(rect)), float(np.std(rect, ddof=1))

    # 2. Calculate 95% CIs for individual variants (t-distribution)
    t_crit = stats.t.ppf(0.975, n - 1)
    
    se_crop = sd_crop / np.sqrt(n)
    ci_low_crop = m_crop - t_crit * se_crop
    ci_high_crop = m_crop + t_crit * se_crop

    se_rect = sd_rect / np.sqrt(n)
    ci_low_rect = m_rect - t_crit * se_rect
    ci_high_rect = m_rect + t_crit * se_rect

    # 3. Paired Comparisons (Pingouin)
    # Note: pg.ttest returns the CI of the DIFFERENCE [crop - rect]
    t_res = pg.ttest(crop, rect, paired=True, alternative="two-sided").iloc[0]
    d_eff = pg.compute_effsize(crop, rect, paired=True, eftype="cohen")

    return pd.DataFrame([{
        "base_key": base_key,
        "n_subjects": n,
        # Crop Stats
        "mean_crop": m_crop,
        "sd_crop": sd_crop,
        "ci95_low_crop": ci_low_crop,
        "ci95_high_crop": ci_high_crop,
        # Rectified Stats
        "mean_rectified": m_rect,
        "sd_rectified": sd_rect,
        "ci95_low_rect": ci_low_rect,
        "ci95_high_rect": ci_high_rect,
        # Difference Stats
        "mean_diff": float(np.mean(diff)),
        "sd_diff": float(np.std(diff, ddof=1)),
        "t": float(t_res["T"]),
        "dof": int(t_res["dof"]),
        "p_val": float(t_res["p-val"]),
        "ci95%_low": float(t_res["CI95%"][0]),  # CI of the Difference
        "ci95%_high": float(t_res["CI95%"][1]), # CI of the Difference
        "cohen_d_paired": float(d_eff),
    }])

# --------------------------- Main ---------------------------

def run():
    # 1) Load CSV
    df = _load_results_csv()
    
    included_models = [
        "puregaze", "puregaze_rectification_unrectified",
        "gazetr", "gazetr_rectification_unrectified",
        "puregaze_gaze360", "puregaze_gaze360_rectification_unrectified",
        "gazetr_gaze360", "gazetr_gaze360_rectification_unrectified",
        "l2cs_padding0fixed_isrgb_False_resize_448", 
        "l2cs_padding0fixed_rectification_isrgb_False_resize_448_unrectified",
        "mcgaze_clip_size_7", "mcgaze_clip_size_7_rectification_unrectified",
        "gaze3d_clip_len_8", "gaze3d_clip_len_8_rectification_unrectified"
    ]
    df = df[df["gaze_model"].isin(included_models)].copy()
    variants, base_keys = zip(*df["gaze_model"].map(_infer_variant_and_base_key_from_key))
    df["variant"] = variants
    df["base_key"] = base_keys

    # 2) Subject-level frame-weighted means
    subj = _subject_level_weighted(df)

    # 3) Paired tests per base_key (only if both variants exist)
    results = []
    for base_key, d in subj.groupby("base_key", sort=True):
        if set(d["variant"].unique()) >= {"crop", "rectified"}:
            res = _paired_one_model(d)
            if not res.empty:
                results.append(res)

    out_df = (pd.concat(results, axis=0, ignore_index=True)
              if results else
              pd.DataFrame(columns=[
                  "base_key","n_subjects",
                  "mean_crop","sd_crop","mean_rectified","sd_rectified",
                  "mean_diff","sd_diff","t","dof","p_val","ci95%_low","ci95%_high","cohen_d_paired"
              ]))

    # 4) Attach pretty base display names ONLY at the end
    base_display_map = _build_base_display_map_from_config()
    out_df["base_model_display"] = out_df["base_key"].map(base_display_map).fillna(out_df["base_key"])

    if not out_df.empty:
        # keep raw p for transparency
        out_df.rename(columns={"p_val": "p_val_raw"}, inplace=True)

        # Holm-Bonferroni across all models' tests
        reject, padj = pg.multicomp(out_df["p_val_raw"].values, method="holm")
        # write adjusted p and decision; also set p_val = adjusted
        out_df["p_val_holm"] = padj
        out_df["reject_holm_0.05"] = reject.astype(bool)
        out_df["p_val"] = out_df["p_val_holm"]
    else:
        out_df["p_val_raw"] = []
        out_df["p_val_holm"] = []
        out_df["reject_holm_0.05"] = []
        out_df["p_val"] = []

    # 5) Save one CSV
    base_dir = config.get_dataset_base_directory()
    outdir = os.path.join(base_dir, "crop_vs_rect")
    os.makedirs(outdir, exist_ok=True)
    out_csv = os.path.join(outdir, "full_dataset_results.csv")

    # Order nicely by known display names if available
    desired_order = ["PureGaze", "GazeTR", "L2CS-Net", "MCGaze", "GaT"]
    out_df["__ord"] = out_df["base_model_display"].apply(lambda x: desired_order.index(x) if x in desired_order else 999)
    out_df.sort_values(["__ord","base_model_display"], inplace=True, ignore_index=True)
    out_df.drop(columns="__ord", inplace=True)

    # Column order (now includes Holm columns)
    cols = [
        "base_model_display", "base_key", "n_subjects",
        "mean_crop","ci95_low_crop","ci95_high_crop","sd_crop",
        "mean_rectified","ci95_low_rect","ci95_high_rect","sd_rectified",
        "mean_diff","sd_diff","t","dof",
        "p_val_raw","p_val_holm","reject_holm_0.05",  # new/adjusted
        "p_val",  # duplicate of p_val_holm for convenience
        "ci95%_low","ci95%_high","cohen_d_paired",
    ]
    out_df = out_df.loc[:, cols]

    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    return out_df


if __name__ == "__main__":
    run()
