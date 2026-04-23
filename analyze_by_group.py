import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config
import numpy as np
import json
from frame_db import load_frame_dataset

BASE_DIR = config.get_dataset_base_directory()
CSV_PATH = os.path.join(BASE_DIR, "gaze_evaluation_results.csv")
AGGREGATE_BY = "video"  # "video" | "frame"
ANALYSIS_ROOT = os.path.join(BASE_DIR, "group_analysis", AGGREGATE_BY)
os.makedirs(ANALYSIS_ROOT, exist_ok=True)

# -------------------- common helpers --------------------
def save_subject_stats(df: pd.DataFrame, out_dir: str):
    subjects = df[["subject", "gender", "glasses"]].drop_duplicates("subject").copy()

    def _canon_gender(x):
        if pd.isna(x): return None
        s = str(x).strip().lower()
        if s in ("female","woman","women","f"): return "women"
        if s in ("male","man","men","m"): return "men"
        return None
    subjects["gender_canon"] = subjects["gender"].map(_canon_gender)

    def _canon_glasses(x):
        if pd.isna(x): return None
        if isinstance(x, (int, float)):
            return "has glasses" if x != 0 else "no glasses"
        s = str(x).strip().lower().replace("-", "_").replace(" ", "_")
        if s in ("true","1","yes","y","glasses","has_glasses","with_glasses","wearing_glasses","wears_glasses"):
            return "has glasses"
        if s in ("false","0","no","n","no_glasses","without_glasses","none"):
            return "no glasses"
        return None
    subjects["glasses_canon"] = subjects["glasses"].map(_canon_glasses)

    stats = {
        "total": int(len(subjects)),
        "women": int((subjects["gender_canon"] == "women").sum()),
        "men": int((subjects["gender_canon"] == "men").sum()),
        "has glasses": int((subjects["glasses_canon"] == "has glasses").sum()),
        "no glasses": int((subjects["glasses_canon"] == "no glasses").sum()),
        "women-has glasses": int(((subjects["gender_canon"] == "women") & (subjects["glasses_canon"] == "has glasses")).sum()),
        "women-no glasses": int(((subjects["gender_canon"] == "women") & (subjects["glasses_canon"] == "no glasses")).sum()),
        "men-has glasses": int(((subjects["gender_canon"] == "men") & (subjects["glasses_canon"] == "has glasses")).sum()),
        "men-no glasses": int(((subjects["gender_canon"] == "men") & (subjects["glasses_canon"] == "no glasses")).sum()),
    }

    out_path = os.path.join(out_dir, "subject_stats.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved subject stats to {out_path}")

def load_data(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    df["angular_error"] = df["error_mean"]
    df["experiment_type"] = df["exp_type"]
    df["subject"] = df["subject_dir"]
    df["num_samples"] = df["num_samples"]
    return df

def plot_subject_mean_error_frequency(df_subject_only: pd.DataFrame, output_dir: str, prefix: str = "", title_model: str | None = None):
    subj_mean = (
        df_subject_only.groupby("subject")[["angular_error", "num_samples"]]
        .apply(lambda d: np.average(d["angular_error"], weights=d["num_samples"]))
        .rename("mean_deg")
    )
    edges = list(np.arange(0, 61, 1)) + [np.inf]
    labels = [f"{a}" for a in range(0, 60)] + ["60+"]

    cats = pd.cut(subj_mean.values, bins=edges, labels=labels, right=False, include_lowest=True)
    counts = pd.Series(cats, index=subj_mean.index).value_counts().sort_index()

    plt.figure(figsize=(12, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("Mean angular error bin (deg)")
    plt.ylabel("# of subjects")
    plt.title("Subjects per 1° mean-error bin" + (f" — {title_model}" if title_model else ""))
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    out_png = os.path.join(output_dir, f"{prefix}angular_error_by_subject.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved subject mean-error frequency plot to {out_png}")

def plot_error_spatial_grids(df_point_only: pd.DataFrame, output_dir: str, prefix: str = "", title_model: str | None = None):
    agg = (
        df_point_only.groupby("point")[["angular_error", "num_samples"]]
        .apply(lambda d: np.average(d["angular_error"], weights=d["num_samples"]))
        .rename("mean_error")
    )
    p_grid_labels = [["p1","p2","p3"], ["p4","p5","p6"], ["p7","p8","p9"]]
    h_grid_labels = [["h1","h2","h3"], ["h4","h5","h6"]]

    p_mat = np.array([[agg.get(lbl, np.nan) for lbl in row] for row in p_grid_labels], dtype=float)
    h_mat = np.array([[agg.get(lbl, np.nan) for lbl in row] for row in h_grid_labels], dtype=float)
    p_mat = np.clip(p_mat, 0.0, 60.0); h_mat = np.clip(h_mat, 0.0, 60.0)
    vmin, vmax = 0.0, 60.0

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(p_mat, annot=True, fmt=".2f", vmin=vmin, vmax=vmax, cmap="viridis",
                     cbar=True, cbar_kws={"ticks": [0,10,20,30,40,50,60]},
                     linewidths=0.5, linecolor="white", square=True)
    ax.collections[0].colorbar.set_label("Mean Gaze Error (deg)")
    ax.set_xticks([0.5,1.5,2.5]); ax.set_xticklabels(["left","middle","right"])
    ax.set_yticks([0.5,1.5,2.5]); ax.set_yticklabels(["top","middle","bottom"], rotation=0)
    ax.set_title("Angular Error (deg) — p-grid" + (f" — {title_model}" if title_model else ""))
    ax.set_xlabel("column"); ax.set_ylabel("row")
    plt.tight_layout()
    out_p = os.path.join(output_dir, f"{prefix}spatial_error_grid_p_points.png")
    plt.savefig(out_p, dpi=200, bbox_inches="tight"); plt.close()

    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(h_mat, annot=True, fmt=".2f", vmin=vmin, vmax=vmax, cmap="viridis",
                     cbar=True, cbar_kws={"ticks": [0,10,20,30,40,50,60]},
                     linewidths=0.5, linecolor="white", square=True)
    ax.collections[0].colorbar.set_label("Mean Gaze Error (deg)")
    ax.set_xticks([0.5,1.5,2.5]); ax.set_xticklabels(["left","middle","right"])
    ax.set_yticks([0.5,1.5]); ax.set_yticklabels(["top","middle"], rotation=0)
    ax.set_title("Angular Error (deg) — h-grid" + (f" — {title_model}" if title_model else ""))
    ax.set_xlabel("column"); ax.set_ylabel("row")
    plt.tight_layout()
    out_h = os.path.join(output_dir, f"{prefix}spatial_error_grid_h_points.png")
    plt.savefig(out_h, dpi=200, bbox_inches="tight"); plt.close()

    print(f"Saved spatial grids: {out_p} | {out_h}")

def analyze_group(df, group_col, output_dir, prefix="", title_model: str | None = None):
    cols_needed = [group_col, "angular_error", "error_std", "num_samples"]
    df = df.loc[:, cols_needed].copy()
    df["_wx"]  = df["angular_error"] * df["num_samples"]
    df["_ws2"] = (df["error_std"] ** 2) * df["num_samples"]
    df["_wm2"] = (df["angular_error"] ** 2) * df["num_samples"]

    agg = df.groupby(group_col).agg(
        total_n=("num_samples","sum"),
        sum_wx=("_wx","sum"),
        sum_ws2=("_ws2","sum"),
        sum_wm2=("_wm2","sum"),
    )
    mean = agg["sum_wx"] / agg["total_n"]
    var  = (agg["sum_ws2"] + agg["sum_wm2"] - agg["total_n"] * (mean ** 2)) / agg["total_n"]
    std  = np.sqrt(var.clip(lower=0))

    group_stats = pd.DataFrame({"mean": mean, "std": std, "count": agg["total_n"]}).sort_index()
    os.makedirs(output_dir, exist_ok=True)
    group_stats.to_csv(os.path.join(output_dir, f"{prefix}angular_error_by_{group_col}.csv"))
    print(f"Saved stats for {group_col} to CSV in {output_dir}")

    order = None
    if group_col == "point":
        desired = [f"p{i}" for i in range(1,10)] + [f"h{i}" for i in range(1,7)] + ["horizontal","vertical"]
        present = set(df[group_col].unique())
        order = [x for x in desired if x in present]
    if group_col == "experiment_type":
        desired = config.get_experiment_types(order="plot_error_by_experiment_type")
        present = set(df[group_col].unique())
        order = [x for x in desired if x in present]

    if group_col == "subject":
        df_subj = df[["subject", "angular_error", "num_samples"]].copy()
        plot_subject_mean_error_frequency(df_subj, output_dir, prefix=prefix, title_model=title_model)
    else:
        plt.figure()
        sns.boxplot(data=df, x=group_col, y="angular_error", order=order)
        title_suffix = f" — {title_model}" if title_model else ""
        plt.title(f"Angular Error by {group_col} {prefix.strip('_')}{title_suffix}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}angular_error_by_{group_col}.png"))
        plt.close()

    if group_col == "point":
        df_for_grid = df[["point", "angular_error", "num_samples"]].copy()
        plot_error_spatial_grids(df_for_grid, output_dir, prefix=prefix, title_model=title_model)

    return group_stats

# -------------------- frame-mode helpers --------------------
def load_frame_db_as_analysis_df() -> pd.DataFrame:
    df = load_frame_dataset()
    if df.empty:
        return df
    df = df.copy()
    df["angular_error"] = df["error"].astype(float)
    df["experiment_type"] = df["exp_type"]
    df["subject"] = df["subject_dir"]
    df["num_samples"] = 1
    for col in ["gender", "glasses", "point", "gaze_model"]:
        if col not in df.columns:
            df[col] = None
    return df

def analyze_group_frame(df: pd.DataFrame, group_col: str, output_dir: str, prefix: str = "", title_model: str | None = None):
    cols_needed = [group_col, "angular_error", "num_samples"]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing cols in frame-mode df: {missing}")
    d = df.loc[:, cols_needed].copy()

    group_stats = (
        d.groupby(group_col)["angular_error"]
        .agg(mean="mean", std=lambda x: float(np.std(x, ddof=0)), count="size")
        .sort_index()
    )
    os.makedirs(output_dir, exist_ok=True)
    group_stats.to_csv(os.path.join(output_dir, f"{prefix}angular_error_by_{group_col}.csv"))
    print(f"[frame] Saved stats for {group_col} to CSV in {output_dir}")

    order = None
    if group_col == "point":
        desired = [f"p{i}" for i in range(1,10)] + [f"h{i}" for i in range(1,7)] + ["horizontal","vertical"]
        present = set(d[group_col].dropna().unique())
        order = [x for x in desired if x in present]

    if group_col == "subject":
        df_subj = df[["subject", "angular_error", "num_samples"]].copy()
        plot_subject_mean_error_frequency(df_subj, output_dir, prefix=prefix, title_model=title_model)
    else:
        plt.figure()
        sns.boxplot(data=df, x=group_col, y="angular_error", order=order)
        title_suffix = f" — {title_model}" if title_model else ""
        plt.title(f"[frame] Angular Error by {group_col} {prefix.strip('_')}{title_suffix}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}angular_error_by_{group_col}.png"))
        plt.close()

    if group_col == "point":
        df_for_grid = df[["point", "angular_error", "num_samples"]].copy()
        plot_error_spatial_grids(df_for_grid, output_dir, prefix=prefix, title_model=title_model)

    return group_stats

# -------------------- main --------------------
def run_analysis():
    if AGGREGATE_BY == "video":
        df = load_data()
    elif AGGREGATE_BY == "frame":
        df = load_frame_db_as_analysis_df()
        if df.empty:
            print("[frame] Frame DB is empty; nothing to analyze.")
            return
    else:
        raise ValueError("AGGREGATE_BY must be 'video' or 'frame'")

    analysis_groups = ["glasses", "gender", "experiment_type", "subject", "point"]
    valid_points = set(list(sum(config.get_point_variations().values(), [])) + config.get_line_movement_types())
    exp_order = config.get_experiment_types(order="plot_error_by_experiment_type")

    comparison_rows = []
    gender_glasses_rows = []
    point_order = [f"p{i}" for i in range(1,10)] + [f"h{i}" for i in range(1,7)] + ["horizontal","vertical"]
    point_tables = {"ALL": []}
    for et in exp_order:
        point_tables[et] = []

    models_in_df = list(df["gaze_model"].dropna().unique()) if "gaze_model" in df.columns else []
    # Keep "All Models" row in BOTH modes for symmetry
    for model in models_in_df + ["all"]:
        if model == "all":
            df_model = df.copy()
            model_disp = "All Models"
        else:
            df_model = df[df["gaze_model"] == model].copy()
            model_disp = config.display_model_name(model)  # use display name in BOTH modes

        if "point" in df_model.columns:
            df_model = df_model[df_model["point"].isin(valid_points)]

        output_dir = os.path.join(ANALYSIS_ROOT, model)
        os.makedirs(output_dir, exist_ok=True)

        # Per-group plots/tables
        for group in analysis_groups:
            try:
                if AGGREGATE_BY == "video":
                    analyze_group(df_model, group, output_dir, title_model=model_disp)
                else:
                    analyze_group_frame(df_model, group, output_dir, title_model=model_disp)
            except Exception as e:
                print(f"Failed to analyze group {group} for model {model}: {e}")

        # ---- NEW/RESTORED: nested per-experiment, then point (inside each model dir) ----
        nested_output_dir = os.path.join(output_dir, "group_by_experiment_first_then_point")
        os.makedirs(nested_output_dir, exist_ok=True)

        for exp_type in config.get_point_variations().keys():
            if exp_type == "horizontal_movement":
                continue
            df_exp = df_model[df_model["experiment_type"] == exp_type]
            if df_exp.empty:
                continue
            try:
                if AGGREGATE_BY == "video":
                    analyze_group(df_exp, "point", nested_output_dir, prefix=f"{exp_type}_", title_model=model_disp)
                else:
                    analyze_group_frame(df_exp, "point", nested_output_dir, prefix=f"{exp_type}_", title_model=model_disp)
            except Exception as e:
                print(f"Failed nested analysis for {exp_type} in model {model}: {e}")

        # ---- Comparison tables & point tables (same filenames across modes) ----
        if AGGREGATE_BY == "video":
            # original weighted pooled stats (unchanged from working version)
            agg = (
                df_model.groupby("experiment_type")[["angular_error", "error_std", "num_samples"]]
                .apply(lambda d: pd.Series({
                    "total_n": d["num_samples"].sum(),
                    "sum_wx": (d["angular_error"] * d["num_samples"]).sum(),
                    "sum_ws2": ((d["error_std"] ** 2) * d["num_samples"]).sum(),
                    "sum_wm2": ((d["angular_error"] ** 2) * d["num_samples"]).sum(),
                }))
            )
            row = {"gaze_model": model_disp}
            for et in exp_order:
                if et in agg.index:
                    total_n = agg.loc[et, "total_n"]
                    mean = agg.loc[et, "sum_wx"] / total_n
                    var  = (agg.loc[et, "sum_ws2"] + agg.loc[et, "sum_wm2"] - total_n * (mean ** 2)) / total_n
                    std  = np.sqrt(max(var, 0))
                    row[et] = f"{mean:.2f}, {std:.2f}"
            comparison_rows.append(row)

            # gender/glasses (weighted)
            row_gg = {"gaze_model": model_disp}
            def _canon_gender(x):
                s = str(x).strip().lower()
                if s in ("female","woman","women","f"): return "women"
                if s in ("male","man","men","m"): return "men"
                return None
            def _canon_glasses(x):
                if pd.isna(x): return None
                if isinstance(x, (int, float)): return "has glasses" if x != 0 else "no glasses"
                s = str(x).strip().lower().replace("-", "_").replace(" ", "_")
                if s in ("true","1","yes","y","glasses","has_glasses","with_glasses","wearing_glasses","wears_glasses"): return "has glasses"
                if s in ("false","0","no","n","no_glasses","without_glasses","none"): return "no glasses"
                return None
            if "gender" in df_model.columns:
                df_model["__gender_canon"] = df_model["gender"].map(_canon_gender)
            else:
                df_model["__gender_canon"] = None
            if "glasses" in df_model.columns:
                df_model["__glasses_canon"] = df_model["glasses"].map(_canon_glasses)
            else:
                df_model["__glasses_canon"] = None

            def _weighted_stats(dsub: pd.DataFrame):
                if dsub.empty: return None
                total_n = dsub["num_samples"].sum()
                if total_n <= 0: return None
                sum_wx  = (dsub["angular_error"] * dsub["num_samples"]).sum()
                sum_ws2 = ((dsub["error_std"] ** 2) * dsub["num_samples"]).sum()
                sum_wm2 = ((dsub["angular_error"] ** 2) * dsub["num_samples"]).sum()
                mean_val = sum_wx / total_n
                var_val  = (sum_ws2 + sum_wm2 - total_n * (mean_val ** 2)) / total_n
                std_val  = float(np.sqrt(max(var_val, 0)))
                return f"{mean_val:.2f}, {std_val:.2f}"

            for lbl in ("women","men"):
                res = _weighted_stats(df_model[df_model["__gender_canon"] == lbl])
                if res is not None:
                    row_gg[lbl] = res
            for lbl in ("has glasses","no glasses"):
                res = _weighted_stats(df_model[df_model["__glasses_canon"] == lbl])
                if res is not None:
                    row_gg[lbl] = res
            gender_glasses_rows.append(row_gg)

            # point tables (weighted)
            agg_pt_all = (
                df_model.groupby("point")[["angular_error", "error_std", "num_samples"]]
                .apply(lambda d: pd.Series({
                    "total_n": d["num_samples"].sum(),
                    "sum_wx": (d["angular_error"] * d["num_samples"]).sum(),
                    "sum_ws2": ((d["error_std"] ** 2) * d["num_samples"]).sum(),
                    "sum_wm2": ((d["angular_error"] ** 2) * d["num_samples"]).sum(),
                }))
            )
            row_all_pts = {"gaze_model": model_disp}
            for pt in point_order:
                if pt in agg_pt_all.index:
                    n = agg_pt_all.loc[pt, "total_n"]
                    mean = agg_pt_all.loc[pt, "sum_wx"] / n
                    var  = (agg_pt_all.loc[pt, "sum_ws2"] + agg_pt_all.loc[pt, "sum_wm2"] - n * (mean ** 2)) / n
                    std  = float(np.sqrt(max(var, 0)))
                    row_all_pts[pt] = f"{mean:.2f}, {std:.2f}"
            point_tables["ALL"].append(row_all_pts)

            for et in exp_order:
                df_e = df_model[df_model["experiment_type"] == et]
                if df_e.empty:
                    point_tables[et].append({"gaze_model": model_disp})
                    continue
                agg_pt = (
                    df_e.groupby("point")[["angular_error", "error_std", "num_samples"]]
                    .apply(lambda d: pd.Series({
                        "total_n": d["num_samples"].sum(),
                        "sum_wx": (d["angular_error"] * d["num_samples"]).sum(),
                        "sum_ws2": ((d["error_std"] ** 2) * d["num_samples"]).sum(),
                        "sum_wm2": ((d["angular_error"] ** 2) * d["num_samples"]).sum(),
                    }))
                )
                row_et = {"gaze_model": model_disp}
                for pt in point_order:
                    if pt in agg_pt.index:
                        n = agg_pt.loc[pt, "total_n"]
                        mean = agg_pt.loc[pt, "sum_wx"] / n
                        var  = (agg_pt.loc[pt, "sum_ws2"] + agg_pt.loc[pt, "sum_wm2"] - n * (mean ** 2)) / n
                        std  = float(np.sqrt(max(var, 0)))
                        row_et[pt] = f"{mean:.2f}, {std:.2f}"
                point_tables[et].append(row_et)

        else:
            # frame-mode simple stats (per-frame, weight=1)
            if "experiment_type" in df_model.columns and not df_model.empty:
                agg_et = (
                    df_model.groupby("experiment_type")["angular_error"]
                    .agg(mean="mean", std=lambda x: float(np.std(x, ddof=0)), count="size")
                )
                row = {"gaze_model": model_disp}
                for et in exp_order:
                    if et in agg_et.index:
                        row[et] = f"{agg_et.loc[et,'mean']:.2f}, {agg_et.loc[et,'std']:.2f}"
                comparison_rows.append(row)

            row_gg = {"gaze_model": model_disp}
            if "gender" in df_model.columns:
                def _canon_gender(x):
                    if pd.isna(x): return None
                    s = str(x).strip().lower()
                    if s in ("female","woman","women","f"): return "women"
                    if s in ("male","man","men","m"): return "men"
                    return None
                df_model["__gender_canon"] = df_model["gender"].map(_canon_gender)
                for lbl in ("women","men"):
                    sub = df_model[df_model["__gender_canon"] == lbl]
                    if not sub.empty:
                        row_gg[lbl] = f"{sub['angular_error'].mean():.2f}, {float(np.std(sub['angular_error'].values, ddof=0)):.2f}"

            if "glasses" in df_model.columns:
                def _canon_glasses(x):
                    if pd.isna(x): return None
                    if isinstance(x, (int,float)): return "has glasses" if x != 0 else "no glasses"
                    s = str(x).strip().lower().replace("-","_").replace(" ","_")
                    if s in ("true","1","yes","y","glasses","has_glasses","with_glasses","wearing_glasses","wears_glasses"): return "has glasses"
                    if s in ("false","0","no","n","no_glasses","without_glasses","none"): return "no glasses"
                    return None
                df_model["__glasses_canon"] = df_model["glasses"].map(_canon_glasses)
                for lbl in ("has glasses","no glasses"):
                    sub = df_model[df_model["__glasses_canon"] == lbl]
                    if not sub.empty:
                        row_gg[lbl] = f"{sub['angular_error'].mean():.2f}, {float(np.std(sub['angular_error'].values, ddof=0)):.2f}"
            gender_glasses_rows.append(row_gg)

            if "point" in df_model.columns and not df_model.empty:
                agg_pt_all = (
                    df_model.groupby("point")["angular_error"]
                    .agg(mean="mean", std=lambda x: float(np.std(x, ddof=0)), count="size")
                )
                row_all_pts = {"gaze_model": model_disp}
                for pt in point_order:
                    if pt in agg_pt_all.index:
                        row_all_pts[pt] = f"{agg_pt_all.loc[pt,'mean']:.2f}, {agg_pt_all.loc[pt,'std']:.2f}"
                point_tables["ALL"].append(row_all_pts)

                for et in exp_order:
                    df_e = df_model[df_model["experiment_type"] == et]
                    if df_e.empty:
                        point_tables[et].append({"gaze_model": model_disp})
                        continue
                    agg_pt = (
                        df_e.groupby("point")["angular_error"]
                        .agg(mean="mean", std=lambda x: float(np.std(x, ddof=0)), count="size")
                    )
                    row_et = {"gaze_model": model_disp}
                    for pt in point_order:
                        if pt in agg_pt.index:
                            mean_val = agg_pt.loc[pt, "mean"]
                            std_val  = agg_pt.loc[pt, "std"]
                            row_et[pt] = f"{mean_val:.2f}, {std_val:.2f}"
                    point_tables[et].append(row_et)

    # ---- write comparison/point tables (filenames WITHOUT __frame) ----
    comp_dir = os.path.join(ANALYSIS_ROOT, "comparison_tables")
    os.makedirs(comp_dir, exist_ok=True)

    comp_df = pd.DataFrame(comparison_rows)
    if not comp_df.empty:
        cols = ["gaze_model"] + [et for et in exp_order if et in comp_df.columns]
        comp_df = comp_df.reindex(columns=cols)
        out_path = os.path.join(comp_dir, "model_vs_experiment_type_mean_error.csv")
        comp_df.to_csv(out_path, index=False)
        print(f"Saved model vs experiment-type mean-error table to {out_path}")

    gg_df = pd.DataFrame(gender_glasses_rows)
    if not gg_df.empty:
        cols_gg = ["gaze_model", "women", "men", "has glasses", "no glasses"]
        gg_df = gg_df.reindex(columns=cols_gg)
        out_path_gg = os.path.join(comp_dir, "model_vs_gender_glasses_mean_error.csv")
        gg_df.to_csv(out_path_gg, index=False)
        print(f"Saved model vs gender+glasses mean-error table to {out_path_gg}")

    df_all_pts = pd.DataFrame(point_tables["ALL"])
    if not df_all_pts.empty:
        cols_all = ["gaze_model"] + [pt for pt in point_order if pt in df_all_pts.columns]
        df_all_pts = df_all_pts.reindex(columns=cols_all)
        out_all = os.path.join(comp_dir, "model_vs_point_mean_error.csv")
        df_all_pts.to_csv(out_all, index=False)
        print(f"Saved overall model vs point table to {out_all}")

    for et in exp_order:
        df_et = pd.DataFrame(point_tables[et])
        if df_et.empty:
            continue
        cols_et = ["gaze_model"] + [pt for pt in point_order if pt in df_et.columns]
        df_et = df_et.reindex(columns=cols_et)
        out_et = os.path.join(comp_dir, f"model_vs_point_mean_error__{et}.csv")
        df_et.to_csv(out_et, index=False)
        print(f"Saved per-experiment model vs point table to {out_et}")

    # SUBJECT STATS: only under BASE_DIR (aggregation-agnostic)
    save_subject_stats(df, BASE_DIR)

if __name__ == "__main__":
    run_analysis()
