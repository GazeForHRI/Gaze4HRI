import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config
import numpy as np

BASE_DIR = config.get_dataset_base_directory()
CSV_PATH = os.path.join(BASE_DIR, "blink_evaluation_results.csv")
ANALYSIS_ROOT = os.path.join(BASE_DIR, "blink_group_analysis")
os.makedirs(ANALYSIS_ROOT, exist_ok=True)

def safe_calc_metrics(tp, fp, fn, tn):
    """Calculates all requested binary classification and IoU metrics safely."""
    
    # Standard Classification Metrics
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    
    # Jaccard / IoU Metrics
    iou_blink = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    iou_non_blink = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0.0
    iou_balanced = (iou_blink + iou_non_blink) / 2.0
    
    return pd.Series({
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "IoU_balanced": iou_balanced,
        "IoU_blink": iou_blink,
        "IoU_non_blink": iou_non_blink
    })

def load_data():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()
    df = pd.read_csv(CSV_PATH)
    
    def _canon_gender(x):
        s = str(x).strip().lower()
        if s in ("female", "woman", "women", "f"): return "women"
        if s in ("male", "man", "men", "m"): return "men"
        return "unknown"

    def _canon_glasses(x):
        s = str(x).strip().lower()
        if s in ("1", "1.0", "true", "yes", "glasses", "has glasses"): return "has glasses"
        return "no glasses"

    if "gender" in df.columns:
        df["gender_canon"] = df["gender"].map(_canon_gender)
    if "glasses" in df.columns:
        df["glasses_canon"] = df["glasses"].map(_canon_glasses)

    # NEW: Add Exp Type / Point combination column
    if "exp_type" in df.columns and "point" in df.columns:
        df["exp_type_point"] = df["exp_type"].astype(str) + " / " + df["point"].astype(str)
    
    return df

def plot_metric_spatial_grids(df_point_only, output_dir, prefix="", title_model=None):
    if df_point_only.empty:
        return

    agg = df_point_only.groupby("point")[["tp", "fp", "fn", "tn"]].sum()
    metrics = agg.apply(lambda r: safe_calc_metrics(r["tp"], r["fp"], r["fn"], r["tn"]), axis=1, result_type='expand')
    
    p_grid_labels = [["p1","p2","p3"], ["p4","p5","p6"], ["p7","p8","p9"]]
    h_grid_labels = [["h1","h2","h3"], ["h4","h5","h6"]]

    p_mat = np.array([[metrics["F1"].get(lbl, np.nan) for lbl in row] for row in p_grid_labels], dtype=float)
    h_mat = np.array([[metrics["F1"].get(lbl, np.nan) for lbl in row] for row in h_grid_labels], dtype=float)

    # Plot P-Grid
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(p_mat, annot=True, fmt=".3f", vmin=0.0, vmax=1.0, cmap="viridis",
                     cbar=True, linewidths=0.5, linecolor="white", square=True)
    ax.collections[0].colorbar.set_label("F1 Score")
    ax.set_xticks([0.5, 1.5, 2.5]); ax.set_xticklabels(["left", "middle", "right"])
    ax.set_yticks([0.5, 1.5, 2.5]); ax.set_yticklabels(["top", "middle", "bottom"], rotation=0)
    ax.set_title("F1 Score — p-grid" + (f" — {title_model}" if title_model else ""))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}spatial_f1_grid_p_points.png"), dpi=200); plt.close()

    # Plot H-Grid
    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(h_mat, annot=True, fmt=".3f", vmin=0.0, vmax=1.0, cmap="viridis",
                     cbar=True, linewidths=0.5, linecolor="white", square=True)
    ax.collections[0].colorbar.set_label("F1 Score")
    ax.set_xticks([0.5, 1.5, 2.5]); ax.set_xticklabels(["left", "middle", "right"])
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["top", "middle"], rotation=0)
    ax.set_title("F1 Score — h-grid" + (f" — {title_model}" if title_model else ""))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}spatial_f1_grid_h_points.png"), dpi=200); plt.close()

def analyze_group(df, group_col, output_dir, prefix="", title_model=None):
    if df.empty:
        return pd.DataFrame()

    # Aggregate raw counts
    agg = df.groupby(group_col)[["tp", "fp", "fn", "tn", "num_frames"]].sum()
    metrics = agg.apply(lambda r: safe_calc_metrics(r["tp"], r["fp"], r["fn"], r["tn"]), axis=1, result_type='expand')
    
    group_stats = pd.concat([agg, metrics], axis=1).sort_index()
    os.makedirs(output_dir, exist_ok=True)
    group_stats.to_csv(os.path.join(output_dir, f"{prefix}metrics_by_{group_col}.csv"))

    if group_col == "point":
        plot_metric_spatial_grids(df, output_dir, prefix=prefix, title_model=title_model)
    elif group_col not in ["subject_dir", "exp_type_point"]:
        df_plot = df.copy()
        res_metrics = df_plot.apply(lambda r: safe_calc_metrics(r["tp"], r["fp"], r["fn"], r["tn"]), axis=1, result_type='expand')
        df_plot[res_metrics.columns] = res_metrics
        
        plt.figure()
        sns.boxplot(data=df_plot, x=group_col, y="F1")
        plt.title(f"F1 Score by {group_col} {prefix.strip('_')}" + (f" — {title_model}" if title_model else ""))
        plt.xticks(rotation=45)
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}f1_by_{group_col}.png"))
        plt.close()

    return group_stats

def run_analysis():
    df = load_data()
    if df.empty:
        print(f"Results CSV not found or empty at {CSV_PATH}. Run blink_structured_results.py first.")
        return
        
    models = df["blink_model"].unique()
    # ADDED: exp_type_point to the analysis groups
    analysis_groups = ["glasses_canon", "gender_canon", "exp_type", "subject_dir", "point", "exp_type_point"]
    valid_points = set(list(sum(config.get_point_variations().values(), [])) + config.get_line_movement_types())

    comparison_rows = []
    comparison_rows_comb = [] # NEW: Table for combination
    global_rows = []

    for model in models:
        df_model = df[df["blink_model"] == model].copy()
        if "point" in df_model.columns:
            df_model = df_model[df_model["point"].isin(valid_points) | (df_model["point"].isna())]
        
        output_dir = os.path.join(ANALYSIS_ROOT, model)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Analyzing model: {model}")

        # 1. Standard Group Analysis
        for group in analysis_groups:
            if group in df_model.columns:
                analyze_group(df_model, group, output_dir, title_model=model)

        # 2. Nested Point Analysis
        nested_output_dir = os.path.join(output_dir, "point_grids_by_experiment")
        for exp_type in df_model["exp_type"].unique():
            df_exp = df_model[df_model["exp_type"] == exp_type]
            analyze_group(df_exp, "point", nested_output_dir, prefix=f"{exp_type}_", title_model=model)

        # 3. Global Dataset-Wide Metrics for this model
        agg_global = df_model[["tp", "fp", "fn", "tn"]].sum()
        metrics_global = safe_calc_metrics(agg_global["tp"], agg_global["fp"], agg_global["fn"], agg_global["tn"])
        
        g_row = {"blink_model": model}
        for metric, val in metrics_global.items():
            g_row[metric] = round(val, 4)
        global_rows.append(g_row)

        # 4. Long Format Comparison Tables
        # View A: By Exp Type
        agg_et = df_model.groupby("exp_type")[["tp", "fp", "fn", "tn"]].sum()
        metrics_et = agg_et.apply(lambda r: safe_calc_metrics(r["tp"], r["fp"], r["fn"], r["tn"]), axis=1, result_type='expand')
        
        for et in metrics_et.index:
            row = {"blink_model": model, "exp_type": et}
            for col in metrics_et.columns:
                row[col] = round(metrics_et.loc[et, col], 4)
            comparison_rows.append(row)

        # View B: By Exp Type / Point Combination (NEW)
        agg_comb = df_model.groupby("exp_type_point")[["tp", "fp", "fn", "tn"]].sum()
        metrics_comb = agg_comb.apply(lambda r: safe_calc_metrics(r["tp"], r["fp"], r["fn"], r["tn"]), axis=1, result_type='expand')

        for comb in metrics_comb.index:
            row = {"blink_model": model, "exp_type_point": comb}
            for col in metrics_comb.columns:
                row[col] = round(metrics_comb.loc[comb, col], 4)
            comparison_rows_comb.append(row)

    # Save Final Tables
    comp_dir = os.path.join(ANALYSIS_ROOT, "comparison_tables")
    os.makedirs(comp_dir, exist_ok=True)
    
    # Save Model vs Experiment Type
    comp_df = pd.DataFrame(comparison_rows)
    if not comp_df.empty:
        out_path = os.path.join(comp_dir, "model_vs_experiment_type_metrics.csv")
        comp_df.to_csv(out_path, index=False)
        print(f"Saved model vs experiment-type metrics table to {out_path}")

    # Save Model vs Exp Type / Point Combination (NEW)
    comp_comb_df = pd.DataFrame(comparison_rows_comb)
    if not comp_comb_df.empty:
        out_path_comb = os.path.join(comp_dir, "model_vs_exp_type_point_metrics.csv")
        comp_comb_df.to_csv(out_path_comb, index=False)
        print(f"Saved model vs exp_type_point metrics table to {out_path_comb}")

    # Save Global Summary
    global_df = pd.DataFrame(global_rows)
    if not global_df.empty:
        out_global_path = os.path.join(comp_dir, "model_global_metrics.csv")
        global_df.to_csv(out_global_path, index=False)
        print(f"Saved overarching global metrics to {out_global_path}")

if __name__ == "__main__":
    run_analysis()
