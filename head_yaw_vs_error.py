# ------------- Hypothesis 1 ------------- #

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from collections import defaultdict
from data_loader import GazeDataLoader
from data_analyer import process_direction_errors
import config
from hypotheses_and_patterns import extract_head_yaw_deviation
import pandas as pd
from data_matcher import match_irregular_to_regular
import traceback

def head_yaw_vs_error(root_dir: str, model: str, plot: bool = False):
    dataloader = GazeDataLoader(
        root_dir=root_dir,
        target_period=config.get_target_period(),
        camera_pose_period=config.get_camera_pose_period(),
        time_diff_max=config.get_time_diff_max(),
        get_latest_subdirectory_by_name=True,
    )
    try:
        err_res = process_direction_errors(
            dataloader=dataloader,
            model=model,
            gaze_target_bias=None,
            save_to_file=False,
        )

        head_cam = dataloader.load_head_poses(frame="camera")
        neutral_R = config.get_neutral_head_orientation_in_cam_frame()

        # Build [ts, yaw] with SIGNED yaw (deg) in camera frame
        yaw_dev = extract_head_yaw_deviation(head_cam, neutral_R)   # shape (Ny,) or (Ny,1)
        yaw_dev = np.asarray(yaw_dev).reshape(-1)
        yaw_ts_full = head_cam[:, 0].astype(float)
        yaw_ts_vals = np.column_stack([yaw_ts_full, yaw_dev])       # [ts, yaw]

        # Align error series (irregular) to head-yaw series (regular)
        camera_period = 1000.0 / float(config.get_rgb_fps())
        err_res_aligned, matched_yaw_vals = match_irregular_to_regular(
            irregular_data=err_res,           # [ts, ang_err, eucl_err] at ~30Hz
            regular_data=yaw_ts_vals,         # [ts, yaw] (signed)
            regular_period_ms=camera_period,
        )

        # Guards
        if err_res_aligned.shape[0] == 0:
            raise ValueError("No aligned samples after stable-start filtering.")
        if matched_yaw_vals.shape[0] != err_res_aligned.shape[0]:
            raise ValueError("Alignment bug: lengths differ after matching.")

        # Use only ALIGNED data
        pose_ts   = err_res_aligned[:, 0]
        align_err = err_res_aligned[:, 1]
        yaw_algn  = matched_yaw_vals[:, 0]  # signed yaw (deg)

        # Split into LEFT (positive yaw) and RIGHT (negative yaw) around 0.0
        left_mask  = yaw_algn > 0.0   # LEFT  = positive yaw
        right_mask = yaw_algn < 0.0   # RIGHT = negative yaw

        # Correlation helper
        def safe_corr(x, y):
            if x.size < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
                return np.nan, np.nan
            r, _ = pearsonr(x, y)
            rho, _ = spearmanr(x, y)
            return r, rho

        # For interpretability: treat both sides as "sideways magnitude" from center.
        # LEFT: already positive; RIGHT: negate to make positive.
        pearson_left,  spearman_left  = safe_corr(yaw_algn[left_mask],              align_err[left_mask])
        pearson_right, spearman_right = safe_corr(-yaw_algn[right_mask],            align_err[right_mask])

        print(
            f"{root_dir} | {model} → "
            f"LEFT r={pearson_left:.3f}, ρ={spearman_left:.3f} | "
            f"RIGHT r={pearson_right:.3f}, ρ={spearman_right:.3f}"
        )

        # Optional plot (keep signed yaw for visualization)
        if plot:
            save_dir = os.path.join(dataloader.get_gaze_estimations_dir(model), "head_yaw_vs_error")
            os.makedirs(save_dir, exist_ok=True)
            t_rel = pose_ts - pose_ts[0]
            plt.figure()
            plt.plot(t_rel, align_err, label="Angular Error (deg)")
            plt.plot(t_rel, yaw_algn,  label="Head-Yaw in Camera Frame (deg, signed)")
            plt.xlabel("Time (ms)")
            plt.ylabel("Degrees")
            plt.title("Hypothesis 1 — Error vs. Head-Yaw in Camera Frame (signed)")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "error_vs_yaw_deviation.png"))
            plt.close()

        point = os.path.basename(root_dir)
        subject_path = os.path.join(*root_dir.split(os.sep)[-4:-2]) + "/"
        # Return LEFT first (positive yaw), then RIGHT (originally negative yaw)
        return subject_path, point, model, pearson_left, spearman_left, pearson_right, spearman_right

    except Exception as e:
        print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        print(f"Skipped {root_dir} ({model}): {e}")
        return None


def calculate_and_save_correlations(subject_dirs, models, output_dir_for_text_files: str, plot=False):
    os.makedirs(output_dir_for_text_files, exist_ok=True)

    results_pearson = []
    results_spearman = []
    exp_types = ["circular_movement"]
    for subj_dir in subject_dirs:
        for exp in exp_types:
            if exp != "circular_movement":
                continue
            for pt in config.get_point_variations().get(exp, []):
                exp_path = os.path.join(subj_dir, exp, pt)
                try:
                    for model in models:
                        res = head_yaw_vs_error(exp_path, model, plot=plot)
                        if res:
                            subj, pt, model, pearson_left, spearman_left, pearson_right, spearman_right = res
                            # record LEFT and RIGHT separately
                            results_pearson.append((subj, pt, model, "left",  pearson_left))
                            results_pearson.append((subj, pt, model, "right", pearson_right))
                            results_spearman.append((subj, pt, model, "left",  spearman_left))
                            results_spearman.append((subj, pt, model, "right", spearman_right))
                except Exception as e:
                    print(f"Warning: {exp_path} skipped: {e}")

    def save_csv_and_json(results, label):
        # CSV with an extra 'side' column
        out_csv = os.path.join(output_dir_for_text_files, f"head_yaw_vs_error_{label}_correlation_coefficients.csv")
        with open(out_csv, "w") as f:
            f.write("subject_dir,point,model,side,corr\n")
            for subj, pt, model, side, corr in results:
                f.write(f"{subj},{pt},{model},{side},{corr:.4f}\n")
        print(f"Written CSV: {out_csv}")

        # Grouped stats, now per side
        grouped = {
            "left": {
                "points": defaultdict(lambda: defaultdict(list)),
                "subjects": defaultdict(lambda: defaultdict(list)),
                "all": defaultdict(list),
            },
            "right": {
                "points": defaultdict(lambda: defaultdict(list)),
                "subjects": defaultdict(lambda: defaultdict(list)),
                "all": defaultdict(list),
            },
        }

        for subj, pt, model, side, corr in results:
            grp = grouped[side]
            grp["points"][pt][model].append(corr)
            grp["points"][pt]["all"].append(corr)
            grp["subjects"][subj][model].append(corr)
            grp["subjects"][subj]["all"].append(corr)
            grp["all"][model].append(corr)
            grp["all"]["all"].append(corr)

        def summarize(vals):
            if not vals:
                return {"mean": None, "median": None, "std": None}
            vals = np.array(vals)
            return {
                "mean": round(float(np.mean(vals)), 4),
                "median": round(float(np.median(vals)), 4),
                "std": round(float(np.std(vals)), 4)
            }

        # Build stats for both sides, structure kept parallel to the old format
        stats = {"left": {"points": {}, "subjects": {}, "all": {}},
                 "right": {"points": {}, "subjects": {}, "all": {}}}

        for side in ["left", "right"]:
            # points
            stats[side]["points"] = {}
            for pt in config.get_point_variations().get("circular_movement", []):
                stats[side]["points"][pt] = {
                    model: summarize(grouped[side]["points"][pt].get(model, []))
                    for model in grouped[side]["points"][pt]
                }
            # subjects
            stats[side]["subjects"] = {
                subj: {
                    model: summarize(grouped[side]["subjects"][subj].get(model, []))
                    for model in grouped[side]["subjects"][subj]
                }
                for subj in grouped[side]["subjects"]
            }
            # all
            stats[side]["all"] = {
                model: summarize(grouped[side]["all"].get(model, []))
                for model in grouped[side]["all"]
            }

        out_json = os.path.join(output_dir_for_text_files, f"head_yaw_vs_error_{label}_corr_coeff_grouped.json")
        with open(out_json, "w") as jf:
            json.dump(stats, jf, indent=2)
        print(f"Written grouped JSON: {out_json}")

    save_csv_and_json(results_pearson, "pearson")
    save_csv_and_json(results_spearman, "spearman")


def save_subject_boxplot_from_corr_df(
    corr_df: pd.DataFrame,
    models: list[str],
    out_png: str,
    title: str = "Subject-level correlation vs model",
    agg: str = "median",   # "median" or "mean"
    show: bool = False,
):
    """
    Build a per-model distribution across subjects by aggregating each subject's
    correlations over all points (i.e., 'subjects' → 'all (points)').

    Args:
        corr_df: DataFrame with columns ['subject_dir','point','model','corr'].
        models: list of model names to include (order respected; missing are skipped).
        out_png: path to save the boxplot.
        title: figure title.
        agg: aggregation across points per subject: 'median' (default) or 'mean'.
        show: plt.show() after saving.

    Output:
        A boxplot where each box is the distribution (over subjects) of the
        aggregated correlation for that model.
    """
    needed = {"subject_dir", "model", "corr"}
    if not needed.issubset(set(corr_df.columns)):
        raise ValueError(f"corr_df must contain columns {needed}, got {corr_df.columns.tolist()}")

    # keep requested models that actually exist
    have_models = [m for m in models if m in corr_df["model"].unique()]
    if not have_models:
        raise ValueError("None of the requested models are present in corr_df.")

    df = corr_df.copy()
    df = df[df["model"].isin(have_models)]
    df = df[np.isfinite(df["corr"])].copy()

    # aggregate per (subject, model) across points
    if agg == "mean":
        subj_model = df.groupby(["subject_dir", "model"], as_index=False)["corr"].mean()
    else:
        subj_model = df.groupby(["subject_dir", "model"], as_index=False)["corr"].median()

    # arrange data per model in requested order
    data = [subj_model[subj_model["model"] == m]["corr"].values for m in have_models]

    # plot (compact, neutral styling; black lines)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(max(6.0, 1.2 * len(have_models)), 4.4))
    # wrap long labels with \n for readability
    wrapped_labels = [m.replace("_", "_\n") if len(m) > 12 else m for m in have_models]
    bp = plt.boxplot(
        data,
        tick_labels=wrapped_labels,
        showfliers=False,
        showmeans=True,
        meanline=False,
        patch_artist=True,
    )
    # styling: all lines black, subtle fill
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfd8e3")
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.25)
    for key in ("whiskers", "caps", "medians", "means"):
        for art in bp[key]:
            art.set_color("black")
            art.set_linewidth(1.25)

    plt.ylabel("Correlation coefficient")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    if show:
        plt.show()
    plt.close()


def boxplot_from_correlation_coefficients(output_dir_for_text_files: str, models):
    # Make per-model boxplots over subjects from the saved CSVs.
    # If 'side' exists, aggregate across sides to preserve the previous behavior.
    pearson_csv = os.path.join(output_dir_for_text_files, "head_yaw_vs_error_pearson_correlation_coefficients.csv")
    spearman_csv = os.path.join(output_dir_for_text_files, "head_yaw_vs_error_spearman_correlation_coefficients.csv")

    try:
        df_p = pd.read_csv(pearson_csv)
        if "side" in df_p.columns:
            # Aggregate across sides to keep one correlation per (subject, model) for boxplot
            df_p = (df_p
                    .groupby(["subject_dir", "model"], as_index=False)["corr"]
                    .mean())
        save_subject_boxplot_from_corr_df(
            df_p,
            models=models,
            out_png=os.path.join(output_dir_for_text_files, "boxplot_subject_corrs__pearson.png"),
            title="Subject-level Pearson correlation vs model (aggregated across points)",
            agg="mean",   # or "median"
            show=False,
        )
    except Exception as e:
        print(f"Pearson boxplot skipped: {e}")

    try:
        df_s = pd.read_csv(spearman_csv)
        if "side" in df_s.columns:
            df_s = (df_s
                    .groupby(["subject_dir", "model"], as_index=False)["corr"]
                    .mean())
        save_subject_boxplot_from_corr_df(
            df_s,
            models=models,
            out_png=os.path.join(output_dir_for_text_files, "boxplot_subject_corrs__spearman.png"),
            title="Subject-level Spearman correlation vs model (aggregated across points)",
            agg="mean",
            show=False,
        )
    except Exception as e:
        print(f"Spearman boxplot skipped: {e}")

def _render_headyaw_vs_error_grid(subject_dirs_batch, point: str, model: str, batch_idx: int, save: bool = False):
    """Renders a 2 x N grid (N = len(subject_dirs_batch)), with consistent scaling across all subplots."""
    n = len(subject_dirs_batch)
    if n == 0:
        return

    # ---------- First pass: load & collect global ranges ----------
    per_subj = []  # list of dicts with keys: subj_name, t_rel, align_err, yaw_abs
    time_max, y_min, y_max = 0.0, np.inf, -np.inf           # for top row (time series)
    yaw_min, yaw_max, err_min, err_max = np.inf, -np.inf, np.inf, -np.inf  # for bottom row (scatter)

    for subj_dir in subject_dirs_batch:
        exp_path = os.path.join(subj_dir, "circular_movement", point)
        try:
            dl = GazeDataLoader(
                root_dir=exp_path,
                target_period=config.get_target_period(),
                camera_pose_period=config.get_camera_pose_period(),
                time_diff_max=config.get_time_diff_max(),
                get_latest_subdirectory_by_name=True,
            )

            # Errors (irregular time, ~30 Hz)
            err_res = process_direction_errors(
                dataloader=dl,
                model=model,
                gaze_target_bias=None,
                save_to_file=False,
            )
            err_res = np.asarray(err_res)
            if err_res.size == 0:
                raise RuntimeError("Empty error results")

            # Head yaw (regular) in camera frame → build [ts, |yaw|]
            head_cam = dl.load_head_poses(frame="camera")
            if head_cam.size == 0:
                raise RuntimeError("Empty head_cam")
            neutral_R = config.get_neutral_head_orientation_in_cam_frame()
            yaw_dev = np.asarray(extract_head_yaw_deviation(head_cam, neutral_R)).reshape(-1)
            yaw_dev_abs_full = np.abs(yaw_dev)
            yaw_ts_full = head_cam[:, 0].astype(float)
            yaw_ts_vals = np.column_stack([yaw_ts_full, yaw_dev_abs_full])  # [ts, |yaw|]

            # Align: err_res (irregular) ↔ yaw_ts_vals (regular)
            camera_period = 1000.0 / float(config.get_rgb_fps())
            err_res_aligned, matched_yaw_vals = match_irregular_to_regular(
                irregular_data=err_res,           # [ts, ang_err, eucl_err]
                regular_data=yaw_ts_vals,         # [ts, |yaw|]
                regular_period_ms=camera_period,
            )
            if err_res_aligned.shape[0] == 0 or matched_yaw_vals.shape[0] == 0:
                raise RuntimeError("No aligned samples after filtering")

            # Use aligned series only
            pose_ts   = err_res_aligned[:, 0]
            align_err = err_res_aligned[:, 1]
            yaw_abs   = matched_yaw_vals[:, 0]   # one value col (|yaw|)

            # Prepare relative time
            t_rel = pose_ts - pose_ts[0]

            # Update global ranges (time row)
            if t_rel.size > 0:
                time_max = max(time_max, float(np.nanmax(t_rel)))
            if align_err.size > 0 and yaw_abs.size > 0:
                y_min = min(y_min, float(np.nanmin([np.nanmin(align_err), np.nanmin(yaw_abs)])))
                y_max = max(y_max, float(np.nanmax([np.nanmax(align_err), np.nanmax(yaw_abs)])))

            # Update global ranges (scatter row)
            if yaw_abs.size > 0:
                yaw_min = min(yaw_min, float(np.nanmin(yaw_abs)))
                yaw_max = max(yaw_max, float(np.nanmax(yaw_abs)))
            if align_err.size > 0:
                err_min = min(err_min, float(np.nanmin(align_err)))
                err_max = max(err_max, float(np.nanmax(align_err)))

            per_subj.append({
                "subj_name": os.path.basename(os.path.normpath(subj_dir)),
                "t_rel": t_rel,
                "align_err": align_err,
                "yaw_abs": yaw_abs,
            })

        except Exception as e:
            print(f"Skipped {exp_path} ({model}): {e}")
            per_subj.append({
                "subj_name": os.path.basename(os.path.normpath(subj_dir)),
                "t_rel": None, "align_err": None, "yaw_abs": None
            })

    # Handle degenerate cases
    if not np.isfinite(time_max): time_max = 1.0
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
        y_min, y_max = 0.0, 1.0
    if not np.isfinite(yaw_min) or not np.isfinite(yaw_max) or yaw_min == yaw_max:
        yaw_min, yaw_max = 0.0, 1.0
    if not np.isfinite(err_min) or not np.isfinite(err_max) or err_min == err_max:
        err_min, err_max = 0.0, 1.0

    # Add a small padding for nicer visuals
    def _pad(lo, hi, frac=0.05):
        span = hi - lo
        if span <= 0: span = 1.0
        return lo - frac*span, hi + frac*span

    y_min, y_max     = _pad(y_min, y_max)
    yaw_min, yaw_max = _pad(yaw_min, yaw_max)
    err_min, err_max = _pad(err_min, err_max)

    # ---------- Second pass: plot with fixed limits ----------
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), sharey='row')
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    top_handles, top_labels = None, None
    bot_handles, bot_labels = None, None

    for col, item in enumerate(per_subj):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        t_rel = item["t_rel"]
        align_err = item["align_err"]
        yaw_abs = item["yaw_abs"]
        subj_name = item["subj_name"]

        if t_rel is None:
            ax_top.axis("off")
            ax_bot.axis("off")
            continue

        # Correlations (robust to NaNs/constant vectors)
        def _safe_corr(x, y):
            x, y = np.asarray(x), np.asarray(y)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 3 or np.nanstd(x[m]) == 0 or np.nanstd(y[m]) == 0:
                return np.nan, np.nan
            r, _ = pearsonr(x[m], y[m])
            rho, _ = spearmanr(x[m], y[m])
            return r, rho

        pear_corr, spear_corr = _safe_corr(yaw_abs, align_err)
        title = f"{subj_name}\nPearson={pear_corr:.2f}, Spearman={spear_corr:.2f}"

        # ----- Top row: time vs curves with global x/y -----
        l1 = ax_top.plot(t_rel, align_err, linewidth=1.2, label="Angular Error (deg)")
        l2 = ax_top.plot(t_rel, yaw_abs,   linewidth=1.2, label="|Head-Yaw| (deg)")
        ax_top.set_xlim(0, time_max)
        ax_top.set_ylim(y_min, y_max)
        ax_top.set_title(title)
        ax_top.set_xlabel("Time (ms)")
        if col == 0:
            ax_top.set_ylabel("Degrees")
        ax_top.grid(True, alpha=0.35)
        if top_handles is None:
            top_handles, top_labels = [l1[0], l2[0]], ["Angular Error (deg)", "|Head-Yaw| (deg)"]

        # ----- Bottom row: scatter with global x/y -----
        s1 = ax_bot.scatter(yaw_abs, align_err, s=6, alpha=0.5, label="Samples")
        # Optional linear fit
        try:
            m = np.isfinite(yaw_abs) & np.isfinite(align_err)
            if m.sum() >= 2:
                k, b = np.polyfit(yaw_abs[m], align_err[m], 1)
                xline = np.linspace(yaw_min, yaw_max, 100)
                lfit = ax_bot.plot(xline, k * xline + b, linewidth=1.0, label="Linear fit")
                if bot_handles is None:
                    bot_handles, bot_labels = [s1, lfit[0]], ["Samples", "Linear fit"]
        except Exception:
            if bot_handles is None:
                bot_handles, bot_labels = [s1], ["Samples"]

        ax_bot.set_xlim(yaw_min, yaw_max)
        ax_bot.set_ylim(err_min, err_max)
        ax_bot.set_xlabel("|Head-Yaw| (deg)")
        if col == 0:
            ax_bot.set_ylabel("Angular Error (deg)")
        ax_bot.grid(True, alpha=0.35)

    # Row-level legends
    if top_handles is not None:
        fig.legend(top_handles, top_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    if bot_handles is not None:
        fig.legend(bot_handles, bot_labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"|Head-Yaw| vs Error — Point {point}, Model={model} (Batch {batch_idx})", y=1.06, fontsize=12)
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.98])

    if save:
        out_dir = os.path.join(config.get_dataset_base_directory(), "head_yaw_vs_error_results", "viz", model, point)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"headyaw_error_grid_{model}_{point}_batch{batch_idx}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.show()  # blocks until closed


# --- NEW: tiny helpers for side logic ---------------------------------------

def _point_side_layout():
    # Camera-perspective layout:
    # [p9-L, p8-L, p8-R, p7-R]
    # [p6-L, p5-L, p5-R, p4-R]
    # [p3-L, p2-L, p2-R, p1-R]
    return [
        ("p9","left"), ("p8","left"), ("p8","right"), ("p7","right"),
        ("p6","left"), ("p5","left"), ("p5","right"), ("p4","right"),
        ("p3","left"), ("p2","left"), ("p2","right"), ("p1","right"),
    ]

def _default_side_for_point(pt: str) -> str:
    """Used by the batched single-point viewer."""
    if pt in {"p1","p4","p7"}:  # C1
        return "right"
    if pt in {"p3","p6","p9"}:  # C3
        return "left"
    # C2 (p2,p5,p8): default to LEFT for single-plot view
    return "left"

def _align_signed_yaw_and_error(exp_path: str, model: str):
    """
    Returns (pose_ts, align_err, yaw_signed_algn) where yaw is SIGNED in the camera frame.
    """
    dl = GazeDataLoader(
        root_dir=exp_path,
        target_period=config.get_target_period(),
        camera_pose_period=config.get_camera_pose_period(),
        time_diff_max=config.get_time_diff_max(),
        get_latest_subdirectory_by_name=True,
    )
    err_res = process_direction_errors(
        dataloader=dl, model=model, gaze_target_bias=None, save_to_file=False
    )
    err_res = np.asarray(err_res)
    if err_res.size == 0:
        raise RuntimeError("Empty error results")

    head_cam = dl.load_head_poses(frame="camera")
    if head_cam.size == 0:
        raise RuntimeError("Empty head_cam")

    neutral_R = config.get_neutral_head_orientation_in_cam_frame()
    yaw_dev = np.asarray(extract_head_yaw_deviation(head_cam, neutral_R)).reshape(-1)  # SIGNED (deg)
    yaw_ts_full = head_cam[:, 0].astype(float)
    yaw_ts_vals = np.column_stack([yaw_ts_full, yaw_dev])  # [ts, signed_yaw]

    camera_period = 1000.0 / float(config.get_rgb_fps())
    err_res_aligned, matched_yaw_vals = match_irregular_to_regular(
        irregular_data=err_res,
        regular_data=yaw_ts_vals,
        regular_period_ms=camera_period,
    )
    if err_res_aligned.shape[0] == 0 or matched_yaw_vals.shape[0] == 0:
        raise RuntimeError("No aligned samples after filtering")

    pose_ts = err_res_aligned[:, 0]
    align_err = err_res_aligned[:, 1]
    yaw_signed_algn = matched_yaw_vals[:, 0]
    return pose_ts, align_err, yaw_signed_algn

def _side_mask_and_magnitude(yaw_signed: np.ndarray, side: str):
    """
    For LEFT: keep yaw>0, x = +yaw (already positive).
    For RIGHT: keep yaw<0, x = -yaw (magnitude).
    Returns mask, x_mag (magnitude for that side).
    """
    if side == "left":
        m = yaw_signed > 0.0
        x = yaw_signed[m]  # already positive
    else:  # "right"
        m = yaw_signed < 0.0
        x = -yaw_signed[m]  # flip to magnitude
    return m, x

# --- UPDATED: batched viewer (single point across many subjects) -------------

from scipy.stats import pearsonr, spearmanr

def visualize_headyaw_vs_error_batched(subject_dirs, point: str, model: str,
                                       max_subjects: int = 4, save: bool = False, start_index: int = 0):
    """
    Iterates subjects in batches of size `max_subjects`. Each batch opens one 2xN window:
      - Top row: time vs [Angular Error, |Head-Yaw|_SIDE]  (SIDE per point: C1->RIGHT, C3->LEFT, C2->LEFT default)
      - Bottom row: |Head-Yaw|_SIDE vs Angular Error (scatter + linear fit) using ONLY that SIDE
    For C2 (p2,p5,p8), this single-point view defaults to LEFT side. Use the 3x4 grid for both sides.
    """
    if not subject_dirs:
        print("visualize_headyaw_vs_error_batched: subject_dirs is empty; pass config.get_dataset_subject_directories().")
        return

    # decide side from point (single-side per point in this viewer)
    side = _default_side_for_point(point)

    i = max(0, int(start_index))
    batch_idx = 1 + (i // max_subjects)
    total = len(subject_dirs)

    while i < total:
        batch = subject_dirs[i : i + max_subjects]
        print(f"Showing subjects {i+1}-{min(i+max_subjects, total)} of {total} (Batch {batch_idx})")

        # Build the 2 x N grid
        n = len(batch)
        fig, axes = plt.subplots(2, n, figsize=(5.0*n, 7.0), squeeze=False)

        # Global ranges for scatter
        yaw_all, err_all = [], []

        # First pass: gather data
        per = []
        for j, subj_dir in enumerate(batch):
            exp_path = os.path.join(subj_dir, "circular_movement", point)
            try:
                pose_ts, align_err, yaw_signed = _align_signed_yaw_and_error(exp_path, model)
                m, x_mag = _side_mask_and_magnitude(yaw_signed, side)
                t_rel = pose_ts - pose_ts[0]
                per.append((t_rel, align_err, yaw_signed, m, x_mag))
                yaw_all.extend(x_mag.tolist())
                err_all.extend(align_err[m].tolist())
            except Exception as e:
                print(f"Skipped {exp_path} ({model}): {e}")
                per.append(None)

        # Compute global limits for scatter
        yaw_min = yaw_max = err_min = err_max = None
        if yaw_all and err_all:
            yaw_min, yaw_max = np.nanmin(yaw_all), np.nanmax(yaw_all)
            err_min, err_max = np.nanmin(err_all), np.nanmax(err_all)
            def _pad(lo, hi, frac=0.05):
                span = hi - lo
                if not np.isfinite(span) or span <= 0: span = 1.0
                return lo - frac*span, hi + frac*span
            yaw_min, yaw_max = _pad(yaw_min, yaw_max)
            err_min, err_max = _pad(err_min, err_max)

        # Second pass: render
        for j, subj_dir in enumerate(batch):
            ax_t = axes[0, j]
            ax_s = axes[1, j]
            if per[j] is None:
                ax_t.text(0.5, 0.5, f"{os.path.basename(subj_dir)}\n(no data)", ha="center", va="center")
                ax_t.axis("off")
                ax_s.axis("off")
                continue

            t_rel, align_err, yaw_signed, m, x_mag = per[j]

            # --- time plot (ONLY that side’s |yaw|) ---
            ax_t.plot(t_rel, align_err, label="Angular Error (deg)")
            # build side-magnitude over full timeline (NaN outside mask)
            ymag_full = np.full_like(yaw_signed, np.nan, dtype=float)
            ymag_full[m] = x_mag
            ax_t.plot(t_rel, ymag_full, label=f"|Head-Yaw| ({side}) (deg)")
            ax_t.set_xlabel("Time (ms)")
            if j == 0: ax_t.set_ylabel("Degrees")
            ax_t.grid(True, alpha=0.35)
            ax_t.legend(loc="upper right", fontsize=8)

            # --- scatter + regression on SIDE subset ---
            ax_s.scatter(x_mag, align_err[m], s=7, alpha=0.5)
            if yaw_min is not None:
                ax_s.set_xlim(yaw_min, yaw_max)
            if err_min is not None:
                ax_s.set_ylim(err_min, err_max)

            # correlation & fit
            r_p = r_s = np.nan
            if x_mag.size > 2 and np.nanstd(x_mag) > 0 and np.nanstd(align_err[m]) > 0:
                r_p, _ = pearsonr(x_mag, align_err[m])
                r_s, _ = spearmanr(x_mag, align_err[m])
                k, b = np.polyfit(x_mag, align_err[m], 1)
                xline = np.linspace(ax_s.get_xlim()[0], ax_s.get_xlim()[1], 100)
                ax_s.plot(xline, k * xline + b, linewidth=1.0)
            ax_s.set_xlabel(f"|Head-Yaw| ({side}) (deg)")
            if j == 0: ax_s.set_ylabel("Angular Error (deg)")
            ax_s.grid(True, alpha=0.35)
            subj_name = os.path.basename(os.path.normpath(subj_dir))
            ax_s.set_title(f"{subj_name}  r={r_p:.2f}, ρ={r_s:.2f}", fontsize=10)

        fig.suptitle(f"{model} — point {point} ({side} side) — batch {batch_idx}", y=1.02, fontsize=12)
        fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.96])

        if save:
            out_dir = os.path.join(config.get_dataset_base_directory(), "head_yaw_vs_error_results",
                                   "viz", model, f"point_{point}_{side}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"batch_{batch_idx:02d}__{point}_{side}.png")
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"Saved: {out_path}")

        plt.show()
        batch_idx += 1

# --- UPDATED: per-subject 3x4 grid (C2 split L/R) ---------------------------

def visualize_subject_point_grid(subject_dir: str, model: str, plot_type: str = "time", save: bool = False):
    """
    Show a 3x4 grid for one subject & model with C2 split into LEFT/RIGHT.
    plot_type:
      - "time"    -> time vs two curves: [Angular Error (deg), |Head-Yaw|_SIDE (deg)]
      - "scatter" -> |Head-Yaw|_SIDE vs Angular Error (scatter + linear fit)
    Ensures consistent scales across all 12 subplots.
    Grid layout (rows = R1..R3):
      [p1-R, p2-L, p2-R, p3-L]
      [p4-R, p5-L, p5-R, p6-L]
      [p7-R, p8-L, p8-R, p9-L]
    """
    assert plot_type in ("time", "scatter"), "plot_type must be 'time' or 'scatter'"

    layout = _point_side_layout()  # 12 panels
    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(19, 12), sharey=(plot_type=="time"))
    subj_name = os.path.basename(os.path.normpath(subject_dir))

    # First pass: collect data and global scatter ranges
    yaw_all, err_all = [], []
    results = []
    for (pt, side) in layout:
        exp_path = os.path.join(subject_dir, "circular_movement", pt)
        try:
            pose_ts, align_err, yaw_signed = _align_signed_yaw_and_error(exp_path, model)
            m, x_mag = _side_mask_and_magnitude(yaw_signed, side)
            t_rel = pose_ts - pose_ts[0]
            results.append((pt, side, t_rel, align_err, yaw_signed, m, x_mag))
            if plot_type == "scatter":
                yaw_all.extend(x_mag.tolist())
                err_all.extend(align_err[m].tolist())
        except Exception as e:
            print(f"Skipped {exp_path} ({model}): {e}")
            results.append((pt, side, None, None, None, None, None))

    # Global ranges for scatter
    yaw_min = yaw_max = err_min = err_max = None
    if plot_type == "scatter" and yaw_all and err_all:
        yaw_min, yaw_max = np.nanmin(yaw_all), np.nanmax(yaw_all)
        err_min, err_max = np.nanmin(err_all), np.nanmax(err_all)
        def _pad(lo, hi, frac=0.05):
            span = hi - lo
            if not np.isfinite(span) or span <= 0: span = 1.0
            return lo - frac*span, hi + frac*span
        yaw_min, yaw_max = _pad(yaw_min, yaw_max)
        err_min, err_max = _pad(err_min, err_max)

    # Render panels
    for idx, (pt, side, t_rel, align_err, yaw_signed, m, x_mag) in enumerate(results):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        if t_rel is None:
            ax.text(0.5, 0.5, f"{pt}-{side}\n(no data)", ha="center", va="center")
            ax.axis("off")
            continue

        if plot_type == "time":
            # Plot error and SIDE-specific magnitude over time
            ax.plot(t_rel, align_err, label="Angular Error (deg)")
            ymag_full = np.full_like(yaw_signed, np.nan, dtype=float)
            ymag_full[m] = x_mag
            ax.plot(t_rel, ymag_full, label=f"|Head-Yaw| ({side}) (deg)")
            if r == nrows-1: ax.set_xlabel("Time (ms)")
            if c == 0: ax.set_ylabel("Degrees")
            ax.grid(True, alpha=0.35)
            ax.legend(loc="upper right", fontsize=8)

            # correlations on SIDE subset
            r_p = r_s = np.nan
            if x_mag.size > 2 and np.nanstd(x_mag) > 0 and np.nanstd(align_err[m]) > 0:
                r_p, _ = pearsonr(x_mag, align_err[m])
                r_s, _ = spearmanr(x_mag, align_err[m])
            ax.set_title(f"{pt}-{side}  r={r_p:.2f}, ρ={r_s:.2f}", fontsize=10)

        else:
            # scatter + fit on SIDE subset
            ax.scatter(x_mag, align_err[m], s=8, alpha=0.55)
            if yaw_min is not None:
                ax.set_xlim(yaw_min, yaw_max)
            if err_min is not None:
                ax.set_ylim(err_min, err_max)

            r_p = r_s = np.nan
            if x_mag.size > 2 and np.nanstd(x_mag) > 0 and np.nanstd(align_err[m]) > 0:
                r_p, _ = pearsonr(x_mag, align_err[m])
                r_s, _ = spearmanr(x_mag, align_err[m])
                k, b = np.polyfit(x_mag, align_err[m], 1)
                xline = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 120)
                ax.plot(xline, k * xline + b, linewidth=1.0)
            if r == nrows-1: ax.set_xlabel(f"|Head-Yaw| ({side}) (deg)")
            if c == 0: ax.set_ylabel("Angular Error (deg)")
            ax.grid(True, alpha=0.35)
            ax.set_title(f"{pt}-{side}  r={r_p:.2f}, ρ={r_s:.2f}", fontsize=10)

    fig.suptitle(f"{subj_name} — {model}  |  3×4 grid (C2 split) — {'Time vs Curves' if plot_type=='time' else 'Scatter + Fit'}",
                 y=1.02, fontsize=13)
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.96])

    if save:
        out_dir = os.path.join(config.get_dataset_base_directory(), "head_yaw_vs_error_results",
                               "viz", model, subj_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{subj_name}_{model}_grid3x4_{plot_type}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.show()


if __name__ == "__main__":
    # 1- calculate and save head_yaw-error correlation coefficients (both individual corr. coeff., as well as their aggregations) as csv.
    BASE_DIR = config.get_dataset_base_directory()
    SUBJECT_DIRS = config.get_dataset_subject_directories()
    MODELS = config.get_currently_analyzed_models()
    OUTPUT_DIR_FOR_TEXT_FILES = os.path.join(BASE_DIR, "head_yaw_vs_error_results")

    calculate_and_save_correlations(SUBJECT_DIRS, MODELS, output_dir_for_text_files=OUTPUT_DIR_FOR_TEXT_FILES, plot=True)
    boxplot_from_correlation_coefficients(OUTPUT_DIR_FOR_TEXT_FILES, MODELS)
    exit()
    # Usage:
    # 2- visualize point_grid per subject. our table has a 3x3 gaze target grid,
    # but we now split C2 into left/right, so the visualization uses a 3x4 layout:
    # [p9-L, p8-L, p8-R, p7-R]
    # [p6-L, p5-L, p5-R, p4-R]
    # [p3-L, p2-L, p2-R, p1-R]
    # The "visualize_subject_point_grid" function shows these 12 plots for a subject,
    # spatially arranged to match the camera-perspective layout.
    SUBJECT_DIRS = config.get_dataset_subject_directories()
    for subject_dir in SUBJECT_DIRS:
        print(subject_dir)
        visualize_subject_point_grid(
            subject_dir=subject_dir,
            model="puregaze_rectification",
            plot_type="scatter", # "time" for "time vs. head_yaw and error", or "scatter" for "head_yaw vs. error"
            save=False,
        )
    exit()
    # 3- visualizes head_yaw-error relationship by displaying the plots for multiple subjects at a time. Allows analyzing subject variability
    # by eyeballing the plots for different subjects in a batch of 4 subjects.
    visualize_headyaw_vs_error_batched(
        subject_dirs=SUBJECT_DIRS,
        point="p6",
        model="puregaze",
        save=False,
    )
    exit()
