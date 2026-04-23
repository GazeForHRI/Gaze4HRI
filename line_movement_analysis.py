
from matplotlib.dates import MO
from data_loader import GazeDataLoader
import matplotlib.pyplot as plt
import numpy as np
import config
from scipy.signal import savgol_filter
import os
from data_analyer import process_direction_errors
from data_matcher import match_irregular_to_regular
from neutral_eye_position_calculation import load_neutral_eye_pose_in_world_frame
import csv
import math
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)

FRAME = "neutral_camera" # neutral_eye or neutral_camera
# Note: neutral cam frame: +Y = left, -Y = right; +Z = up, -Z = down. neutral eye frame: +Y = up, -Y = down; +Z = right, -Z = left
RESULTS_DIR = f"line_movement_by_position_results/{FRAME}" # relative path

def _get_axis(is_line_horizontal, frame):
    # Note: neutral cam frame: +Y = left, -Y = right; +Z = up, -Z = down. neutral eye frame: +Y = up, -Y = down; +Z = right, -Z = left
    if FRAME == "neutral_eye":
        return 1 if not is_line_horizontal else 2
    else: # FRAME == neutral_camera frame
        return 1 if is_line_horizontal else 2

def _get_horizontal_signs(frame):
    if frame == "neutral_eye":
        return "left", "right"
    else:
        return "right", "left"

def print_pretty(name: str, array: np.ndarray):
    with np.printoptions(
        precision=4,                # 4 decimal places
        suppress=True,              # Avoid scientific notation
        threshold=np.inf,           # Show all elements
        linewidth=np.inf,           # Don’t break lines
        formatter={'float_kind': lambda x: ("+" if x >= 0 else "-") + f"{abs(x):.4f}"}
    ):
        print(name)
        print(array)

def extract_intersection_points(dataloader: GazeDataLoader, line_movement_speed: str, line_movement_type: str, visualize_for_debug: bool=False):
    """Break down the camera movement into sections for analysis.

    Args:
        dataloader (GazeDataLoader): The dataloader instance.
        line_movement_speed (str): The speed of line movement (e.g., "fast", "slow").
        line_movement_type (str): The type of line movement (e.g., "horizontal", "vertical").
    """
    if line_movement_speed not in ["fast", "slow"]:
        raise ValueError(f"Unknown line movement speed: {line_movement_speed}")
    
    if line_movement_type not in config.get_line_movement_types():
        raise ValueError(f"Unknown line movement type: {line_movement_type}")

    # a point of intersection is a corner point that is at the intersection of a horizontal line and a vertical line. At an intersection point, the camera momentarily stops; and then continues by moving horizontally (if it had been moving vertically before the stop), or by moving vertically (if it had been moving horizontally before the stop).
    # since they are not at the intersection of any two lines, the starting point and ending point of the movement does not count as an intersection point, and thus will not be included in this array.
    points_of_intersection = []
    
    # since all experiments are programatically executed, we approximately know where the points of intersections will be within these ranges (i.e. ranges_for_intersections), which have been empirically determined by inspecting the plots from a few experiments.
    # To pinpoint the points of intersection, we will search within these ranges.
    # Note: fast movement is twice as fast as slow, so we naturally observe double the time (index) for the slow version compared to the fast version.
    if line_movement_speed == "fast":
        range_margin = 30 # margin of error, unit: index
        if line_movement_type == "horizontal":
            ranges_for_intersections = [(i-range_margin, i+range_margin) for i in [500, 600, 1100, 1200]] # list of tuples, where each tuple is a range.
        else: # line_movement_type == "vertical":
            ranges_for_intersections = [(i-range_margin, i+range_margin) for i in [200, 450, 650, 900]]
    else: # line_movement_speed == "slow"
        range_margin = 30 # margin of error, unit: index
        if line_movement_type == "horizontal":
            ranges_for_intersections = [(i-range_margin, i+range_margin) for i in [1000, 1200, 2200, 2400]]
        else: # line_movement_type == "vertical":
            ranges_for_intersections = [(i-range_margin, i+range_margin) for i in [400, 900, 1300, 1800]]

    # Camera pose data are in 100hz, but it actually seems to be in 20Hz (see print_camera_poses_for_debug for the details), so we will multiply the window size using this multiplier rather than just using the window_size as is.
    camera_poses = dataloader.load_camera_poses() # (n, 17), where 17 is [timestamp_in_ms, *flattened_homogeneous_transformation_matrix], in meters
    camera_positions = camera_poses[:, (0,4,8,12)] # (n, 4), where 4 is [timestamp_in_ms, x, y, z], in meters
    camera_velocities = 1000 * np.diff(camera_positions[:,1:], axis=0,n=1) / config.get_camera_pose_period() # (n-1, 3), in m/s
    camera_velocities = savgol_filter(camera_velocities, window_length=21, polyorder=3, axis=0) # smooth the data since it has a lot of sensor related 0 values and noise.
    # find the absolute minimum point within this range (i.e. find the row vector that is the closest to the 0 vector).
    n_vel = camera_velocities.shape[0]
    for s, e in ranges_for_intersections:
        start = max(0, int(s))
        end   = min(int(e), n_vel - 1)
        if start > end:
            continue

        window = camera_velocities[start:end+1]  # (m, 3)
        if window.size == 0:
            continue

        # Use squared norm (cheaper, same argmin)
        sq_speeds = np.einsum('ij,ij->i', window, window)
        local_min = int(np.nanargmin(sq_speeds))
        vel_idx = start + local_min
        points_of_intersection.append(vel_idx)

    def visualize_ranges_for_debug(camera_vels, ranges):
        # --- Visualization of ranges ---
        plt.figure(figsize=(12, 6))
        t = np.arange(camera_vels.shape[0])
        plt.plot(t, camera_vels[:, 0], label='X Velocity')
        plt.plot(t, camera_vels[:, 1], label='Y Velocity')
        plt.plot(t, camera_vels[:, 2], label='Z Velocity')

        # Pick 4 distinct colors
        colors = ['red', 'green', 'blue', 'purple']

        for i, (s, e) in enumerate(ranges):
            c = colors[i % len(colors)]
            plt.axvline(x=s, color=c, linestyle='--', linewidth=1.2, alpha=0.7, label=f'Range {i+1} start' if i==0 else None)
            plt.axvline(x=e, color=c, linestyle='-.', linewidth=1.2, alpha=0.7, label=f'Range {i+1} end' if i==0 else None)

        plt.xlabel('Frame index')
        plt.ylabel('Velocity (m/s)')
        plt.title('Camera Velocities with Search Ranges')
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_points_of_intersection_for_debug(camera_vels, points):
        # --- Visualization of points of intersection ---
        plt.figure(figsize=(12, 6))
        t = np.arange(camera_vels.shape[0])
        plt.plot(t, camera_vels[:, 0], label='X Velocity')
        plt.plot(t, camera_vels[:, 1], label='Y Velocity')
        plt.plot(t, camera_vels[:, 2], label='Z Velocity')

        # Mark points of intersection
        for idx in points:
            plt.axvline(x=idx, color='orange', linestyle='--', linewidth=1.2, alpha=0.7, label='Point of Intersection' if idx==points[0] else None)

        plt.xlabel('Frame index')
        plt.ylabel('Velocity (m/s)')
        plt.title('Camera Velocities with Points of Intersection')
        plt.legend()
        plt.grid(True)
        plt.show()

    if visualize_for_debug:
        visualize_ranges_for_debug(camera_velocities, ranges_for_intersections)
        visualize_points_of_intersection_for_debug(camera_velocities, points_of_intersection)

    return points_of_intersection


def extract_intersection_points_for_all_subjects(print_every_intersection_point: bool = False):
    """
    This has been written to make sure the intersection points are very similar (almost identical) for all subjects.
    """
    SUBJECT_DIRS = config.get_dataset_subject_directories()

    print("Number of subjects: ", len(SUBJECT_DIRS))

    # we will do all 4 combinations of line_movement (fast/slow, horizontal/vertical)
    for line_movement_speed in ["fast", "slow"]:
        for line_movement_type in config.get_line_movement_types():
            print(f"Started processing {line_movement_speed}, {line_movement_type}")
            arr = np.zeros((len(SUBJECT_DIRS), 4), dtype=np.int32)
            
            for i, subject_dir in enumerate(SUBJECT_DIRS):
                exp_dir = f"{subject_dir}/line_movement_{line_movement_speed}/{line_movement_type}"
                dataloader = GazeDataLoader(
                    root_dir=exp_dir,
                    target_period=config.get_target_period(),
                    camera_pose_period=config.get_camera_pose_period(),
                    time_diff_max=config.get_time_diff_max(),
                    get_latest_subdirectory_by_name=True
                )
                intersection_points = extract_intersection_points(dataloader=dataloader, line_movement_speed=line_movement_speed, line_movement_type=line_movement_type)
                if print_every_intersection_point:
                    print(f"{intersection_points} are the intersection points for {subject_dir}, {line_movement_speed}, {line_movement_type}")
                arr[i] = np.array(intersection_points)
            
            for col in range(arr.shape[1]):
                carr = arr[:, col]
                mean, std = np.mean(carr, axis=0), np.std(carr, axis=0)
                print(f"Intersection Point No: {col + 1} - Mean, std: {mean:.2f}, {std:.2f}")
            print(f"Finished processing {line_movement_speed}, {line_movement_type}")

def _save_corr_csv(results, out_csv_path: str, corr_label: str):
    """
    results: list of dicts with keys:
        subject_dir, speed, movement_type, line_segment, model, pearson, spearman, n
    corr_label: "pearson" or "spearman"
    """
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w") as f:
        f.write("subject_dir,speed,movement_type,line_segment,model,corr,n\n")
        for r in results:
            val = r["pearson"] if corr_label == "pearson" else r["spearman"]
            f.write(
                f"{r['subject_dir']},{r['speed']},{r['movement_type']},{r['line_segment']},"
                f"{r['model']},{'' if np.isnan(val) else f'{val:.4f}'},{r['n']}\n"
            )
    print(f"Wrote CSV: {out_csv_path}")

SIDE_MIN_RATIO = 0.33  # e.g., require at least 20% of samples on both sides for analyze_by_position.

def analyze_by_position(
    dataloader: GazeDataLoader,
    model: str,
    line_movement_speed: str,
    line_movement_type: str,
    plot: bool = False,
):
    """
    Split each line segment into two subsets by the sign of the position in the neutral cam frame:
      - Horizontal segments (Y axis):  pos -> left,  neg -> right
      - Vertical   segments (Z axis):  pos -> up,    neg -> down

    Returns:
        List[Tuple[int, str, float, float, int]]: 
            [(line_segment_index, side, pearson_r, spearman_rho, n_samples), ...]
            where side ∈ {"left","right"} or {"up","down"} depending on axis.
    """
    if FRAME == "neutral_eye":
        neutral_eye_pose_in_world_frame = load_neutral_eye_pose_in_world_frame(subject_dir=dataloader.get_subject_dir(), csv_path=config.get_neutral_eye_position_per_subject_csv_path())
        target_positions = dataloader.load_target_positions(frame="neutral_eye", neutral_eye_pose_in_world_frame=neutral_eye_pose_in_world_frame) # [ts_ms, x, y, z], in meters, in the neutral eye frame of the subject.
    else: # FRAME == neutral_camera frame
        target_positions = dataloader.load_target_positions(frame="neutral_camera") # [ts_ms, x, y, z], in meters, in the neutral eye frame of the subject.
    # gaze estimation errors
    err_res = process_direction_errors(
        dataloader=dataloader,
        model=model,
        gaze_target_bias=None,
        save_to_file=False,
    )

    # 4 intersections → 5 segments
    intersection_points = extract_intersection_points(
        dataloader,
        line_movement_speed=line_movement_speed,
        line_movement_type=line_movement_type,
        visualize_for_debug=False
    )

    def _corr_safe(x: np.ndarray, y: np.ndarray):
        """Return (pearson_r, spearman_rho) or (nan, nan) if degenerate."""
        m = min(len(x), len(y))
        if m < 2:
            return np.nan, np.nan
        x = x[:m]; y = y[:m]
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]; y = y[mask]
        if x.size < 2:
            return np.nan, np.nan
        if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
            return np.nan, np.nan
        pearson_r = float(np.corrcoef(x, y)[0, 1])
        from scipy.stats import spearmanr
        rho, _ = spearmanr(x, y)
        rho = float(rho) if np.isfinite(rho) else np.nan
        return pearson_r, rho

    def _side_labels(is_horizontal: bool, sign_positive: bool) -> str:
        if FRAME == "neutral_eye":
            if is_horizontal:
                return "right" if sign_positive else "left"
            else:
                return "up" if sign_positive else "down"
        else: # FRAME == neutral_camera frame
            if is_horizontal:
                return "right" if not sign_positive else "left"
            else:
                return "up" if sign_positive else "down"
        

    results = []

    def process(is_line_horizontal: bool, line_segment_index: int, target_positions: np.ndarray = target_positions, err_res: np.ndarray = err_res):
        s_index = 0 if line_segment_index == 0 else intersection_points[line_segment_index - 1]
        e_index = (intersection_points[line_segment_index]
                   if line_segment_index < len(intersection_points)
                   else target_positions.shape[0])

        # time-window subset
        target_positions_on_line_seg = target_positions[s_index:e_index]
        s_ts, e_ts = target_positions[s_index, 0], target_positions[e_index - 1, 0]

        # errors in the same time window
        err_res_on_line_seg = err_res[(err_res[:, 0] >= s_ts) & (err_res[:, 0] <= e_ts)]

        # UPDATED: align and unpack (filtered irregular rows + matched regular values)
        err_res_aligned, matched_target_vals = match_irregular_to_regular(
            irregular_data=err_res_on_line_seg,          # timestamps in col 0
            regular_data=target_positions_on_line_seg,   # [t, x, y, z]
            regular_period_ms=dataloader.target_period
        )

        # Guard: empty after filtering
        if err_res_aligned.shape[0] == 0:
            n_pos = n_neg = 0
            # (If caller expects arrays, you can also set target_positions = errors = np.empty((0,), dtype=float))
        else:
            axis = _get_axis(is_line_horizontal=is_line_horizontal, frame=FRAME)  # 0,1,2 over [x,y,z]
            target_positions = matched_target_vals[:, axis]   # regular VALUES (no timestamp)
            errors = err_res_aligned[:, 1]                    # angular error column from irregular side
            pos_mask = target_positions > 0
            neg_mask = target_positions < 0
            n_pos, n_neg = np.sum(pos_mask), np.sum(neg_mask)
        total = n_pos + n_neg

        if total < 2:
            return  # too few samples

        min_ratio = min(n_pos, n_neg) / total

        if min_ratio >= SIDE_MIN_RATIO:
            # Balanced enough → keep both sides
            if n_pos > 0:
                pr_pos, sr_pos = _corr_safe(np.abs(target_positions[pos_mask]), errors[pos_mask])
                results.append((line_segment_index, _side_labels(is_line_horizontal, True), pr_pos, sr_pos, int(n_pos)))
            if n_neg > 0:
                pr_neg, sr_neg = _corr_safe(np.abs(target_positions[neg_mask]), errors[neg_mask])
                results.append((line_segment_index, _side_labels(is_line_horizontal, False), pr_neg, sr_neg, int(n_neg)))
        else:
            # Too skewed → only use the majority side
            if n_pos >= n_neg:
                pr, sr = _corr_safe(np.abs(target_positions[pos_mask]), errors[pos_mask])
                results.append((line_segment_index, _side_labels(is_line_horizontal, True), pr, sr, int(n_pos)))
            else:
                pr, sr = _corr_safe(np.abs(target_positions[neg_mask]), errors[neg_mask])
                results.append((line_segment_index, _side_labels(is_line_horizontal, False), pr, sr, int(n_neg)))

        # Optional plot (entire segment as before; no side coloring to keep it minimal)
        if plot:
            plt.figure()
            plt.plot(100.0 * target_positions, errors)  # cm
            hor_minus, hor_plus = _get_horizontal_signs(frame=FRAME)
            if is_line_horizontal:
                plt.xlabel(f"Horizontal Position (cm)  (- is {hor_minus}, + is {hor_plus} in the {FRAME} frame)")
            else:
                plt.xlabel(f"Vertical Position (cm)    (- is down, + is up in the {FRAME} frame)")
            plt.ylabel("Gaze Estimation Error (deg)")
            plt.title(
                f"Error vs {'Horizontal' if is_line_horizontal else 'Vertical'} Position "
                f"(type={line_movement_type}, speed={line_movement_speed}, line={line_segment_index})"
            )
            plt.grid(True)
            plt.show()

    for line_segment_index in range(len(intersection_points) + 1):
        is_line_horizontal = (
            (line_movement_type == "horizontal" and line_segment_index % 2 == 0) or
            (line_movement_type == "vertical"   and line_segment_index % 2 == 1)
        )
        process(is_line_horizontal=is_line_horizontal, line_segment_index=line_segment_index)

    return results  # [(seg_idx, side, pearson, spearman, n)]

def _save_corr_csv(results, out_csv_path: str, corr_label: str):
    """
    results: list of dicts with keys:
        subject_dir, speed, movement_type, line_segment, side, model, pearson, spearman, n
    corr_label: "pearson" or "spearman"
    """
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w") as f:
        f.write("subject_dir,speed,movement_type,line_segment,side,model,corr,n\n")
        for r in results:
            val = r["pearson"] if corr_label == "pearson" else r["spearman"]
            f.write(
                f"{r['subject_dir']},{r['speed']},{r['movement_type']},{r['line_segment']},"
                f"{r['side']},{r['model']},{'' if np.isnan(val) else f'{val:.4f}'},{r['n']}\n"
            )
    print(f"Wrote CSV: {out_csv_path}")


def line_movement_analysis_by_position(
    subject_dirs: list[str],
    models: list[str],
    output_dir_for_text_files: str,
    speeds: list[str] = ("fast", "slow"),
    movement_types: list[str] = None,
    plot: bool = False,
):
    """
    For each subject and each combo (fast/slow × horizontal/vertical),
    compute correlations per line segment and per side (left/right or up/down).
    Produces two CSVs: pearson and spearman, each with a 'side' column.
    """
    if movement_types is None:
        movement_types = list(config.get_line_movement_types())

    all_rows = []  # one row per (subject, combo, segment, side, model)

    for subj_dir in subject_dirs:
        for speed in speeds:
            for mv_type in movement_types:
                exp_dir = os.path.join(subj_dir, f"line_movement_{speed}", mv_type)
                try:
                    dataloader = GazeDataLoader(
                        root_dir=exp_dir,
                        target_period=config.get_target_period(),
                        camera_pose_period=config.get_camera_pose_period(),
                        time_diff_max=config.get_time_diff_max(),
                        get_latest_subdirectory_by_name=True
                    )

                    for model in models:
                        seg_results = analyze_by_position(
                            dataloader=dataloader,
                            model=model,
                            line_movement_speed=speed,
                            line_movement_type=mv_type,
                            plot=plot
                        )
                        # seg_results: List[(seg_idx, side, pearson, spearman, n)]
                        for seg_idx, side, pr, sr, n in seg_results:
                            all_rows.append({
                                "subject_dir": subj_dir,
                                "speed": speed,
                                "movement_type": mv_type,
                                "line_segment": seg_idx,
                                "side": side,
                                "model": model,
                                "pearson": pr,
                                "spearman": sr,
                                "n": n,
                            })
                except Exception as e:
                        print(f"Skip {exp_dir}: {e}")
                        continue
    pearson_csv = os.path.join(output_dir_for_text_files, "line_movement_by_position_pearson.csv")
    spearman_csv = os.path.join(output_dir_for_text_files, "line_movement_by_position_spearman.csv")
    _save_corr_csv(all_rows, pearson_csv, corr_label="pearson")
    _save_corr_csv(all_rows, spearman_csv, corr_label="spearman")

    return all_rows

def _read_corr_csv_file(path: str):
    """
    Reads a correlation CSV produced by _save_corr_csv (with 'side' column).
    Returns: List[dict] with parsed types; drops rows where corr is empty/NaN.
    Expected headers:
      subject_dir,speed,movement_type,line_segment,side,model,corr,n
    """
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            corr_str = (r.get("corr") or "").strip()
            if corr_str == "":
                continue  # skip empty correlation cells
            try:
                corr_val = float(corr_str)
            except ValueError:
                continue

            try:
                n_val = int(r.get("n", "0"))
            except ValueError:
                n_val = 0

            # line_segment may be blank for malformed rows; coerce to -1 then
            seg = r.get("line_segment", "").strip()
            seg_val = int(seg) if seg != "" else -1

            rows.append({
                "subject_dir": r.get("subject_dir", ""),
                "speed": r.get("speed", ""),
                "movement_type": r.get("movement_type", ""),
                "line_segment": seg_val,
                "side": r.get("side", ""),
                "model": r.get("model", ""),
                "corr": corr_val,
                "n": n_val,
            })
    return rows


def _weighted_avg_std(vals: np.ndarray, weights: np.ndarray):
    """
    Return (wmean, wstd). If weights are invalid/zero, returns (np.nan, np.nan).
    """
    if vals.size == 0 or weights.size == 0 or np.all(weights <= 0):
        return np.nan, np.nan
    try:
        wmean = np.average(vals, weights=weights)
        var = np.average((vals - wmean) ** 2, weights=weights)
        return float(wmean), float(np.sqrt(var))
    except Exception:
        return np.nan, np.nan


def _aggregate_rows(rows, group_keys):
    """
    rows: list of dicts from _read_corr_csv_file
    group_keys: tuple/list of keys to group by, e.g. 
        ("movement_type", "speed", "line_segment", "side", "model") or
        ("movement_type", "speed", "side", "model")
    Returns: list of aggregated dicts with stats.
    Stats:
      count_rows, sum_n, mean, median, std, wmean_by_n, wstd_by_n
    """
    from collections import defaultdict

    buckets = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        buckets[key].append(r)

    out = []
    for key, items in buckets.items():
        vals = np.array([it["corr"] for it in items], dtype=float)
        ns   = np.array([it["n"]    for it in items], dtype=float)

        mean   = float(np.mean(vals))   if vals.size else np.nan
        median = float(np.median(vals)) if vals.size else np.nan
        std    = float(np.std(vals))    if vals.size else np.nan
        wmean, wstd = _weighted_avg_std(vals, ns)

        agg_row = {k: v for k, v in zip(group_keys, key)}
        agg_row.update({
            "count_rows": int(vals.size),
            "sum_n": int(np.sum(ns)) if ns.size else 0,
            "mean": round(mean, 6) if np.isfinite(mean) else "",
            "median": round(median, 6) if np.isfinite(median) else "",
            "std": round(std, 6) if np.isfinite(std) else "",
            "wmean_by_n": round(wmean, 6) if np.isfinite(wmean) else "",
            "wstd_by_n": round(wstd, 6) if np.isfinite(wstd) else "",
        })
        out.append(agg_row)

    out.sort(key=lambda d: tuple(d[k] for k in group_keys))
    return out


def _write_agg_csv(rows, out_path: str, group_keys):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    headers = list(group_keys) + ["count_rows", "sum_n", "mean", "median", "std", "wmean_by_n", "wstd_by_n"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow({h: r.get(h, "") for h in headers})
    print(f"Wrote aggregation CSV: {out_path}")


def aggregate_line_movement_csv(
    corr_csv_path: str,
    output_dir_for_text_files: str,
    label: str,  # "pearson" or "spearman"
):
    """
    Load a saved correlation CSV and emit FOUR aggregation CSVs with model in the grouping:
      1) (movement_type, speed, line_segment, side, model)
      2) (movement_type, speed, side, model)
      3) (movement_type, line_segment, side, model)
      4) (movement_type, side, model)
    """
    rows = _read_corr_csv_file(corr_csv_path)
    if not rows:
        print(f"No valid rows found in {corr_csv_path}. Skipping.")
        return

    groupings = [
        ("movement_type", "speed", "line_segment", "side", "model"),
        ("movement_type", "speed", "side", "model"),
        ("movement_type", "line_segment", "side", "model"),
        ("movement_type", "side", "model"),
    ]

    suffixes = [
        "agg_by_type_speed_segment_side_model",
        "agg_by_type_speed_side_model",
        "agg_by_type_segment_side_model",
        "agg_by_type_side_model",
    ]

    for grp_keys, suffix in zip(groupings, suffixes):
        agg_rows = _aggregate_rows(rows, grp_keys)
        out_path = os.path.join(
            output_dir_for_text_files,
            f"line_movement_by_position_{label}__{suffix}.csv"
        )
        _write_agg_csv(agg_rows, out_path, grp_keys)


def aggregate_all_saved_line_movement_results(output_dir_for_text_files: str):
    """
    Reads your standard pearson/spearman CSVs and writes 8 aggregated CSVs
    (four groupings × two correlation types), all including 'model' in the groups.
    """
    pearson_csv = os.path.join(output_dir_for_text_files, "line_movement_by_position_pearson.csv")
    spearman_csv = os.path.join(output_dir_for_text_files, "line_movement_by_position_spearman.csv")

    if os.path.isfile(pearson_csv):
        aggregate_line_movement_csv(pearson_csv, output_dir_for_text_files, label="pearson")
    else:
        print(f"Not found: {pearson_csv}")

    if os.path.isfile(spearman_csv):
        aggregate_line_movement_csv(spearman_csv, output_dir_for_text_files, label="spearman")
    else:
        print(f"Not found: {spearman_csv}")

def _safe_float(x):
    if x is None:
        return math.nan
    s = str(x).strip()
    if s == "":
        return math.nan
    try:
        return float(s)
    except Exception:
        return math.nan

def _read_agg_csv_rows(path: str):
    """
    Reads an aggregated CSV produced by _write_agg_csv.
    Returns a list[dict] with numeric fields coerced where possible.
    Expected headers include:
      movement_type, speed (maybe absent depending on grouping), line_segment (maybe),
      side, model,
      count_rows, sum_n, mean, median, std, wmean_by_n, wstd_by_n
    """
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # keep strings for grouping columns; coerce numerics
            r_out = dict(r)
            # Normalize numerics
            for k in ("line_segment", "count_rows", "sum_n"):
                if k in r_out:
                    try:
                        r_out[k] = int(r_out[k])
                    except Exception:
                        r_out[k] = None
            for k in ("mean", "median", "std", "wmean_by_n", "wstd_by_n"):
                if k in r_out:
                    r_out[k] = _safe_float(r_out[k])
            rows.append(r_out)
    return rows

def _pick_metric(row, prefer_weighted=True):
    """
    Returns (y, yerr, metric_name) choosing weighted if available; else unweighted.
    yerr returns NaN if not available.
    """
    if prefer_weighted:
        y = row.get("wmean_by_n", math.nan)
        yerr = row.get("wstd_by_n", math.nan)
        if not math.isnan(y):
            return y, yerr, "wmean_by_n"
    # fallback
    y = row.get("mean", math.nan)
    yerr = row.get("std", math.nan)
    return y, yerr, "mean"

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# 1) Fast vs Slow: grouped by (movement_type, speed, line_segment, side, model)
def compare_fast_vs_slow_by_segment_side_model(
    agg_csv_path: str,
    out_dir: str,
    prefer_weighted: bool = True,
    file_label: str = "pearson"  # only used in filename
):
    """
    For an aggregated CSV grouped by (movement_type, speed, line_segment, side, model),
    create a 2-bar plot (FAST vs SLOW) for every tuple (movement_type, line_segment, side, model)
    that has BOTH speeds present. Save PNGs; do not show.
    """
    _ensure_dir(out_dir)
    rows = _read_agg_csv_rows(agg_csv_path)

    # bucket by key WITHOUT speed
    from collections import defaultdict
    buckets = defaultdict(dict)  # key -> {"fast": row, "slow": row}
    for r in rows:
        key = (
            r.get("movement_type", ""),
            r.get("line_segment", None),
            r.get("side", ""),
            r.get("model", ""),
        )
        sp = r.get("speed", "").lower()
        if sp in ("fast", "slow"):
            buckets[key][sp] = r

    for key, d in buckets.items():
        if "fast" not in d or "slow" not in d:
            continue  # need both to compare

        mt, seg, side, model = key

        # pull metrics
        y_fast, err_fast, metric_name = _pick_metric(d["fast"], prefer_weighted=prefer_weighted)
        y_slow, err_slow, _ = _pick_metric(d["slow"], prefer_weighted=prefer_weighted)

        # make plot
        fig = plt.figure(figsize=(4.6, 3.2))
        xs = [0, 1]
        ys = [y_fast, y_slow]
        yerrs = [err_fast if not math.isnan(err_fast) else 0.0,
                 err_slow if not math.isnan(err_slow) else 0.0]

        plt.bar(xs, ys, yerr=yerrs, capsize=3)
        plt.xticks(xs, ["FAST", "SLOW"])
        plt.ylabel(f"{file_label.upper()} ({metric_name})")
        title = f"{mt}, seg={seg}, side={side}, model={model}"
        plt.title(title)
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()

        safe_model = str(model).replace("/", "_")
        fname = f"{file_label}_fast_vs_slow__{mt}__seg{seg}__side-{side}__model-{safe_model}.png"
        fpath = os.path.join(out_dir, fname)
        plt.savefig(fpath, dpi=150)
        plt.close(fig)

#2) Fast vs Slow: grouped by (movement_type, speed, side, model)
def compare_fast_vs_slow_by_side_model(
    agg_csv_path: str,
    out_dir: str,
    prefer_weighted: bool = True,
    file_label: str = "pearson"
):
    """
    For an aggregated CSV grouped by (movement_type, speed, side, model),
    create a 2-bar plot (FAST vs SLOW) for every tuple (movement_type, side, model)
    that has BOTH speeds present. Save PNGs; do not show.
    """
    _ensure_dir(out_dir)
    rows = _read_agg_csv_rows(agg_csv_path)

    from collections import defaultdict
    buckets = defaultdict(dict)  # key -> {"fast": row, "slow": row}
    for r in rows:
        key = (
            r.get("movement_type", ""),
            r.get("side", ""),
            r.get("model", ""),
        )
        sp = r.get("speed", "").lower()
        if sp in ("fast", "slow"):
            buckets[key][sp] = r

    for key, d in buckets.items():
        if "fast" not in d or "slow" not in d:
            continue

        mt, side, model = key
        y_fast, err_fast, metric_name = _pick_metric(d["fast"], prefer_weighted=prefer_weighted)
        y_slow, err_slow, _ = _pick_metric(d["slow"], prefer_weighted=prefer_weighted)

        fig = plt.figure(figsize=(4.6, 3.2))
        xs = [0, 1]
        ys = [y_fast, y_slow]
        yerrs = [err_fast if not math.isnan(err_fast) else 0.0,
                 err_slow if not math.isnan(err_slow) else 0.0]

        plt.bar(xs, ys, yerr=yerrs, capsize=3)
        plt.xticks(xs, ["FAST", "SLOW"])
        plt.ylabel(f"{file_label.upper()} ({metric_name})")
        title = f"{mt}, side={side}, model={model}"
        plt.title(title)
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()

        safe_model = str(model).replace("/", "_")
        fname = f"{file_label}_fast_vs_slow__{mt}__side-{side}__model-{safe_model}.png"
        fpath = os.path.join(out_dir, fname)
        plt.savefig(fpath, dpi=150)
        plt.close(fig)

def _extract_segment_series_for_subject(
    subject_dir: str,
    model: str,
    line_movement_speed: str,
    line_movement_type: str,
    line_segment_index: int,
):
    """
    Returns (x_cm, err_deg) for a single subject and segment using the SAME pipeline as analyze_by_position:
      - target positions in neutral eye frame
      - same intersection detection
      - same timestamp-windowing
      - same irregular→regular matching
      - axis selection identical to analyze_by_position (vertical→Y=1, horizontal→Z=2)
    On any failure, returns (None, None) so the caller can skip.
    """
    try:
        exp_dir = os.path.join(subject_dir, f"line_movement_{line_movement_speed}", line_movement_type)
        dataloader = GazeDataLoader(
            root_dir=exp_dir,
            target_period=config.get_target_period(),
            camera_pose_period=config.get_camera_pose_period(),
            time_diff_max=config.get_time_diff_max(),
            get_latest_subdirectory_by_name=True
        )

        if FRAME == "neutral_eye":
            # load neutral eye frame + target positions (exactly as in analyze_by_position)
            neutral_eye_pose_in_world_frame = load_neutral_eye_pose_in_world_frame(
                subject_dir=dataloader.get_subject_dir(),
                csv_path=config.get_neutral_eye_position_per_subject_csv_path()
            )
            target_positions_full = dataloader.load_target_positions(
                frame="neutral_eye",
                neutral_eye_pose_in_world_frame=neutral_eye_pose_in_world_frame
            )  # [ts_ms, x, y, z] (m), neutral eye frame
        else:
            target_positions_full = dataloader.load_target_positions(
                frame="neutral_camera",
            )  # [ts_ms, x, y, z] (m), neutral eye frame  

        # errors
        err_res = process_direction_errors(
            dataloader=dataloader,
            model=model,
            gaze_target_bias=None,
            save_to_file=False,
        )

        # segment boundaries
        intersection_points = extract_intersection_points(
            dataloader,
            line_movement_speed=line_movement_speed,
            line_movement_type=line_movement_type,
            visualize_for_debug=False
        )
        # compute indices for this segment (same logic)
        s_index = 0 if line_segment_index == 0 else intersection_points[line_segment_index - 1]
        e_index = (intersection_points[line_segment_index]
                   if line_segment_index < len(intersection_points)
                   else target_positions_full.shape[0])

        if s_index >= e_index or e_index <= 0:
            return None, None

        # time window subset
        target_positions_seg = target_positions_full[s_index:e_index]
        s_ts, e_ts = target_positions_full[s_index, 0], target_positions_full[e_index - 1, 0]

        # error subset
        err_res_seg = err_res[(err_res[:, 0] >= s_ts) & (err_res[:, 0] <= e_ts)]

        if err_res_seg.shape[0] < 2:
            return None, None

        # match to handle frequency discrepancy (unchanged)
        err_res_seg, matched_target_positions = match_irregular_to_regular(
            irregular_data=err_res_seg,
            regular_data=target_positions_seg,
            regular_period_ms=dataloader.target_period   # <-- FIX: use target period, not camera pose period
        )

        is_line_horizontal = line_movement_type == "horizontal"
        axis = _get_axis(is_line_horizontal=is_line_horizontal, frame=FRAME)
        x = matched_target_positions[:, axis]  # meters; signed
        err = err_res_seg[:, 1]                # degrees

        # return signed positions in cm (like your plotting), and error
        return 100.0 * x, err

    except Exception as e:
        print(f"[visualization] Skip subject {subject_dir} ({line_movement_speed}, {line_movement_type}, seg={line_segment_index}, model={model}): {e}")
        return None, None


def visualize_subject_variability_for_segment(
    subject_dirs: list[str],
    model: str,
    line_movement_speed: str,
    line_movement_type: str,
    line_segment_index: int,
    plot: bool = False,
):
    """
    Same as before, but each 4-up window shares identical x/y limits so you can
    eyeball variability fairly. No files saved; only shown if plot=True.
    Returns list of dicts: {subject_dir, x_cm, err_deg, n}.
    """
    max_per_window = 4
    # --- collect series (unchanged) ---
    collected = []
    for subj in subject_dirs:
        x_cm, err_deg = _extract_segment_series_for_subject(
            subject_dir=subj,
            model=model,
            line_movement_speed=line_movement_speed,
            line_movement_type=line_movement_type,
            line_segment_index=line_segment_index,
        )
        if x_cm is None or err_deg is None:
            continue
        n = min(len(x_cm), len(err_deg))
        if n < 2:
            continue
        collected.append({
            "subject_dir": subj,
            "x_cm": x_cm[:n],
            "err_deg": err_deg[:n],
            "n": n,
        })

    if not collected:
        print("[visualization] No valid series to plot.")
        return collected

    import math as _math
    num = len(collected)
    chunks = (num + max_per_window - 1) // max_per_window
    idx = 0

    for _ in range(chunks):
        # slice the entries that will appear in THIS window
        entries = collected[idx: idx + max_per_window]
        if not entries:
            break

        # compute common axis limits for this window
        x_min = min(float(np.min(e["x_cm"])) for e in entries)
        x_max = max(float(np.max(e["x_cm"])) for e in entries)
        y_min = min(float(np.min(e["err_deg"])) for e in entries)
        y_max = max(float(np.max(e["err_deg"])) for e in entries)

        # add small padding so lines aren’t glued to borders
        def _pad(lo, hi, frac=0.05):
            span = hi - lo
            if span <= 0:
                # degenerate: pad by fixed small amount
                return lo - 1.0, hi + 1.0
            pad = frac * span
            return lo - pad, hi + pad

        x_lo, x_hi = _pad(x_min, x_max, 0.03)
        y_lo, y_hi = _pad(y_min, y_max, 0.08)

        # build figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes = axes.flatten()

        filled = 0
        for k, entry in enumerate(entries):
            ax = axes[k]
            ax.plot(entry["x_cm"], entry["err_deg"], linewidth=1.2)
            # show last two path components (e.g., 2025-07-29/Name) for clarity
            parts = entry["subject_dir"].rstrip("/").split(os.sep)
            title = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            ax.set_title(title)
            ax.set_xlabel("Position (cm)")
            ax.set_ylabel("Gaze Error (deg)")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            filled += 1

        # hide unused subplots
        for k in range(filled, len(axes)):
            axes[k].axis("off")

        fig.suptitle(
            f"{line_movement_type}, {line_movement_speed}, segment={line_segment_index}, model={model}",
            fontsize=12
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if plot:
            plt.show()
        else:
            plt.close(fig)

        idx += max_per_window

    return collected

def visualize_speed_variability_for_segment(
    subject_dirs: list[str],
    model: str,
    line_movement_type: str,
    line_segment_index: int,
    plot: bool = False,
):
    """
    Compare FAST vs SLOW for the same (movement_type, segment, model) across subjects.
    Layout: max_per_window rows × 2 cols  [FAST | SLOW]
    - Uses the exact same extraction path as analyze_by_position (neutral-eye frame, same matching, same segment boundaries).
    - Only includes subjects that have BOTH speeds.
    - Enforces identical x/y limits across the entire window for eyeballing.
    - Does NOT save; shows only if plot=True.

    Returns: List[Dict] with keys:
      subject_dir, x_cm_fast, err_deg_fast, n_fast, x_cm_slow, err_deg_slow, n_slow
    """
    max_per_window = 3
    # collect paired series (must have both speeds)
    paired = []
    for subj in subject_dirs:
        # FAST
        x_f, e_f = _extract_segment_series_for_subject(
            subject_dir=subj,
            model=model,
            line_movement_speed="fast",
            line_movement_type=line_movement_type,
            line_segment_index=line_segment_index,
        )
        # SLOW
        x_s, e_s = _extract_segment_series_for_subject(
            subject_dir=subj,
            model=model,
            line_movement_speed="slow",
            line_movement_type=line_movement_type,
            line_segment_index=line_segment_index,
        )

        if x_f is None or e_f is None or x_s is None or e_s is None:
            continue

        n_f = min(len(x_f), len(e_f))
        n_s = min(len(x_s), len(e_s))
        if n_f < 2 or n_s < 2:
            continue

        paired.append({
            "subject_dir": subj,
            "x_cm_fast": x_f[:n_f],
            "err_deg_fast": e_f[:n_f],
            "n_fast": n_f,
            "x_cm_slow": x_s[:n_s],
            "err_deg_slow": e_s[:n_s],
            "n_slow": n_s,
        })

    if not paired:
        print("[speed-variability] No valid FAST/SLOW pairs to plot.")
        return paired

    # chunk into windows
    total = len(paired)
    windows = (total + max_per_window - 1) // max_per_window
    idx = 0

    for _ in range(windows):
        entries = paired[idx: idx + max_per_window]
        if not entries:
            break

        # compute common limits across BOTH speeds for ALL entries in this window
        x_min = min(float(np.min(np.concatenate((e["x_cm_fast"], e["x_cm_slow"])))) for e in entries)
        x_max = max(float(np.max(np.concatenate((e["x_cm_fast"], e["x_cm_slow"])))) for e in entries)
        y_min = min(float(np.min(np.concatenate((e["err_deg_fast"], e["err_deg_slow"])))) for e in entries)
        y_max = max(float(np.max(np.concatenate((e["err_deg_fast"], e["err_deg_slow"])))) for e in entries)

        def _pad(lo, hi, frac=0.05):
            span = hi - lo
            if span <= 0:
                return lo - 1.0, hi + 1.0
            pad = frac * span
            return lo - pad, hi + pad

        x_lo, x_hi = _pad(x_min, x_max, 0.03)
        y_lo, y_hi = _pad(y_min, y_max, 0.08)

        # build figure
        fig, axes = plt.subplots(len(entries), 2, figsize=(10, 3.2 * len(entries)))
        if len(entries) == 1:
            axes = np.array([axes])  # normalize to 2D [rows, 2]

        # column headers
        axes[0, 0].set_title("FAST")
        axes[0, 1].set_title("SLOW")

        for r, entry in enumerate(entries):
            # nicer subject label: last two path components
            parts = entry["subject_dir"].rstrip("/").split(os.sep)
            subj_label = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]

            # FAST
            ax_f = axes[r, 0]
            ax_f.plot(entry["x_cm_fast"], entry["err_deg_fast"], linewidth=1.2)
            ax_f.set_xlim(x_lo, x_hi)
            ax_f.set_ylim(y_lo, y_hi)
            ax_f.set_ylabel("Gaze Error (deg)")
            ax_f.set_xlabel("Position (cm)")
            ax_f.grid(True, alpha=0.3)
            ax_f.text(0.02, 0.92, subj_label, transform=ax_f.transAxes, fontsize=10, ha="left", va="top")

            # SLOW
            ax_s = axes[r, 1]
            ax_s.plot(entry["x_cm_slow"], entry["err_deg_slow"], linewidth=1.2)
            ax_s.set_xlim(x_lo, x_hi)
            ax_s.set_ylim(y_lo, y_hi)
            ax_s.set_ylabel("Gaze Error (deg)")
            ax_s.set_xlabel("Position (cm)")
            ax_s.grid(True, alpha=0.3)

        fig.suptitle(f"{line_movement_type}, segment={line_segment_index}, model={model}  —  FAST vs SLOW", fontsize=12)
        fig.tight_layout(rect=[0, 0.04, 1, 0.96])

        if plot:
            plt.show()
        else:
            plt.close(fig)

        idx += max_per_window

    return paired

def visualize_error_field_yz(
    subject_dir: str,
    model: str,
    line_movement_speed: str,
    line_movement_type: str,
    bins=(40, 40),            # (Z bins, Y bins)
    show_plot: bool = False,
    save_plot: bool = True,
):
    """
    Visualize mean gaze error over the Y–Z plane (neutral eye frame) for a single subject's run.
    - Uses the SAME pipeline as analyze_by_position: neutral-eye targets, identical matching.
    - Aggregates the entire experiment window (all segments) for the given speed/type.
    - Produces a heatmap (default) or a 3D surface of mean error per (Z,Y) bin.
    - Axes are in centimeters. Error is in degrees.

    Returns:
        dict with keys: {"Z_edges_cm","Y_edges_cm","mean_error","counts","subject_dir","model",
                         "movement_type","speed","kind","fig_path"}  (fig_path None if not saved)
    """
    # --- Build loader exactly like analyze_by_position ---
    exp_dir = os.path.join(subject_dir, f"line_movement_{line_movement_speed}", line_movement_type)
    dataloader = GazeDataLoader(
        root_dir=exp_dir,
        target_period=config.get_target_period(),
        camera_pose_period=config.get_camera_pose_period(),
        time_diff_max=config.get_time_diff_max(),
        get_latest_subdirectory_by_name=True
    )
    out_dir = dataloader.get_gaze_estimations_dir(model=model)
    kind = "heatmap"

    if FRAME == "neutral_eye":
        # Neutral eye frame targets (identical to analyze_by_position)
        neutral_eye_pose_in_world_frame = load_neutral_eye_pose_in_world_frame(
            subject_dir=dataloader.get_subject_dir(),
            csv_path=config.get_neutral_eye_position_per_subject_csv_path()
        )
        target_positions_full = dataloader.load_target_positions(
            frame="neutral_eye",
            neutral_eye_pose_in_world_frame=neutral_eye_pose_in_world_frame
        )  # [ts_ms, x, y, z] in meters
    else:
        target_positions_full = dataloader.load_target_positions(
            frame="neutral_camera",
        )  # [ts_ms, x, y, z] in meters   

    # Errors
    err_res = process_direction_errors(
        dataloader=dataloader,
        model=model,
        gaze_target_bias=None,
        save_to_file=False,
    )
    if err_res.shape[0] < 2 or target_positions_full.shape[0] < 2:
        print("[error-field] Not enough data points.")
        return {
            "Z_edges_cm": None, "Y_edges_cm": None, "mean_error": None, "counts": None,
            "subject_dir": subject_dir, "model": model, "movement_type": line_movement_type,
            "speed": line_movement_speed, "kind": kind, "fig_path": None,
        }

    # Match targets (regular) to error timestamps (irregular) — unchanged
    err_res, matched_target_positions = match_irregular_to_regular(
        irregular_data=err_res,
        regular_data=target_positions_full,
        regular_period_ms=dataloader.camera_pose_period
    )

    # Extract (vertical) and (horizontal) in the frame;
    Y_m = matched_target_positions[:, _get_axis(is_line_horizontal=False, frame=FRAME)]
    Z_m = matched_target_positions[:, _get_axis(is_line_horizontal=True, frame=FRAME)]
    ERR = err_res[:, 1]

    # To centimeters for plotting/interpretation
    Y = 100.0 * Y_m
    Z = 100.0 * Z_m

    # 2D binning: mean error per (Z,Y) bin using histogram2d (sum/ count)
    z_bins, y_bins = (int(bins[0]), int(bins[1]))
    Z_edges = np.linspace(np.nanmin(Z), np.nanmax(Z), z_bins + 1)
    Y_edges = np.linspace(np.nanmin(Y), np.nanmax(Y), y_bins + 1)
    # indexing = [None, Z_edges, Y_edges] # I will use this since I am updating the code and I do not want to rewrite everything.
    # Z_edges, Y_edges = indexing[_get_axis(is_line_horizontal=False, frame=FRAME)] , indexing[_get_axis(is_line_horizontal=True, frame=FRAME)]
    # Sum of errors in each bin
    sum_err, _, _ = np.histogram2d(Z, Y, bins=[Z_edges, Y_edges], weights=ERR)
    # Count per bin
    counts, _, _ = np.histogram2d(Z, Y, bins=[Z_edges, Y_edges])

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_err = sum_err / counts
    mean_err[counts == 0] = np.nan  # no data → NaN

    # Plot
    fig_path = None
    if show_plot or save_plot:
        hor_minus, hor_plus = _get_horizontal_signs(frame=FRAME)
        fig = plt.figure(figsize=(7.2, 5.6))
        # pcolormesh expects edges; transpose so Y is vertical axis
        ZZ, YY = np.meshgrid(Z_edges, Y_edges, indexing="ij")
        pcm = plt.pcolormesh(ZZ, YY, mean_err, shading="auto")
        cbar = plt.colorbar(pcm)
        cbar.set_label("Mean Gaze Error (deg)")
        plt.xlabel(f"Horizontal position (cm)   [− {hor_minus}, + {hor_plus} in the {FRAME} frame]")
        plt.ylabel("Vertical position (cm)    [− down, + up]")
        subj_tail = "/".join(subject_dir.rstrip("/").split(os.sep)[-2:])
        plt.title(f"Error field (mean) — {line_movement_type}, {line_movement_speed}, model={model}\n{subj_tail}")
        plt.grid(False)
        plt.tight_layout()

        if save_plot:
            os.makedirs(out_dir, exist_ok=True)
            safe_model = str(model).replace("/", "_")
            subj_tail = "_".join(subject_dir.rstrip("/").split(os.sep)[-2:])
            fname = f"error_field_yz__{line_movement_type}__{line_movement_speed}__model-{safe_model}__{subj_tail}__{kind}_{FRAME}.png"
            fig_path = os.path.join(out_dir, fname)
            plt.savefig(fig_path, dpi=160)
        if show_plot:
            plt.show()
        else:
            plt.close()

    return {
        "Z_edges_cm": Z_edges,
        "Y_edges_cm": Y_edges,
        "mean_error": mean_err,   # shape: [z_bins, y_bins]
        "counts": counts,
        "subject_dir": subject_dir,
        "model": model,
        "movement_type": line_movement_type,
        "speed": line_movement_speed,
        "kind": kind,
        "fig_path": fig_path,
    }

def visualize_error_field_yz_for_all(subject_dirs, models):
    for subject_dir in subject_dirs:
        for line_movement_speed in ["fast", "slow"]:
            for line_movement_type in config.get_line_movement_types():
                for model in models:
                    visualize_error_field_yz(
                        subject_dir=subject_dir,
                        model=model,
                        line_movement_speed=line_movement_speed,
                        line_movement_type=line_movement_type,
                        bins=(48, 48),
                        show_plot=False,
                        save_plot=True,
                    )

from collections import defaultdict
from scipy.ndimage import gaussian_filter

def _nan_gaussian(arr, sigma, min_w=1e-6):
    """Gaussian smooth a grid with NaNs by smoothing both values and a validity mask."""
    if sigma is None or sigma <= 0:
        return arr
    valid = np.isfinite(arr).astype(np.float32)
    arr0  = np.nan_to_num(arr, nan=0.0)
    num = gaussian_filter(arr0, sigma=sigma, mode="nearest")
    den = gaussian_filter(valid, sigma=sigma, mode="nearest")
    out = num / np.maximum(den, min_w)
    out[den < min_w] = np.nan
    return out


from collections import defaultdict as _dd
from scipy.ndimage import gaussian_filter  # (kept if you later want light smoothing)
import numpy as np
import os
import matplotlib.pyplot as plt

def visualize_error_field_yz_across_subjects(
    subject_dirs: list,
    model: str,
    line_movement_speed: str,
    line_movement_type: str,          # "horizontal" or "vertical"
    bins=(40, 40),                    # (horizontal_bins, vertical_bins) for the *continuous* axis
    show_plot: bool = False,
    save_plot: bool = True,
    v_band_cm: float = 2.0,           # half-width around vertical rails  (-10, 0, +10)
    h_band_cm: float = 2.0,           # half-width around horizontal rails (-25, 0, +25)
):
    """
    Across-subject error field restricted to canonical line segments ("rails"), producing thin stripes.

    - FRAME-aware (neutral_eye / neutral_camera).
    - Horizontal runs: keep only vertical ∈ {-10,0,+10} ± v_band_cm; horizontal axis is continuous.
    - Vertical   runs: keep only horizontal ∈ {-25,0,+25} ± h_band_cm; vertical   axis is continuous.

    Returns dict with edges, mean/median arrays (unclipped), counts, and figure paths.
    """
    if not subject_dirs:
        print("[error-field/all] subject_dirs is empty.")
        return None

    # Collect per-subject series (already frame-aware) --------------------------------------------
    all_H, all_V, all_ERR = [], [], []
    first_loader_for_save = None

    # canonical rails (in cm)
    V_RAILS = np.array([-10.0, 0.0, 10.0], dtype=float)   # vertical rails (Y in neutral_eye, Z in neutral_camera)
    H_RAILS = np.array([-25.0, 0.0, 25.0], dtype=float)   # horizontal rails (Z in neutral_eye, Y in neutral_camera)

    for subj in subject_dirs:
        try:
            exp_dir = os.path.join(subj, f"line_movement_{line_movement_speed}", line_movement_type)
            dataloader = GazeDataLoader(
                root_dir=exp_dir,
                target_period=config.get_target_period(),
                camera_pose_period=config.get_camera_pose_period(),
                time_diff_max=config.get_time_diff_max(),
                get_latest_subdirectory_by_name=True
            )
            if first_loader_for_save is None:
                first_loader_for_save = dataloader

            # Load target positions in requested FRAME
            if FRAME == "neutral_eye":
                neutral_eye_pose_in_world_frame = load_neutral_eye_pose_in_world_frame(
                    subject_dir=dataloader.get_subject_dir(),
                    csv_path=config.get_neutral_eye_position_per_subject_csv_path()
                )
                target_positions_full = dataloader.load_target_positions(
                    frame="neutral_eye",
                    neutral_eye_pose_in_world_frame=neutral_eye_pose_in_world_frame
                )  # [ts, x, y, z] m
            else:
                target_positions_full = dataloader.load_target_positions(frame="neutral_camera")  # [ts, x, y, z] m

            # Errors (deg)
            err_res = process_direction_errors(
                dataloader=dataloader,
                model=model,
                gaze_target_bias=None,
                save_to_file=False,
            )
            if err_res.shape[0] < 2 or target_positions_full.shape[0] < 2:
                continue

            # Timestamp match
            err_res, matched_target_positions = match_irregular_to_regular(
                irregular_data=err_res,
                regular_data=target_positions_full,
                regular_period_ms=dataloader.camera_pose_period
            )

            # Select axes (FRAME-aware)
            V_m = matched_target_positions[:, _get_axis(is_line_horizontal=False, frame=FRAME)]
            H_m = matched_target_positions[:, _get_axis(is_line_horizontal=True,  frame=FRAME)]
            ERR = err_res[:, 1]

            # meters → cm
            V = 100.0 * V_m
            H = 100.0 * H_m
            msk = np.isfinite(V) & np.isfinite(H) & np.isfinite(ERR)
            if not np.any(msk):
                continue
            V, H, ERR = V[msk], H[msk], ERR[msk]

            # Keep only samples close to the relevant rails; drop everything else
            if line_movement_type == "horizontal":
                # vertical must be near one of V_RAILS
                # compute distance to nearest rail
                d = np.abs(V[:, None] - V_RAILS[None, :]).min(axis=1)
                keep = d <= v_band_cm
            else:  # "vertical"
                # horizontal must be near one of H_RAILS
                d = np.abs(H[:, None] - H_RAILS[None, :]).min(axis=1)
                keep = d <= h_band_cm

            if not np.any(keep):
                continue

            all_V.append(V[keep])
            all_H.append(H[keep])
            all_ERR.append(ERR[keep])

        except Exception as e:
            print(f"[error-field/all] Skipping subject {subj}: {e}")
            continue

    if not all_H:
        print("[error-field/all] No valid rail-aligned data collected.")
        return None

    # Build bin edges -----------------------------------------------------------------------------
    # One axis is continuous (use 'bins' for resolution), the other is *rail-banded*.
    all_H_cat = np.concatenate(all_H, axis=0)
    all_V_cat = np.concatenate(all_V, axis=0)

    if line_movement_type == "horizontal":
        # Horizontal axis continuous across observed range
        h_bins = int(bins[0])
        Z_edges = np.linspace(np.nanmin(all_H_cat), np.nanmax(all_H_cat), h_bins + 1)  # horizontal edges

        # Vertical axis only narrow bands around rails
        bw = float(v_band_cm)
        # produce edges like [-12,-8,  -2,2,  8,12] → exactly 3 vertical bins → thin stripes
        Y_edges = np.array(
            [V_RAILS[0]-bw, V_RAILS[0]+bw,
             V_RAILS[1]-bw, V_RAILS[1]+bw,
             V_RAILS[2]-bw, V_RAILS[2]+bw],
            dtype=float
        )
        y_bins = len(Y_edges) - 1  # should be 3
    else:  # "vertical"
        # Vertical axis continuous across observed range
        y_bins = int(bins[1])
        Y_edges = np.linspace(np.nanmin(all_V_cat), np.nanmax(all_V_cat), y_bins + 1)  # vertical edges

        # Horizontal axis only narrow bands around rails
        bw = float(h_band_cm)
        Z_edges = np.array(
            [H_RAILS[0]-bw, H_RAILS[0]+bw,
             H_RAILS[1]-bw, H_RAILS[1]+bw,
             H_RAILS[2]-bw, H_RAILS[2]+bw],
            dtype=float
        )
        h_bins = len(Z_edges) - 1  # should be 3

    # Aggregators (z=horizontal bins count, y=vertical bins count)
    sum_err = np.zeros((len(Z_edges)-1, len(Y_edges)-1), dtype=float)
    counts  = np.zeros_like(sum_err)
    err_lists = _dd(list)

    # Bin per subject -----------------------------------------------------------------------------
    for H, V, ERR in zip(all_H, all_V, all_ERR):
        # Mean aggregation via histogram2d
        s, _, _ = np.histogram2d(H, V, bins=[Z_edges, Y_edges], weights=ERR)
        c, _, _ = np.histogram2d(H, V, bins=[Z_edges, Y_edges])
        sum_err += s
        counts  += c

        # Median aggregation (append per bin)
        zi = np.digitize(H, Z_edges) - 1
        yi = np.digitize(V, Y_edges) - 1
        valid = (zi >= 0) & (zi < (len(Z_edges)-1)) & (yi >= 0) & (yi < (len(Y_edges)-1)) & np.isfinite(ERR)
        if not np.any(valid):
            continue
        ziv = zi[valid]; yiv = yi[valid]; ev = ERR[valid]
        flat = ziv * (len(Y_edges)-1) + yiv
        uniq_flat, inv = np.unique(flat, return_inverse=True)
        for k_idx, f in enumerate(uniq_flat):
            err_lists[int(f)].extend(ev[inv == k_idx].tolist())

    # Mean / Median --------------------------------------------------------------------------------
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_err = sum_err / counts
    mean_err[counts == 0] = np.nan

    median_err = np.full_like(mean_err, np.nan, dtype=float)
    y_bins_eff = len(Y_edges) - 1
    for f, lst in err_lists.items():
        if not lst:
            continue
        zi = f // y_bins_eff
        yi = f %  y_bins_eff
        median_err[zi, yi] = float(np.median(np.asarray(lst, dtype=float)))

    # Clip for display
    VMIN, VMAX = 0.0, 30.0
    mean_plot   = np.clip(mean_err,   VMIN, VMAX)
    median_plot = np.clip(median_err, VMIN, VMAX)

    # Plot & save ----------------------------------------------------------------------------------
    fig_paths = {"mean": None, "median": None}
    if show_plot or save_plot:
        base_save_dir = os.path.join(config.get_dataset_base_directory(), RESULTS_DIR, "error_field_yz_across_subjects")
        os.makedirs(base_save_dir, exist_ok=True)
        hor_minus, hor_plus = _get_horizontal_signs(frame=FRAME)

        def _plot_and_maybe_save(A, label):
            fig = plt.figure(figsize=(7.8, 5.8))
            ZZ, YY = np.meshgrid(Z_edges, Y_edges, indexing="ij")
            pcm = plt.pcolormesh(ZZ, YY, A, shading="auto", vmin=VMIN, vmax=VMAX, cmap="viridis")
            cbar = plt.colorbar(pcm)
            cbar.set_label(f"{label.title()} Gaze Error (deg)")
            plt.xlabel(f"Horizontal position (cm)   [− {hor_minus}, + {hor_plus} in the {FRAME} frame]")
            plt.ylabel("Vertical position (cm)     [− down, + up]")
            plt.title(f"Across-subject error field ({label}) — {line_movement_type}, {line_movement_speed}, model={model}")
            plt.tight_layout()
            out_path = None
            if save_plot:
                fname = (
                    f"error_field_yz_ACROSS_SUBJECTS__{line_movement_type}"
                    f"__{line_movement_speed}__model-{str(model).replace('/', '_')}"
                    f"__{label}_{FRAME}.png"
                )
                out_path = os.path.join(base_save_dir, fname)
                plt.savefig(out_path, dpi=170)
            if show_plot:
                plt.show()
            else:
                plt.close()
            return out_path

        fig_paths["mean"]   = _plot_and_maybe_save(mean_plot,   "mean")
        fig_paths["median"] = _plot_and_maybe_save(median_plot, "median")

    return {
        "Z_edges_cm": Z_edges,            # horizontal edges (rail-banded or continuous)
        "Y_edges_cm": Y_edges,            # vertical edges (rail-banded or continuous)
        "mean_error": mean_err,           # unclipped
        "median_error": median_err,       # unclipped
        "counts": counts,
        "figure_paths": fig_paths,
        "model": model,
        "movement_type": line_movement_type,
        "speed": line_movement_speed,
    }


def visualize_error_field_yz_across_subjects_for_all(
    subject_dirs: list,
    models: list,
):
    for line_movement_speed in ["fast", "slow"]:
        for line_movement_type in config.get_line_movement_types():
            for model in models:
                visualize_error_field_yz_across_subjects(
                    subject_dirs=subject_dirs,
                    model=model,
                    line_movement_speed=line_movement_speed,
                    line_movement_type=line_movement_type,
                    bins=(48,48),
                    show_plot=False,
                    save_plot=True
                )

def visualize_error_field_yz_across_subjects_overlaid(
    subject_dirs: list,
    model: str,
    line_movement_speed: str,      # "fast" | "slow"
    bins=(48, 48),
    v_band_cm: float = 2.0,
    h_band_cm: float = 2.0,
    show_plot: bool = False,
    save_plot: bool = True,
):
    """
    ONE figure that overlays horizontal-rails heatmap and vertical-rails heatmap
    in the SAME axes (shared color scale). This is the “combined” look you mocked.

    Internals:
    - Calls visualize_error_field_yz_across_subjects(...) twice (H and V).
    - Uses pcolormesh twice on the same axes with masked NaNs, so only bins that
      have data are drawn (no transparency artifacts).
    """
    # Build both maps without saving their individual figures
    out_h = visualize_error_field_yz_across_subjects(
        subject_dirs=subject_dirs,
        model=model,
        line_movement_speed=line_movement_speed,
        line_movement_type="horizontal",
        bins=bins,
        show_plot=False,
        save_plot=False,
        v_band_cm=v_band_cm,
        h_band_cm=h_band_cm,
    )
    out_v = visualize_error_field_yz_across_subjects(
        subject_dirs=subject_dirs,
        model=model,
        line_movement_speed=line_movement_speed,
        line_movement_type="vertical",
        bins=bins,
        show_plot=False,
        save_plot=False,
        v_band_cm=v_band_cm,
        h_band_cm=h_band_cm,
    )
    if out_h is None or out_v is None:
        print("[overlaid] one of the movement types returned no data; skipping.")
        return None

    # Pull edges + arrays
    Z_h, Y_h, A_h = out_h["Z_edges_cm"], out_h["Y_edges_cm"], out_h["mean_error"]
    Z_v, Y_v, A_v = out_v["Z_edges_cm"], out_v["Y_edges_cm"], out_v["mean_error"]

    x_min = float(min(np.nanmin(Z_h), np.nanmin(Z_v)))
    x_max = float(max(np.nanmax(Z_h), np.nanmax(Z_v)))
    y_min = float(min(np.nanmin(Y_h), np.nanmin(Y_v)))
    y_max = float(max(np.nanmax(Y_h), np.nanmax(Y_v)))

    # Shared color scale (match your singles)
    VMIN, VMAX = 0.0, 30.0
    Ah = np.clip(A_h, VMIN, VMAX)
    Av = np.clip(A_v, VMIN, VMAX)

    # Mask NaNs so pcolormesh leaves holes (white) where no data exist
    Ah = np.ma.array(Ah, mask=~np.isfinite(Ah))
    Av = np.ma.array(Av, mask=~np.isfinite(Av))

    # Axes labels consistent with FRAME
    hor_minus, hor_plus = _get_horizontal_signs(frame=FRAME)

    # Single-axes figure
    fig = plt.figure(figsize=(7.8, 5.8))
    ax = fig.add_subplot(111)

    # First draw the horizontal-run stripes
    ZZ_h, YY_h = np.meshgrid(Z_h, Y_h, indexing="ij")
    pcm_h = ax.pcolormesh(ZZ_h, YY_h, Ah, shading="auto",
                          vmin=VMIN, vmax=VMAX, cmap="viridis", zorder=1)

    # Then draw the vertical-run stripes (so the center pillar appears on top)
    ZZ_v, YY_v = np.meshgrid(Z_v, Y_v, indexing="ij")
    pcm_v = ax.pcolormesh(ZZ_v, YY_v, Av, shading="auto",
                          vmin=VMIN, vmax=VMAX, cmap="viridis", zorder=2)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')  # 1 cm on X == 1 cm on Y

    from matplotlib.ticker import MultipleLocator
    ax.set_xlim(-25.0, 25.0)                    # show the full −25…25 span
    ax.xaxis.set_major_locator(MultipleLocator(5.0))  # tick every 5 cm
    # (optional) match Y to 5-cm steps too:
    # ax.yaxis.set_major_locator(MultipleLocator(5.0))


    # Labels / title
    ax.set_xlabel(f"Horizontal position (cm)   [− {hor_plus}, + {hor_minus}]") # we got a  ax.set_xlabel(f"Horizontal position (cm)   [− {hor_minus}, + {hor_plus} in the {FRAME} frame]")
    ax.set_ylabel("Vertical position (cm)     [− down, + up]")
    model_name = config.display_model_name(model_id=model, clean_display_name=True)
    ax.set_title(model_name)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Create an axis on the right exactly matching the height of ax
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)

    cbar = fig.colorbar(pcm_v, cax=cax)
    cbar.set_label("Angular Error (deg)")

    fig.tight_layout()

    out_path = None
    if save_plot:
        base_dir = os.path.join(config.get_dataset_base_directory(), RESULTS_DIR, "error_field_yz_across_subjects")
        os.makedirs(base_dir, exist_ok=True)
        fname = (
            f"error_field_yz_ACROSS_SUBJECTS__OVERLAID__{line_movement_speed}"
            f"__model-{str(model).replace('/', '_')}__mean_{FRAME}.pdf"
        )
        out_path = os.path.join(base_dir, fname)
        plt.savefig(out_path, dpi=170, bbox_inches="tight", pad_inches=0)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "figure_path": out_path,
        "horizontal": out_h,
        "vertical": out_v,
    }

def visualize_error_field_yz_across_subjects_overlaid_for_all(subject_dirs: list, models: list):
    for speed in ["fast", "slow"]:
        for model in models:
            visualize_error_field_yz_across_subjects_overlaid(
                subject_dirs=subject_dirs,
                model=model,
                line_movement_speed=speed,
                bins=(48, 48),
                v_band_cm=0.5,
                h_band_cm=1.0,
                show_plot=False,
                save_plot=True,
            )



if __name__ == "__main__":
    # Warning: You must run these in the following order, also you must have run the "neutral_eye_position_calculation.py" script before this script.
    # 1- extract_intersection_points usage: 
    # extract_intersection_points_for_all_subjects(print_every_intersection_point=False)
    # exit()
    # # break down into sections for a particular line movement experiment of a single subject:
    # line_movement_speed, line_movement_type = "fast", "horizontal"
    # exp_dir = f"dataset_base_dir/2025-07-31/subj_xxxx/line_movement_{line_movement_speed}/{line_movement_type}"
    # dataloader = GazeDataLoader(
    #     root_dir=exp_dir,
    #     target_period=config.get_target_period(),
    #     camera_pose_period=config.get_camera_pose_period(),
    #     time_diff_max=config.get_time_diff_max(),
    #     get_latest_subdirectory_by_name=True
    # )
    # extract_intersection_points(dataloader, line_movement_speed=line_movement_speed, line_movement_type=line_movement_type, visualize_for_debug=True)
    # exit()

    # 2- analyze_by_position
    # 2.1- calculate and save analyze_by_position data.
    BASE_DIR = config.get_dataset_base_directory()
    # SUBJECT_DIRS = config.get_dataset_subject_directories(rnd=True, n=20, seed=42) # get a subset of subjects for faster experimentation.
    # MODELS = ["padding_20_puregaze", "l2cs", "gazetr", "mcgaze_clip_size_3", "mcgaze_clip_size_7"]
    # OUT_DIR = os.path.join(BASE_DIR, RESULTS_DIR)
    # # Runs all 4 combos: (fast|slow) × (horizontal|vertical), 5 segments each, per model
    # line_movement_analysis_by_position(
    #     subject_dirs=SUBJECT_DIRS,
    #     models=MODELS,
    #     output_dir_for_text_files=OUT_DIR,
    #     plot=False,  # keep False unless you really want the per-segment plots
    # )
    # aggregate_all_saved_line_movement_results(OUT_DIR)
    #2.2- compare by slow vs fast and save plots
    # BASE_DIR = config.get_dataset_base_directory()
    # OUT_DIR  = os.path.join(BASE_DIR, RESULTS_DIR, "plots_fast_vs_slow")
    # os.makedirs(OUT_DIR, exist_ok=True)

    # # Use the aggregated CSVs you already write:
    # pearson_by_seg_side = os.path.join(
    #     BASE_DIR, RESULTS_DIR,
    #     "line_movement_by_position_pearson__agg_by_type_speed_segment_side_model.csv"
    # )
    # spearman_by_seg_side = os.path.join(
    #     BASE_DIR, RESULTS_DIR,
    #     "line_movement_by_position_spearman__agg_by_type_speed_segment_side_model.csv"
    # )

    # pearson_by_side = os.path.join(
    #     BASE_DIR, RESULTS_DIR,
    #     "line_movement_by_position_pearson__agg_by_type_speed_side_model.csv"
    # )
    # spearman_by_side = os.path.join(
    #     BASE_DIR, RESULTS_DIR,
    #     "line_movement_by_position_spearman__agg_by_type_speed_side_model.csv"
    # )

    # # Create plots (weighted metrics by default)
    # compare_fast_vs_slow_by_segment_side_model(pearson_by_seg_side, OUT_DIR, prefer_weighted=True, file_label="pearson")
    # compare_fast_vs_slow_by_segment_side_model(spearman_by_seg_side, OUT_DIR, prefer_weighted=True, file_label="spearman")

    # compare_fast_vs_slow_by_side_model(pearson_by_side, OUT_DIR, prefer_weighted=True, file_label="pearson")
    # compare_fast_vs_slow_by_side_model(spearman_by_side, OUT_DIR, prefer_weighted=True, file_label="spearman")
    #3- visualize_subject_variability_for_segment (to visualize subject variability):
    # BASE_DIR = config.get_dataset_base_directory()
    # SUBJECT_DIRS = config.get_dataset_subject_directories(rnd=True, n=20,seed=42) # get a subset of subjects for faster experimentation.
    # # Pick one condition to inspect:
    # visualize_subject_variability_for_segment(
    #     subject_dirs=SUBJECT_DIRS,
    #     model="padding_20_puregaze",
    #     line_movement_speed="fast",
    #     line_movement_type="vertical",
    #     line_segment_index=0,
    #     plot=True,   # set True to display; False does the extraction only
    # )
    #4- visualize_speed_variability_for_segment (to visualize speed variability for the same subject)
    # BASE_DIR = config.get_dataset_base_directory()
    # SUBJECT_DIRS = config.get_dataset_subject_directories(rnd=True, n=20,seed=42) # get a subset of subjects for faster experimentation.
    # # Pick one condition to inspect:
    # visualize_speed_variability_for_segment(
    #     subject_dirs=SUBJECT_DIRS,
    #     model="padding_20_puregaze",
    #     line_movement_type="vertical",
    #     line_segment_index=0,
    #     plot=True,   # set True to display; False does the extraction only
    # )
    #5-visualize gaze estimation error wrt position on the yz plane (rather than position on a single axis) as a 2D heatmap or 3D surface.
    #5.1-visualize per subject
    # SUBJECT_DIRS = config.get_dataset_subject_directories(rnd=True, n=20,seed=42) # get a subset of subjects for faster experimentation.
    # MODELS = ["padding_20_puregaze", "l2cs", "gazetr", "mcgaze_clip_size_3", "mcgaze_clip_size_7"]
    # visualize_error_field_yz_for_all(subject_dirs=SUBJECT_DIRS, models=MODELS)
    #5.2-visualize the average of all subjects. since this data is in the eye frame, the boundaries of the field will differ from person to person.
    # WARNING: This does not seem to be working, the visualization seems very off.
    SUBJECT_DIRS = config.get_dataset_subject_directories() # get a subset of subjects for faster experimentation during dev.
    MODELS = config.get_currently_analyzed_models()
    # visualize_error_field_yz_across_subjects_for_all(subject_dirs=SUBJECT_DIRS, models=MODELS)
    visualize_error_field_yz_across_subjects_overlaid_for_all(SUBJECT_DIRS, MODELS)# New single-axes combined plot per speed × model:
