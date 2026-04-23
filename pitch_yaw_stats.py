import os
import math
import hashlib
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import config
from data_loader import GazeDataLoader

import os
import config
from data_loader import GazeDataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy.ndimage import gaussian_filter

try:
    # prefer pyarrow engine if available (faster, smaller files)
    import pyarrow  # noqa: F401
    _PARQUET_ENGINE = "pyarrow"
except Exception:  # pragma: no cover
    _PARQUET_ENGINE = "auto"


_LABEL_UNCERTAIN = 2
_LABEL_BLINK = 3

# ===================== helpers =====================

def _vec_to_pitch_yaw(x: float, y: float, z: float) -> Tuple[float, float]:
    """
    Standard CV convention (+Z forward, +X right, +Y down):
      pitch = atan2(-y, sqrt(x^2+z^2))
      yaw   = atan2(x, z)   # implemented piecewise (no math.atan2)
    Returns degrees.
    """
    # normalize
    n = math.sqrt(x*x + y*y + z*z)
    if n < 1e-12:
        return float("nan"), float("nan")
    x, y, z = x/n, y/n, z/n

    # pitch
    pitch_rad = math.atan2(-y, math.sqrt(x*x + z*z))

    # yaw via piecewise atan2(u, v) with u=x, v=z
    u, v = x, z
    eps = 1e-12
    if abs(v) > eps:
        base = math.atan(u / v)  # arctan(u/v)
        if v > 0:
            yaw_rad = base
        else:  # v < 0
            yaw_rad = base + (math.pi if u >= 0 else -math.pi)
    else:
        # v == 0 -> ±pi/2 depending on sign of u (undefined if u == 0)
        yaw_rad = (math.pi/2) if u > 0 else ((-math.pi/2) if u < 0 else float("nan"))

    # normalize yaw to (-pi, pi]
    if math.isfinite(yaw_rad):
        yaw_rad = (yaw_rad + math.pi) % (2*math.pi) - math.pi

    # the following part is to fix the yaw problem which causes 30 deg right of the camera to be -150 deg, 30 deg left of the camera to be +150 deg, and 0 deg to be app. +180 deg or app. -180
    yaw_rad += math.pi
    yaw_rad = (yaw_rad + math.pi) % (2*math.pi) - math.pi

    return math.degrees(-pitch_rad), math.degrees(-yaw_rad)

def _parse_meta_from_loader(dataloader: GazeDataLoader) -> Tuple[str, str, str, str]:
    """Extract (subject_dir, experiment_type, point, timestamp) from the loader's cwd.

    Loader directory layout: <base_dir>/<subject_dir>/<exp_type>/<point>/<timestamp>
    """
    cwd = dataloader.get_cwd()  # absolute exp_dir path
    base_dir = config.get_dataset_base_directory()

    # Get relative path under base_dir
    rel = os.path.relpath(cwd, start=base_dir)
    parts = rel.split(os.sep)

    if len(parts) < 4:
        raise ValueError(f"Unexpected exp_dir structure: {cwd}")

    subject_dir = os.path.join(parts[0], parts[1]) if len(parts) > 4 else parts[0]
    exp_type = parts[-3]
    point = parts[-2]
    timestamp = parts[-1]

    return subject_dir, exp_type, point, timestamp

def _db_root_dir(db_root: Optional[str]) -> str:
    return db_root or os.path.join(config.get_dataset_base_directory(), "pitch_yaw_stats_db")


def _stable_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


# ===================== core function =====================

def save_pitch_yaw_framewise(dataloader: GazeDataLoader, db_root: Optional[str] = None) -> str:
    """Compute and persist per-*frame* GT gaze & head directions in **camera frame**.

    Updates per your clarifications:
      • Uses camera_period = 1000.0 / config.get_rgb_fps().
      • Aligns GT gaze (irregular) to head poses (regular) via match_irregular_to_regular.

    Saved columns per row (one per RGB-aligned GT sample):
      - uid, subject, experiment_type, point, session, exp_dir_rel
      - frame_idx (nearest RGB), timestamp_ms
      - gaze_vec ("x y z"), gaze_pitch, gaze_yaw
      - head_vec ("x y z"), head_pitch, head_yaw

    Returns:
        str: Path of the Parquet file written for this video session.
    """
    # Local import to avoid changing global imports in this module
    try:
        from data_matcher import match_irregular_to_regular
    except ImportError:
        # support relative import if packaged
        from .data_matcher import match_irregular_to_regular  # type: ignore

    # ---------- metadata & base arrays ----------
    subject, exp_type, point, session = _parse_meta_from_loader(dataloader)
    exp_dir_abs = dataloader.get_cwd()
    base_dir = config.get_dataset_base_directory()
    exp_dir_rel = os.path.relpath(exp_dir_abs, start=base_dir)

    # RGB timestamps used for deriving frame_idx by nearest-neighbor
    rgb_ts = dataloader.load_rgb_timestamps().astype(float).reshape(-1)

    # Ground-truth gaze in CAMERA frame: (Ng, 4) => [ts, gx, gy, gz]
    gt_cam = dataloader.load_gaze_ground_truths(frame="camera")

    # Head poses in CAMERA frame (regular stream at camera fps): (Nh, 17) => [ts, 16]
    head_cam = dataloader.load_head_poses(frame="camera")

    # Period for the regular stream = camera/RGB period in ms
    camera_period = 1000.0 / float(config.get_rgb_fps())
    
    try:
        blink_ann = dataloader.get_blink_annotations()  # (T,) array
    except Exception as e:
        print(f"[WARN] Could not load blink annotations for {exp_dir_rel}: {e}. Defaulting all to valid.")
        blink_ann = None

    # ---------- align: gaze (irregular) → head poses (regular @ camera_period) ----------
    gt_cam, matched_head_for_gt = match_irregular_to_regular(
        irregular_data=gt_cam,
        regular_data=head_cam,
        regular_period_ms=camera_period,
    )

    # ---------- build rows ----------
    rows = []
    for i in range(gt_cam.shape[0]):
        ts = float(gt_cam[i, 0])
        gx, gy, gz = map(float, gt_cam[i, 1:4])
        #a, b, c -> 2, 0, 1 = c,a,b -> 1,2,0 = a,b,c
        gx, gy, gz = gy, gz, gx
        # matched head pose row corresponds to this GT timestamp
        flat16 = matched_head_for_gt[i]
        mat = flat16.reshape(4, 4)
        from scipy.spatial.transform import Rotation as R
        rot = R.from_matrix(mat[:3, :3])
        hx, hy, hz = rot.apply([1.0, 0.0, 0.0])
        nrm = float(np.linalg.norm([hx, hy, hz]))
        if nrm > 0:
            hx, hy, hz = hx / nrm, hy / nrm, hz / nrm

        hx, hy, hz = hy, hz, hx

        gp, gyaw = _vec_to_pitch_yaw(gx, gy, gz)
        hp, hyaw = _vec_to_pitch_yaw(hx, hy, hz)

        # nearest RGB frame index for this timestamp (for consistent per-frame indexing)
        if rgb_ts.size:
            r = int(np.argmin(np.abs(rgb_ts - ts)))
        else:
            r = i  # fallback
            
        is_valid = 1  # Default to valid
        if blink_ann is not None:
            if 0 <= r < len(blink_ann):
                label = int(blink_ann[r])
                # Mimic data_analyzer.py: INVALID if {UNCERTAIN=2, BLINK=3}
                if label == _LABEL_UNCERTAIN or label == _LABEL_BLINK:
                    is_valid = 0
            else:
                # This case might happen if array lengths mismatch
                print(f"[WARN] frame_idx {r} out of bounds for blink_ann (len={len(blink_ann)}). Defaulting to valid.")

        rows.append({
            "uid": f"{_stable_hash(exp_dir_rel)}::{r}",
            "subject": subject,
            "experiment_type": exp_type,
            "point": point,
            "session": session,
            "exp_dir_rel": exp_dir_rel,
            "frame_idx": r,
            "timestamp_ms": ts,
            "gaze_vec": " ".join(f"{v:.4f}" for v in (gx, gy, gz)),
            "gaze_pitch": float(gp),
            "gaze_yaw": float(gyaw),
            "head_vec": " ".join(f"{v:.4f}" for v in (hx, hy, hz)),
            "head_pitch": float(hp),
            "head_yaw": float(hyaw),
            "is_valid": is_valid  # <-- ADD THIS LINE
        })

    if not rows:
        raise RuntimeError("No rows produced; check input arrays.")

    df = pd.DataFrame(rows)

    # ---------- write Parquet shard (one file per video session) ----------
    root = _db_root_dir(db_root)
    os.makedirs(root, exist_ok=True)

    fname = f"{_stable_hash(exp_dir_rel)}.parquet"
    out_path = os.path.join(root, fname)

    # Idempotent append/overwrite based on uid
    try:
        tmp_path = out_path + ".tmp"
        df.to_parquet(tmp_path, index=False, engine=_PARQUET_ENGINE)
        os.replace(tmp_path, out_path)
        return out_path
    except Exception:
        # Safe fallback: overwrite if anything odd happens
        df.to_parquet(out_path, index=False, engine=_PARQUET_ENGINE)
        return out_path

# ---- same helpers as data_analyzer_batch ----
def get_latest_subdirectory_by_name(parent_directory):
    try:
        subdirs = [d for d in os.listdir(parent_directory)
                   if os.path.isdir(os.path.join(parent_directory, d))]
        if not subdirs:
            raise Exception(f"No subdirectories found in '{parent_directory}'")
        return max(subdirs)
    except FileNotFoundError:
        raise Exception(f"The directory '{parent_directory}' does not exist.")

def get_all_experiment_paths(subject_dirs, exp_types):
    paths = []
    for subj_path in subject_dirs:
        for exp in exp_types:
            if exp == "horizontal_movement":
                parent_dir = os.path.join(subj_path, exp)
                try:
                    latest_subdir = get_latest_subdirectory_by_name(parent_dir)
                    exp_dir = os.path.join(parent_dir, latest_subdir)
                    paths.append(exp_dir)
                except Exception as e:
                    print(f"Warning: failed to resolve path under {parent_dir}: {e}")
                continue

            if exp in ("line_movement_slow", "line_movement_fast"):
                for pt in config.get_line_movement_types():
                    parent_dir = os.path.join(subj_path, exp, pt)
                    try:
                        latest_subdir = get_latest_subdirectory_by_name(parent_dir)
                        exp_dir = os.path.join(parent_dir, latest_subdir)
                        paths.append(exp_dir)
                    except Exception as e:
                        print(f"Warning: failed to resolve path under {parent_dir}: {e}")
                continue

            for pt in config.get_point_variations().get(exp, []):
                parent_dir = os.path.join(subj_path, exp, pt)
                try:
                    latest_subdir = get_latest_subdirectory_by_name(parent_dir)
                    exp_dir = os.path.join(parent_dir, latest_subdir)
                    paths.append(exp_dir)
                except Exception as e:
                    print(f"Warning: failed to resolve path under {parent_dir}: {e}")
                    continue
    return paths

# ---- batch runner for pitch_yaw_stats ----
def run_pitch_yaw_for_all(subject_dirs, exp_types, log_suffix="pitch_yaw"):
    all_dirs = get_all_experiment_paths(subject_dirs, exp_types)
    log_path = os.path.join(os.getcwd(), f"{log_suffix}_batch_log.txt")
    processed_status = {}

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            for line in f:
                path, status = line.strip().split("||")
                processed_status[path] = status

    with open(log_path, "a") as log_file, tqdm(total=len(all_dirs), desc="pitch_yaw_stats", unit="dir") as pbar:
        for exp_dir in all_dirs:
            if exp_dir in processed_status:
                print(f"Skipping already processed: {exp_dir}")
                pbar.update(1)
                continue

            try:
                dataloader = GazeDataLoader(
                    root_dir=exp_dir,
                    target_period=config.get_target_period(),
                    camera_pose_period=config.get_camera_pose_period(),
                    time_diff_max=config.get_time_diff_max(),
                    get_latest_subdirectory_by_name=False,
                )
                out_path = save_pitch_yaw_framewise(dataloader)
                print(f"[ok] {exp_dir} -> {out_path}")
                log_file.write(f"{exp_dir}||SUCCESSFUL\n")
                log_file.flush()
            except Exception as e:
                print(f"[err] {exp_dir}: {e}")
                log_file.write(f"{exp_dir}||FAILED\n")
                log_file.flush()
            finally:
                pbar.update(1)


# ---------- DB load ----------
def load_pitch_yaw_db(db_root: Optional[str] = None) -> pd.DataFrame:
    """
    Read all Parquet shards under pitch_yaw_stats_db and return a single DataFrame.
    Columns expected from save_pitch_yaw_framewise:
      uid, subject, experiment_type, point, session, exp_dir_rel, frame_idx, timestamp_ms,
      gaze_vec, gaze_pitch, gaze_yaw, head_vec, head_pitch, head_yaw
    """
    root = _db_root_dir(db_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"DB root not found: {root}")

    rows = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".parquet"):
                continue
            full = os.path.join(dirpath, fn)
            try:
                df = pd.read_parquet(full, engine=_PARQUET_ENGINE)
                rows.append(df)
            except Exception as e:
                print(f"[warn] failed to read {full}: {e}")

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    # enforce dtypes for numeric analysis
    for c in ("gaze_pitch","gaze_yaw","head_pitch","head_yaw"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------- stats helpers ----------
def _series_stats(x: pd.Series) -> dict:
    """Compute mean, median, std (ddof=0), IQR, selected percentiles, range."""
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return {
            "mean": np.nan, "median": np.nan, "std": np.nan, "iqr": np.nan,
            "p01": np.nan, "p05": np.nan, "p10": np.nan,
            "p90": np.nan, "p95": np.nan, "p99": np.nan,
            "min": np.nan, "max": np.nan, "range": np.nan, "count": 0
        }
    q = np.percentile(x.values, [1,5,10,90,95,99])
    q25, q75 = np.percentile(x.values, [25, 75])
    mn, mx = float(np.min(x.values)), float(np.max(x.values))
    return {
        "mean": float(np.mean(x.values)),
        "median": float(np.median(x.values)),
        "std": float(np.std(x.values, ddof=0)),
        "iqr": float(q75 - q25),
        "p01": float(q[0]), "p05": float(q[1]), "p10": float(q[2]),
        "p90": float(q[3]), "p95": float(q[4]), "p99": float(q[5]),
        "min": mn, "max": mx, "range": float(mx - mn),
        "count": int(x.size)
    }

def aggregate_pitch_yaw_stats(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Frame-level aggregation (no weighting): compute stats for
      - gaze_pitch, gaze_yaw, head_pitch, head_yaw
    grouped by the given columns.

    Returns a wide DataFrame with hierarchical columns:
      (<signal>, <angle>, <metric>) e.g. ('gaze','pitch','mean')
    """
    if df.empty:
        return pd.DataFrame()

    needed = group_cols + ["gaze_pitch","gaze_yaw","head_pitch","head_yaw"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for aggregation: {missing}")

    def _pack(group: pd.DataFrame) -> pd.Series:
        out = {}
        for sig, angle_col in (("gaze","gaze_pitch"), ("gaze","gaze_yaw"),
                               ("head","head_pitch"), ("head","head_yaw")):
            stats = _series_stats(group[angle_col])
            for k, v in stats.items():
                out[(sig, angle_col.split('_')[1], k)] = v
        return pd.Series(out)

    agg = df.groupby(group_cols, dropna=False, sort=True).apply(_pack, include_groups=False)

    # neat column order
    agg.columns = pd.MultiIndex.from_tuples(agg.columns)
    level_order = [
        ("gaze","pitch"), ("gaze","yaw"),
        ("head","pitch"), ("head","yaw"),
    ]
    metrics = ["mean","median","std","iqr","p01","p05","p10","p90","p95","p99","min","max","range","count"]
    new_cols = []
    for (sig,ang) in level_order:
        for m in metrics:
            col = (sig, ang, m)
            if col in agg.columns:
                new_cols.append(col)
    agg = agg.reindex(columns=new_cols)
    return agg

def stats_by_experiment_type(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_pitch_yaw_stats(df, ["experiment_type"])

def stats_by_point(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_pitch_yaw_stats(df, ["point"])

def stats_by_experiment_then_point(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_pitch_yaw_stats(df, ["experiment_type", "point"])

def save_stats_tables(df: pd.DataFrame, out_dir: str) -> None:
    """
    Write the four aggregations to CSVs under out_dir.
    Also write a 4-col summary CSV (mean ± std | [p01, p99]) for each table.
    """
    os.makedirs(out_dir, exist_ok=True)
    tables = {
        "by_experiment_type.csv":       stats_by_experiment_type(df),
        "by_point.csv":                 stats_by_point(df),
        "by_experiment_then_point.csv": stats_by_experiment_then_point(df),
    }
    for name, tbl in tables.items():
        # ---- existing wide CSV ----
        flat_cols = ["__".join(col) for col in tbl.columns.to_flat_index()]
        out = tbl.copy()
        out.columns = flat_cols
        out.to_csv(os.path.join(out_dir, name))
        print(f"Saved {name} to {out_dir}")

        # ---- NEW: 4-col summary CSV ----
        # Expect these metrics to exist in the MultiIndex columns
        def _fmt(mean, std, p01, p99):
            return f"{mean:.3f} ± {std:.3f} | [{p01:.3f}, {p99:.3f}]"

        # Build a compact 4-column dataframe, preserving the same index (groups)
        compact = pd.DataFrame(index=tbl.index)
        compact["gaze_pitch"] = [
            _fmt(m, s, lo, hi) for m, s, lo, hi in zip(
                tbl[("gaze", "pitch", "mean")],
                tbl[("gaze", "pitch", "std")],
                tbl[("gaze", "pitch", "p01")],
                tbl[("gaze", "pitch", "p99")]
            )
        ]
        compact["gaze_yaw"] = [
            _fmt(m, s, lo, hi) for m, s, lo, hi in zip(
                tbl[("gaze", "yaw", "mean")],
                tbl[("gaze", "yaw", "std")],
                tbl[("gaze", "yaw", "p01")],
                tbl[("gaze", "yaw", "p99")]
            )
        ]
        compact["head_pitch"] = [
            _fmt(m, s, lo, hi) for m, s, lo, hi in zip(
                tbl[("head", "pitch", "mean")],
                tbl[("head", "pitch", "std")],
                tbl[("head", "pitch", "p01")],
                tbl[("head", "pitch", "p99")]
            )
        ]
        compact["head_yaw"] = [
            _fmt(m, s, lo, hi) for m, s, lo, hi in zip(
                tbl[("head", "yaw", "mean")],
                tbl[("head", "yaw", "std")],
                tbl[("head", "yaw", "p01")],
                tbl[("head", "yaw", "p99")]
            )
        ]

        # Save next to the original, with a clear suffix
        compact_name = name.replace(".csv", "_summary.csv")
        compact.to_csv(os.path.join(out_dir, compact_name))
        print(f"Saved {compact_name} to {out_dir}")


# ===================== plotting =====================

def _hist2d_density(x_deg: np.ndarray, y_deg: np.ndarray,
                    bins: Tuple[int, int],
                    rng: Tuple[Tuple[float, float], Tuple[float, float]]) -> np.ndarray:
    """
    Return normalized 2D density (sum = 1). x=Yaw, y=Pitch (degrees).
    """
    H, xedges, yedges = np.histogram2d(x_deg, y_deg, bins=bins, range=rng, density=False)
    H = H.astype(np.float64)
    s = H.sum()
    if s > 0:
        H /= s
    return H


def _balanced_density(df: pd.DataFrame,
                      group_cols: list[str] | None,
                      bins_gaze=(160, 160),
                      bins_head=(160, 160),
                      range_gaze=(( -120.0, 120.0), ( -120.0, 120.0)),
                      range_head=((  -80.0,  80.0), (  -80.0,  80.0)),
                      two_stage: bool = False) -> tuple[np.ndarray, np.ndarray]:

    """
    Compute 2D pitch–yaw densities for gaze (top) and head (bottom).

    Modes:
      - group_cols is None  -> use all rows (unbalanced)
      - group_cols provided -> average per-group normalized histograms (each group has equal weight)
      - two_stage=True      -> first give equal weight to each experiment_type, then equal-share its points
                               (use group_cols=['experiment_type','point'])
    """
    def _accumulate(groups, weights):
        G = np.zeros(bins_gaze, dtype=np.float64)
        H = np.zeros(bins_head, dtype=np.float64)
        for (gdf, w) in zip(groups, weights):
            # gaze
            G += w * _hist2d_density(
                x_deg=gdf["gaze_yaw"].to_numpy(dtype=float),
                y_deg=gdf["gaze_pitch"].to_numpy(dtype=float),
                bins=bins_gaze, rng=range_gaze
            )
            # head
            H += w * _hist2d_density(
                x_deg=gdf["head_yaw"].to_numpy(dtype=float),
                y_deg=gdf["head_pitch"].to_numpy(dtype=float),
                bins=bins_head, rng=range_head
            )
        # renormalize to sum=1 (numeric safety)
        if G.sum() > 0: G /= G.sum()
        if H.sum() > 0: H /= H.sum()

        if G.size:
            G = gaussian_filter(G, sigma=1.8, mode="nearest")
            if G.sum() > 0: G /= G.sum()
        if H.size:
            H = gaussian_filter(H, sigma=1.8, mode="nearest")
            if H.sum() > 0: H /= H.sum()
        return G, H

    # Full (unbalanced)
    if not group_cols:
        return _accumulate([df], [1.0])

    # Balanced across groups
    if not two_stage:
        groups = [g for _, g in df.groupby(group_cols, dropna=False, sort=False)]
        if len(groups) == 0:
            return _accumulate([df], [1.0])
        w = np.full(len(groups), 1.0 / len(groups), dtype=np.float64)
        return _accumulate(groups, w)

    # Two-stage: equal per experiment_type; within each, equal per point
    assert group_cols == ["experiment_type", "point"], "two_stage expects ['experiment_type','point']"
    top = []
    weights = []
    for et, et_df in df.groupby("experiment_type", dropna=False, sort=False):
        point_groups = [g for _, g in et_df.groupby("point", dropna=False, sort=False)]
        if len(point_groups) == 0:
            continue
        # weight for this exp_type
        w_et = 1.0
        # split equally among its points
        w_each = (w_et / len(point_groups))
        top.extend(point_groups)
        weights.extend([w_each] * len(point_groups))
    if not top:
        return _accumulate([df], [1.0])
    # normalize weights so they sum to 1 across all exp_types
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    return _accumulate(top, weights)

def _plot_single_panel(D: np.ndarray, fname: str, title: str,
                       xlim: Tuple[float, float], ylim: Tuple[float, float],
                       tick_step: int):
    """
    Save ONE heatmap figure. D is a normalized 2D density with shape (bins_x, bins_y).
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=220, constrained_layout=True)

    # imshow: transpose because our histogram is (x,y) but imshow expects [row(y), col(x)]
    vals = D[D > 0]
    if vals.size:
        vmax = float(np.quantile(vals, 0.970))   # clip the brightest ~1.5%
        if vmax <= 0:
            vmax = float(vals.max())
    else:
        vmax = None  # fallback to matplotlib autoscale

    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax) if vmax is not None else PowerNorm(gamma=0.5)

    im = ax.imshow(
        D.T, origin="lower",
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        interpolation="bicubic", rasterized=True,
        norm=norm,
        cmap="viridis"
    )

    # Force square plotting region (grid cells become squares)
    ax.set_box_aspect(1)           # square axes box
    ax.set_aspect("equal", "box")  # keep data units equal on x/y

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("Yaw (°)")
    ax.set_ylabel("Pitch (°)")
    ax.set_title(title, fontsize=10, pad=4)

    ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.1, tick_step))
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.1, tick_step))
    ax.grid(alpha=0.25, linestyle=":", linewidth=0.7)

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)

# --- REPLACE the previous make_pitch_yaw_distribution_plots(...) with THIS ---
def _safe_name(s: str) -> str:
    """Filesystem-friendly name."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(s))

def make_pitch_yaw_distribution_plots(df: pd.DataFrame, out_root: str) -> dict[str, dict[str, str]]:
    """
    Save BOTH gaze and head distribution heatmaps under:
      distribution_plots/
        full_dataset_gaze.png
        full_dataset_head.png
        by_experiment_type/<exp_type>__gaze.png
        by_experiment_type/<exp_type>__head.png
        by_experiment_type_then_point/<exp_type>/<point>__gaze.png
        by_experiment_type_then_point/<exp_type>/<point>__head.png

    Returns a flat dict of created file paths keyed by a human-readable tag.
    """
    created: dict[str, dict[str, str] | str] = {}

    base = os.path.join(out_root, "distribution_plots")
    os.makedirs(base, exist_ok=True)

    # ---------------- 1) Full dataset ----------------
    G_full, H_full = _balanced_density(df, group_cols=None)

    f_full_g = os.path.join(base, "full_dataset_gaze.png")
    _plot_single_panel(G_full, f_full_g, "Gaze (distribution)",
                       xlim=(-120, 120), ylim=(-120, 120), tick_step=40)

    f_full_h = os.path.join(base, "full_dataset_head.png")
    _plot_single_panel(H_full, f_full_h, "Head (distribution)",
                       xlim=(-80, 80), ylim=(-80, 80), tick_step=20)

    created["full_dataset"] = {"gaze": f_full_g, "head": f_full_h}

    # ---------------- 2) By experiment type ----------------
    dir_by_exp = os.path.join(base, "by_experiment_type")
    os.makedirs(dir_by_exp, exist_ok=True)

    for et, df_et in df.groupby("experiment_type", dropna=False, sort=True):
        if df_et.empty:
            continue
        safe_et = _safe_name(et)

        G_et, H_et = _balanced_density(df_et, group_cols=None)

        f_et_g = os.path.join(dir_by_exp, f"{safe_et}__gaze.png")
        _plot_single_panel(G_et, f_et_g, "Gaze (distribution)",
                           xlim=(-120, 120), ylim=(-120, 120), tick_step=40)

        f_et_h = os.path.join(dir_by_exp, f"{safe_et}__head.png")
        _plot_single_panel(H_et, f_et_h, "Head (distribution)",
                           xlim=(-80, 80), ylim=(-80, 80), tick_step=20)

        created[f"by_experiment_type/{et}"] = {"gaze": f_et_g, "head": f_et_h}

    # ---------------- 3) By experiment type THEN point ----------------
    root_e_p = os.path.join(base, "by_experiment_type_then_point")
    for et, df_et in df.groupby("experiment_type", dropna=False, sort=True):
        if df_et.empty:
            continue
        safe_et = _safe_name(et)
        dir_et = os.path.join(root_e_p, safe_et)
        os.makedirs(dir_et, exist_ok=True)

        for pt, df_ep in df_et.groupby("point", dropna=False, sort=True):
            if df_ep.empty:
                continue
            safe_pt = _safe_name(pt)

            G_ep, H_ep = _balanced_density(df_ep, group_cols=None)

            f_ep_g = os.path.join(dir_et, f"{safe_pt}__gaze.png")
            _plot_single_panel(G_ep, f_ep_g, "Gaze (distribution)",
                               xlim=(-120, 120), ylim=(-120, 120), tick_step=40)

            f_ep_h = os.path.join(dir_et, f"{safe_pt}__head.png")
            _plot_single_panel(H_ep, f_ep_h, "Head (distribution)",
                               xlim=(-80, 80), ylim=(-80, 80), tick_step=20)

            created[f"by_experiment_type_then_point/{et}/{pt}"] = {"gaze": f_ep_g, "head": f_ep_h}

    return created


def main():
    # 1- Run pitch yaw for all data to create the db.
    # SUBJECT_DIRS = config.get_dataset_subject_directories()
    # EXP_TYPES = [
    #     "lighting_10", "lighting_25", "lighting_50", "lighting_100",
    #     "circular_movement",
    #     "head_pose_middle", "head_pose_left", "head_pose_right",
    #     "line_movement_slow", "line_movement_fast",
    # ]
    # run_pitch_yaw_for_all(SUBJECT_DIRS, EXP_TYPES)
    # print("pitch_yaw_stats completed for all experiments.")

    # 2- Analyze and save analysis results (requires an existing db).
    df = load_pitch_yaw_db()  # reads <BASE>/pitch_yaw_stats_db
    print(df.columns)

    # write to disk
    out_dir = os.path.join(config.get_dataset_base_directory(), "pitch_yaw_stats_analysis")
    save_stats_tables(df, out_dir)

    # new distributions (three files; each one has two subplots: gaze on top, head below)
    make_pitch_yaw_distribution_plots(df, out_dir)



if __name__ == "__main__":
    main()
