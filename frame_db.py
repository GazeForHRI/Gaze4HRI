# frame_db.py
import os
import json
import hashlib
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

import config

BASE_DIR = config.get_dataset_base_directory()
FRAME_DB_DIR = os.path.join(BASE_DIR, "frame_db", "parquet")
DATASET_DIR = os.path.join(FRAME_DB_DIR, "gaze_evaluation_results_by_frame")

REQUIRED_COLUMNS = [
    "subject_dir", "exp_type", "point", "subject_name", "gender", "birthyear",
    "glasses", "height_cm", "gaze_model", "error", "is_valid", "frame_idx", "exp_dir"
]

def _ensure_dirs():
    os.makedirs(DATASET_DIR, exist_ok=True)

def _rel_to_base(abs_path: str) -> str:
    ap = os.path.normpath(abs_path)
    bp = os.path.normpath(BASE_DIR)
    if ap.startswith(bp):
        rel = ap[len(bp):].lstrip(os.sep)
    else:
        # Fall back: return normalized path if outside BASE_DIR (shouldn't happen)
        rel = ap
    return rel.replace("\\", "/")

def _parse_exp_dir(exp_dir_abs: str) -> Tuple[str, str, str]:
    """
    Returns (subject_dir, exp_type, point).
    subject_dir := "<date>/<subject>"
    exp_type := directory name after subject (e.g., 'circular_movement', 'lighting_25', ...)
    point := 'p1'..'p9', 'h1'..'h6', 'horizontal'/'vertical', or '' if N/A
    """
    rel = _rel_to_base(exp_dir_abs)
    parts = rel.split("/")
    if len(parts) < 3:
        raise ValueError(f"Invalid exp_dir structure relative to BASE_DIR: {rel}")

    date_str, subject_name, exp_type = parts[0], parts[1], parts[2]
    point = ""
    if len(parts) >= 4:
        # handle line_movement_* and standard point experiments
        point = parts[3]

    subject_dir = f"{date_str}/{subject_name}"
    return subject_dir, exp_type, point

def _load_subject_meta(subject_dir: str) -> Dict[str, Any]:
    """
    Locate subject_info.json under BASE_DIR/<date>/<subject>/... and return first 'subjects[0]'.
    We try a shallow walk just under subject root.
    """
    subject_root = os.path.join(BASE_DIR, subject_dir)
    # Fast path: common locations
    candidates = []
    for root, dirs, files in os.walk(subject_root):
        if "subject_info.json" in files:
            candidates.append(os.path.join(root, "subject_info.json"))
        # Don't walk too deep for speed
        if root.count(os.sep) - subject_root.count(os.sep) > 2:
            dirs[:] = []  # prune

    meta = {
        "subject_name": None,
        "gender": None,
        "birthyear": None,
        "glasses": None,
        "height_cm": None,
    }
    if not candidates:
        return meta

    # Take the closest one (shortest path)
    candidates.sort(key=lambda p: len(p))
    try:
        with open(candidates[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "subjects" in data and data["subjects"]:
            subj = data["subjects"][0]
            meta["subject_name"] = subj.get("name")
            meta["gender"] = subj.get("gender")
            meta["birthyear"] = subj.get("birth_year")
            meta["glasses"] = subj.get("glasses")
            meta["height_cm"] = subj.get("height_cm")
    except Exception:
        pass
    return meta

def _build_frame_df(
    exp_dir_abs: str,
    gaze_model: str,
    error_results_sorted: List[Tuple[float, float, float, float]]
) -> pd.DataFrame:
    """
    error_results_sorted: list of [timestamp_ms, angular_error_deg, euclidean_error, is_valid]
                          (euclidean_error is ignored here)
    """
    subject_dir, exp_type, point = _parse_exp_dir(exp_dir_abs)
    meta = _load_subject_meta(subject_dir)

    # Enumerate frame_idx
    rows = []
    rel_exp_dir = _rel_to_base(exp_dir_abs)
    for idx, (_ts, ang_err, _eu, is_valid) in enumerate(error_results_sorted):
        rows.append({
            "subject_dir": subject_dir,
            "exp_type": exp_type,
            "point": point or "",
            "subject_name": meta.get("subject_name"),
            "gender": meta.get("gender"),
            "birthyear": meta.get("birthyear"),
            "glasses": meta.get("glasses"),
            "height_cm": meta.get("height_cm"),
            "gaze_model": gaze_model,
            "error": float(ang_err),
            "is_valid": int(1 if float(is_valid) > 0.5 else 0),
            "frame_idx": int(idx),
            "exp_dir": rel_exp_dir,
        })

    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    return df

def _stable_part_path(exp_dir_abs: str, gaze_model: str) -> str:
    rel = _rel_to_base(exp_dir_abs)
    key = f"{rel}|{gaze_model}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(DATASET_DIR, f"part-{h}.parquet")

def append_frame_results(exp_dir_abs: str, gaze_model: str, error_results_sorted):
    if not error_results_sorted:
        return ""

    _ensure_dirs()
    df = _build_frame_df(exp_dir_abs, gaze_model, error_results_sorted)

    out_path      = _stable_part_path(exp_dir_abs, gaze_model)
    tmp_out_path  = out_path + ".tmp"

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, tmp_out_path)
    # atomic replace to avoid torn writes if interrupted
    os.replace(tmp_out_path, out_path)
    return out_path

def load_frame_dataset() -> pd.DataFrame:
    """
    Load ALL parts into a single pandas DataFrame.
    For ~650k rows this is fast enough.
    """
    if not os.path.isdir(DATASET_DIR):
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    parts = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith(".parquet")]
    if not parts:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    dfs = [pq.read_table(p).to_pandas() for p in parts]
    df = pd.concat(dfs, ignore_index=True)

    # Ensure required columns & order
    df = df.reindex(columns=REQUIRED_COLUMNS)
    return df


def compact_deduplicate(write_backup=True) -> str:
    """
    Load all parts, drop duplicates on (exp_dir, gaze_model, frame_idx), rewrite as fresh single file.
    Optionally back up old parts into a timestamped directory.
    Returns path to the compacted parquet file.
    """
    _ensure_dirs()
    df = load_frame_dataset()
    if df.empty:
        return ""

    before = len(df)
    df = df.drop_duplicates(subset=["exp_dir", "gaze_model", "frame_idx"])
    after = len(df)
    print(f"[frame_db] Compaction: {before} -> {after} rows after de-dup.")

    # Backup old parts
    if write_backup:
        backup_dir = os.path.join(DATASET_DIR, f"_backup_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}")
        os.makedirs(backup_dir, exist_ok=True)
        for f in os.listdir(DATASET_DIR):
            if f.endswith(".parquet"):
                os.replace(os.path.join(DATASET_DIR, f), os.path.join(backup_dir, f))

    # Write one consolidated parquet
    out_path = os.path.join(DATASET_DIR, "part-compact.parquet")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_path)
    return out_path

def export_csv(out_csv_path: str) -> str:
    """
    Optional: export a CSV snapshot for sharing.
    """
    df = load_frame_dataset()
    if df.empty:
        return ""
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    return out_csv_path
