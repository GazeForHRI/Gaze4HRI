import os
import shutil
import csv
from typing import Iterable, List, Literal
from gaze_estimation_batch import get_all_experiment_paths
import config
import re

# ----------------------------- Utilities ----------------------------- #

def _safe_remove(path: str):
    """Remove file or directory at path if it exists."""
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.unlink(path)


def _safe_copy(src: str, dst: str):
    """
    Copy a file or directory from src to dst.
    - If dst exists, it will be removed first.
    - Parent directories of dst are created as needed.
    """
    _safe_remove(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src) and not os.path.islink(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


# --------------------------- General (new) --------------------------- #
# New design: you pass explicit relative item paths (strings) under each exp_dir.
# Examples:
#   items = [f"gaze_estimations/{m}" for m in models]               # directories
#   items = ["head_bboxes.npy"]                                     # files
#   items = ["metrics/summary.csv", "aux/notes.txt", "images"]      # mixed / nested
#
# The flattener mirrors those relative paths under <output>/<id>/...
# The mapping CSV stores: id, exp_dir_relpath, item_relpath
# Unflattening copies <output>/<id>/<item_relpath> back to <base>/<exp_dir_relpath>/<item_relpath>.

def flatten(
    subject_dirs: Iterable[str],
    exp_types: Iterable[str],
    output_dir: str,
    base_dir: str,
    items: List[str],
):
    """
    General flattener that copies specific relative item paths from each exp_dir.

    Args:
        subject_dirs: iterable of subject directories to scan.
        exp_types: iterable of experiment types to include.
        output_dir: where flattened copies and mapping CSV are written.
        base_dir: dataset base directory; mapping stores paths relative to this.
        items: list of relative paths (under each exp_dir) to copy. Each entry may be
               a file or a directory (and may be nested like "foo/bar/baz.npy").
    """
    os.makedirs(output_dir, exist_ok=True)
    mapping_path = os.path.join(output_dir, "path_mapping.csv")
    all_exp_dirs = get_all_experiment_paths(subject_dirs=subject_dirs, exp_types=exp_types)

    with open(mapping_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "exp_dir_relpath", "item_relpath"])
        idx = 1

        for exp_dir in all_exp_dirs:
            # Which of the requested items actually exist for this exp_dir?
            existing_items = []
            for rel in items:
                src_path = os.path.join(exp_dir, rel)
                if os.path.exists(src_path):
                    existing_items.append(rel)

            if not existing_items:
                continue  # nothing to copy for this experiment

            # Prepare flat bucket for this experiment
            flat_dir = os.path.join(output_dir, str(idx))
            os.makedirs(flat_dir, exist_ok=True)

            # Copy each item, preserving its relative path under the flat bucket
            exp_dir_relpath = os.path.relpath(exp_dir, base_dir)
            for rel in existing_items:
                src_path = os.path.join(exp_dir, rel)
                dst_path = os.path.join(flat_dir, rel)
                _safe_copy(src_path, dst_path)

                # Record one row per (id, exp_dir_relpath, item_relpath)
                writer.writerow([idx, exp_dir_relpath, rel])

            idx += 1

    print(f"Flattened {idx - 1} experiment directories with at least one matched item.")


def _unflatten_from_new_csv(output_dir: str, base_dir: str, reader: csv.DictReader):
    """
    Handle CSV with header: id, exp_dir_relpath, item_relpath
    """
    for row in reader:
        idx = row["id"]
        # Normalize paths: converts Windows '\' to Linux '/'
        exp_dir_rel = row["exp_dir_relpath"].replace("\\", "/")
        item_rel = row["item_relpath"].replace("\\", "/")

        src_item = os.path.join(output_dir, idx, item_rel)
        dst_item = os.path.join(base_dir, exp_dir_rel, item_rel)

        if not os.path.exists(src_item):
            print(f"Warning: missing source '{src_item}', skipping.")
            continue

        _safe_copy(src_item, dst_item)


def _unflatten_from_old_csv(output_dir: str, base_dir: str, reader: csv.DictReader):
    """
    Handle CSV with header: id, relative_path
    """
    for row in reader:
        idx = row["id"]
        # Normalize path: converts Windows '\' to Linux '/'
        rel_path = row["relative_path"].replace("\\", "/")
        original_path = os.path.join(base_dir, rel_path)

        src_dir = os.path.join(output_dir, idx)
        if not os.path.isdir(src_dir):
            print(f"Warning: {src_dir} does not exist or is not a directory. Skipping.")
            continue

        os.makedirs(original_path, exist_ok=True)

        for name in os.listdir(src_dir):
            src_child = os.path.join(src_dir, name)
            dst_child = os.path.join(original_path, name)

            _safe_copy(src_child, dst_child)


def unflatten(output_dir: str, base_dir: str):
    """
    Unflattener that restores files/dirs back to their original exp_dir locations.

    Auto-detects CSV schema:
      - NEW: id, exp_dir_relpath, item_relpath
      - OLD: id, relative_path

    Copies:
      NEW: <output>/<id>/<item_relpath> -> <base>/<exp_dir_relpath>/<item_relpath>
      OLD: <output>/<id>/* (children)   -> <base>/<relative_path>/* (merge, overwrite)
    """
    mapping_path = os.path.join(output_dir, "path_mapping.csv")
    if not os.path.isfile(mapping_path):
        raise FileNotFoundError(f"No path_mapping.csv found at {mapping_path}")

    with open(mapping_path, "r", newline="") as csvfile:
        # Peek header
        header_line = csvfile.readline()
        if not header_line:
            raise ValueError("Empty path_mapping.csv")

        # Reset to start and create DictReader
        csvfile.seek(0)
        reader = csv.DictReader(csvfile)
        fieldnames = [fn.strip() for fn in (reader.fieldnames or [])]

        if set(fieldnames) == {"id", "exp_dir_relpath", "item_relpath"}:
            _unflatten_from_new_csv(output_dir, base_dir, reader)
        elif set(fieldnames) == {"id", "relative_path"}:
            _unflatten_from_old_csv(output_dir, base_dir, reader)
        else:
            raise ValueError(
                f"Unrecognized mapping CSV header: {fieldnames}. "
                "Expected either ['id','exp_dir_relpath','item_relpath'] or ['id','relative_path']."
            )

    print("Unflattening complete.")


# --------------------- Gaze-specific wrappers (old API) --------------------- #
# Keep old signatures for drop-in replacement, but allow choosing CSV format.

def flatten_gaze_estimations(
    subject_dirs: Iterable[str],
    exp_types: Iterable[str],
    output_dir: str,
    models_to_include: Iterable[str],
    base_dir: str,
    csv_format: Literal["new", "old"] = "new",
):
    """
    Flattens specified gaze estimation model folders.

    csv_format:
      - "new": writes the general new CSV with (id,exp_dir_relpath,item_relpath)
               where item_relpath = f"gaze_estimations/{model}"
      - "old": writes the legacy CSV with (id,relative_path) where relative_path
               points to ".../<exp>/gaze_estimations", and the flattened bucket
               contains all selected model subfolders (exactly like the legacy script).
               ONLY suited for directory-based gaze estimations (not files).
    """
    if csv_format == "new":
        items = [f"gaze_estimations/{m}" for m in models_to_include]
        return flatten(
            subject_dirs=subject_dirs,
            exp_types=exp_types,
            output_dir=output_dir,
            base_dir=base_dir,
            items=items,
        )

    # ----- Legacy "old" CSV path -----
    os.makedirs(output_dir, exist_ok=True)
    mapping_path = os.path.join(output_dir, "path_mapping.csv")
    all_exp_dirs = get_all_experiment_paths(subject_dirs=subject_dirs, exp_types=exp_types)

    with open(mapping_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "relative_path"])
        idx = 1

        for exp_dir in all_exp_dirs:
            gaze_estimations_path = os.path.join(exp_dir, "gaze_estimations")

            if not os.path.isdir(gaze_estimations_path):
                continue

            valid_models = [
                m for m in models_to_include
                if os.path.isdir(os.path.join(gaze_estimations_path, m))
            ]
            if not valid_models:
                continue

            flat_dir = os.path.join(output_dir, str(idx))
            os.makedirs(flat_dir, exist_ok=True)

            # Copy each selected model directory under the bucket
            for model in valid_models:
                src = os.path.join(gaze_estimations_path, model)
                dst = os.path.join(flat_dir, model)
                _safe_copy(src, dst)

            # In old format, we store the *gaze_estimations* directory as relative path
            rel_path = os.path.relpath(gaze_estimations_path, base_dir)
            writer.writerow([idx, rel_path])

            idx += 1

    print(f"(OLD CSV) Flattened {idx - 1} gaze-estimation directories.")


def flatten_blink_annotations(
    subject_dirs: Iterable[str],
    exp_types: Iterable[str],
    output_dir: str,
    annotators_to_include: Iterable[str],
    base_dir: str,
):
    """
    Flattens only the specified blink annotation files made by the specified annotators.
    Uses the NEW CSV because old CSV can't represent arbitrary files safely.
    """
    items = [f"blink_annotations_by_{a}.npy" for a in annotators_to_include]
    return flatten(
        subject_dirs=subject_dirs,
        exp_types=exp_types,
        output_dir=output_dir,
        base_dir=base_dir,
        items=items,
    )


# --------------------------------- Main --------------------------------- #

if __name__ == "__main__":
    BASE_DIR = config.get_dataset_base_directory()
    SUBJECT_DIRS = config.get_dataset_subject_directories()

    EXP_TYPES = [
        "lighting_10", "lighting_25", "lighting_50", "lighting_100",
        "circular_movement", "head_pose_middle", "head_pose_left", "head_pose_right",
        "line_movement_slow", "line_movement_fast"
    ]

    # ---------------- Example 1: GAZE ESTIMATIONS ----------------
    # MODELS_TO_INCLUDE = ["gaze3d_clip_len_8_rectification"]
    # OUTPUT_DIR_GAZE = f"{BASE_DIR}/flattened_gaze_estimations (gaze3d_clip_len_8_rectification)"
    # flatten_gaze_estimations(SUBJECT_DIRS, EXP_TYPES, OUTPUT_DIR_GAZE, MODELS_TO_INCLUDE, base_dir=BASE_DIR)
    # unflatten(OUTPUT_DIR_GAZE, base_dir=BASE_DIR)

    # ---------------- Example 2: FILE OR MIXED ITEMS (NEW CSV ONLY) ----------------
    # OTHER_ITEMS = ["head_bboxes.npy"]
    # OUTPUT_DIR_OTHER = f"{BASE_DIR}/flattened_head_bboxes"
    # flatten(SUBJECT_DIRS, EXP_TYPES, OUTPUT_DIR_OTHER, base_dir=BASE_DIR, items=OTHER_ITEMS)
    # unflatten(OUTPUT_DIR_OTHER, base_dir=BASE_DIR)
