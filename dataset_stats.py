# dataset_stats.py
import argparse
import os
from typing import Tuple

import pandas as pd

import frame_db


def _dedupe_to_frames(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse per-(exp_dir, frame_idx) across gaze models so we don't double-count.
    A frame is considered valid if ANY source row marks it valid.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["exp_dir", "frame_idx", "subject_dir", "exp_type", "point", "is_valid_any"]
        )

    # Keep stable metadata from the first occurrence; validity = max across rows
    agg = {
        "subject_dir": "first",
        "exp_type": "first",
        "point": "first",
        "is_valid": "max",
    }
    frames = (
        df.groupby(["exp_dir", "frame_idx"], as_index=False)
          .agg(agg)
          .rename(columns={"is_valid": "is_valid_any"})
    )
    return frames


def _summarize_by_video(frames: pd.DataFrame, fps: float) -> pd.DataFrame:
    """
    One row per video (exp_dir) with frame counts and durations (all vs valid).
    """
    if frames.empty:
        return pd.DataFrame(
            columns=[
                "exp_dir", "subject_dir", "exp_type",
                "num_frames_all", "num_frames_valid",
                "duration_all_sec", "duration_valid_sec",
            ]
        )

    vids = (
        frames.groupby("exp_dir", as_index=False)
              .agg(
                  subject_dir=("subject_dir", "first"),
                  exp_type=("exp_type", "first"),
                  num_frames_all=("frame_idx", "nunique"),
                  num_frames_valid=("is_valid_any", "sum"),
              )
    )
    vids["duration_all_sec"] = vids["num_frames_all"] / fps
    vids["duration_valid_sec"] = vids["num_frames_valid"] / fps
    return vids


def _totals_by_exp_type(vids: pd.DataFrame, fps: float) -> pd.DataFrame:
    """
    Totals per experiment type (videos, frames, durations).
    """
    if vids.empty:
        return pd.DataFrame(
            columns=[
                "exp_type", "num_subjects", "num_videos",
                "num_frames_all", "num_frames_valid",
                "duration_all_sec", "duration_valid_sec",
            ]
        )

    # subjects per exp_type = unique subject_dir that have at least one video in that exp_type
    grp = vids.groupby("exp_type")
    totals = pd.DataFrame({
        "num_subjects": grp["subject_dir"].nunique(),
        "num_videos": grp["exp_dir"].nunique(),
        "num_frames_all": grp["num_frames_all"].sum(),
        "num_frames_valid": grp["num_frames_valid"].sum(),
    }).reset_index()

    totals["duration_all_sec"] = totals["num_frames_all"] / fps
    totals["duration_valid_sec"] = totals["num_frames_valid"] / fps
    return totals.sort_values("exp_type").reset_index(drop=True)


def _per_subject_per_exp_type(vids: pd.DataFrame, fps: float) -> pd.DataFrame:
    """
    Per subject & exp_type breakdown: videos, frames, durations.
    """
    if vids.empty:
        return pd.DataFrame(
            columns=[
                "subject_dir", "exp_type", "num_videos",
                "num_frames_all", "num_frames_valid",
                "duration_all_sec", "duration_valid_sec",
            ]
        )

    grp = vids.groupby(["subject_dir", "exp_type"], as_index=False).agg(
        num_videos=("exp_dir", "nunique"),
        num_frames_all=("num_frames_all", "sum"),
        num_frames_valid=("num_frames_valid", "sum"),
    )
    grp["duration_all_sec"] = grp["num_frames_all"] / fps
    grp["duration_valid_sec"] = grp["num_frames_valid"] / fps
    return grp.sort_values(["subject_dir", "exp_type"]).reset_index(drop=True)


def _overall_counts(frames: pd.DataFrame, vids: pd.DataFrame, fps: float) -> pd.DataFrame:
    """
    Overall totals across the entire dataset.
    """
    if frames.empty:
        return pd.DataFrame(
            [{"num_subjects": 0, "num_videos": 0,
              "num_frames_all": 0, "num_frames_valid": 0,
              "duration_all_sec": 0.0, "duration_valid_sec": 0.0}]
        )

    num_subjects = frames["subject_dir"].nunique()
    num_videos = vids["exp_dir"].nunique() if not vids.empty else 0
    num_frames_all = len(frames.index)  # already unique per (exp_dir, frame_idx)
    num_frames_valid = int(frames["is_valid_any"].sum())

    return pd.DataFrame([{
        "num_subjects": num_subjects,
        "num_videos": num_videos,
        "num_frames_all": num_frames_all,
        "num_frames_valid": num_frames_valid,
        "duration_all_sec": num_frames_all / fps,
        "duration_valid_sec": num_frames_valid / fps,
    }])


def main(outdir: str, fps: float, quiet: bool):
    df = frame_db.load_frame_dataset()
    if df.empty:
        print("[dataset_stats] frame_db is empty. Nothing to summarize.")
        return

    # Reduce to unique frames across models
    frames = _dedupe_to_frames(df)

    frames = frames[~frames["exp_dir"].apply(config.is_experiment_directory_excluded_from_eval)]

    # Per-video summary (distinct exp_dir)
    vids = _summarize_by_video(frames, fps=fps)

    # High-level counts
    overall = _overall_counts(frames, vids, fps=fps)

    # Totals per exp_type
    by_exp = _totals_by_exp_type(vids, fps=fps)

    # Per subject & exp_type
    by_subject_exp = _per_subject_per_exp_type(vids, fps=fps)

    # Optional: per-subject totals (across all exp_types)
    by_subject = (
        by_subject_exp.groupby("subject_dir", as_index=False)
        .agg(
            num_videos=("num_videos", "sum"),
            num_frames_all=("num_frames_all", "sum"),
            num_frames_valid=("num_frames_valid", "sum"),
            duration_all_sec=("duration_all_sec", "sum"),
            duration_valid_sec=("duration_valid_sec", "sum"),
        )
        .sort_values("subject_dir")
        .reset_index(drop=True)
    )

    # Ensure outdir
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        overall.to_csv(os.path.join(outdir, "overall_totals.csv"), index=False)
        by_exp.to_csv(os.path.join(outdir, "totals_by_experiment_type.csv"), index=False)
        by_subject_exp.to_csv(os.path.join(outdir, "per_subject_per_experiment_type.csv"), index=False)
        by_subject.to_csv(os.path.join(outdir, "per_subject_totals.csv"), index=False)
        vids.to_csv(os.path.join(outdir, "per_video.csv"), index=False)

    if not quiet:
        print("\n=== OVERALL ===")
        print(overall.to_string(index=False))

        print("\n=== TOTALS BY EXPERIMENT TYPE ===")
        print(by_exp.to_string(index=False))

        print("\n=== PER SUBJECT (TOTALS ACROSS EXP TYPES) ===")
        print(by_subject.to_string(index=False))

        print("\n=== PER SUBJECT × EXPERIMENT TYPE ===")
        # Keep it readable if very long
        if len(by_subject_exp) > 200:
            print(by_subject_exp.head(200).to_string(index=False))
            print(f"... ({len(by_subject_exp)} rows total; full table saved to CSV if outdir specified)")
        else:
            print(by_subject_exp.to_string(index=False))

        print("\n=== PER VIDEO (EXP_DIR) ===")
        if len(vids) > 200:
            print(vids.head(200).to_string(index=False))
            print(f"... ({len(vids)} rows total; full table saved to CSV if outdir specified)")
        else:
            print(vids.to_string(index=False))

    # Convenience: echo headline numbers explicitly
    if not quiet:
        n_subjects = int(overall.loc[0, "num_subjects"])
        n_videos = int(overall.loc[0, "num_videos"])
        print(f"\n[Headline] Number of subjects: {n_subjects}")
        print(f"[Headline] Number of videos (total, distinct exp_dir): {n_videos}")


if __name__ == "__main__":
    import config
    base_dir = config.get_dataset_base_directory()
    # If user passes outdir as empty string, don't save files.
    outdir = os.path.join(base_dir, "dataset_stats")
    fps = config.get_rgb_fps()
    main(outdir=outdir, fps=fps, quiet=True)
