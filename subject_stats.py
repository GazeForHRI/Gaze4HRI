import os
import json
import pandas as pd
import config

# -------------------- exclusion helper --------------------
def _is_excluded_subject_dir(subject_dir: str) -> bool:
    """subject_dir is 'YYYY-MM-DD/SubjectName'"""
    try:
        b = bool(config.is_subject_directory_excluded_from_eval(subject_dir))
        if b:
            print("Excluding subject_dir: ", subject_dir)
        return b
    except AttributeError:
        return False


# -------------------- gather subject metadata --------------------
def gather_subject_info(subject_dirs=None):
    if subject_dirs is None:
        subject_dirs = config.get_dataset_subject_directories()

    subject_metadata = {}
    for subj_dir in subject_dirs:
        for root, dirs, files in os.walk(subj_dir):
            if "subject_info.json" in files:
                json_path = os.path.join(root, "subject_info.json")
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if "subjects" in data and data["subjects"]:
                            subj = data["subjects"][0]  # assuming 1 subject per file
                            name = subj["name"].strip().lower()

                            # subject_dir key: "<date>/<subject>"
                            relative_path = os.path.normpath(root).split(os.sep)[-2:]
                            subject_dir_key = relative_path[0] + "/" + relative_path[1]

                            # Skip excluded subjects
                            if _is_excluded_subject_dir(subject_dir_key):
                                continue

                            if name in subject_metadata:
                                print(f"Warning: Duplicate subject name found: {name}. Overwriting existing metadata.")

                            subject_metadata[subject_dir_key] = {
                                "name": name,
                                "gender": subj.get("gender"),
                                "birthyear": subj.get("birth_year"),
                                "glasses": subj.get("glasses"),
                                "height_cm": subj.get("height_cm"),
                            }
                except Exception as e:
                    print(f"Error reading {json_path}: {e}")

    return subject_metadata


# -------------------- save metadata JSON --------------------
def save_subject_metadata(subject_metadata: dict, out_dir: str):
    out_path = os.path.join(out_dir, "subject_metadata.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(subject_metadata, f, indent=2)
    print(f"Saved subject metadata to {out_path}")


# -------------------- save stats JSON --------------------
def save_subject_stats(subject_metadata: dict, out_dir: str):
    # Each entry in subject_metadata corresponds to one subject_dir (YYYY-MM-DD/SubjectName)
    # -> treat EACH as a distinct subject for stats.
    df = pd.DataFrame.from_dict(subject_metadata, orient="index").reset_index()
    df.rename(columns={"index": "subject_dir"}, inplace=True)

    subjects = df[["subject_dir", "gender", "glasses"]].copy()  # no dedup by name

    def _canon_gender(x):
        if pd.isna(x): return None
        s = str(x).strip().lower()
        if s in ("female","woman","women","f"): return "women"
        if s in ("male","man","men","m"): return "men"
        return None
    subjects["gender_canon"] = subjects["gender"].map(_canon_gender)

    def _canon_glasses(x):
        if pd.isna(x): return None
        if isinstance(x, (int, float, bool)):
            return "has glasses" if bool(x) else "no glasses"
        s = str(x).strip().lower().replace("-", "_").replace(" ", "_")
        if s in ("true","1","yes","y","glasses","has_glasses","with_glasses","wearing_glasses","wears_glasses"):
            return "has glasses"
        if s in ("false","0","no","n","no_glasses","without_glasses","none"):
            return "no glasses"
        return None
    subjects["glasses_canon"] = subjects["glasses"].map(_canon_glasses)

    stats = {
        "total": int(len(subjects)),  # equals len(subject_metadata) (e.g., 52)
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
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved subject stats to {out_path} (counted {stats['total']} subjects)")


# -------------------- main --------------------
def main():
    base_dir = config.get_dataset_base_directory()
    subject_dirs = config.get_dataset_subject_directories()

    subject_metadata = gather_subject_info(subject_dirs=subject_dirs)
    print(f"Gathered {len(subject_metadata)} subjects' metadata.")

    save_subject_metadata(subject_metadata, base_dir)
    save_subject_stats(subject_metadata, base_dir)


if __name__ == "__main__":
    main()
