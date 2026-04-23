import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import json
import pickle
from pathlib import Path
from tqdm import tqdm
from exordium.video.tddfa_v2 import TDDFA_V2
import exordium.video.iris as ex_iris

# --- EXORDIUM BUG FIX 1: UNBATCHING ---
def _patched_eye_to_features(self, eye):
    if isinstance(eye, (str, Path, os.PathLike)):
        eye_original = cv2.imread(str(eye))
        eye_original = cv2.cvtColor(eye_original, cv2.COLOR_BGR2RGB)
    else:
        eye_original = eye
    
    eye_resized = cv2.resize(eye_original, (64, 64), interpolation=cv2.INTER_AREA)
    
    eye_landmarks_batch, iris_landmarks_batch = self([eye_resized])
    
    eye_landmarks = eye_landmarks_batch[0]
    iris_landmarks = iris_landmarks_batch[0]
    
    iris_diameters = ex_iris.calculate_iris_diameters(iris_landmarks)
    eyelid_pupil_distances = ex_iris.calculate_eyelid_pupil_distances(iris_landmarks, eye_landmarks)
    ear = ex_iris.calculate_eye_aspect_ratio(eye_landmarks)
    
    return {
        'eye_original': eye_original,
        'eye': eye_resized,
        'landmarks': eye_landmarks,
        'iris_landmarks': iris_landmarks,
        'iris_diameters': iris_diameters,
        'eyelid_pupil_distances': eyelid_pupil_distances,
        'ear': ear
    }

ex_iris.IrisWrapper.eye_to_features = _patched_eye_to_features


# --- EXORDIUM BUG FIX 2: SCIPY 1D STRICTNESS ---
original_euclidean = ex_iris.distance.euclidean

def _safe_euclidean(u, v, **kwargs):
    u, v = np.asarray(u), np.asarray(v)
    if u.ndim > 1:
        u = np.mean(u, axis=0)
    if v.ndim > 1:
        v = np.mean(v, axis=0)
    return original_euclidean(u, v, **kwargs)

ex_iris.distance.euclidean = _safe_euclidean
# -----------------------------------------------

def get_bounding_boxes(w, h, face_mesh_results):
    if not face_mesh_results.multi_face_landmarks:
        return None, None, None

    landmarks = face_mesh_results.multi_face_landmarks[0].landmark
    
    x_coords = [lm.x * w for lm in landmarks]
    y_coords = [lm.y * h for lm in landmarks]
    face_bbox = [
        max(0, int(min(x_coords))), 
        max(0, int(min(y_coords))), 
        min(w, int(max(x_coords))), 
        min(h, int(max(y_coords)))
    ]

    left_eye_indices = [33, 160, 158, 133, 153, 144]
    right_eye_indices = [362, 385, 387, 263, 373, 380]

    def get_square_eye_bbox(indices, pad=15):
        ex = [landmarks[i].x * w for i in indices]
        ey = [landmarks[i].y * h for i in indices]
        cx, cy = int(np.mean(ex)), int(np.mean(ey))
        
        ew = max(ex) - min(ex)
        eh = max(ey) - min(ey)
        size = int(max(ew, eh)) + pad * 2
        half = size // 2
        
        return [
            max(0, cx - half), 
            max(0, cy - half), 
            min(w, cx + half), 
            min(h, cy + half)
        ]

    left_eye_bbox = get_square_eye_bbox(left_eye_indices)
    right_eye_bbox = get_square_eye_bbox(right_eye_indices)

    return face_bbox, left_eye_bbox, right_eye_bbox

def get_dummy_features():
    """Provides a zeroed-out dictionary to prevent PyTorch KeyErrors on missing frames."""
    return {
        'eye_original': np.zeros((64, 64, 3), dtype=np.uint8),
        'eye': np.zeros((64, 64, 3), dtype=np.uint8),
        'landmarks': np.zeros((71, 2), dtype=np.float32),
        'iris_landmarks': np.zeros((5, 2), dtype=np.float32),
        'iris_diameters': np.zeros((2,), dtype=np.float32),
        'eyelid_pupil_distances': np.zeros((2,), dtype=np.float32),
        'ear': 0.0
    }

def extract_and_save_exordium_features(dataloader, face_mesh_detector=None, face_model=None, iris_model=None):
    exp_dir = dataloader.get_cwd()
    out_dir = os.path.join(exp_dir, "exordium_landmarks2")
    crops_dir = os.path.join(out_dir, "crops")
    
    os.makedirs(os.path.join(crops_dir, "face"), exist_ok=True)
    os.makedirs(os.path.join(crops_dir, "left_eye"), exist_ok=True)
    os.makedirs(os.path.join(crops_dir, "right_eye"), exist_ok=True)

    print("Loading video frames...")
    video_frames_raw = dataloader.load_rgb_video(as_numpy=True)
    if not video_frames_raw:
        raise ValueError(f"No frames found in {exp_dir}")

    if face_mesh_detector is None:
        face_mesh_detector = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5
        )

    valid_frames = []
    
    print("\n--- PHASE 1: CROPPING & SAVING ---")
    sys.stdout.flush()

    for i, frame_raw in enumerate(tqdm(video_frames_raw, desc="Cropping Faces & Eyes")):
        if frame_raw is None:
            continue

        h, w, _ = frame_raw.shape
        
        # RESTORED: The array is natively BGR. We convert to RGB for MediaPipe.
        frame_rgb_mp = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
        results = face_mesh_detector.process(frame_rgb_mp)
        face_bbox, left_bbox, right_bbox = get_bounding_boxes(w, h, results)

        if face_bbox is None:
            continue

        # RESTORED: Slice and save directly from the native BGR frame
        face_crop = frame_raw[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
        left_crop = frame_raw[left_bbox[1]:left_bbox[3], left_bbox[0]:left_bbox[2]]
        right_crop = frame_raw[right_bbox[1]:right_bbox[3], right_bbox[0]:right_bbox[2]]

        if face_crop.size == 0 or left_crop.size == 0 or right_crop.size == 0:
            continue

        left_crop = cv2.resize(left_crop, (64, 64), interpolation=cv2.INTER_CUBIC)
        right_crop = cv2.resize(right_crop, (64, 64), interpolation=cv2.INTER_CUBIC)

        face_path = os.path.join(crops_dir, "face", f"{i:06d}.jpg")
        left_path = os.path.join(crops_dir, "left_eye", f"{i:06d}.jpg")
        right_path = os.path.join(crops_dir, "right_eye", f"{i:06d}.jpg")

        cv2.imwrite(face_path, face_crop)
        cv2.imwrite(left_path, left_crop)
        cv2.imwrite(right_path, right_crop)

        valid_frames.append({
            'frame_idx': i,
            'face_path': face_path,
            'left_path': left_path,
            'right_path': right_path,
            'face_bbox': face_bbox,
            'left_bbox': left_bbox,
            'right_bbox': right_bbox
        })

    print("\n--- PHASE 2: EXORDIUM EXTRACTION ---")
    sys.stdout.flush()

    if face_model is None:
        face_model = TDDFA_V2()
    if iris_model is None:
        iris_model = ex_iris.IrisWrapper()
    
    total_frames = len(video_frames_raw)
    extracted_data = {}

    for item in tqdm(valid_frames, desc="Extracting Features"):
        idx = item['frame_idx']
        try:
            headpose_dict = face_model(item['face_path'])
            left_eye_features = iris_model.eye_to_features(item['left_path'])
            right_eye_features = iris_model.eye_to_features(item['right_path'])

            extracted_data[idx] = {
                'face_bbox': item['face_bbox'],
                'left_bbox': item['left_bbox'],
                'right_bbox': item['right_bbox'],
                'headpose': headpose_dict['headpose'],
                'left_features': left_eye_features,
                'right_features': right_eye_features
            }
        except Exception as e:
            raise RuntimeError(f"Exordium crashed on frame {idx} with error: {e}")

    print("\n--- PHASE 3: COMPILING DATA ---")
    
    samples = {}
    for i in range(total_frames):
        if i in extracted_data:
            samples[i] = {
                'id': i,
                'face_bbox': extracted_data[i]['face_bbox'],
                'left_bbox': extracted_data[i]['left_bbox'],
                'right_bbox': extracted_data[i]['right_bbox'],
                'tddfa-retinaface_headpose': extracted_data[i]['headpose'],
                'annotation_left_eye_features': extracted_data[i]['left_features'],
                'annotation_right_eye_features': extracted_data[i]['right_features']
            }
        else:
            # WORK ON HERE :: FIND A BETTER DUMMY FEATURE ACQUIRING METHOD
            samples[i] = {
                'id': i,
                'face_bbox': [-1, -1, -1, -1],
                'left_bbox': [-1, -1, -1, -1],
                'right_bbox': [-1, -1, -1, -1],
                'tddfa-retinaface_headpose': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'annotation_left_eye_features': get_dummy_features(),
                'annotation_right_eye_features': get_dummy_features()
            }
    pkl_path = os.path.join(out_dir, "exordium_data.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(samples, f)

    print(f"Data saved successfully to {pkl_path}. {len(extracted_data)}/{total_frames} frames fully processed.")
    return len(extracted_data), total_frames


def fix_exordium(dataloader, cheap_limit=5, hermite_coefficent=0.5):
    # 1. Get the base experiment directory
    exp_dir = dataloader.get_cwd() 
    
    # 2. Define the landmarks folder path explicitly
    landmarks_dir = os.path.join(exp_dir, "exordium_landmarks2")
    
    # 3. Reference files within that folder
    pkl_path = os.path.join(landmarks_dir, "exordium_data.pkl")
    out_path = os.path.join(landmarks_dir, "exordium_data_fixed.pkl")
    log_path = os.path.join(landmarks_dir, "missing_frames_log.json")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Could not find: {pkl_path}")

    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)

    if not isinstance(samples, dict) or len(samples) == 0:
        raise ValueError("Loaded samples are empty or not a dictionary.")

    print("Loading raw video frames for real-pixel cropping...")
    video_frames_raw = dataloader.load_rgb_video(as_numpy=True)
    frame_h, frame_w = video_frames_raw[0].shape[:2]

    sorted_indices = sorted(samples.keys())
    total_frames = len(sorted_indices)

    def is_dummy(sample):
        if sample is None:
            return True
        f_bbox = list(sample.get("face_bbox", [-1, -1, -1, -1]))
        l_bbox = list(sample.get("left_bbox", [-1, -1, -1, -1]))
        r_bbox = list(sample.get("right_bbox", [-1, -1, -1, -1]))
        return (f_bbox == [-1, -1, -1, -1] or 
                l_bbox == [-1, -1, -1, -1] or 
                r_bbox == [-1, -1, -1, -1])

    def find_dummy_runs(indices):
        runs = []
        run_start = None
        prev_idx = None
        for idx in indices:
            if is_dummy(samples[idx]):
                if run_start is None: run_start = idx
                elif prev_idx is not None and idx != prev_idx + 1:
                    runs.append((run_start, prev_idx))
                    run_start = idx
            else:
                if run_start is not None:
                    runs.append((run_start, prev_idx))
                    run_start = None
            prev_idx = idx
        if run_start is not None:
            runs.append((run_start, prev_idx))
        return runs

    dummy_runs = find_dummy_runs(sorted_indices)
    
    # --- NEW: Logging Gaps for Statistical Analysis ---
    log_data = {
        "exp_dir": exp_dir,
        "total_frames": total_frames,
        "total_missing": sum(end - start + 1 for start, end in dummy_runs),
        "missing_runs": dummy_runs
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=4)
    # ------------------------------------------------

    if not dummy_runs:
        with open(out_path, "wb") as f:
            pickle.dump(samples, f)
        return total_frames, total_frames, total_frames

    # Mathematical Type Casting Helpers
    def to_numpy(x): return x if isinstance(x, np.ndarray) else np.asarray(x)
    def cast_like(ref, value):
        if isinstance(ref, np.ndarray):
            out = np.asarray(value, dtype=ref.dtype)
            if np.issubdtype(ref.dtype, np.integer):
                info = np.iinfo(ref.dtype)
                out = np.clip(np.rint(out), info.min, info.max).astype(ref.dtype)
            else: out = out.astype(ref.dtype)
            return out
        if isinstance(ref, (list, tuple)):
            arr = np.asarray(value)
            if arr.ndim == 0: arr = np.asarray([arr.item()])
            if len(ref) > 0:
                ref_arr = np.asarray(ref)
                if np.issubdtype(ref_arr.dtype, np.integer):
                    arr = np.rint(arr).astype(ref_arr.dtype)
                else: arr = arr.astype(ref_arr.dtype)
            out_list = arr.tolist()
            return tuple(out_list) if isinstance(ref, tuple) else out_list
        if isinstance(ref, (int, np.integer)): return int(round(float(np.asarray(value))))
        if isinstance(ref, (float, np.floating)): return float(np.asarray(value))
        return value

    # Interpolation Helpers
    def estimate_derivative(prev2_val, prev_val, next_val, next2_val, side="left"):
        if side == "left":
            if prev2_val is not None: return to_numpy(prev_val) - to_numpy(prev2_val)
            return to_numpy(next_val) - to_numpy(prev_val)
        else:
            if next2_val is not None: return to_numpy(next2_val) - to_numpy(next_val)
            return to_numpy(next_val) - to_numpy(prev_val)

    def hermite_interp(y0, y1, m0, m1, t):
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        return h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1

    def interpolate_value(prev2_val, prev_val, next_val, next2_val, ref_prev, t):
        try:
            y0, y1 = to_numpy(prev_val).astype(np.float64), to_numpy(next_val).astype(np.float64)
            p2 = to_numpy(prev2_val).astype(np.float64) if prev2_val is not None else None
            n2 = to_numpy(next2_val).astype(np.float64) if next2_val is not None else None
            m0 = estimate_derivative(p2, y0, y1, n2, side="left")
            m1 = estimate_derivative(p2, y0, y1, n2, side="right")
            interp = hermite_coefficent * hermite_interp(y0, y1, m0, m1, t) + (1 - hermite_coefficent) * ((1.0 - t) * y0 + t * y1)
            return cast_like(ref_prev, interp)
        except Exception:
            return ref_prev

    # --- NEW: Extrapolation Helper for Edge Cases ---
    def extrapolate_value(base_val, ref_val, ref_out, step):
        if isinstance(ref_out, dict):
            return {k: extrapolate_value(base_val.get(k), ref_val.get(k) if ref_val else None, ref_out[k], step) for k in ref_out.keys()}
        try:
            b = to_numpy(base_val).astype(np.float64)
            if ref_val is not None:
                r = to_numpy(ref_val).astype(np.float64)
                # Linear extrapolation: base + step * (base - reference)
                extrap = b + step * (b - r)
            else:
                extrap = b # Degradation to Copy (Zero-Order Hold)
            return cast_like(ref_out, extrap)
        except Exception:
            return ref_out
    # ------------------------------------------------

    def interpolate_numeric_features(p2_f, p_f, n_f, n2_f, t):
        keys = ['landmarks', 'iris_landmarks', 'iris_diameters', 'eyelid_pupil_distances', 'ear']
        return {k: interpolate_value(p2_f[k] if p2_f else None, p_f[k], n_f[k], n2_f[k] if n2_f else None, p_f[k], t) for k in keys}

    def extrapolate_numeric_features(base_f, ref_f, step):
        keys = ['landmarks', 'iris_landmarks', 'iris_diameters', 'eyelid_pupil_distances', 'ear']
        return {k: extrapolate_value(base_f[k], ref_f[k] if ref_f else None, base_f[k], step) for k in keys}

    fixed_samples = pickle.loads(pickle.dumps(samples))

    for start_idx, end_idx in dummy_runs:
        run_len = end_idx - start_idx + 1
        left_idx, right_idx = start_idx - 1, end_idx + 1
        left2_idx, right2_idx = start_idx - 2, end_idx + 2

        left_exists = left_idx in samples and not is_dummy(samples[left_idx])
        right_exists = right_idx in samples and not is_dummy(samples[right_idx])

        if not left_exists and not right_exists: continue
        
        if run_len > cheap_limit:
            print(f"CHEAP SOLUTION WONT CUT IT for run {start_idx}->{end_idx} (len={run_len})")
            continue

        prev_sample = samples[left_idx] if left_exists else None
        next_sample = samples[right_idx] if right_exists else None
        prev2_sample = samples[left2_idx] if left2_idx in samples and not is_dummy(samples[left2_idx]) else None
        next2_sample = samples[right2_idx] if right2_idx in samples and not is_dummy(samples[right2_idx]) else None

        for j, idx in enumerate(range(start_idx, end_idx + 1), start=1):
            
            # --- Routing Logic: Interpolate vs. Extrapolate ---
            if left_exists and right_exists:
                # INTERNAL GAP: Interpolate
                t = j / (run_len + 1)
                i_face_bbox = interpolate_value(prev2_sample["face_bbox"] if prev2_sample else None, prev_sample["face_bbox"], next_sample["face_bbox"], next2_sample["face_bbox"] if next2_sample else None, prev_sample["face_bbox"], t)
                i_left_bbox = interpolate_value(prev2_sample["left_bbox"] if prev2_sample else None, prev_sample["left_bbox"], next_sample["left_bbox"], next2_sample["left_bbox"] if next2_sample else None, prev_sample["left_bbox"], t)
                i_right_bbox = interpolate_value(prev2_sample["right_bbox"] if prev2_sample else None, prev_sample["right_bbox"], next_sample["right_bbox"], next2_sample["right_bbox"] if next2_sample else None, prev_sample["right_bbox"], t)
                i_headpose = interpolate_value(prev2_sample["tddfa-retinaface_headpose"] if prev2_sample else None, prev_sample["tddfa-retinaface_headpose"], next_sample["tddfa-retinaface_headpose"], next2_sample["tddfa-retinaface_headpose"] if next2_sample else None, prev_sample["tddfa-retinaface_headpose"], t)
                new_left_features = interpolate_numeric_features(prev2_sample["annotation_left_eye_features"] if prev2_sample else None, prev_sample["annotation_left_eye_features"], next_sample["annotation_left_eye_features"], next2_sample["annotation_left_eye_features"] if next2_sample else None, t)
                new_right_features = interpolate_numeric_features(prev2_sample["annotation_right_eye_features"] if prev2_sample else None, prev_sample["annotation_right_eye_features"], next_sample["annotation_right_eye_features"], next2_sample["annotation_right_eye_features"] if next2_sample else None, t)
            
            elif not left_exists and right_exists:
                # LEFT EDGE GAP: Extrapolate backwards from the right
                step = right_idx - idx
                i_face_bbox = extrapolate_value(next_sample["face_bbox"], next2_sample["face_bbox"] if next2_sample else None, next_sample["face_bbox"], step)
                i_left_bbox = extrapolate_value(next_sample["left_bbox"], next2_sample["left_bbox"] if next2_sample else None, next_sample["left_bbox"], step)
                i_right_bbox = extrapolate_value(next_sample["right_bbox"], next2_sample["right_bbox"] if next2_sample else None, next_sample["right_bbox"], step)
                i_headpose = extrapolate_value(next_sample["tddfa-retinaface_headpose"], next2_sample["tddfa-retinaface_headpose"] if next2_sample else None, next_sample["tddfa-retinaface_headpose"], step)
                new_left_features = extrapolate_numeric_features(next_sample["annotation_left_eye_features"], next2_sample["annotation_left_eye_features"] if next2_sample else None, step)
                new_right_features = extrapolate_numeric_features(next_sample["annotation_right_eye_features"], next2_sample["annotation_right_eye_features"] if next2_sample else None, step)

            elif left_exists and not right_exists:
                # RIGHT EDGE GAP: Extrapolate forwards from the left
                step = idx - left_idx
                i_face_bbox = extrapolate_value(prev_sample["face_bbox"], prev2_sample["face_bbox"] if prev2_sample else None, prev_sample["face_bbox"], step)
                i_left_bbox = extrapolate_value(prev_sample["left_bbox"], prev2_sample["left_bbox"] if prev2_sample else None, prev_sample["left_bbox"], step)
                i_right_bbox = extrapolate_value(prev_sample["right_bbox"], prev2_sample["right_bbox"] if prev2_sample else None, prev_sample["right_bbox"], step)
                i_headpose = extrapolate_value(prev_sample["tddfa-retinaface_headpose"], prev2_sample["tddfa-retinaface_headpose"] if prev2_sample else None, prev_sample["tddfa-retinaface_headpose"], step)
                new_left_features = extrapolate_numeric_features(prev_sample["annotation_left_eye_features"], prev2_sample["annotation_left_eye_features"] if prev2_sample else None, step)
                new_right_features = extrapolate_numeric_features(prev_sample["annotation_right_eye_features"], prev2_sample["annotation_right_eye_features"] if prev2_sample else None, step)

            # Crop Real Images using the estimated bounding box
            frame_raw = video_frames_raw[idx]
            
            def safe_crop(frame, bbox):
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_w, x2), min(frame_h, y2)
                if x1 >= x2 or y1 >= y2: return np.zeros((64, 64, 3), dtype=np.uint8)
                return frame[y1:y2, x1:x2]

            left_crop_rgb = cv2.resize(cv2.cvtColor(safe_crop(frame_raw, i_left_bbox), cv2.COLOR_BGR2RGB), (64, 64), interpolation=cv2.INTER_CUBIC)
            right_crop_rgb = cv2.resize(cv2.cvtColor(safe_crop(frame_raw, i_right_bbox), cv2.COLOR_BGR2RGB), (64, 64), interpolation=cv2.INTER_CUBIC)

            new_left_features['eye'] = left_crop_rgb
            new_left_features['eye_original'] = left_crop_rgb 
            new_right_features['eye'] = right_crop_rgb
            new_right_features['eye_original'] = right_crop_rgb

            fixed_samples[idx] = {
                "id": idx,
                "face_bbox": i_face_bbox,
                "left_bbox": i_left_bbox,
                "right_bbox": i_right_bbox,
                "tddfa-retinaface_headpose": i_headpose,
                "annotation_left_eye_features": new_left_features,
                "annotation_right_eye_features": new_right_features,
            }

    with open(out_path, "wb") as f:
        pickle.dump(fixed_samples, f)

    previous_processed_frames = sum(1 for idx in sorted_indices if not is_dummy(samples[idx]))
    processed_frames = sum(1 for idx in sorted_indices if not is_dummy(fixed_samples[idx]))

    return processed_frames, total_frames, previous_processed_frames


def _expand_missing_runs_to_index_set(missing_runs):
    missing_indices = set()
    for run in missing_runs:
        if not isinstance(run, (list, tuple)) or len(run) != 2:
            continue
        start_idx, end_idx = int(run[0]), int(run[1])
        for idx in range(start_idx, end_idx + 1):
            missing_indices.add(idx)
    return missing_indices


def _safe_get_eye_preview(ann_eye_features):
    if not isinstance(ann_eye_features, dict):
        return np.zeros((64, 64, 3), dtype=np.uint8)

    eye_img = ann_eye_features.get("eye", None)
    if eye_img is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    eye_img = np.asarray(eye_img)
    if eye_img.size == 0:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    if eye_img.ndim == 2:
        eye_img = cv2.cvtColor(eye_img, cv2.COLOR_GRAY2BGR)
    elif eye_img.ndim == 3 and eye_img.shape[2] == 3:
        # stored as RGB in your pipeline, convert for OpenCV rendering
        eye_img = cv2.cvtColor(eye_img, cv2.COLOR_RGB2BGR)
    else:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    return cv2.resize(eye_img, (128, 128), interpolation=cv2.INTER_NEAREST)


def _put_multiline_text(img, lines, x, y, line_gap=22, color=(255, 255, 255), scale=0.55, thickness=1):
    cy = y
    for line in lines:
        cv2.putText(
            img,
            str(line),
            (x, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA
        )
        cy += line_gap


def _format_eye_feature_lines(ann_eye_features, prefix="L"):
    if not isinstance(ann_eye_features, dict):
        return [f"{prefix} eye: N/A"]

    iris_d = np.asarray(ann_eye_features.get("iris_diameters", [0.0, 0.0])).reshape(-1)
    eyelid_pupil = np.asarray(ann_eye_features.get("eyelid_pupil_distances", [0.0, 0.0])).reshape(-1)
    ear = ann_eye_features.get("ear", 0.0)

    iris_d_0 = float(iris_d[0]) if iris_d.size > 0 else 0.0
    iris_d_1 = float(iris_d[1]) if iris_d.size > 1 else 0.0
    ep_0 = float(eyelid_pupil[0]) if eyelid_pupil.size > 0 else 0.0
    ep_1 = float(eyelid_pupil[1]) if eyelid_pupil.size > 1 else 0.0
    ear = float(ear)

    return [
        f"{prefix} EAR: {ear:.3f}",
        f"{prefix} Iris D: {iris_d_0:.2f}, {iris_d_1:.2f}",
        f"{prefix} Lid-Pupil: {ep_0:.2f}, {ep_1:.2f}",
    ]

def _draw_points_from_64x64_to_frame(frame, points, bbox, color, radius=2, thickness=-1):
    """
    Draw landmark points that are defined on a 64x64 resized eye crop
    back onto the original full-frame image.

    points: array-like of shape (N, 2) or (N, >=2)
    bbox: (x1, y1, x2, y2) in full-frame coordinates
    """
    if points is None:
        return

    points = np.asarray(points)
    if points.size == 0:
        return
    if points.ndim != 2 or points.shape[1] < 2:
        return

    x1, y1, x2, y2 = map(float, bbox)
    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return

    sx = bw / 64.0
    sy = bh / 64.0

    for pt in points:
        px = int(round(x1 + pt[0] * sx))
        py = int(round(y1 + pt[1] * sy))

        if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
            cv2.circle(frame, (px, py), radius, color, thickness)


def visualize_landmarks(dataloader):
    exp_dir = dataloader.get_cwd()
    out_dir = os.path.join(exp_dir, "exordium_landmarks2")
    os.makedirs(out_dir, exist_ok=True)

    fixed_pkl_path = os.path.join(out_dir, "exordium_data_fixed.pkl")
    raw_pkl_path = os.path.join(out_dir, "exordium_data.pkl")
    log_path = os.path.join(out_dir, "missing_frames_log.json")

    if os.path.exists(fixed_pkl_path):
        pkl_path = fixed_pkl_path
    else:
        pkl_path = raw_pkl_path

    if not os.path.exists(pkl_path):
        extract_and_save_exordium_features(dataloader)

    print(f"Loading landmarks from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    missing_indices = set()
    if os.path.exists(log_path):
        print(f"Loading missing frame log from {log_path}...")
        with open(log_path, "r") as f:
            missing_log = json.load(f)
        missing_indices = _expand_missing_runs_to_index_set(missing_log.get("missing_runs", []))

    video_frames_raw = dataloader.load_rgb_video(as_numpy=True)
    output_mp4_path = os.path.join(out_dir, "visualization.mp4")

    h, w = video_frames_raw[0].shape[:2]

    panel_w = 320
    total_w = w + panel_w

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(output_mp4_path, fourcc, 30.0, (total_w, h))

    border_color = (0, 255, 0)
    border_thickness = 5

    for i, frame_raw in enumerate(tqdm(video_frames_raw, desc="Rendering Video")):
        if frame_raw is None:
            continue

        display_frame = frame_raw.copy()
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        item = data[i]
        face_bbox = item.get("face_bbox", [-1, -1, -1, -1])

        # Mark frames that were originally missing before fix
        if i in missing_indices:
            cv2.rectangle(
                display_frame,
                (0, 0),
                (w - 1, h - 1),
                border_color,
                border_thickness
            )
            cv2.putText(
                display_frame,
                "ORIGINALLY MISSING",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                border_color,
                2,
                cv2.LINE_AA
            )

        ann_left = item.get("annotation_left_eye_features", {})
        ann_right = item.get("annotation_right_eye_features", {})

        left_preview = _safe_get_eye_preview(ann_left)
        right_preview = _safe_get_eye_preview(ann_right)

        cv2.putText(panel, f"Frame: {i}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        if i in missing_indices:
            cv2.putText(panel, "Missing before fix: YES", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_color, 2, cv2.LINE_AA)
        else:
            cv2.putText(panel, "Missing before fix: NO", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

        # Eye previews
        panel[75:203, 15:143] = left_preview
        panel[75:203, 177:305] = right_preview

        cv2.putText(panel, "Left Eye", (35, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, "Right Eye", (190, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        left_lines = _format_eye_feature_lines(ann_left, "L")
        right_lines = _format_eye_feature_lines(ann_right, "R")

        _put_multiline_text(panel, left_lines, 15, 255, line_gap=22, color=(255, 255, 255), scale=0.52, thickness=1)
        _put_multiline_text(panel, right_lines, 15, 335, line_gap=22, color=(255, 255, 255), scale=0.52, thickness=1)

        if face_bbox[0] != -1:
            fx1, fy1, fx2, fy2 = map(int, face_bbox)
            lx1, ly1, lx2, ly2 = map(int, item["left_bbox"])
            rx1, ry1, rx2, ry2 = map(int, item["right_bbox"])

            # TDDFA outputs: Yaw, Pitch, Roll
            y, p, r = item["tddfa-retinaface_headpose"]

            # Draw bounding boxes
            cv2.rectangle(display_frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
            cv2.rectangle(display_frame, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)
            cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)

            text = f"Pitch: {p:.2f} | Yaw: {y:.2f} | Roll: {r:.2f}"
            cv2.putText(
                display_frame,
                text,
                (fx1, max(fy1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # LEFT EYE LANDMARKS
            left_landmarks = ann_left.get("landmarks", None)
            left_iris_landmarks = ann_left.get("iris_landmarks", None)

            _draw_points_from_64x64_to_frame(
                display_frame,
                left_landmarks,
                (lx1, ly1, lx2, ly2),
                color=(0, 255, 255),
                radius=2,
                thickness=-1,
            )

            _draw_points_from_64x64_to_frame(
                display_frame,
                left_iris_landmarks,
                (lx1, ly1, lx2, ly2),
                color=(255, 0, 255),
                radius=3,
                thickness=-1,
            )

            # RIGHT EYE LANDMARKS
            right_landmarks = ann_right.get("landmarks", None)
            right_iris_landmarks = ann_right.get("iris_landmarks", None)

            _draw_points_from_64x64_to_frame(
                display_frame,
                right_landmarks,
                (rx1, ry1, rx2, ry2),
                color=(255, 255, 0),
                radius=2,
                thickness=-1,
            )

            _draw_points_from_64x64_to_frame(
                display_frame,
                right_iris_landmarks,
                (rx1, ry1, rx2, ry2),
                color=(0, 165, 255),
                radius=3,
                thickness=-1,
            )

        composed = np.concatenate([display_frame, panel], axis=1)
        out_video.write(composed)

    out_video.release()
    print(f"Visualization saved to: {os.path.abspath(output_mp4_path)}")

if __name__ == "__main__":
    import config
    from data_loader import GazeDataLoader

    base_dir = config.get_dataset_base_directory()
    exp_dir = f"{base_dir}/2025-07-30/subj_xxxx/lighting_100/p7"
    dataloader = GazeDataLoader(
        root_dir=exp_dir,
        target_period=config.get_target_period(),
        camera_pose_period=config.get_camera_pose_period(),
        time_diff_max=config.get_time_diff_max(),
        get_latest_subdirectory_by_name=True
    )

    visualize_landmarks(dataloader=dataloader)

