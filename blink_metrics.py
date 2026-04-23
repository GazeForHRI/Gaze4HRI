# blink_metrics.py  (LOCAL stats + CONSENSUS)

import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# ---- Per-model blink timing offsets (milliseconds) ----
# Positive = move predicted blink later; Negative = move earlier, in milliseconds
MODEL_BLINK_OFFSET_MS: dict[str, float] = {
    "mcgaze_clip_size_3": -100.0, # 3 times 33.3ms (each frame comes with 33.3ms intervals since camera fps is 30Hz)
    "mcgaze_clip_size_7": -233.1, # 7 times 33.3ms 
}

def get_model_blink_offsets() -> dict[str, float]:
    """Public accessor so other files can import if needed."""
    return MODEL_BLINK_OFFSET_MS.copy()

# ----------------- small utils -----------------

def _median_filter_1d(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if k is None or k <= 1:
        return x.copy()
    if k % 2 == 0:
        k += 1
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        pad = k // 2
        xp = np.pad(x, (pad, pad), mode="edge")
        w = sliding_window_view(xp, k)
        return np.median(w, axis=-1)
    except Exception:
        pad = k // 2
        xp = np.pad(x, (pad, pad), mode="edge")
        out = np.empty_like(x)
        for i in range(x.shape[0]):
            out[i] = np.median(xp[i:i+k])
        return out

def _rolling_median_and_mad(x: np.ndarray, win: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rolling median and MAD with edge padding. If win<=1, returns global median/MAD.
    """
    x = np.asarray(x, dtype=float)
    if win <= 1:
        med = np.full_like(x, np.median(x))
        mad = np.full_like(x, np.median(np.abs(x - med[0])))
        return med, mad

    if win % 2 == 0:
        win += 1
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        pad = win // 2
        xp = np.pad(x, (pad, pad), mode="edge")
        W = sliding_window_view(xp, win)  # (N, win)
        med = np.median(W, axis=1)
        mad = np.median(np.abs(W - med[:, None]), axis=1)
        return med, mad
    except Exception:
        # Fallback (slower)
        pad = win // 2
        xp = np.pad(x, (pad, pad), mode="edge")
        med = np.empty_like(x)
        mad = np.empty_like(x)
        for i in range(x.shape[0]):
            w = xp[i:i+win]
            m = np.median(w)
            med[i] = m
            mad[i] = np.median(np.abs(w - m))
        return med, mad

def _robust_z_local(x: np.ndarray, win: int, eps: float = 1e-9) -> np.ndarray:
    med, mad = _rolling_median_and_mad(x, win)
    return (x - med) / (1.4826 * (mad + eps))

def _hysteresis(score: np.ndarray, on: float, off: float) -> np.ndarray:
    on_flag = False
    mask = np.zeros(score.shape[0], dtype=bool)
    for i, s in enumerate(score):
        if not on_flag and s >= on:
            on_flag = True
        elif on_flag and s <= off:
            on_flag = False
        mask[i] = on_flag
    return mask

def _segments_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    segs, n, i = [], mask.shape[0], 0
    while i < n:
        if mask[i]:
            s = i
            while i + 1 < n and mask[i + 1]:
                i += 1
            e = i
            segs.append((s, e))
        i += 1
    return segs

def _reconstruct_mask_from_segments(n: int, segments: List[Tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for s, e in segments:
        mask[s:e+1] = True
    return mask

def _prune_and_merge_segments(segments, t_ms, min_dur_ms, merge_gap_ms):
    if not segments:
        return []
    kept = [(s, e) for (s, e) in segments if (t_ms[e] - t_ms[s]) >= min_dur_ms]
    if not kept:
        return []
    merged = [kept[0]]
    for s, e in kept[1:]:
        ps, pe = merged[-1]
        if (t_ms[s] - t_ms[pe]) < merge_gap_ms:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged

def _estimate_frame_period_ms(t_ms: np.ndarray) -> float:
    """Median dt (ms). Falls back to 33.33 if too short."""
    if t_ms.shape[0] < 3:
        return 33.333
    d = np.diff(t_ms)
    return float(np.median(d)) if np.isfinite(np.median(d)) else 33.333

def _segments_from_mask_safe(mask: np.ndarray) -> list[tuple[int,int]]:
    if mask.size == 0:
        return []
    return _segments_from_mask(mask)

def _shift_segments_by_ms(segments: list[tuple[int,int]], t_ms: np.ndarray, offset_ms: float) -> list[tuple[int,int]]:
    """Shift segments in index-space by an offset in milliseconds."""
    if not segments or abs(offset_ms) < 1e-6:
        return segments
    dt = _estimate_frame_period_ms(t_ms)  # ~frame period in ms
    shift = int(np.round(offset_ms / max(1e-6, dt)))

    if shift == 0:
        return segments

    N = int(t_ms.shape[0])
    out = []
    for s, e in segments:
        ns = s + shift
        ne = e + shift
        if ne < 0 or ns >= N:  # fully out of range
            continue
        ns = max(0, ns)
        ne = min(N - 1, ne)
        if ns <= ne:
            out.append((ns, ne))

    # No pruning/merging gaps here; keep original connectivity.
    # If you prefer, you can merge adjacents:
    if not out:
        return []
    out.sort()
    merged = [out[0]]
    for s, e in out[1:]:
        ps, pe = merged[-1]
        if s <= pe + 1:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged

def _events_from_segments(
    segs: list[tuple[int,int]],
    t_ms: np.ndarray,
    z: Optional[np.ndarray] = None,
    value: Optional[np.ndarray] = None,
    *,
    value_key: str = "peak_ang_deg"  # or "peak_dev_deg"
) -> list[dict]:
    events = []
    for s, e in segs:
        start_ms = float(t_ms[s]); end_ms = float(t_ms[e])
        payload = {
            "start_idx": int(s),
            "end_idx": int(e),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms,
        }
        if z is not None and z.size > 0:
            peak_idx = int(s + np.argmax(z[s:e+1]))
            payload["peak_z"] = float(z[peak_idx])
            if value is not None and value.size > 0:
                payload[value_key] = float(value[peak_idx])
        events.append(payload)
    return events


# ----------------- step 1: use your process_direction_errors -----------------

def compute_model_ang_error(dataloader, model: str) -> Dict[str, np.ndarray]:
    """
    Returns:
      {'t_ms': (N,), 'ang_deg': (N,), 'euc': (N,)}
    Uses data_analyer.process_direction_errors to get timestamped angular errors
    (aligned 30Hz est to 100Hz GT).
    """
    from data_analyer import process_direction_errors

    errs = process_direction_errors(dataloader, model=model, gaze_target_bias=None, save_to_file=False)
    if errs is None or errs.size == 0:
        return {"t_ms": np.array([]), "ang_deg": np.array([]), "euc": np.array([])}

    errs = np.asarray(errs, dtype=float)
    errs = errs[np.argsort(errs[:, 0])]
    t_ms = errs[:, 0] - errs[0, 0]
    ang_deg = errs[:, 1]
    euc = errs[:, 2]
    return {"t_ms": t_ms, "ang_deg": ang_deg, "euc": euc}

# ----------------- step 2: local-stat blink mask (angular error only) -----------------

def build_blink_mask_from_ang_error_local(
    t_ms: np.ndarray,
    ang_deg: np.ndarray,
    *,
    window_ms: float = 1000.0,   # LOCAL stats window (ms). Try 600–1500 depending on motion.
    smooth_window: int = 3,      # optional small median on ang before z
    z_on: float = 3.0,
    z_off: float = 2.0,
    min_dur_ms: float = 120.0,
    merge_gap_ms: float = 120.0,
    min_peak_deg: Optional[float] = None   # e.g., 12–15 to suppress tiny bumps
) -> Dict[str, Any]:
    """
    Local robust-z blink detection from angular error only.
    - smooth with a tiny median filter (3–5)
    - local rolling median & MAD over 'window_ms'
    - hysteresis + prune + merge
    """
    N = ang_deg.shape[0]
    if N == 0:
        return {"mask": np.zeros(0, bool), "z": np.zeros(0, float), "events": []}

    # small pre-smoothing to reduce single-frame spikes
    ang_s = _median_filter_1d(ang_deg, smooth_window)

    # rolling window length in frames
    dt = _estimate_frame_period_ms(t_ms)
    win = max(3, int(round(window_ms / max(1e-6, dt))))
    if win % 2 == 0:
        win += 1

    z = _robust_z_local(ang_s, win=win)
    raw = _hysteresis(z, on=z_on, off=z_off)
    segs = _segments_from_mask(raw)
    segs = _prune_and_merge_segments(segs, t_ms, min_dur_ms=min_dur_ms, merge_gap_ms=merge_gap_ms)

    # optional min-peak filter
    if min_peak_deg is not None and len(segs) > 0:
        filtered = []
        for s, e in segs:
            peak = float(np.max(ang_deg[s:e+1]))
            if peak >= min_peak_deg:
                filtered.append((s, e))
        segs = filtered

    mask = _reconstruct_mask_from_segments(N, segs)

    events = []
    for s, e in segs:
        start_ms = float(t_ms[s]); end_ms = float(t_ms[e])
        peak_idx = int(s + np.argmax(z[s:e+1]))
        events.append({
            "start_idx": int(s),
            "end_idx": int(e),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms,
            "peak_z": float(z[peak_idx]),
            "peak_ang_deg": float(ang_deg[peak_idx]),
        })

    return {"mask": mask, "z": z, "events": events}

# ----------------- consensus utilities -----------------

def build_consensus_from_models(
    masks: Dict[str, np.ndarray],
    t_ms: np.ndarray,
    *,
    k_of_m: Optional[int] = None,     # None => majority
    min_dur_ms: float = 120.0,
    merge_gap_ms: float = 120.0
) -> Dict[str, Any]:
    """
    Create union, intersection, and K-of-M (consensus) masks & events.
    Returns:
      {
        'union_mask': bool[N],
        'intersection_mask': bool[N],
        'consensus_mask': bool[N],
        'events': [ ... ],  # events of consensus_mask
      }
    """
    if not masks:
        return {
            "union_mask": np.zeros(0, bool),
            "intersection_mask": np.zeros(0, bool),
            "consensus_mask": np.zeros(0, bool),
            "events": []
        }
    model_list = list(masks.keys())
    stack = np.stack([masks[m] for m in model_list], axis=0)  # (M, N)
    M, N = stack.shape

    union_mask = np.any(stack, axis=0)
    intersection_mask = np.all(stack, axis=0)

    if k_of_m is None:
        k_of_m = (M // 2) + 1  # majority by default
    counts = np.sum(stack, axis=0)
    consensus_mask = counts >= k_of_m

    # Build events on consensus_mask
    segs = _segments_from_mask(consensus_mask)
    segs = _prune_and_merge_segments(segs, t_ms, min_dur_ms=min_dur_ms, merge_gap_ms=merge_gap_ms)
    consensus_mask = _reconstruct_mask_from_segments(N, segs)

    events = []
    for s, e in segs:
        start_ms = float(t_ms[s]); end_ms = float(t_ms[e])
        events.append({
            "start_idx": int(s),
            "end_idx": int(e),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms,
            "support": int(np.max(counts[s:e+1]))  # max #models agreeing within event
        })

    return {
        "union_mask": union_mask,
        "intersection_mask": intersection_mask,
        "consensus_mask": consensus_mask,
        "events": events
    }

# ----------------- wrapper: steps 1 & 2 with consensus -----------------

def analyze_sequence_errors_only(
    dataloader,
    models: List[str],
    *,
    # local stats + detector params:
    window_ms: float = 1000.0,
    smooth_window: int = 3,
    z_on: float = 3.0,
    z_off: float = 2.0,
    min_dur_ms: float = 120.0,
    merge_gap_ms: float = 120.0,
    min_peak_deg: Optional[float] = None,
    # consensus:
    consensus_k: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Runs Steps 1 & 2 (angular error only, with LOCAL stats) for each model,
    and builds a K-of-M consensus.

    Returns:
      results[model] = {
        't_ms','ang_deg','z','mask','events'
      }
      results['_consensus'] = {
        'union_mask','intersection_mask','consensus_mask','events'
      }
    """
    out: Dict[str, Dict[str, Any]] = {}
    t_axes = []

    # Per-model signals & masks
    for m in models:
        sig = compute_model_ang_error(dataloader, m)  # {'t_ms','ang_deg','euc'}
        det = build_blink_mask_from_ang_error_local(
            sig["t_ms"], sig["ang_deg"],
            window_ms=window_ms,
            smooth_window=smooth_window,
            z_on=z_on, z_off=z_off,
            min_dur_ms=min_dur_ms, merge_gap_ms=merge_gap_ms,
            min_peak_deg=min_peak_deg
        )

        # --- NEW: apply per-model blink timing offset (in ms) ---
        offset_ms = MODEL_BLINK_OFFSET_MS.get(m, 0.0)
        segs = _segments_from_mask_safe(det["mask"])
        segs = _shift_segments_by_ms(segs, sig["t_ms"], offset_ms)
        mask_shifted = _reconstruct_mask_from_segments(sig["t_ms"].shape[0], segs)
        events_shifted = _events_from_segments(
            segs, sig["t_ms"], z=det.get("z", None), value=sig["ang_deg"], value_key="peak_ang_deg"
        )

        out[m] = {
            "t_ms": sig["t_ms"],
            "ang_deg": sig["ang_deg"],
            "z": det["z"],
            "mask": mask_shifted,
            "events": events_shifted
        }
        t_axes.append(sig["t_ms"])

    # Choose a common t_ms for consensus (use the first model’s)
    if len(models) > 0 and out[models[0]]["t_ms"].size > 0:
        t_ms_ref = out[models[0]]["t_ms"]
        # Trim all masks to common N (min length)
        N = min([out[m]["mask"].shape[0] for m in models])
        masks = {m: out[m]["mask"][:N] for m in models}
        t_ms_ref = t_ms_ref[:N]
        cons = build_consensus_from_models(
            masks, t_ms_ref, k_of_m=consensus_k,
            min_dur_ms=min_dur_ms, merge_gap_ms=merge_gap_ms
        )
        out["_consensus"] = cons
        # Also include the time axis for convenience
        out["_consensus"]["t_ms"] = t_ms_ref
    else:
        out["_consensus"] = {
            "union_mask": np.array([], dtype=bool),
            "intersection_mask": np.array([], dtype=bool),
            "consensus_mask": np.array([], dtype=bool),
            "events": [],
            "t_ms": np.array([], dtype=float)
        }

    return out


# === Deviation-based blink detection (angle to optical axis (1,0,0) in camera frame) ===

import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Reuse the helper functions already defined in blink_metrics.py:
# _median_filter_1d, _rolling_median_and_mad, _robust_z_local,
# _hysteresis, _segments_from_mask, _reconstruct_mask_from_segments,
# _prune_and_merge_segments, _estimate_frame_period_ms,
# and build_consensus_from_models

def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=float); v = np.asarray(v, dtype=float)
    nu = np.linalg.norm(u) + 1e-12; nv = np.linalg.norm(v) + 1e-12
    dot = np.clip(np.dot(u/nu, v/nv), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))

def compute_model_dev_to_axis(
    dataloader,
    model: str,
    axis: Tuple[float, float, float] = (1.0, 0.0, 0.0),   # +X is out-of-camera in your convention
    frame: str = "camera",
) -> Dict[str, np.ndarray]:
    """
    Returns:
      {'t_ms': (N,), 'dev_deg': (N,)}
    where dev_deg[i] = angle( est_cam[i], axis ).
    Uses camera-frame estimations directly; no GT involved.
    """
    est = dataloader.load_gaze_estimations(model=model, frame=frame)  # Nx4 [t_ms, gx, gy, gz]
    if est.shape[0] == 0:
        return {"t_ms": np.array([]), "dev_deg": np.array([])}

    # Time axis (zeroed)
    t_ms = est[:, 0] - est[0, 0]

    # Angle to the camera optical axis (+X)
    axis_vec = np.asarray(axis, dtype=float)
    dev = np.empty(est.shape[0], dtype=np.float32)
    for i in range(est.shape[0]):
        dev[i] = _angle_deg(est[i, 1:], axis_vec)

    return {"t_ms": t_ms.astype(np.float64), "dev_deg": dev}

def build_blink_mask_from_deviation_local(
    t_ms: np.ndarray,
    dev_deg: np.ndarray,
    *,
    window_ms: float = 1000.0,
    smooth_window: int = 3,
    z_on: float = 3.0,
    z_off: float = 2.0,
    min_dur_ms: float = 120.0,
    merge_gap_ms: float = 120.0,
    # Peak guards:
    min_peak_rel_deg: float | None = None,   # NEW: min |dev - local_median| within event (degrees)
    min_peak_abs_deg: float | None = None    # (optional) min absolute dev within event (degrees)
) -> Dict[str, Any]:
    """
    Local robust-z blink detection using deviation to +X.
    Peak guards:
      - min_peak_rel_deg: require a minimum absolute deviation from local median (deg)
      - min_peak_abs_deg: (optional) require a minimum absolute deviation value (deg)
    """
    N = dev_deg.shape[0]
    if N == 0:
        return {"mask": np.zeros(0, bool), "z": np.zeros(0, float), "events": []}

    # Pre-smooth small spikes
    dev_s = _median_filter_1d(dev_deg, smooth_window)

    # Rolling local stats
    dt = _estimate_frame_period_ms(t_ms)
    win = max(3, int(round(window_ms / max(1e-6, dt))))
    if win % 2 == 0:
        win += 1

    # Local median & MAD
    med, mad = _rolling_median_and_mad(dev_s, win)
    abs_change = np.abs(dev_s - med)         # <-- relative amplitude (deg)
    z = abs_change / (1.4826 * (mad + 1e-9)) # robust |z| symmetric about the median

    # Hysteresis + segments
    raw = _hysteresis(z, on=z_on, off=z_off)
    segs = _segments_from_mask(raw)
    segs = _prune_and_merge_segments(segs, t_ms, min_dur_ms=min_dur_ms, merge_gap_ms=merge_gap_ms)

    # --- Peak guards ---
    if (min_peak_rel_deg is not None) or (min_peak_abs_deg is not None):
        kept = []
        for s, e in segs:
            ok = True
            if min_peak_rel_deg is not None:
                peak_rel = float(np.max(abs_change[s:e+1]))
                ok &= (peak_rel >= min_peak_rel_deg)
            if min_peak_abs_deg is not None:
                peak_abs = float(np.max(dev_deg[s:e+1]))
                ok &= (peak_abs >= min_peak_abs_deg)
            if ok:
                kept.append((s, e))
        segs = kept

    mask = _reconstruct_mask_from_segments(N, segs)

    # Events with both absolute & relative peaks
    events = []
    for s, e in segs:
        start_ms = float(t_ms[s]); end_ms = float(t_ms[e])
        z_peak_idx = int(s + np.argmax(z[s:e+1]))
        events.append({
            "start_idx": int(s),
            "end_idx": int(e),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms,
            "peak_z": float(z[z_peak_idx]),
            "peak_dev_deg": float(dev_deg[z_peak_idx]),              # absolute deviation value
            "peak_abs_change_deg": float(abs_change[z_peak_idx]),    # deviation from local median
        })

    return {"mask": mask, "z": z, "events": events}

def analyze_sequence_deviation_only(
    dataloader,
    models: List[str],
    *,
    axis: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    window_ms: float = 1000.0,
    smooth_window: int = 3,
    z_on: float = 3.0,
    z_off: float = 2.0,
    min_dur_ms: float = 120.0,
    merge_gap_ms: float = 120.0,
    min_peak_rel_deg: float | None = None,   # NEW
    min_peak_abs_deg: float | None = None,   # optional
    consensus_k: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Deviation-based Steps 1&2 for multiple models + K-of-M consensus.
    Returns:
      results[model] = {'t_ms','dev_deg','z','mask','events'}
      results['_consensus'] = {'union_mask','intersection_mask','consensus_mask','events','t_ms'}
    """
    out: Dict[str, Dict[str, Any]] = {}

    for m in models:
        sig = compute_model_dev_to_axis(dataloader, m, axis=axis, frame="camera")
        det = build_blink_mask_from_deviation_local(
            sig["t_ms"], sig["dev_deg"],
            window_ms=window_ms,
            smooth_window=smooth_window,
            z_on=z_on, z_off=z_off,
            min_dur_ms=min_dur_ms, merge_gap_ms=merge_gap_ms,
            min_peak_rel_deg=min_peak_rel_deg,
            min_peak_abs_deg=min_peak_abs_deg,
        )

        # --- NEW: apply per-model blink timing offset (in ms) ---
        offset_ms = MODEL_BLINK_OFFSET_MS.get(m, 0.0)
        segs = _segments_from_mask_safe(det["mask"])
        segs = _shift_segments_by_ms(segs, sig["t_ms"], offset_ms)
        mask_shifted = _reconstruct_mask_from_segments(sig["t_ms"].shape[0], segs)
        events_shifted = _events_from_segments(
            segs, sig["t_ms"], z=det.get("z", None), value=sig["dev_deg"], value_key="peak_dev_deg"
        )

        out[m] = {
            "t_ms": sig["t_ms"],
            "dev_deg": sig["dev_deg"],
            "z": det["z"],
            "mask": mask_shifted,
            "events": events_shifted,
        }

    # Build consensus on common length/time axis (use first model’s as ref)
    if len(models) > 0 and out[models[0]]["t_ms"].size > 0:
        N = min([out[m]["mask"].shape[0] for m in models])
        masks = {m: out[m]["mask"][:N] for m in models}
        t_ms_ref = out[models[0]]["t_ms"][:N]

        cons = build_consensus_from_models(
            masks, t_ms_ref, k_of_m=consensus_k,
            min_dur_ms=min_dur_ms, merge_gap_ms=merge_gap_ms
        )
        cons["t_ms"] = t_ms_ref
        out["_consensus"] = cons
    else:
        out["_consensus"] = {
            "union_mask": np.array([], dtype=bool),
            "intersection_mask": np.array([], dtype=bool),
            "consensus_mask": np.array([], dtype=bool),
            "events": [],
            "t_ms": np.array([], dtype=float)
        }

    return out
