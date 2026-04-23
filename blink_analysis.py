# blink_analysis.py

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import config
from data_loader import GazeDataLoader
import json

# -----------------------------------------
# Main: interactive blink inspection player
# -----------------------------------------

def play_blink_inspector(dataloader,
                         models: list,
                         fps: float = 30.0,
                         window_title: str = "Blink Inspector",
                         show_current_value: bool = True):
    """
    Synchronized viewer:
      - Left: head crop video (RGB, ~30 Hz)
      - Right: progressive angular-error timeline up to current time,
               one curve per model.

    IMPORTANT: Each model's 30 Hz estimations are timestamp-matched to the 100 Hz GT
               using match_irregular_to_regular before computing errors.

    Keys:
      SPACE: pause/resume
      ←/→  : step 1 frame (paused)
      ↑/↓  : jump 30 frames (paused)
      q    : quit
    """
    assert isinstance(models, (list, tuple)) and len(models) > 0, "models must be a non-empty list of strings"

    from data_matcher import match_irregular_to_regular

    # --- Load shared data ---
    frames = dataloader.load_rgb_video(as_numpy=True)                     # list[np.ndarray], BGR
    gt = dataloader.load_gaze_ground_truths(frame="camera")               # Mx4 [t_ms, gx, gy, gz] at ~100 Hz
    try:
        rgb_ts = dataloader.load_rgb_timestamps().reshape(-1)             # Nx at ~30 Hz
    except Exception:
        rgb_ts = None

    # Load estimations per model (each ~30 Hz)
    est_by_model = {m: dataloader.load_gaze_estimations(model=m, frame="camera") for m in models}

    # Determine playback length N at RGB cadence
    N = len(frames)
    if rgb_ts is not None:
        N = min(N, rgb_ts.shape[0])
    for m in models:
        N = min(N, est_by_model[m].shape[0])
    if N <= 0:
        print("No data to play.")
        return

    frames = frames[:N]
    if rgb_ts is not None:
        t_ms = rgb_ts[:N] - rgb_ts[0]
    else:
        # Fallback: use the first model's timestamps (still 30 Hz)
        first_model = models[0]
        t_ms = est_by_model[first_model][:N, 0] - est_by_model[first_model][0, 0]

    # --- For each model: match its 30 Hz estimations to 100 Hz GT and compute per-frame errors ---
    def _direction_error(est_vec, gt_vec):
        est = np.array(est_vec, dtype=float)
        lab = np.array(gt_vec, dtype=float)
        est /= (np.linalg.norm(est) + 1e-12)
        lab /= (np.linalg.norm(lab) + 1e-12)
        dot = np.clip(np.dot(est, lab), -1.0, 1.0)
        ang = np.degrees(np.arccos(dot))
        return ang

    ang_err_by_model = {}
    all_errs_concat = []
    for m in models:
        est = est_by_model[m][:N]  # Nx4 [t_ms, gx, gy, gz] at 30 Hz

        est_aligned, matched_gt_vecs = match_irregular_to_regular(
            irregular_data=est,             # timestamps in col 0
            regular_data=gt,                # high-rate GT
            regular_period_ms=dataloader.target_period
        )

        # Guard: nothing to evaluate if alignment dropped everything
        if est_aligned.shape[0] == 0:
            ang_err_by_model[m] = np.empty((0,), dtype=np.float32)
            continue

        # Compute angular error per aligned estimation
        errs = np.empty(est_aligned.shape[0], dtype=np.float32)
        for i in range(est_aligned.shape[0]):
            errs[i] = _direction_error(est_aligned[i, 1:], matched_gt_vecs[i])

        ang_err_by_model[m] = errs
        all_errs_concat.append(errs)

    all_errs = np.concatenate(all_errs_concat) if all_errs_concat else np.array([0.0])

    # --- Head crops (RGB-sized boxes if missing) ---
    head_bboxes = dataloader.load_head_bboxes()
    h0, w0 = frames[0].shape[:2]
    if not (isinstance(head_bboxes, np.ndarray) and head_bboxes.ndim == 2 and head_bboxes.shape[0] >= N):
        head_bboxes = np.tile([0, 0, w0, h0, -1, -1], (N, 1))

    # --- Matplotlib setup ---
    plt.ion()
    fig = plt.figure(figsize=(13, 6))
    fig.canvas.manager.set_window_title(window_title)

    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.4], wspace=0.25)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_plot = fig.add_subplot(gs[0, 1])

    # Image panel
    ax_img.axis('off')
    x1, y1, x2, y2, *_ = map(int, head_bboxes[0])
    crop0 = frames[0][max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop0.size == 0:
        crop0 = frames[0]
    im_artist = ax_img.imshow(cv.cvtColor(crop0, cv.COLOR_BGR2RGB))
    ax_img.set_title("Head Crop")

    # Plot panel
    ax_plot.set_title("Angular Error Over Time")
    ax_plot.set_xlabel("Time (ms)")
    ax_plot.set_ylabel("Angular Error (degrees)")
    ax_plot.set_xlim(0, float(t_ms[-1]))
    ypad = max(5.0, 0.1 * (float(np.nanmax(all_errs)) - float(np.nanmin(all_errs)) + 1e-6))
    ax_plot.set_ylim(max(0.0, float(np.nanmin(all_errs) - ypad)),
                     float(np.nanmax(all_errs) + ypad))
    ax_plot.grid(True)

    # Progressive lines & current markers per model
    prog_lines = {}
    cur_dots = {}
    for m in models:
        (pl,) = ax_plot.plot([], [], linewidth=2.0, label=m)
        (cd,) = ax_plot.plot([], [], marker='o')
        prog_lines[m] = pl
        cur_dots[m] = cd

    ax_plot.legend(loc="upper right", framealpha=0.85)

    # Current value readout
    txt = None
    if show_current_value:
        txt = ax_plot.text(
            0.02, 0.95, "", transform=ax_plot.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.8)
        )

    # --- Playback state ---
    idx = 0
    paused = False
    step_once = 0

    def on_key(event):
        nonlocal paused, step_once, idx
        if event.key == ' ':
            paused = not paused
        elif event.key in ('q', 'escape'):
            plt.close(fig)
        elif paused:
            if event.key == 'right':
                step_once = +1
            elif event.key == 'left':
                step_once = -1
            elif event.key == 'up':
                step_once = +30
            elif event.key == 'down':
                step_once = -30

    fig.canvas.mpl_connect('key_press_event', on_key)

    # --- Main loop ---
    frame_period = 1.0 / max(1e-6, fps)
    last_time = time.time()

    while plt.fignum_exists(fig.number):
        # Left: crop
        x1, y1, x2, y2, *_ = map(int, head_bboxes[idx])
        crop = frames[idx][max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            crop = frames[idx]
        im_artist.set_data(cv.cvtColor(crop, cv.COLOR_BGR2RGB))

        # Right: progressive lines up to current RGB time index
        for m in models:
            errs = ang_err_by_model[m]
            prog_lines[m].set_data(t_ms[:idx+1], errs[:idx+1])
            cur_dots[m].set_data([t_ms[idx]], [errs[idx]])

        ax_plot.set_xlim(0, float(t_ms[idx]) if idx > 0 else float(t_ms[0]))

        if txt is not None:
            lines = [f"t = {t_ms[idx]:.0f} ms"]
            for m in models:
                lines.append(f"{m}: {ang_err_by_model[m][idx]:.2f}°")
            txt.set_text("\n".join(lines))

        fig.canvas.draw_idle()
        plt.pause(0.001)

        # Advance
        if paused:
            if step_once != 0:
                idx = int(np.clip(idx + step_once, 0, N - 1))
                step_once = 0
            else:
                time.sleep(0.01)
        else:
            now = time.time()
            to_wait = frame_period - (now - last_time)
            if to_wait > 0:
                time.sleep(to_wait)
            last_time = time.time()
            idx += 1
            if idx >= N:
                break

    plt.ioff()
    plt.close(fig)


def play_blink_inspector_with_blinks(
    dataloader,
    models: list,
    *,
    fps: float = 30.0,
    window_title: str = "Blink Inspector (with detection overlay)",
    show_current_value: bool = True,
    # blink detection params (match blink_metrics.py defaults)
    smooth_window: int = 5,
    z_on: float = 3.0,
    z_off: float = 2.0,
    min_dur_ms: float = 80.0,
    merge_gap_ms: float = 100.0,
    show_consensus: bool = True,
    consensus_k: int | None = None,  # None -> majority
):
    """
    Visual player (like play_blink_inspector) + live blink overlay.

    - Per model:
        * angular error timeline retrieved via process_direction_errors(...)
        * blink mask from build_blink_mask_from_ang_error(...)
    - Plot:
        * progressive error lines (one per model)
        * semi-transparent shaded regions where a model is in blink state
        * optional consensus (K-of-M) band at bottom

    Controls:
      SPACE: pause/resume
      ←/→  : step 1 frame (paused)
      ↑/↓  : jump 30 frames (paused)
      q    : quit
    """
    assert isinstance(models, (list, tuple)) and len(models) > 0, "models must be a non-empty list"

    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2 as cv

    # --- Step 1 & 2: compute angular-error series and blink masks ---
    # We import the small wrappers we made in blink_metrics.py
    from blink_metrics import compute_model_ang_error, build_blink_mask_from_ang_error

    # Visual data
    frames = dataloader.load_rgb_video(as_numpy=True)  # list[np.ndarray], BGR
    if len(frames) == 0:
        print("No RGB frames.")
        return

    # Per-model signals and masks
    per_model = {}
    lengths = [len(frames)]
    for m in models:
        sig = compute_model_ang_error(dataloader, m)  # {'t_ms','ang_deg','euc'}
        blink = build_blink_mask_from_ang_error(
            sig["t_ms"],
            sig["ang_deg"],
            smooth_window=smooth_window,
            z_on=z_on,
            z_off=z_off,
            min_dur_ms=min_dur_ms,
            merge_gap_ms=merge_gap_ms,
        )
        per_model[m] = {
            "t_ms": sig["t_ms"],
            "ang_deg": sig["ang_deg"],
            "mask": blink["mask"],
            "z": blink["z"],
            "events": blink["events"],
        }
        lengths.append(sig["t_ms"].shape[0])

    # Use a common length across models & frames so that indexing is consistent
    N = min(lengths)
    if N <= 0:
        print("Empty series after alignment.")
        return

    # Build a common time axis (take the first model’s time; they are already zeroed)
    ref_model = models[0]
    t_ms = per_model[ref_model]["t_ms"][:N]

    # Trim everything to N
    frames = frames[:N]
    for m in models:
        per_model[m]["t_ms"] = per_model[m]["t_ms"][:N]
        per_model[m]["ang_deg"] = per_model[m]["ang_deg"][:N]
        per_model[m]["mask"] = per_model[m]["mask"][:N]
        per_model[m]["z"] = per_model[m]["z"][:N]

    # Optional consensus
    consensus_mask = None
    if show_consensus:
        stack = np.stack([per_model[m]["mask"] for m in models], axis=0)  # (M, N)
        if consensus_k is None:
            consensus_k = (len(models) // 2) + 1  # majority by default
        consensus_mask = (np.sum(stack, axis=0) >= consensus_k)

    # Head crops (fallback to full frame)
    head_bboxes = dataloader.load_head_bboxes()
    h0, w0 = frames[0].shape[:2]
    if not (isinstance(head_bboxes, np.ndarray) and head_bboxes.ndim == 2 and head_bboxes.shape[0] >= N):
        head_bboxes = np.tile([0, 0, w0, h0, -1, -1], (N, 1))

    # --- Matplotlib setup ---
    plt.ion()
    fig = plt.figure(figsize=(13, 6))
    try:
        fig.canvas.manager.set_window_title(window_title)
    except Exception:
        pass

    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5], wspace=0.25)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_plot = fig.add_subplot(gs[0, 1])

    # Image panel
    ax_img.axis('off')
    x1, y1, x2, y2, *_ = map(int, head_bboxes[0])
    crop0 = frames[0][max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop0.size == 0:
        crop0 = frames[0]
    im_artist = ax_img.imshow(cv.cvtColor(crop0, cv.COLOR_BGR2RGB))
    ax_img.set_title("Head Crop")

    # Plot panel
    # Y-limits from all errors (for stable baseline)
    all_errs = np.concatenate([per_model[m]["ang_deg"] for m in models], axis=0)
    ypad = max(5.0, 0.1 * (float(np.nanmax(all_errs)) - float(np.nanmin(all_errs)) + 1e-6))

    ax_plot.set_title("Angular Error Over Time (+ blink overlay)")
    ax_plot.set_xlabel("Time (ms)")
    ax_plot.set_ylabel("Angular Error (degrees)")
    ax_plot.set_xlim(0, float(t_ms[-1]))
    ax_plot.set_ylim(max(0.0, float(np.nanmin(all_errs) - ypad)),
                     float(np.nanmax(all_errs) + ypad))
    ax_plot.grid(True)

    # Line & point artists per model
    prog_lines = {}
    cur_dots = {}
    for m in models:
        (pl,) = ax_plot.plot([], [], linewidth=2.0, label=m)
        (cd,) = ax_plot.plot([], [], marker='o')
        prog_lines[m] = pl
        cur_dots[m] = cd

    # Blink overlay containers (we’ll redraw them per frame for simplicity)
    blink_fills = []          # list of PolyCollections (to remove/update)
    consensus_fill = None     # PolyCollection for consensus (if enabled)

    ax_plot.legend(loc="upper right", framealpha=0.85)

    # On-plot readout
    txt = None
    if show_current_value:
        txt = ax_plot.text(
            0.02, 0.95, "", transform=ax_plot.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85)
        )

    # --- playback state ---
    idx = 0
    paused = False
    step_once = 0

    def on_key(event):
        nonlocal paused, step_once, idx
        if event.key == ' ':
            paused = not paused
        elif event.key in ('q', 'escape'):
            plt.close(fig)
        elif paused:
            if event.key == 'right':
                step_once = +1
            elif event.key == 'left':
                step_once = -1
            elif event.key == 'up':
                step_once = +30
            elif event.key == 'down':
                step_once = -30

    fig.canvas.mpl_connect('key_press_event', on_key)

    frame_period = 1.0 / max(1e-6, fps)
    last_time = time.time()

    def _draw_blinks_up_to(k: int):
        """(Re)draw shaded regions up to index k for each model (and consensus)."""
        nonlocal blink_fills, consensus_fill
        # Remove old fills
        for f in blink_fills:
            try:
                f.remove()
            except Exception:
                pass
        blink_fills = []

        ymax = ax_plot.get_ylim()[1]
        # Per-model regions
        for m in models:
            t = t_ms[:k+1]
            mask = per_model[m]["mask"][:k+1]
            # We draw a light fill under the curve where mask==True
            fill = ax_plot.fill_between(
                t, 0, ymax,
                where=mask,
                step="pre",
                alpha=0.12,
                label=None
            )
            blink_fills.append(fill)

        # Consensus band (thin strip near the bottom)
        if show_consensus and consensus_mask is not None:
            if consensus_fill is not None:
                try: consensus_fill.remove()
                except Exception: pass
            t = t_ms[:k+1]
            c = consensus_mask[:k+1]
            # Draw a small band (e.g., 0..0.02*ymax) when consensus True
            h = 0.03 * ymax
            consensus_fill = ax_plot.fill_between(
                t, 0, h, where=c, step="pre", alpha=0.25, color="k", label=None
            )

    while plt.fignum_exists(fig.number):
        # Left pane: crop
        x1, y1, x2, y2, *_ = map(int, head_bboxes[idx])
        crop = frames[idx][max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            crop = frames[idx]
        im_artist.set_data(cv.cvtColor(crop, cv.COLOR_BGR2RGB))

        # Right pane: progressive lines and current dots
        for m in models:
            errs = per_model[m]["ang_deg"]
            prog_lines[m].set_data(t_ms[:idx+1], errs[:idx+1])
            cur_dots[m].set_data([t_ms[idx]], [errs[idx]])

        # Draw blink overlays up to current idx
        _draw_blinks_up_to(idx)

        # Progressive xlim
        ax_plot.set_xlim(0, float(t_ms[idx]) if idx > 0 else float(t_ms[0]))

        if txt is not None:
            lines = [f"t = {t_ms[idx]:.0f} ms"]
            for m in models:
                ang = per_model[m]["ang_deg"][idx]
                state = " BLINK" if per_model[m]["mask"][idx] else ""
                lines.append(f"{m}: {ang:.2f}°{state}")
            if show_consensus and consensus_mask is not None:
                lines.append(f"consensus: {'ON' if consensus_mask[idx] else 'off'}")
            txt.set_text("\n".join(lines))

        fig.canvas.draw_idle()
        plt.pause(0.001)

        # Advance
        if paused:
            if step_once != 0:
                idx = int(np.clip(idx + step_once, 0, N - 1))
                step_once = 0
            else:
                time.sleep(0.01)
        else:
            now = time.time()
            to_wait = frame_period - (now - last_time)
            if to_wait > 0:
                time.sleep(to_wait)
            last_time = time.time()
            idx += 1
            if idx >= N:
                break

    plt.ioff()
    plt.close(fig)

def play_blink_inspector_with_blinks_local(
    dataloader,
    models: list,
    *,
    fps: float = 30.0,
    window_title: str = "Blink Inspector (local stats + consensus)",
    show_current_value: bool = True,
    # detector params (LOCAL stats)
    window_ms: float = 1000.0,     # rolling median/MAD window; try 600–1500
    smooth_window: int = 3,        # small median on error before z
    z_on: float = 3.0,
    z_off: float = 2.0,
    min_dur_ms: float = 120.0,
    merge_gap_ms: float = 120.0,
    min_peak_deg: float | None = None,   # e.g., 12–15 to suppress tiny bumps
    # consensus
    consensus_k: int | None = None       # None => majority; or set e.g. 2
):
    """
    Visualizes progressive angular error with blink overlays (LOCAL stats) and K-of-M consensus.
    Controls:
      SPACE pause/resume, ←/→ step 1, ↑/↓ step 30, q quit
    """
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2 as cv

    # --- compute per-model masks + consensus using LOCAL stats ---
    from blink_metrics import analyze_sequence_errors_only

    results = analyze_sequence_errors_only(
        dataloader, models,
        window_ms=window_ms,
        smooth_window=smooth_window,
        z_on=z_on, z_off=z_off,
        min_dur_ms=min_dur_ms, merge_gap_ms=merge_gap_ms,
        min_peak_deg=min_peak_deg,
        consensus_k=consensus_k
    )

    # Common time axis for consensus & trimming
    cons = results["_consensus"]
    t_ms = cons["t_ms"]
    if t_ms.size == 0:
        print("No time axis found.")
        return

    # Load frames and head boxes to same length
    frames = dataloader.load_rgb_video(as_numpy=True)
    if len(frames) == 0:
        print("No RGB frames.")
        return
    N = min(len(frames), t_ms.shape[0], *[results[m]["ang_deg"].shape[0] for m in models])
    frames = frames[:N]
    t_ms = t_ms[:N]

    head_bboxes = dataloader.load_head_bboxes()
    h0, w0 = frames[0].shape[:2]
    if not (isinstance(head_bboxes, np.ndarray) and head_bboxes.ndim == 2 and head_bboxes.shape[0] >= N):
        head_bboxes = np.tile([0, 0, w0, h0, -1, -1], (N, 1))

    # Trim per-model series to N
    for m in models:
        results[m]["ang_deg"] = results[m]["ang_deg"][:N]
        results[m]["mask"]    = results[m]["mask"][:N]

    # Consensus masks (already cleaned)
    union_mask        = cons["union_mask"][:N]
    intersection_mask = cons["intersection_mask"][:N]
    consensus_mask    = cons["consensus_mask"][:N]

    # --- Matplotlib setup ---
    plt.ion()
    fig = plt.figure(figsize=(13, 6))
    try:
        fig.canvas.manager.set_window_title(window_title)
    except Exception:
        pass

    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5], wspace=0.25)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_plot = fig.add_subplot(gs[0, 1])

    # Left: first crop
    ax_img.axis('off')
    x1, y1, x2, y2, *_ = map(int, head_bboxes[0])
    crop0 = frames[0][max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
    if crop0.size == 0:
        crop0 = frames[0]
    im_artist = ax_img.imshow(cv.cvtColor(crop0, cv.COLOR_BGR2RGB))
    ax_img.set_title("Head Crop")

    # Right: axes and limits
    all_errs = np.concatenate([results[m]["ang_deg"] for m in models], axis=0)
    ypad = max(5.0, 0.1 * (float(np.nanmax(all_errs)) - float(np.nanmin(all_errs)) + 1e-6))
    ax_plot.set_title("Angular Error Over Time (+ local blink overlay)")
    ax_plot.set_xlabel("Time (ms)")
    ax_plot.set_ylabel("Angular Error (degrees)")
    ax_plot.set_xlim(0, float(t_ms[-1]))
    ax_plot.set_ylim(max(0.0, float(np.nanmin(all_errs) - ypad)),
                     float(np.nanmax(all_errs) + ypad))
    ax_plot.grid(True)

    # Lines & points per model
    prog_lines, cur_dots = {}, {}
    for m in models:
        (pl,) = ax_plot.plot([], [], linewidth=2.0, label=m)
        (cd,) = ax_plot.plot([], [], marker='o')
        prog_lines[m] = pl
        cur_dots[m]   = cd
    ax_plot.legend(loc="upper right", framealpha=0.85)

    # Overlays
    blink_fills = []      # per-model shaded regions
    consensus_fill = None # thin band

    # HUD
    txt = None
    if show_current_value:
        txt = ax_plot.text(
            0.02, 0.95, "", transform=ax_plot.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85)
        )

    # Playback state
    idx, paused, step_once = 0, False, 0
    def on_key(event):
        nonlocal paused, step_once, idx
        if event.key == ' ':
            paused = not paused
        elif event.key in ('q', 'escape'):
            plt.close(fig)
        elif paused:
            if event.key == 'right': step_once = +1
            elif event.key == 'left': step_once = -1
            elif event.key == 'up': step_once = +30
            elif event.key == 'down': step_once = -30
    fig.canvas.mpl_connect('key_press_event', on_key)

    frame_period = 1.0 / max(1e-6, fps)
    last_time = time.time()

    def _draw_blinks_up_to(k: int):
        nonlocal blink_fills, consensus_fill
        for f in blink_fills:
            try: f.remove()
            except Exception: pass
        blink_fills = []

        ymax = ax_plot.get_ylim()[1]
        # per-model fills
        for m in models:
            t = t_ms[:k+1]
            mask = results[m]["mask"][:k+1]
            fill = ax_plot.fill_between(
                t, 0, ymax,
                where=mask, step="pre",
                alpha=0.12
            )
            blink_fills.append(fill)

        # consensus band
        if consensus_mask.size:
            if consensus_fill is not None:
                try: consensus_fill.remove()
                except Exception: pass
            t = t_ms[:k+1]
            c = consensus_mask[:k+1]
            h = 0.03 * ymax
            consensus_fill = ax_plot.fill_between(
                t, 0, h, where=c, step="pre",
                alpha=0.25, color="k"
            )

    # main loop
    while plt.fignum_exists(fig.number):
        # Left: crop
        x1, y1, x2, y2, *_ = map(int, head_bboxes[idx])
        crop = frames[idx][max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
        if crop.size == 0:
            crop = frames[idx]
        im_artist.set_data(cv.cvtColor(crop, cv.COLOR_BGR2RGB))

        # Right: lines and points
        for m in models:
            ang = results[m]["ang_deg"]
            prog_lines[m].set_data(t_ms[:idx+1], ang[:idx+1])
            cur_dots[m].set_data([t_ms[idx]], [ang[idx]])

        _draw_blinks_up_to(idx)
        ax_plot.set_xlim(0, float(t_ms[idx]) if idx > 0 else float(t_ms[0]))

        if txt is not None:
            lines = [f"t = {t_ms[idx]:.0f} ms",
                     f"consensus: {'ON' if consensus_mask[idx] else 'off'}"]
            for m in models:
                lines.append(f"{m}: {results[m]['ang_deg'][idx]:.2f}°"
                             + (" BLINK" if results[m]['mask'][idx] else ""))
            txt.set_text("\n".join(lines))

        fig.canvas.draw_idle()
        plt.pause(0.001)

        # advance
        if paused:
            if step_once:
                idx = int(np.clip(idx + step_once, 0, N - 1))
                step_once = 0
            else:
                time.sleep(0.01)
        else:
            now = time.time()
            to_wait = frame_period - (now - last_time)
            if to_wait > 0: time.sleep(to_wait)
            last_time = time.time()
            idx += 1
            if idx >= N: break

    plt.ioff()
    plt.close(fig)


def play_blink_inspector_with_dev_local(
    dataloader,
    models: list,
    *,
    fps: float = 30.0,
    window_title: str = "Blink Inspector (deviation to camera axis + consensus + annotations)",
    show_current_value: bool = True,
    # detector params:
    axis: tuple = (1.0, 0.0, 0.0),  # +X out of camera
    window_ms: float = 1000.0,
    smooth_window: int = 3,
    z_on: float = 3.0,
    z_off: float = 2.0,
    min_dur_ms: float = 120.0,
    merge_gap_ms: float = 120.0,
    min_peak_rel_deg: float | None = None,
    min_peak_abs_deg: float | None = None,
    consensus_k: int | None = None,
    # NEW: whose annotations to show (if available)
    annotator: str | None = None,
    # NEW: style for annotation lines
    ann_line_width: float = 1.6,
    ann_line_alpha: float = 0.95,
):
    import os
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2 as cv
    from blink_metrics import analyze_sequence_deviation_only

    # --- blink label constants (mirrors blink_annotation.py) ---
    NOT_LABELED    = 0
    LABEL_NO_BLINK = 1
    LABEL_UNCERTAIN= 2
    LABEL_BLINK    = 3

    # --- run deviation-based analyzer (unchanged) ---
    res = analyze_sequence_deviation_only(
        dataloader, models,
        axis=axis,
        window_ms=window_ms,
        smooth_window=smooth_window,
        z_on=z_on, z_off=z_off,
        min_dur_ms=min_dur_ms, merge_gap_ms=merge_gap_ms,
        min_peak_rel_deg=min_peak_rel_deg,
        min_peak_abs_deg=min_peak_abs_deg,
        consensus_k=consensus_k
    )

    cons = res["_consensus"]
    t_ms = cons["t_ms"]
    if t_ms.size == 0:
        print("No time axis found.")
        return

    frames = dataloader.load_rgb_video(as_numpy=True)
    if len(frames) == 0:
        print("No RGB frames.")
        return

    N = min(len(frames), t_ms.shape[0], *[res[m]["dev_deg"].shape[0] for m in models])
    frames = frames[:N]
    t_ms = t_ms[:N]

    head_bboxes = dataloader.load_head_bboxes()
    h0, w0 = frames[0].shape[:2]
    if not (isinstance(head_bboxes, np.ndarray) and head_bboxes.ndim == 2 and head_bboxes.shape[0] >= N):
        head_bboxes = np.tile([0, 0, w0, h0, -1, -1], (N, 1))

    for m in models:
        res[m]["dev_deg"] = res[m]["dev_deg"][:N]
        res[m]["mask"]    = res[m]["mask"][:N]

    consensus_mask = cons["consensus_mask"][:N]

    # --- try to load manual blink annotations (optional) ---
    ann = None
    if annotator is not None:
        try:
            # preferred path if dataloader provides it
            ann = dataloader.get_blink_annotations(annotator)
        except Exception:
            # fallback to the file we save in annotate_blinks
            try:
                save_path = os.path.join(dataloader.get_cwd(), f"blink_annotations_by_{annotator}.npy")
                ann = np.load(save_path)
            except Exception:
                ann = None

    # Prepare starts/ends from annotations if present
    ann_start_idx = np.array([], dtype=int)
    ann_end_idx   = np.array([], dtype=int)
    if isinstance(ann, np.ndarray) and ann.size > 0:
        ann = ann.astype(np.int16)[:N]
        # Map according to your rule:
        # - 0 (unlabeled) -> treat as OPEN (1)
        # - 2 (uncertain) -> treat as CLOSED (3)
        mapped = ann.copy()
        mapped[mapped == NOT_LABELED] = LABEL_NO_BLINK
        # Closed if 3 or 2; Open otherwise
        closed = (mapped == LABEL_BLINK) | (mapped == LABEL_UNCERTAIN)
        closed_i8 = closed.astype(np.int8)
        # diff with padding to capture edges
        diff = np.diff(closed_i8, prepend=0, append=0)
        # start where 0->1, end where 1->0 (end index is i-1 from the diff placement)
        ann_start_idx = np.flatnonzero(diff == 1)
        ann_end_idx   = np.flatnonzero(diff == -1) - 1
        # Guard against negatives if no open→closed transition at start
        ann_end_idx = ann_end_idx[ann_end_idx >= 0]

    # --- plotting ---
    plt.ion()
    fig = plt.figure(figsize=(13, 6))
    try:
        fig.canvas.manager.set_window_title(window_title)
    except Exception:
        pass

    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5], wspace=0.25)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_plot = fig.add_subplot(gs[0, 1])

    # Left: crop
    ax_img.axis('off')
    x1, y1, x2, y2, *_ = map(int, head_bboxes[0])
    crop0 = frames[0][max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
    if crop0.size == 0:
        crop0 = frames[0]
    im_artist = ax_img.imshow(cv.cvtColor(crop0, cv.COLOR_BGR2RGB))
    ax_img.set_title("Head Crop")

    # Right: deviation plot
    all_vals = np.concatenate([res[m]["dev_deg"] for m in models], axis=0)
    ypad = max(5.0, 0.1 * (float(np.nanmax(all_vals)) - float(np.nanmin(all_vals)) + 1e-6))
    ax_plot.set_title("Deviation to Camera Axis (+ local blink overlay + annotations)")
    ax_plot.set_xlabel("Time (ms)")
    ax_plot.set_ylabel("Angle to +X (degrees)")
    ax_plot.set_xlim(0, float(t_ms[-1]))
    ax_plot.set_ylim(max(0.0, float(np.nanmin(all_vals) - ypad)),
                     float(np.nanmax(all_vals) + ypad))
    ax_plot.grid(True)

    # Lines for models
    prog_lines, cur_dots = {}, {}
    for m in models:
        (pl,) = ax_plot.plot([], [], linewidth=2.0, label=m)
        (cd,) = ax_plot.plot([], [], marker='o')
        prog_lines[m] = pl
        cur_dots[m]   = cd
    ax_plot.legend(loc="upper right", framealpha=0.85)

    # Shaded blink regions (detector) + consensus band
    blink_fills = []
    consensus_fill = None

    # --- draw manual annotation vertical lines (full extent, static) ---
    # Only draw the *starts* and *ends* of closed segments.
    if ann_start_idx.size and ann_end_idx.size:
        # Align to time axis; we draw the full set once.
        for i0 in ann_start_idx:
            if 0 <= i0 < N:
                ax_plot.axvline(
                    float(t_ms[i0]),
                    color='r', linestyle='-', linewidth=ann_line_width, alpha=ann_line_alpha
                )
        for i1 in ann_end_idx:
            if 0 <= i1 < N:
                ax_plot.axvline(
                    float(t_ms[i1]),
                    color='r', linestyle='-', linewidth=ann_line_width, alpha=ann_line_alpha
                )

    # HUD
    txt = None
    if show_current_value:
        txt = ax_plot.text(
            0.02, 0.95, "", transform=ax_plot.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85)
        )

    # Playback controls
    idx, paused, step_once = 0, False, 0
    def on_key(event):
        nonlocal paused, step_once, idx
        if event.key == ' ': paused = not paused
        elif event.key in ('q','escape'): plt.close(fig)
        elif paused:
            if event.key == 'right': step_once = +1
            elif event.key == 'left': step_once = -1
            elif event.key == 'up': step_once = +30
            elif event.key == 'down': step_once = -30
    fig.canvas.mpl_connect('key_press_event', on_key)

    frame_period = 1.0 / max(1e-6, fps)
    last_time = time.time()

    def _draw_blinks_up_to(k: int):
        nonlocal blink_fills, consensus_fill
        for f in blink_fills:
            try: f.remove()
            except Exception: pass
        blink_fills = []
        ymax = ax_plot.get_ylim()[1]
        # per-model filled blink masks (from detector)
        for m in models:
            t = t_ms[:k+1]
            mask = res[m]["mask"][:k+1]
            fill = ax_plot.fill_between(t, 0, ymax, where=mask, step="pre", alpha=0.12)
            blink_fills.append(fill)
        # consensus band
        if consensus_mask.size:
            if consensus_fill is not None:
                try: consensus_fill.remove()
                except Exception: pass
            t = t_ms[:k+1]
            c = consensus_mask[:k+1]
            h = 0.03 * ymax
            consensus_fill = ax_plot.fill_between(t, 0, h, where=c, step="pre", alpha=0.25, color="k")

    while plt.fignum_exists(fig.number):
        # Left
        x1, y1, x2, y2, *_ = map(int, head_bboxes[idx])
        crop = frames[idx][max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
        if crop.size == 0: crop = frames[idx]
        im_artist.set_data(cv.cvtColor(crop, cv.COLOR_BGR2RGB))

        # Right
        for m in models:
            vals = res[m]["dev_deg"]
            prog_lines[m].set_data(t_ms[:idx+1], vals[:idx+1])
            cur_dots[m].set_data([t_ms[idx]], [vals[idx]])

        _draw_blinks_up_to(idx)
        ax_plot.set_xlim(0, float(t_ms[idx]) if idx > 0 else float(t_ms[0]))

        if txt is not None:
            lines = [f"t = {t_ms[idx]:.0f} ms",
                     f"consensus: {'ON' if consensus_mask[idx] else 'off'}"]
            for m in models:
                val = res[m]['dev_deg'][idx]
                lines.append(f"{m}: {val:.2f}°" + (" BLINK" if res[m]['mask'][idx] else ""))
            # If we have annotations, indicate if current idx is inside a closed annotated segment
            if ann_start_idx.size and ann_end_idx.size:
                in_ann = False
                for s, e in zip(ann_start_idx, ann_end_idx):
                    if s <= idx <= e:
                        in_ann = True
                        break
                lines.append(f"annotated: {'CLOSED' if in_ann else 'open'}")
            txt.set_text("\n".join(lines))

        fig.canvas.draw_idle()
        plt.pause(0.001)

        if paused:
            if step_once:
                idx = int(np.clip(idx + step_once, 0, N - 1)); step_once = 0
            else:
                time.sleep(0.01)
        else:
            now = time.time()
            tw = frame_period - (now - last_time)
            if tw > 0: time.sleep(tw)
            last_time = time.time()
            idx += 1
            if idx >= N: break

    plt.ioff(); plt.close(fig)


def evaluate_blink_overlap_metrics_dev_local(
    dataloader,
    models: list,
    annotator: str,
    *,
    # prediction mask selection
    mask_mode: str = "consensus",  # "consensus" | "union" | "intersection" | "model"
    model_for_mask: str | None = None,  # used only if mask_mode == "model"
    # detector params (same family as play_blink_inspector_with_dev_local)
    axis: tuple = (1.0, 0.0, 0.0),
    window_ms: float = 1000.0,
    smooth_window: int = 3,
    z_on: float = 3.0,
    z_off: float = 2.0,
    min_dur_ms: float = 120.0,
    merge_gap_ms: float = 120.0,
    min_peak_rel_deg: float | None = None,
    min_peak_abs_deg: float | None = None,
    consensus_k: int | None = None,
    # IoU convention
    penalize_overshoot: bool = True,  # False => coverage-only (union clipped to the reference interval)
) -> dict:
    """
    Compare manual blink annotations to deviation-based predictions and return overlap metrics.

    Label policy for annotations:
      0 (NOT_LABELED) and 1 (LABEL_NO_BLINK) => OPEN
      2 (LABEL_UNCERTAIN) and 3 (LABEL_BLINK) => CLOSED

    mask_mode:
      - "consensus": use K-of-M consensus from analyze_sequence_deviation_only (K from consensus_k or majority)
      - "union": closed if any model closed
      - "intersection": closed only if all models closed
      - "model": use a specific model's mask (provide model_for_mask)

    Returns a dict with:
      - per-annotation IoUs and summary
      - per-prediction IoUs and summary
      - global durations, counts (#annotation periods, #prediction periods)
      - global Jaccards (blink, non-blink, balanced), recall/precision, TPR/TNR, prevalence, balanced accuracy
    """
    import os
    import numpy as np
    from blink_metrics import analyze_sequence_deviation_only
    import math

    # ---- constants (mirror blink_annotation.py) ----
    NOT_LABELED    = 0
    LABEL_NO_BLINK = 1
    LABEL_UNCERTAIN= 2
    LABEL_BLINK    = 3

    # ---------- helpers ----------
    def _mask_to_intervals(t_ms: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
        """Convert boolean mask (True=closed) and time vector to merged half-open intervals [s,e)."""
        t = t_ms.astype(float)
        m = mask.astype(bool)
        N = len(m)
        if N == 0:
            return []
        starts = []
        ends_ex = []
        up   = np.flatnonzero((~m[:-1]) & ( m[1:])) + 1
        down = np.flatnonzero(( m[:-1]) & (~m[1:])) + 1
        if m[0]:
            starts = [0, *up.tolist()]
        else:
            starts = up.tolist()
        if m[-1]:
            ends_ex = [*down.tolist(), N]
        else:
            ends_ex = down.tolist()

        dts = np.diff(t)
        dt_default = float(np.median(dts)) if dts.size else 0.0

        out = []
        for s, e in zip(starts, ends_ex):
            start_time = float(t[s])
            end_time = float(t[e]) if e < N else float(t[-1] + dt_default)
            if end_time > start_time:
                out.append((start_time, end_time))
        return out

    def _merge_intervals(iv: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not iv:
            return []
        iv = sorted(iv, key=lambda x: (x[0], x[1]))
        merged = [iv[0]]
        for s, e in iv[1:]:
            ms, me = merged[-1]
            if s <= me:
                merged[-1] = (ms, max(me, e))
            else:
                merged.append((s, e))
        return merged

    def _total_len(iv: list[tuple[float, float]]) -> float:
        return sum(max(0.0, e - s) for s, e in iv)

    def _clip(A: tuple[float, float], B: tuple[float, float]):
        s = max(A[0], B[0]); e = min(A[1], B[1])
        return (s, e) if e > s else None

    # ---------- predictions via deviation-only analysis ----------
    res = analyze_sequence_deviation_only(
        dataloader, models,
        axis=axis,
        window_ms=window_ms,
        smooth_window=smooth_window,
        z_on=z_on, z_off=z_off,
        min_dur_ms=min_dur_ms, merge_gap_ms=merge_gap_ms,
        min_peak_rel_deg=min_peak_rel_deg,
        min_peak_abs_deg=min_peak_abs_deg,
        consensus_k=consensus_k
    )
    cons = res["_consensus"]
    t_ms = cons["t_ms"]

    # --- compute sequence duration (ms) from t_ms ---
    if t_ms.size >= 2:
        dts = np.diff(t_ms.astype(float))
        dt_default = float(np.median(dts)) if dts.size else 0.0
        # include the duration of the last sample using median dt
        sequence_duration_ms = float((t_ms[-1] - t_ms[0]) + dt_default)
    elif t_ms.size == 1:
        dts = np.array([], dtype=float)
        dt_default = 0.0
        sequence_duration_ms = 0.0
    else:
        dts = np.array([], dtype=float)
        dt_default = 0.0
        sequence_duration_ms = 0.0

    if t_ms.size == 0:
        return {
            "per_annotation": [],
            "per_prediction": [],
            "summary": {
                "n_annotations": 0,
                "n_predictions": 0,
                "mean_iou_annotations": 0.0,
                "median_iou_annotations": 0.0,
                "duration_weighted_iou_annotations": 0.0,
                "mean_iou_predictions": 0.0,
                "median_iou_predictions": 0.0,
                "duration_weighted_iou_predictions": 0.0,
                "total_annot_duration_ms": 0.0,
                "total_pred_duration_ms": 0.0,
                "total_intersection_ms": 0.0,
                "global_jaccard": 0.0,
                "jaccard_pos": 0.0,
                "jaccard_neg": 0.0,
                "jaccard_balanced": 0.0,
                "blink_prevalence": 0.0,
                "tpr_blink": float("nan"),
                "tnr_noblink": float("nan"),
                "balanced_accuracy": float("nan"),
                "recall_annot_covered": 0.0,
                "precision_pred_purity": 0.0,
                "sequence_duration_ms": 0.0,
                "TP_ms": 0.0,
                "FP_ms": 0.0,
                "FN_ms": 0.0,
                "TN_ms": 0.0,
            }
        }

    # choose prediction mask
    stack = np.stack([res[m]["mask"] for m in models], axis=0)  # (M, N) boolean
    if mask_mode == "union":
        pred_mask = np.any(stack, axis=0)
    elif mask_mode == "intersection":
        pred_mask = np.all(stack, axis=0)
    elif mask_mode == "model":
        if model_for_mask in res and "mask" in res[model_for_mask]:
            pred_mask = res[model_for_mask]["mask"]
        else:
            # fallback to union if model missing
            pred_mask = np.any(stack, axis=0)
    else:  # "consensus" (default)
        pred_mask = cons["consensus_mask"]

    # ---------- annotations ----------
    frames_len = t_ms.shape[0]
    try:
        ann = dataloader.get_blink_annotations(annotator)
    except Exception:
        # fallback to default file name used by the annotator tool
        save_path = os.path.join(dataloader.get_cwd(), f"blink_annotations_by_{annotator}.npy")
        ann = np.load(save_path) if os.path.exists(save_path) else None

    if not isinstance(ann, np.ndarray) or ann.size == 0:
        ann_mask = np.zeros((frames_len,), dtype=bool)
    else:
        ann = ann.astype(np.int16)[:frames_len]
        # 0,1 => open; 2,3 => closed
        ann_mask = (ann == LABEL_UNCERTAIN) | (ann == LABEL_BLINK)

    pred_mask = pred_mask[:frames_len]

    # ---------- intervals (for per-event IoUs) ----------
    ann_iv  = _mask_to_intervals(t_ms, ann_mask)
    pred_iv = _mask_to_intervals(t_ms, pred_mask)

    # counts & totals for events
    n_annotations = len(ann_iv)
    n_predictions = len(pred_iv)
    total_ann_len  = _total_len(ann_iv)
    total_pred_len = _total_len(pred_iv)

    # ---------- per-annotation IoU ----------
    per_ann = []
    for A in ann_iv:
        overlaps = [P for P in pred_iv if max(A[0], P[0]) < min(A[1], P[1])]
        if not overlaps:
            U = (A[1] - A[0]) if not penalize_overshoot else (A[1] - A[0])
            per_ann.append({
                "start_ms": A[0], "end_ms": A[1], "dur_ms": A[1]-A[0],
                "intersection_ms": 0.0, "union_ms": U, "iou": 0.0
            })
            continue

        clips = []
        for P in overlaps:
            C = _clip(A, P)
            if C: clips.append(C)
        clips = _merge_intervals(clips)
        inter_len = _total_len(clips)

        if penalize_overshoot:
            U_iv = _merge_intervals([A, *overlaps])
            union_len = _total_len(U_iv)
        else:
            union_len = (A[1] - A[0])  # coverage-only

        iou = (inter_len / union_len) if union_len > 0 else 0.0
        per_ann.append({
            "start_ms": A[0], "end_ms": A[1], "dur_ms": A[1]-A[0],
            "intersection_ms": inter_len, "union_ms": union_len, "iou": iou
        })

    # ---------- per-prediction IoU (symmetric) ----------
    per_pred = []
    for P in pred_iv:
        overlaps = [A for A in ann_iv if max(A[0], P[0]) < min(A[1], P[1])]
        if not overlaps:
            U = (P[1] - P[0]) if not penalize_overshoot else (P[1] - P[0])
            per_pred.append({
                "start_ms": P[0], "end_ms": P[1], "dur_ms": P[1]-P[0],
                "intersection_ms": 0.0, "union_ms": U, "iou": 0.0
            })
            continue

        clips = []
        for A in overlaps:
            C = _clip(P, A)
            if C: clips.append(C)
        clips = _merge_intervals(clips)
        inter_len = _total_len(clips)

        if penalize_overshoot:
            U_iv = _merge_intervals([P, *overlaps])
            union_len = _total_len(U_iv)
        else:
            union_len = (P[1] - P[0])

        iou = (inter_len / union_len) if union_len > 0 else 0.0
        per_pred.append({
            "start_ms": P[0], "end_ms": P[1], "dur_ms": P[1]-P[0],
            "intersection_ms": inter_len, "union_ms": union_len, "iou": iou
        })

    # ---------- duration-based confusion totals (for robust global metrics) ----------
    # Segment durations per sample i (covering [t[i], t[i+1])) and last sample with median dt.
    if frames_len >= 2:
        seg_dur = np.concatenate([np.diff(t_ms.astype(float)), [dt_default]])
    elif frames_len == 1:
        seg_dur = np.array([dt_default], dtype=float)
    else:
        seg_dur = np.array([], dtype=float)

    # Masks over segments use the state at index i.
    P = pred_mask.astype(bool)
    A = ann_mask.astype(bool)
    # Align lengths
    P = P[:seg_dur.size]
    A = A[:seg_dur.size]

    TP_ms = float(seg_dur[(P)  & (A)].sum())
    FP_ms = float(seg_dur[(P)  & (~A)].sum())
    FN_ms = float(seg_dur[(~P) & (A)].sum())
    TN_ms = float(seg_dur[(~P) & (~A)].sum())
    total_ms = float(seg_dur.sum())
    # --- denominators for metric-aware weighting ---
    denom_accuracy_ms        = total_ms
    denom_tpr_ms             = TP_ms + FN_ms                # blink (positive) time
    denom_tnr_ms             = TN_ms + FP_ms                # no-blink (negative) time
    denom_precision_ms       = TP_ms + FP_ms                # predicted blink time
    denom_npv_ms             = TN_ms + FN_ms                # predicted no-blink time
    denom_jaccard_pos_ms     = TP_ms + FP_ms + FN_ms
    denom_jaccard_neg_ms     = TN_ms + FP_ms + FN_ms
    denom_blink_prev_ms      = total_ms
    # Duration-weighted accuracy
    accuracy = ((TP_ms + TN_ms) / total_ms) if total_ms > 0 else float("nan")

    # Blink prevalence
    blink_prevalence = ((TP_ms + FN_ms) / total_ms) if total_ms > 0 else 0.0

    # Positive-class (blink) Jaccard with empty/empty -> 1.0
    if (TP_ms == 0.0) and (FP_ms == 0.0) and (FN_ms == 0.0):
        jaccard_pos = 1.0
    else:
        denom_pos = TP_ms + FP_ms + FN_ms
        jaccard_pos = (TP_ms / denom_pos) if denom_pos > 0 else 0.0

    # Negative-class (no-blink) Jaccard
    if (TN_ms == 0.0) and (FP_ms == 0.0) and (FN_ms == 0.0):
        jaccard_neg = 1.0
    else:
        denom_neg = TN_ms + FP_ms + FN_ms
        jaccard_neg = (TN_ms / denom_neg) if denom_neg > 0 else 0.0

    jaccard_balanced = 0.5 * (jaccard_pos + jaccard_neg)

    # Recall/Precision (blink) based on durations
    recall  = (TP_ms / (TP_ms + FN_ms)) if (TP_ms + FN_ms) > 0 else (1.0 if (TP_ms == 0.0 and FN_ms == 0.0) else float("nan"))
    precision = (TP_ms / (TP_ms + FP_ms)) if (TP_ms + FP_ms) > 0 else (1.0 if (TP_ms == 0.0 and FP_ms == 0.0) else float("nan"))

    # TPR (blink) and TNR (no-blink)
    tpr_blink = recall
    tnr_noblink = (TN_ms / (TN_ms + FP_ms)) if (TN_ms + FP_ms) > 0 else (1.0 if (TN_ms == 0.0 and FP_ms == 0.0) else float("nan"))
    # Balanced accuracy
    if math.isnan(tpr_blink) and math.isnan(tnr_noblink):
        balanced_accuracy = float("nan")
    elif math.isnan(tpr_blink):
        balanced_accuracy = tnr_noblink
    elif math.isnan(tnr_noblink):
        balanced_accuracy = tpr_blink
    else:
        balanced_accuracy = 0.5 * (tpr_blink + tnr_noblink)

    # Keep intersection/union totals for reporting consistency (using interval-based computation)
    def _union_len(iv_a: list[tuple[float, float]], iv_b: list[tuple[float, float]]) -> float:
        return _total_len(_merge_intervals([*iv_a, *iv_b]))

    # total intersection across whole timeline = union of all pairwise clips
    all_clips = []
    for Aiv in ann_iv:
        for Piv in pred_iv:
            C = _clip(Aiv, Piv)
            if C: all_clips.append(C)
    all_clips = _merge_intervals(all_clips)
    total_inter_len = _total_len(all_clips)
    union_global = _union_len(ann_iv, pred_iv)
    # Historic 'global_jaccard' equals positive-class Jaccard; keep field for backward compatibility.
    global_jaccard = jaccard_pos

    # summaries of per-event IoUs
    def _summarize(per_list: list[dict]) -> tuple[float, float, float]:
        if not per_list:
            return 0.0, 0.0, 0.0
        ious = np.array([x["iou"] for x in per_list], dtype=float)
        unions = np.array([x["union_ms"] for x in per_list], dtype=float)
        inters = np.array([x["intersection_ms"] for x in per_list], dtype=float)
        mean_iou = float(np.mean(ious))
        median_iou = float(np.median(ious))
        dur_weighted = float((inters.sum() / unions.sum())) if unions.sum() > 0 else 0.0
        return mean_iou, median_iou, dur_weighted

    meanA, medA, dwiA = _summarize(per_ann)
    meanP, medP, dwiP = _summarize(per_pred)

    return {
        "per_annotation": per_ann,
        "per_prediction": per_pred,
        "summary": {
            "n_annotations": n_annotations,
            "n_predictions": n_predictions,
            "mean_iou_annotations": meanA,
            "median_iou_annotations": medA,
            "duration_weighted_iou_annotations": dwiA,
            "mean_iou_predictions": meanP,
            "median_iou_predictions": medP,
            "duration_weighted_iou_predictions": dwiP,
            "total_annot_duration_ms": float(total_ann_len),
            "total_pred_duration_ms": float(total_pred_len),
            "total_intersection_ms": float(total_inter_len),
            "global_jaccard": float(global_jaccard),
            "jaccard_pos": float(jaccard_pos),
            "jaccard_neg": float(jaccard_neg),
            "jaccard_balanced": float(jaccard_balanced),
            "blink_prevalence": float(blink_prevalence),
            "tpr_blink": float(tpr_blink) if not math.isnan(tpr_blink) else float("nan"),
            "tnr_noblink": float(tnr_noblink) if not math.isnan(tnr_noblink) else float("nan"),
            "balanced_accuracy": float(balanced_accuracy) if not math.isnan(balanced_accuracy) else float("nan"),
            "recall_annot_covered": float(recall) if not math.isnan(recall) else float("nan"),
            "precision": float(precision) if not math.isnan(precision) else float("nan"),
            "sequence_duration_ms": float(sequence_duration_ms),
            "union_duration_ms": float(union_global),
            "accuracy": float(accuracy) if not math.isnan(accuracy) else float("nan"),
            "denom_accuracy_ms": float(denom_accuracy_ms),
            "denom_tpr_ms": float(denom_tpr_ms),
            "denom_tnr_ms": float(denom_tnr_ms),
            "denom_precision_ms": float(denom_precision_ms),
            "denom_npv_ms": float(denom_npv_ms),
            "denom_jaccard_pos_ms": float(denom_jaccard_pos_ms),
            "denom_jaccard_neg_ms": float(denom_jaccard_neg_ms),
            "denom_blink_prevalence_ms": float(denom_blink_prev_ms),
            "TP_ms": float(TP_ms),
            "FP_ms": float(FP_ms),
            "FN_ms": float(FN_ms),
            "TN_ms": float(TN_ms),
        }
    }


import csv
import os
from typing import Optional, Iterable

def save_blink_eval_to_csv(
    metrics: dict,
    out_csv_path: str,
    *,
    method_name: str,
    params: Optional[dict] = None,
    data_id: Optional[str] = None,
    append: bool = True,
) -> str:
    """
    Persist a single evaluation run (returned by evaluate_blink_overlap_metrics_dev_local)
    to a CSV. Also stores run-identifying info and your evaluation parameters.
    """
    params = params or {}
    summary = metrics.get("summary", {})

    # Pull core summary numbers (defaults safe if missing)
    n_annotations = int(summary.get("n_annotations", 0))
    n_predictions = int(summary.get("n_predictions", 0))
    total_annot_ms = float(summary.get("total_annot_duration_ms", 0.0))
    total_pred_ms  = float(summary.get("total_pred_duration_ms", 0.0))
    total_inter_ms = float(summary.get("total_intersection_ms", 0.0))
    sequence_duration_ms = float(summary.get("sequence_duration_ms", 0.0))

    # Metrics to persist
    metric_fields = {
        # Jaccards
        "jaccard_pos":              float(summary.get("jaccard_pos", 0.0)),
        "jaccard_neg":              float(summary.get("jaccard_neg", 0.0)),
        "jaccard_balanced":         float(summary.get("jaccard_balanced", 0.0)),

        # Prevalence & rates
        "blink_prevalence":         float(summary.get("blink_prevalence", 0.0)),
        "tpr_blink":                float(summary.get("tpr_blink", float("nan"))),
        "tnr_noblink":              float(summary.get("tnr_noblink", float("nan"))),
        "balanced_accuracy":        float(summary.get("balanced_accuracy", float("nan"))),

        # Precision (renamed) and Accuracy
        # NOTE: read "precision" primarily; fall back to old "precision_pred_purity"
        "precision":                float(
            summary.get("precision", summary.get("precision_pred_purity", 0.0))
        ),
        "accuracy":                 float(summary.get("accuracy", float("nan"))),
    }
    denom_fields = {
        "denom_accuracy_ms":        float(summary.get("denom_accuracy_ms", 0.0)),
        "denom_tpr_ms":             float(summary.get("denom_tpr_ms", 0.0)),
        "denom_tnr_ms":             float(summary.get("denom_tnr_ms", 0.0)),
        "denom_precision_ms":       float(summary.get("denom_precision_ms", 0.0)),
        "denom_npv_ms":             float(summary.get("denom_npv_ms", 0.0)),
        "denom_jaccard_pos_ms":     float(summary.get("denom_jaccard_pos_ms", 0.0)),
        "denom_jaccard_neg_ms":     float(summary.get("denom_jaccard_neg_ms", 0.0)),
        "denom_blink_prevalence_ms":float(summary.get("denom_blink_prevalence_ms", 0.0)),
    }
    conf_ms_fields = {
        "TP_ms": float(summary.get("TP_ms", 0.0)),
        "FP_ms": float(summary.get("FP_ms", 0.0)),
        "FN_ms": float(summary.get("FN_ms", 0.0)),
        "TN_ms": float(summary.get("TN_ms", 0.0)),
    }
    # Base row with identifiers & durations
    base_row = {
        "method_name": method_name,
        "data_id": data_id or "",
        # counts
        "n_annotations": n_annotations,
        "n_predictions": n_predictions,
        # durations
        "total_annot_duration_ms": total_annot_ms,
        "total_pred_duration_ms": total_pred_ms,
        "total_intersection_ms": total_inter_ms,
        "sequence_duration_ms": sequence_duration_ms,
        "weight_duration_ms": sequence_duration_ms,
    }

    # Flatten params into columns (stringify consistently)
    param_cols = {}
    for k, v in sorted(params.items()):
        if isinstance(v, (list, tuple)):
            param_cols[k] = ",".join(map(str, v))
        else:
            param_cols[k] = v

    # Compose full header ordering (stable & readable)
    fixed_order = list(base_row.keys()) + list(metric_fields.keys()) + list(denom_fields.keys()) + list(conf_ms_fields.keys())
    header = fixed_order + list(param_cols.keys())

    # Ensure directory exists
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)

    # If creating a new file, write header first
    file_exists = os.path.isfile(out_csv_path)
    mode = "a" if (append and file_exists) else "w"
    with open(out_csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if mode == "w":
            writer.writeheader()

        row = {}
        row.update(base_row)
        row.update(metric_fields)
        row.update(param_cols)
        row.update(denom_fields)
        row.update(conf_ms_fields)
        writer.writerow(row)

    return out_csv_path


def aggregate_blink_eval_csv(
    csv_paths: Iterable[str],
    *,
    # default fallback weight if a metric-specific denom is missing in a row
    default_weight_col: str = "weight_duration_ms",
    # OPTIONAL: override / extend mapping from metric -> weight column
    metric_weight_cols: Optional[dict] = None,
    metrics_to_aggregate: Optional[Iterable[str]] = None,
) -> dict:
    """
    Aggregate many saved evaluation rows with metric-specific denominators.

    - accuracy, blink_prevalence: weighted by total duration
    - tpr_blink (recall):         weighted by TP+FN (blink time)
    - tnr_noblink (specificity):  weighted by TN+FP (no-blink time)
    - precision:                  weighted by TP+FP (predicted blink time)
    - jaccard_pos:                weighted by TP+FP+FN
    - jaccard_neg:                weighted by TN+FP+FN
    - jaccard_balanced:           weighted by (denom_pos + denom_neg)  (see note)
    """
    import csv, math, os

    # Which metrics to aggregate (same set you print now)
    default_metrics = [
        "jaccard_pos",
        "jaccard_neg",
        "jaccard_balanced",
        "blink_prevalence",
        "tpr_blink",
        "tnr_noblink",
        "balanced_accuracy",
        "precision",
        "accuracy",
        "jaccard_pos_micro",         # NEW
        "jaccard_neg_micro",         # NEW
        "jaccard_balanced_micro",    # NEW
    ]
    metrics_list = list(metrics_to_aggregate) if metrics_to_aggregate else default_metrics

    # Default mapping from metric -> weight column (denominator).
    default_weight_map = {
        "accuracy":           "denom_accuracy_ms",
        "blink_prevalence":   "denom_blink_prevalence_ms",
        "tpr_blink":          "denom_tpr_ms",
        "tnr_noblink":        "denom_tnr_ms",
        "precision":          "denom_precision_ms",
        # if you add "npv" later: "npv": "denom_npv_ms",
        "jaccard_pos":        "denom_jaccard_pos_ms",
        "jaccard_neg":        "denom_jaccard_neg_ms",
        # For balanced Jaccard, weight by sum of pos+neg denominators (see note below)
        "jaccard_balanced":   None,  # handled specially below
        # Balanced accuracy is an average of TPR & TNR; there is no single clean denom.
        # We recommend fallback to total duration for BA, or leave unweighted (=macro).
        "balanced_accuracy":  default_weight_col,
    }
    if metric_weight_cols:
        default_weight_map.update(metric_weight_cols)

    # Load rows
    rows = []
    for path in csv_paths:
        if not os.path.isfile(path):
            continue
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows.extend(list(reader))
    if not rows:
        return {"n_rows": 0, "total_weight": 0.0, "by_metric": {}}

    # Helpers
    def wmean(vec_w_vals):
        num = sum(w * x for w, x in vec_w_vals if not math.isnan(x) and w > 0)
        den = sum(w for w, x in vec_w_vals if not math.isnan(x) and w > 0)
        return (num / den) if den > 0 else float("nan")

    def wstd(vec_w_vals, mu):
        num = sum(w * (x - mu) ** 2 for w, x in vec_w_vals if not math.isnan(x) and w > 0)
        den = sum(w for w, x in vec_w_vals if not math.isnan(x) and w > 0)
        return math.sqrt(num / den) if den > 0 else float("nan")

    def wmedian(vec_w_vals):
        vals = [(x, w) for w, x in vec_w_vals if not math.isnan(x) and w > 0]
        if not vals:
            return float("nan")
        vals.sort(key=lambda t: t[0])
        total_w = sum(w for _, w in vals)
        if total_w <= 0:
            return float("nan")
        cutoff = 0.5 * total_w
        acc = 0.0
        for x, w in vals:
            acc += w
            if acc >= cutoff:
                return x
        return vals[-1][0]

    # --- MICRO totals across all rows ---
    try:
        sum_TP = sum(float(r.get("TP_ms", "0") or 0.0) for r in rows)
        sum_FP = sum(float(r.get("FP_ms", "0") or 0.0) for r in rows)
        sum_FN = sum(float(r.get("FN_ms", "0") or 0.0) for r in rows)
        sum_TN = sum(float(r.get("TN_ms", "0") or 0.0) for r in rows)
    except Exception:
        sum_TP = sum_FP = sum_FN = sum_TN = 0.0

    den_pos = sum_TP + sum_FP + sum_FN
    den_neg = sum_TN + sum_FP + sum_FN

    if den_pos > 0:
        jaccard_pos_micro = sum_TP / den_pos
    else:
        # convention: if truly no positives across dataset, set to 1.0
        jaccard_pos_micro = 1.0

    if den_neg > 0:
        jaccard_neg_micro = sum_TN / den_neg
    else:
        jaccard_neg_micro = 1.0

    jaccard_balanced_micro = 0.5 * (jaccard_pos_micro + jaccard_neg_micro)


    by_metric = {}

    # --- micro metrics already computed above ---
    by_metric["jaccard_pos_micro"] = {
        "weighted_mean": jaccard_pos_micro,
        "weighted_median": float("nan"),
        "weighted_std": float("nan"),
    }
    by_metric["jaccard_neg_micro"] = {
        "weighted_mean": jaccard_neg_micro,
        "weighted_median": float("nan"),
        "weighted_std": float("nan"),
    }
    by_metric["jaccard_balanced_micro"] = {
        "weighted_mean": jaccard_balanced_micro,
        "weighted_median": float("nan"),
        "weighted_std": float("nan"),
    }

    micro_names = {"jaccard_pos_micro", "jaccard_neg_micro", "jaccard_balanced_micro"}

    for m in metrics_list:
        if m in micro_names:
            continue

        vec = []
        for r in rows:
            # value
            v_raw = r.get(m, None)
            try:
                x = float(v_raw) if (v_raw is not None and v_raw != "") else math.nan
            except Exception:
                x = math.nan

            # STRICT weighting by natural denominators for class-conditional metrics
            if m in {"jaccard_pos", "jaccard_neg"}:
                weight_col = default_weight_map[m]  # denom_jaccard_pos_ms / denom_jaccard_neg_ms
                w = float(r.get(weight_col, "0") or 0)

            elif m in {"tpr_blink", "tnr_noblink", "precision"}:
                # Use their own denominators only; NO fallback to duration.
                weight_col = default_weight_map[m]  # denom_tpr_ms / denom_tnr_ms / denom_precision_ms
                w = float(r.get(weight_col, "0") or 0)

            elif m == "jaccard_balanced":
                # Sum of pos+neg denominators; NO fallback.
                wp = float(r.get("denom_jaccard_pos_ms", "0") or 0)
                wn = float(r.get("denom_jaccard_neg_ms", "0") or 0)
                w = max(0.0, wp + wn)

            else:
                # Other metrics may keep the fallback behavior if you want.
                weight_col = default_weight_map.get(m, default_weight_col) or default_weight_col
                w = float(r.get(weight_col, "0") or 0)
                if w <= 0:
                    w = float(r.get(default_weight_col, "0") or 0)

            vec.append((w, x))

        mu = wmean(vec)
        by_metric[m] = {
            "weighted_mean":  mu,
            "weighted_median": wmedian(vec),
            "weighted_std":   wstd(vec, mu),
        }


    # Report n_rows and (optional) total weight of the fallback column for reference
    try:
        total_fallback_weight = sum(float(r.get(default_weight_col, "0") or 0) for r in rows)
    except Exception:
        total_fallback_weight = 0.0

    return {
        "n_rows": len(rows),
        "total_weight": total_fallback_weight,
        "by_metric": by_metric,
    }


def blink_comparison_per_dataloader(dataloader, blink_comparison_out_csv_path, params, method_name):
    metrics = evaluate_blink_overlap_metrics_dev_local(
        dataloader=dataloader,
        models=params["models"],
        annotator=params["annotator"],
        mask_mode=params["mask_mode"],
        window_ms=params["window_ms"],
        smooth_window=params["smooth_window"],
        z_on=params["z_on"], z_off=params["z_off"],
        min_dur_ms=params["min_dur_ms"], merge_gap_ms=params["merge_gap_ms"],
        min_peak_rel_deg=params["min_peak_rel_deg"],
        min_peak_abs_deg=params["min_peak_abs_deg"],
        consensus_k=params["consensus_k"],
        penalize_overshoot=params["penalize_overshoot"],
    )
    save_blink_eval_to_csv(
        metrics,
        blink_comparison_out_csv_path,
        method_name=method_name,
        data_id=dataloader.get_cwd(),
        params=params,
    )

def play_blink_annotations(
    dataloader,
    *,
    fps: float = config.get_rgb_fps(),
    window_title: str = "Blink Annotations",
    show_current_value: bool = True,
    annotator: str | None = None,
    # style for annotation visuals
    ann_fill_alpha: float = 0.22,
    ann_line_width: float = 1.6,
    ann_line_alpha: float = 0.95,
):
    """
    Interactive player that visualizes ONLY manual blink annotations over time.

    Left  : head crop video
    Right : time axis with CLOSED segments shaded; optional vertical lines at starts/ends

    Controls:
      SPACE: pause/resume
      ←/→  : step 1 frame (paused)
      ↑/↓  : jump 30 frames (paused)
      q    : quit
    """
    import os
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2 as cv

    # --- blink label constants (mirrors blink_annotation.py) ---
    NOT_LABELED     = 0
    LABEL_NO_BLINK  = 1
    LABEL_UNCERTAIN = 2
    LABEL_BLINK     = 3

    if annotator is None:
        print("play_blink_annotations: please provide 'annotator' to load annotations.")
        return

    # --- load frames & timestamps ---
    frames = dataloader.load_rgb_video(as_numpy=True)  # list[np.ndarray], BGR
    if len(frames) == 0:
        print("No RGB frames.")
        return
    N_frames = len(frames)

    try:
        rgb_ts = dataloader.load_rgb_timestamps().reshape(-1)
        if rgb_ts.size == 0:
            raise ValueError("empty timestamps")
        t_ms_full = rgb_ts - rgb_ts[0]
    except Exception:
        # Fallback to uniform spacing from fps
        dt_ms = 1000.0 / max(1e-6, fps)
        t_ms_full = np.arange(N_frames, dtype=np.float64) * dt_ms

    # --- load manual annotations ---
    ann = None
    try:
        ann = dataloader.get_blink_annotations(annotator)
    except Exception:
        try:
            save_path = os.path.join(dataloader.get_cwd(), f"blink_annotations_by_{annotator}.npy")
            if os.path.exists(save_path):
                ann = np.load(save_path)
        except Exception:
            ann = None

    if not isinstance(ann, np.ndarray) or ann.size == 0:
        print(f"No annotations found for annotator '{annotator}'.")
        return

    # Align lengths
    N = int(min(N_frames, ann.shape[0], t_ms_full.shape[0]))
    frames = frames[:N]
    t_ms   = t_ms_full[:N]
    ann    = ann.astype(np.int16)[:N]

    # Map labels to CLOSED mask:
    # 0 (unlabeled) -> open (treat as LABEL_NO_BLINK)
    # 1 -> open, 2 -> closed, 3 -> closed
    mapped = ann.copy()
    mapped[mapped == NOT_LABELED] = LABEL_NO_BLINK
    closed_mask = (mapped == LABEL_BLINK) | (mapped == LABEL_UNCERTAIN)

    # Derive start/end indices for optional vertical lines
    cm = closed_mask.astype(np.int8)
    diff = np.diff(cm, prepend=0, append=0)
    ann_start_idx = np.flatnonzero(diff == 1)
    ann_end_idx   = np.flatnonzero(diff == -1) - 1
    ann_end_idx   = ann_end_idx[ann_end_idx >= 0]

    # --- head crops (fallback to full frame) ---
    head_bboxes = dataloader.load_head_bboxes()
    h0, w0 = frames[0].shape[:2]
    if not (isinstance(head_bboxes, np.ndarray) and head_bboxes.ndim == 2 and head_bboxes.shape[0] >= N):
        head_bboxes = np.tile([0, 0, w0, h0, -1, -1], (N, 1))

    # --- plotting setup ---
    plt.ion()
    fig = plt.figure(figsize=(13, 6))
    try:
        fig.canvas.manager.set_window_title(window_title)
    except Exception:
        pass

    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5], wspace=0.25)
    ax_img  = fig.add_subplot(gs[0, 0])
    ax_plot = fig.add_subplot(gs[0, 1])

    # Left: first crop
    ax_img.axis('off')
    x1, y1, x2, y2, *_ = map(int, head_bboxes[0])
    crop0 = frames[0][max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop0.size == 0:
        crop0 = frames[0]
    im_artist = ax_img.imshow(cv.cvtColor(crop0, cv.COLOR_BGR2RGB))
    ax_img.set_title("Head Crop")

    # Right: only annotations
    ax_plot.set_title(f"Blink Annotations (annotator: {annotator})")
    ax_plot.set_xlabel("Time (ms)")
    ax_plot.set_yticks([])
    ax_plot.set_ylim(0.0, 1.0)  # binary band for fills
    ax_plot.set_xlim(0, float(t_ms[-1]))
    ax_plot.grid(True, axis="x", alpha=0.3)

    # Pre-draw vertical lines at starts/ends (static)
    for i0 in ann_start_idx:
        if 0 <= i0 < N:
            ax_plot.axvline(float(t_ms[i0]), color='r', linestyle='-', linewidth=ann_line_width, alpha=ann_line_alpha)
    for i1 in ann_end_idx:
        if 0 <= i1 < N:
            ax_plot.axvline(float(t_ms[i1]), color='r', linestyle='-', linewidth=ann_line_width, alpha=ann_line_alpha)

    # Shaded fills up to current index (we’ll redraw progressively)
    ann_fill = None

    # HUD
    txt = None
    if show_current_value:
        txt = ax_plot.text(
            0.02, 0.95, "", transform=ax_plot.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85)
        )

    # --- controls/state ---
    idx = 0
    paused = False
    step_once = 0

    def on_key(event):
        nonlocal paused, step_once, idx
        if event.key == ' ':
            paused = not paused
        elif event.key in ('q', 'escape'):
            plt.close(fig)
        elif paused:
            if event.key == 'right': step_once = +1
            elif event.key == 'left':  step_once = -1
            elif event.key == 'up':    step_once = +30
            elif event.key == 'down':  step_once = -30

    fig.canvas.mpl_connect('key_press_event', on_key)

    frame_period = 1.0 / max(1e-6, fps)
    last_time = time.time()

    def _draw_annotations_up_to(k: int):
        nonlocal ann_fill
        # Remove prior fill
        if ann_fill is not None:
            try: ann_fill.remove()
            except Exception: pass
            ann_fill = None

        t = t_ms[:k+1]
        m = closed_mask[:k+1]
        # Fill 0..1 band where closed == True
        ann_fill = ax_plot.fill_between(
            t, 0.0, 1.0, where=m, step="pre", alpha=ann_fill_alpha, color="red"
        )

    # --- main loop ---
    while plt.fignum_exists(fig.number):
        # Left: crop
        x1, y1, x2, y2, *_ = map(int, head_bboxes[idx])
        crop = frames[idx][max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            crop = frames[idx]
        im_artist.set_data(cv.cvtColor(crop, cv.COLOR_BGR2RGB))

        # Right: annotations only
        _draw_annotations_up_to(idx)
        ax_plot.set_xlim(0, float(t_ms[idx]) if idx > 0 else float(t_ms[0]))

        if txt is not None:
            txt.set_text(
                "t = {:.0f} ms\nannotated: {}".format(
                    float(t_ms[idx]),
                    "CLOSED" if closed_mask[idx] else "open"
                )
            )

        fig.canvas.draw_idle()
        plt.pause(0.001)

        # Advance
        if paused:
            if step_once != 0:
                idx = int(np.clip(idx + step_once, 0, N - 1))
                step_once = 0
            else:
                time.sleep(0.01)
        else:
            now = time.time()
            to_wait = frame_period - (now - last_time)
            if to_wait > 0:
                time.sleep(to_wait)
            last_time = time.time()
            idx += 1
            if idx >= N:
                break

    plt.ioff()
    plt.close(fig)


def save_agg_results_to_json(agg: dict, out_path: str) -> str:
    """
    Save aggregated blink evaluation results (from aggregate_blink_eval_csv)
    to a JSON file.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    return out_path

# ----------------
# Example runnable
# ----------------
if __name__ == "__main__":
    annotator = "gorkem" # update this to the annotator's name
    base_dir = config.get_dataset_base_directory()
    exp_dir = f"{base_dir}/2025-08-06/subj_xxxx/circular_movement/p2" # set this to the experiment directory you want
    fps = config.get_rgb_fps() // 1 # play at 30 // 6 = 5 FPS to precisely check annotations.
    dataloader = GazeDataLoader(
        root_dir=exp_dir,
        target_period=config.get_target_period(),
        camera_pose_period=config.get_camera_pose_period(),
        time_diff_max=config.get_time_diff_max(),
        get_latest_subdirectory_by_name=True
    )
    play_blink_annotations( # shows blink periods in red.
        dataloader=dataloader,
        fps=fps,
        annotator=annotator,
        window_title="Blink Annotations",
        show_current_value=True,
        ann_fill_alpha=0.22,
        ann_line_width=1.6,
        ann_line_alpha=0.95,
    )
    exit()

    method_name="3_models_min_peak_rel_deg_5_smooth_window_3_merge_gap_ms_400_z_on_2_0"
    params={
        "models": ["puregaze_rectification","gazetr_rectification", "mcgaze_clip_size_3"],
        "annotator": "berk",
        "mask_mode": "consensus",
        "window_ms": 1000,
        "smooth_window": 3,
        "z_on": 2.0, "z_off": 1.0,
        "min_dur_ms": 0, "merge_gap_ms": 400,
        "min_peak_rel_deg": 5,
        "min_peak_abs_deg": None,
        "consensus_k": None,
        "penalize_overshoot": True,
        # you can add 'axis', 'model_for_mask', etc., if relevant for the run
    }
    blink_comparison_out_dir = f"{base_dir}/blink_comparison_results"
    blink_comparison_out_csv_path = f"{blink_comparison_out_dir}/{method_name}_blink_eval_runs.csv"
    blink_comparison_agg_results_path = f"{blink_comparison_out_dir}/{method_name}_blink_eval_agg.json"

    for subject_dir in subject_dirs:
        exp_dirs += config.get_all_exp_directories_under_a_subject_directory(subject_dir)
    for exp_dir in exp_dirs:
        dataloader = GazeDataLoader(
            root_dir=exp_dir,
            target_period=config.get_target_period(),
            camera_pose_period=config.get_camera_pose_period(),
            time_diff_max=config.get_time_diff_max(),
            get_latest_subdirectory_by_name=True
        )
        blink_comparison_per_dataloader(
            dataloader=dataloader,
            blink_comparison_out_csv_path=blink_comparison_out_csv_path,
            params=params,
            method_name=method_name,
        )
    agg = aggregate_blink_eval_csv(
        [blink_comparison_out_csv_path],
    )

    save_agg_results_to_json(agg, blink_comparison_agg_results_path)

    import pprint
    pprint.pprint(agg) # TODO: later save this to a csv rather than printing
    exit()




    # ---- Blink evaluation entry point ----
    metrics = evaluate_blink_overlap_metrics_dev_local(
        dataloader=dataloader,
        models=["puregaze_rectification","gazetr_rectification", "mcgaze_clip_size_3"],
        annotator="berk",
        # prediction mask selection:
        mask_mode="model",     # "consensus" | "union" | "intersection" | "model"
        model_for_mask="mcgaze_clip_size_3",       # used if mask_mode == "model"
        # LOCAL stats params:
        window_ms=1000,            # try 600–1500 depending on motion
        smooth_window=3,
        z_on=2.0, z_off=1.0,
        min_dur_ms=0, merge_gap_ms=200,
        min_peak_rel_deg=5,
        min_peak_abs_deg=None,
        consensus_k=None,          # None => majority; or e.g. 2 for 2-of-M
        # IoU convention
        penalize_overshoot=True
    )
    import pprint
    pprint.pprint(metrics["summary"])


    fps = config.get_rgb_fps() // 2
    play_blink_inspector_with_dev_local(
        dataloader=dataloader,
        models= ["mcgaze_clip_size_3"],# ["puregaze_rectification","gazetr_rectification","l2cs","mcgaze_clip_size_3"],
        fps=fps,
        # LOCAL stats params:
        window_ms=1000,       # try 600–1500 depending on motion
        smooth_window=3,
        z_on=2.0, z_off=1.0,
        min_dur_ms=0, merge_gap_ms=200,
        min_peak_rel_deg=5,
        min_peak_abs_deg=None,
        # consensus:
        consensus_k=None,      # None => majority; or set e.g. 2 for 2-of-M
        annotator="berk",
    )
    exit()

    # analyzes based on deviation in the angular error
    play_blink_inspector_with_blinks_local(
        dataloader=dataloader,
        models= ["puregaze_rectification","gazetr_rectification", "mcgaze_clip_size_3"],# ["puregaze_rectification","gazetr_rectification","l2cs","mcgaze_clip_size_3"],
        fps=fps,
        # LOCAL stats params:
        window_ms=1000,       # try 600–1500 depending on motion
        smooth_window=3,
        z_on=2.0, z_off=1.0,
        min_dur_ms=80, merge_gap_ms=200,
        min_peak_deg=12,      # optional guard; remove if you like
        # consensus:
        consensus_k=None      # None => majority; or set e.g. 2 for 2-of-M
    )
