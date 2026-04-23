import os
import sys
import argparse
from typing import Optional

import numpy as np
import cv2 as cv

# NEW: for detecting Shift being held
from pynput import keyboard  # pip install pynput

# Local imports (your files)
import config
from data_loader import GazeDataLoader

NOT_LABELED = 0 # not labeled by the annotator.
LABEL_NO_BLINK = 1    # open eyes
LABEL_UNCERTAIN = 2
LABEL_BLINK = 3       # closed eyes

KEY_TO_LABEL = {
    ord('1'): LABEL_NO_BLINK,
    ord('2'): LABEL_UNCERTAIN,
    ord('3'): LABEL_BLINK,
}

WINDOW_NAME = "Blink Annotator"


def _draw_hud(img, text_lines, anchor=(10, 20)):
    """Draw multiline HUD text on BGR image."""
    x, y = anchor
    for i, line in enumerate(text_lines):
        cv.putText(
            img, line, (x, y + 18 * i),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv.LINE_AA
        )
        cv.putText(
            img, line, (x, y + 18 * i),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA
        )
    return img


def _safe_crop(frame, bbox):
    """Crop with clamping to image bounds. bbox: [x1,y1,x2,y2,...]."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox[:4])
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return frame  # fallback
    return frame[y1:y2, x1:x2]


def _center_on_canvas(crop: np.ndarray, canvas_wh: tuple[int, int]) -> np.ndarray:
    """Place crop at the center of a black canvas of size (W,H)."""
    W, H = canvas_wh
    canvas = np.zeros((H, W, 3), dtype=crop.dtype)
    ch, cw = crop.shape[:2]
    x0 = max(0, (W - cw) // 2)
    y0 = max(0, (H - ch) // 2)
    x1 = min(W, x0 + cw)
    y1 = min(H, y0 + ch)
    canvas[y0:y1, x0:x1] = crop[0:(y1 - y0), 0:(x1 - x0)]
    return canvas

def _first_unlabeled_indices(blinks: np.ndarray, k: int = 10) -> np.ndarray:
    """Return first k indices (0-based) that are NOT_LABELED."""
    return np.flatnonzero(blinks == NOT_LABELED)[:k]

def _confirm_exit(blinks: np.ndarray, want_save: bool) -> bool:
    """
    If there are unlabeled frames, show a confirmation panel and
    return True to proceed, False to cancel. If none, returns True.
    """
    unlabeled_idx = np.flatnonzero(blinks == NOT_LABELED)
    if unlabeled_idx.size == 0:
        return True  # no need to confirm

    # Build a small panel window with info
    panel_h, panel_w = 380, 1000
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    title = "Confirm: Save & Quit" if want_save else "Confirm: Quit WITHOUT Saving"
    lines = [
        title,
        f"Unlabeled frames: {unlabeled_idx.size} / {blinks.size}",
        "First {} (1-based) unlabeled frames: {}".format(
            min(10, unlabeled_idx.size),
            ", ".join(str(i + 1) for i in unlabeled_idx[:10])
        ),
        "",
        "Press Y to confirm, N (or Esc) to cancel"
    ]

    # Draw text
    y = 40
    for li, line in enumerate(lines):
        # bold white with black outline
        cv.putText(panel, line, (30, y), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv.LINE_AA)
        cv.putText(panel, line, (30, y), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA)
        y += 60 if li in (0, 3) else 40

    cv.imshow("Confirm Exit", panel)

    while True:
        k = cv.waitKey(1) & 0xFF  # 1ms polling so Ctrl+C can interrupt
        if k in (ord('y'), ord('Y')):
            cv.destroyWindow("Confirm Exit")
            return True
        if k in (ord('n'), ord('N'), 27):  # N or Esc cancels
            cv.destroyWindow("Confirm Exit")
            return False


def annotate_blinks(
    dataloader: GazeDataLoader,
    annotator: str,
    use_head_crops: bool = False,
    playback_fps: Optional[float] = 30.0,
    center_crop_fullscreen: bool = False,
) -> str:
    """
    Open a playback window and annotate blinks frame-by-frame.

    Saves: <cwd>/blink_annotations_by_{annotator}.npy  (int16, shape=(num_frames,))
    Labels: 1.0=no blink, 2.0=uncertain, 3.0=blink. Unlabeled frames are NOT_LABELED until set.

    Controls:
      Space/k: play/pause
      j / l: ±1 frame
      Shift+j / Shift+l (J/L): ±10 frames
      , / .: ±10 frames (extra)
      Home/End: jump to start/end
      1/2/3: label current frame (1=no blink, 2=uncertain, 3=blink)
      [: mark range start, ]: mark range end and apply last class to range
      c: toggle head-crop view (if head_bboxes available)
      s: save & quit (ALWAYS)
      q or Esc: quit WITHOUT saving (ALWAYS)
      Hold Shift: **2× playback while playing** (no effect when paused)

    End-of-video: playback stops advancing and waits; same s/q behavior applies.
    Returns:
      The path to the saved .npy file.
    """
    # --- Load frames & timestamps ---
    frames = dataloader.load_rgb_video(as_numpy=True)
    if len(frames) == 0:
        raise RuntimeError("No frames loaded from video. Check your data directory.")
    num_frames = len(frames)
    frame_h, frame_w = frames[0].shape[:2]

    # Try timestamps (ms). If absent, fallback to synthetic timestamps.
    try:
        rgb_ts = dataloader.load_rgb_timestamps().astype(np.float64).reshape(-1)
        if len(rgb_ts) != num_frames:
            raise ValueError("Timestamp length mismatch.")
        ts = rgb_ts
    except Exception:
        period = 1000.0 / (playback_fps if playback_fps and playback_fps > 0 else 30.0)
        ts = np.arange(num_frames, dtype=np.float64) * period

    # Optional head bboxes for crop visualization
    head_bboxes = None
    try:
        head_bboxes = dataloader.load_head_bboxes()
        if head_bboxes is None or head_bboxes.shape[0] != num_frames:
            head_bboxes = None
    except Exception:
        head_bboxes = None

    # --- Initialize/Load annotations ---
    save_dir = dataloader.get_cwd()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"blink_annotations_by_{annotator}.npy")

    if os.path.exists(save_path):
        blinks = np.load(save_path).astype(np.int16)
        if blinks.shape[0] != num_frames:
            new_blinks = np.full((num_frames,), NOT_LABELED, dtype=np.int16)
            nmin = min(num_frames, blinks.shape[0])
            new_blinks[:nmin] = blinks[:nmin]
            blinks = new_blinks
    else:
        # Use unlabeled to indicate "unlabeled" until set; saved as int16
        blinks = np.full((num_frames,), NOT_LABELED, dtype=np.int16)

    # --- UI state ---
    i = 0
    playing = True
    use_crops = use_head_crops and (head_bboxes is not None)
    last_class: Optional[int] = None

    # CHANGED: we now track a pending blink-start index
    blink_start_idx: Optional[int] = None

    step_small = 1
    step_big = 10
    delay_ms = int(round(1000.0 / (playback_fps if playback_fps and playback_fps > 0 else 30.0)))

    # Window
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    if center_crop_fullscreen and use_crops:
        try:
            cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        except Exception:
            pass  # some backends may not support fullscreen

    def save_now():
        np.save(save_path, blinks.astype(np.int16))
        print(f"[Saved] {save_path}")

    def apply_label(idx: int, label: int):
        nonlocal last_class
        blinks[idx] = np.int16(label)
        last_class = int(label)

    def apply_range(i0: int, i1: int, label: int):
        a = max(0, min(i0, i1))
        b = min(num_frames - 1, max(i0, i1))
        blinks[a:b + 1] = np.int16(label)

    def apply_range(i0: int, i1: int, label: float):
        a = max(0, min(i0, i1))
        b = min(num_frames - 1, max(i0, i1))
        blinks[a:b + 1] = np.int16(label)

    playing = False # do not start playing until user manually starts playback.

    # --- NEW: 2× speed via Shift key (hold to speed up while playing) ---
    shift_held = False

    def _on_press(key):
        nonlocal shift_held
        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            shift_held = True

    def _on_release(key):
        nonlocal shift_held
        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            shift_held = False

    kb_listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    kb_listener.daemon = True
    kb_listener.start()

    # --- Main loop ---
    try:
        while True:
            frame = frames[i]
            shown = frame
            # if current frame is unlabeled (0), auto-set to OPEN (1) on visit/play
            if blinks[i] == NOT_LABELED:
                blinks[i] = np.int16(LABEL_NO_BLINK)
            if use_crops:
                if head_bboxes is not None:
                    crop = _safe_crop(frame, head_bboxes[i])
                    if center_crop_fullscreen:
                        # Center crop on a black canvas of original frame size
                        shown = _center_on_canvas(crop, (frame_w, frame_h))
                    else:
                        shown = crop

            cur_label = int(blinks[i])
            if cur_label == NOT_LABELED:
                cur_label_str = "—"
            elif cur_label == LABEL_NO_BLINK:
                cur_label_str = "NO BLINK (1)"
            elif cur_label == LABEL_UNCERTAIN:
                cur_label_str = "UNCERTAIN (2)"
            elif cur_label == LABEL_BLINK:
                cur_label_str = "BLINK (3)"
            else:
                cur_label_str = str(cur_label)

            labeled_pct = (np.count_nonzero(blinks != NOT_LABELED) * 100.0) / num_frames
            last_key_str = (
                "NO BLINK" if last_class == LABEL_NO_BLINK else
                "UNCERTAIN" if last_class == LABEL_UNCERTAIN else
                "BLINK" if last_class == LABEL_BLINK else "None"
            )
            at_end = (i == num_frames - 1)
            # CHANGED: reflect pending blink start
            range_str = "None" if blink_start_idx is None else f"blink_start={blink_start_idx}"

            hud = [
                f"Frame {i+1}/{num_frames} | t={ts[i]:.1f} ms" + ("  [END]" if at_end else ""),
                f"Playback: {'PLAY' if playing else 'PAUSE'} | Speed: {'2×' if (playing and shift_held) else '1×'} (hold Shift)",  # NEW
                f"Current label: {cur_label_str}",
                f"Last class key: {last_key_str}",
                f"Pending: {range_str}",
                f"Mode: {'HEAD-CROP' if use_crops else 'FULL'}"
                + (", Centered Fullscreen" if (use_crops and center_crop_fullscreen) else "")
                + f" | Labeled: {labeled_pct:.1f}%",
                # Updated help text with Shift=2×
                "Keys: Space/k=Play/Pause, j/l=±1, J/L or ,/.=±10, 1/2/3=label, [=blink_start ]=blink_end→CLOSED(3), c=crop, s=save&quit, q/Esc=quit(no save), hold Shift=2× (while playing)",
                "End state: playback stops; s=save&quit, q/Esc=quit(no save)"
            ]

            shown_hud = shown.copy()
            _draw_hud(shown_hud, hud)
            cv.imshow(WINDOW_NAME, shown_hud)

            # Advance if playing (stop at last frame)
            if playing and not at_end:
                i = min(num_frames - 1, i + 1)

            # Wait logic — halve the delay to get 2× when Shift is held (only when playing)
            speed_mult = 2 if (playing and shift_held) else 1
            wait_time = 1 if (not playing or at_end) else max(1, int(round(delay_ms / speed_mult)))
            try:
                key = cv.waitKey(wait_time) & 0xFFFF
            except KeyboardInterrupt:
                # Ctrl+C during waitKey: exit immediately without saving
                try:
                    kb_listener.stop()
                except Exception:
                    pass
                cv.destroyAllWindows()
                return save_path
            if key == 0xFFFF:
                # No key pressed in paused/end wait; continue loop
                continue

            # --- Global quit/save behavior (ALWAYS, with confirmation if unlabeled) ---
            if key in (ord('q'), 27):  # q or ESC => quit WITHOUT saving
                if _confirm_exit(blinks, want_save=False):
                    break
                else:
                    continue  # back to annotator
            if key in (ord('s'), ord('S')):  # s => SAVE and quit
                if _confirm_exit(blinks, want_save=True):
                    save_now()
                    break
                else:
                    continue

            # Toggle play/pause (Space or k/K)
            if key in (32, ord('k'), ord('K')):
                # If at end and currently playing, toggling to playing would do nothing,
                # but we'll allow toggling to paused; user can navigate or label.
                playing = not playing
                continue

            # Crop toggle
            if key in (ord('c'), ord('C')):
                if head_bboxes is not None:
                    use_crops = not use_crops
                    if center_crop_fullscreen and use_crops:
                        try:
                            cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
                        except Exception:
                            pass
                    else:
                        try:
                            cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)
                        except Exception:
                            pass
                else:
                    print("[Info] head_bboxes not found; staying on full-frame.")
                continue

            # Navigation: j (prev), l (next) by 1 frame
            if key == ord('j'):
                i = max(0, i - 1)
                playing = False
                continue
            if key == ord('l'):
                i = min(num_frames - 1, i + 1)
                playing = False
                continue

            # Shift+j / Shift+l (uppercase J/L): ±10 frames
            if key == ord('J'):
                i = max(0, i - 10)
                playing = False
                continue
            if key == ord('L'):
                i = min(num_frames - 1, i + 10)
                playing = False
                continue

            # Big steps with , and .
            if key == ord(','):
                i = max(0, i - 10)
                playing = False
                continue
            if key == ord('.'):
                i = min(num_frames - 1, i + 10)
                playing = False
                continue

            # Home/End (jump)
            if key == 65360:  # Home
                i = 0
                playing = False
                continue
            if key == 65367:  # End
                i = num_frames - 1
                playing = False
                continue

            # Label keys
            if key in KEY_TO_LABEL:
                apply_label(i, KEY_TO_LABEL[key])
                continue

            # Blink range selection via brackets:
            if key == ord('['):  # blink_start
                blink_start_idx = i
                continue

            if key == ord(']'):  # blink_end → apply CLOSED (3) over [start, end]
                if blink_start_idx is not None:
                    apply_range(blink_start_idx, i, LABEL_BLINK)
                    blink_start_idx = None
                else:
                    # no start set; you can optionally beep/log, but harmless to ignore
                    pass
                continue

            # Fallback: ignore unknown keys
    except KeyboardInterrupt:
        # Ctrl+C during annotation: exit immediately without saving
        print("\n[Interrupted] Ctrl+C detected — exiting without saving current video.")
    finally:
        # Always stop the keyboard listener
        try:
            kb_listener.stop()
        except Exception:
            pass

    cv.destroyAllWindows()
    return save_path


if __name__ == "__main__":
    annotator = "gorkem" # update this to the annotator's name
    base_dir = config.get_dataset_base_directory()
    use_head_crops = True  # use head crops if they exist, defaults to full frame if they do not exist.
    fps = config.get_rgb_fps()
    subject_dir = f"{base_dir}2025-08-06/subj_xxxx/" # set this to the experiment directory you want to annotate.
    exp_dirs = config.get_all_exp_directories_under_a_subject_directory(subject_dir)
    exp_dirs = list(reversed(exp_dirs))  # reverse order so that we do lighting_10 as the last exp type.
    import time
    start_time = time.time()
    try:
        for exp_dir in exp_dirs:
            dataloader = GazeDataLoader(
                root_dir=exp_dir,
                target_period=config.get_target_period(),
                camera_pose_period=config.get_camera_pose_period(),
                time_diff_max=config.get_time_diff_max(),
                get_latest_subdirectory_by_name=True,
            )

            annotate_blinks(
                dataloader=dataloader,
                annotator=annotator,
                use_head_crops=use_head_crops,
                playback_fps=fps,
                center_crop_fullscreen=True,
            )

            annotations = dataloader.get_blink_annotations(annotator)
            print(f"\nLABEL_NO_BLINK: {LABEL_NO_BLINK}, LABEL_BLINK: {LABEL_BLINK}\nSaved annotations:\n{annotations}")
    except KeyboardInterrupt:
        print("\n[Interrupted] Ctrl+C detected — stopping batch.")
    finally:
        cv.destroyAllWindows()
    duration = (time.time() - start_time) / 60.0 # in minutes
    print(f"duration: {duration:.1f} minutes", )
