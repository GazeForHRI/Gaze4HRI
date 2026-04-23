import os
import time
from typing import Optional

import numpy as np
import cv2 as cv

import config
from data_loader import GazeDataLoader


# ----------------------------
# Existing blink labels
# ----------------------------
OLD_NOT_LABELED = 0
OLD_NO_BLINK = 1
OLD_UNCERTAIN = 2
OLD_BLINK = 3


# ----------------------------
# New phase labels
# ----------------------------
PHASE_UNCERTAIN = 0
PHASE_OPEN = 1
PHASE_CLOSED = 2
PHASE_PARTIAL = 3

PHASE_LABEL_TO_TEXT = {
    PHASE_UNCERTAIN: "UNCERTAIN",
    PHASE_OPEN: "OPEN",
    PHASE_CLOSED: "CLOSED",
    PHASE_PARTIAL: "PARTIAL",
}

# cycle: uncertain -> open -> closed -> partial -> uncertain
NEXT_PHASE_LABEL = {
    PHASE_UNCERTAIN: PHASE_OPEN,
    PHASE_OPEN: PHASE_CLOSED,
    PHASE_CLOSED: PHASE_PARTIAL,
    PHASE_PARTIAL: PHASE_UNCERTAIN,
}

# BGR colors
LABEL_TO_BORDER_COLOR = {
    PHASE_UNCERTAIN: (160, 160, 160),   # gray
    PHASE_OPEN: (0, 200, 0),            # green
    PHASE_CLOSED: (0, 0, 220),          # red
    PHASE_PARTIAL: (0, 200, 255),       # yellow-ish
}

HOVER_BORDER_COLOR = (255, 255, 255)
WINDOW_NAME = "Blink Phase Annotator"


def _safe_crop(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox[:4])

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return frame.copy()
    return frame[y1:y2, x1:x2]


def _resize_keep_aspect(img, target_w, target_h):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _draw_text(img, text, org, scale=0.5, color=(255, 255, 255), thickness=1):
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv.LINE_AA)
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv.LINE_AA)


def _draw_hud(img, lines, x=10, y=20, line_h=22):
    out = img.copy()
    for k, line in enumerate(lines):
        _draw_text(out, line, (x, y + k * line_h), scale=0.55)
    return out


def _find_blink_segments(blinks: np.ndarray):
    segments = []
    n = len(blinks)
    i = 0
    while i < n:
        if blinks[i] == OLD_BLINK:
            s = i
            while i + 1 < n and blinks[i + 1] == OLD_BLINK:
                i += 1
            e = i
            segments.append((s, e))
        i += 1
    return segments


def _build_event_windows(blinks: np.ndarray, pad_open: int = 2):
    segments = _find_blink_segments(blinks)
    windows = []
    n = len(blinks)
    for s, e in segments:
        ws = max(0, s - pad_open)
        we = min(n - 1, e + pad_open)
        windows.append({
            "blink_start": s,
            "blink_end": e,
            "win_start": ws,
            "win_end": we,
            "indices": list(range(ws, we + 1)),
        })
    return windows


def _initialize_phase_labels_from_old(blinks: np.ndarray):
    phase = np.full_like(blinks, PHASE_UNCERTAIN, dtype=np.int16)
    phase[blinks == OLD_NO_BLINK] = PHASE_OPEN
    phase[blinks == OLD_BLINK] = PHASE_CLOSED
    phase[blinks == OLD_UNCERTAIN] = PHASE_UNCERTAIN
    phase[blinks == OLD_NOT_LABELED] = PHASE_UNCERTAIN
    return phase.astype(np.int16)


def _make_event_canvas(
    frames,
    head_bboxes,
    phase_labels,
    event_indices,
    visible_start,
    batch_size=5,
    canvas_w=1920,
    canvas_h=1080,
    gap=20,
    hud_h=150,
    bottom_pad=250,
    text_h=90,
    hover_frame_idx=None,
):
    visible_indices = event_indices[visible_start:visible_start + batch_size]
    nvis = len(visible_indices)

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    usable_w = canvas_w - gap * (nvis + 1)
    tile_w = usable_w // max(1, nvis)

    tile_h = canvas_h - hud_h - bottom_pad
    tile_h = max(160, tile_h)

    crop_h = max(80, tile_h - text_h)

    hitboxes = []

    for col, frame_idx in enumerate(visible_indices):
        tile_x1 = gap + col * (tile_w + gap)
        tile_y1 = hud_h
        tile_x2 = tile_x1 + tile_w
        tile_y2 = tile_y1 + tile_h

        crop_area_x1 = tile_x1
        crop_area_y1 = tile_y1
        crop_area_x2 = tile_x2
        crop_area_y2 = tile_y1 + crop_h

        text_y1 = crop_area_y2
        text_y2 = tile_y2

        raw_crop = _safe_crop(frames[frame_idx], head_bboxes[frame_idx])

        # Resize while preserving aspect ratio, but also keep the actual placed rect
        h, w = raw_crop.shape[:2]
        if h == 0 or w == 0:
            placed_crop = np.zeros((crop_h, tile_w, 3), dtype=np.uint8)
            disp_x1, disp_y1, disp_x2, disp_y2 = crop_area_x1, crop_area_y1, crop_area_x2, crop_area_y2
        else:
            scale = min(tile_w / w, crop_h / h)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            resized = cv.resize(raw_crop, (new_w, new_h), interpolation=cv.INTER_LINEAR)

            placed_crop = np.zeros((crop_h, tile_w, 3), dtype=np.uint8)
            off_x = (tile_w - new_w) // 2
            off_y = (crop_h - new_h) // 2
            placed_crop[off_y:off_y + new_h, off_x:off_x + new_w] = resized

            disp_x1 = crop_area_x1 + off_x
            disp_y1 = crop_area_y1 + off_y
            disp_x2 = disp_x1 + new_w
            disp_y2 = disp_y1 + new_h

        label = int(phase_labels[frame_idx])
        label_text = PHASE_LABEL_TO_TEXT.get(label, str(label))
        border_color = LABEL_TO_BORDER_COLOR.get(label, (180, 180, 180))

        # draw crop area
        canvas[crop_area_y1:crop_area_y2, crop_area_x1:crop_area_x2] = placed_crop

        # separator
        cv.line(canvas, (tile_x1, text_y1), (tile_x2, text_y1), (100, 100, 100), 1)

        # text under crop
        frame_text = f"Frame {frame_idx}"
        label_text_line = f"Label: {label_text}"

        (frame_tw, _), _ = cv.getTextSize(frame_text, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        (label_tw, _), _ = cv.getTextSize(label_text_line, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)

        _draw_text(
            canvas,
            frame_text,
            (tile_x1 + (tile_w - frame_tw) // 2, text_y1 + 30),
            scale=0.8,
            thickness=2,
        )
        _draw_text(
            canvas,
            label_text_line,
            (tile_x1 + (tile_w - label_tw) // 2, text_y1 + 65),
            scale=0.8,
            thickness=2,
        )

        # ---- custom outline ----
        # top part follows actual displayed crop
        cv.line(canvas, (disp_x1, disp_y1), (disp_x2, disp_y1), border_color, 4)  # top
        cv.line(canvas, (disp_x1, disp_y1), (disp_x1, text_y1), border_color, 4)  # left upper
        cv.line(canvas, (disp_x2, disp_y1), (disp_x2, text_y1), border_color, 4)  # right upper

        # connect from crop width to full tile width at text boundary
        cv.line(canvas, (tile_x1, text_y1), (disp_x1, text_y1), border_color, 4)
        cv.line(canvas, (disp_x2, text_y1), (tile_x2, text_y1), border_color, 4)

        # lower part uses full tile width
        cv.line(canvas, (tile_x1, text_y1), (tile_x1, tile_y2), border_color, 4)
        cv.line(canvas, (tile_x2, text_y1), (tile_x2, tile_y2), border_color, 4)
        cv.line(canvas, (tile_x1, tile_y2), (tile_x2, tile_y2), border_color, 4)

        # hover outline with same shape
        if hover_frame_idx is not None and frame_idx == hover_frame_idx:
            hover_pad = 3
            hx1, hy1, hx2, hy2 = disp_x1 - hover_pad, disp_y1 - hover_pad, disp_x2 + hover_pad, disp_y2 + hover_pad
            tx1, ty1, tx2, ty2 = tile_x1 - hover_pad, text_y1, tile_x2 + hover_pad, tile_y2 + hover_pad

            cv.line(canvas, (hx1, hy1), (hx2, hy1), HOVER_BORDER_COLOR, 2)
            cv.line(canvas, (hx1, hy1), (hx1, ty1), HOVER_BORDER_COLOR, 2)
            cv.line(canvas, (hx2, hy1), (hx2, ty1), HOVER_BORDER_COLOR, 2)

            cv.line(canvas, (tx1, ty1), (hx1, ty1), HOVER_BORDER_COLOR, 2)
            cv.line(canvas, (hx2, ty1), (tx2, ty1), HOVER_BORDER_COLOR, 2)

            cv.line(canvas, (tx1, ty1), (tx1, ty2), HOVER_BORDER_COLOR, 2)
            cv.line(canvas, (tx2, ty1), (tx2, ty2), HOVER_BORDER_COLOR, 2)
            cv.line(canvas, (tx1, ty2), (tx2, ty2), HOVER_BORDER_COLOR, 2)

        hitboxes.append((tile_x1, tile_y1, tile_x2, tile_y2, frame_idx))

    return canvas, hitboxes, visible_indices


def _confirm_save(event_count: int, save_path: str) -> bool:
    panel = np.zeros((220, 1100, 3), dtype=np.uint8)
    lines = [
        "Save phase annotations?",
        f"Blink event windows found: {event_count}",
        f"Output: {save_path}",
        "Press Y to confirm, N or Esc to cancel",
    ]
    y = 40
    for line in lines:
        _draw_text(panel, line, (25, y), scale=0.8)
        y += 45

    cv.imshow("Confirm Save", panel)
    while True:
        key = cv.waitKey(1) & 0xFF
        if key in (ord("y"), ord("Y")):
            cv.destroyWindow("Confirm Save")
            return True
        if key in (ord("n"), ord("N"), 27):
            cv.destroyWindow("Confirm Save")
            return False


def annotate_blink_phases(
    dataloader: GazeDataLoader,
    annotator: str,
    batch_size: int = 5,
    pad_open: int = 2,
) -> str:
    """
    Mouse:
    left click on crop  -> cycle label
    right click         -> go to next event
    mouse wheel up/down -> slide visible batch left/right

    Keyboard:
    a / d        : shift visible batch left/right by 1 crop
    A / D        : shift visible batch left/right by 5 crops
    j / l        : previous/next event window
    home / end   : first/last event
    n            : next event
    p            : previous event
    enter        : next event
    s            : save & quit
    q / esc      : quit without saving
    """

    frames = dataloader.load_rgb_video(as_numpy=True)
    if len(frames) == 0:
        raise RuntimeError("No frames loaded.")

    num_frames = len(frames)

    head_bboxes = dataloader.load_head_bboxes()
    if head_bboxes is None or head_bboxes.shape[0] != num_frames:
        raise RuntimeError("Head bboxes are required for this annotation mode.")

    old_blinks = dataloader.get_blink_annotations(annotator).astype(np.int16)
    if old_blinks.shape[0] != num_frames:
        raise RuntimeError("Blink annotation length mismatch with video length.")

    event_windows = _build_event_windows(old_blinks, pad_open=pad_open)
    if len(event_windows) == 0:
        raise RuntimeError("No blink events found in existing annotations.")

    save_dir = dataloader.get_cwd()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"blink_phase_annotations_by_{annotator}.npy")

    if os.path.exists(save_path):
        phase_labels = np.load(save_path).astype(np.int16)
        if phase_labels.shape[0] != num_frames:
            new_phase = _initialize_phase_labels_from_old(old_blinks)
            nmin = min(num_frames, phase_labels.shape[0])
            new_phase[:nmin] = phase_labels[:nmin]
            phase_labels = new_phase
    else:
        phase_labels = _initialize_phase_labels_from_old(old_blinks)

    current_event_idx = 0
    visible_start = 0
    current_hitboxes = []
    hover_frame_idx = None

    state = {
        "phase_labels": phase_labels,
    }

    # total frames that belong to event windows
    total_event_frames = sum(len(ev["indices"]) for ev in event_windows)

    def reset_visible_start():
        nonlocal visible_start
        event = event_windows[current_event_idx]
        event_len = len(event["indices"])
        visible_start = max(0, min(visible_start, max(0, event_len - batch_size)))

    def save_now():
        np.save(save_path, state["phase_labels"].astype(np.int16))
        print(f"[Saved] {save_path}")

    def goto_next_event():
        nonlocal current_event_idx, visible_start
        current_event_idx = min(len(event_windows) - 1, current_event_idx + 1)
        visible_start = 0

    def goto_prev_event():
        nonlocal current_event_idx, visible_start
        current_event_idx = max(0, current_event_idx - 1)
        visible_start = 0

    def slide_left(step=1):
        nonlocal visible_start
        visible_start = max(0, visible_start - step)

    def slide_right(event_len, step=1):
        nonlocal visible_start
        visible_start = min(max(0, event_len - batch_size), visible_start + step)

    def find_hovered_frame(x, y):
        for x1, y1, x2, y2, frame_idx in current_hitboxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return frame_idx
        return None

    def on_mouse(event, x, y, flags, param):
        nonlocal current_hitboxes, hover_frame_idx

        hover_frame_idx = find_hovered_frame(x, y)

        # Left click: relabel only
        if event == cv.EVENT_LBUTTONDOWN:
            frame_idx = find_hovered_frame(x, y)
            if frame_idx is not None:
                cur = int(state["phase_labels"][frame_idx])
                nxt = NEXT_PHASE_LABEL[cur]
                state["phase_labels"][frame_idx] = np.int16(nxt)

        # Right click: go next event
        elif event == cv.EVENT_RBUTTONDOWN:
            goto_next_event()

        # Ignore wheel events entirely
        elif event == cv.EVENT_MOUSEWHEEL:
            pass

    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    cv.resizeWindow(WINDOW_NAME, 1920, 1080)
    cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        while True:
            event = event_windows[current_event_idx]
            event_indices = event["indices"]
            event_len = len(event_indices)
            reset_visible_start()

            canvas, hitboxes, visible_indices = _make_event_canvas(
                frames=frames,
                head_bboxes=head_bboxes,
                phase_labels=state["phase_labels"],
                event_indices=event_indices,
                visible_start=visible_start,
                batch_size=batch_size,
                hover_frame_idx=hover_frame_idx,
            )
            current_hitboxes = hitboxes

            blink_start = event["blink_start"]
            blink_end = event["blink_end"]

            remaining_in_event = max(0, event_len - (visible_start + len(visible_indices)))
            remaining_events = len(event_windows) - (current_event_idx + 1)

            # Remaining work = all frames in current window + all later windows
            remaining_total_frames = sum(len(ev["indices"]) for ev in event_windows[current_event_idx:])
            remaining_pct = (100.0 * remaining_total_frames / total_event_frames) if total_event_frames > 0 else 0.0

            hud = [
                f"Event {current_event_idx + 1}/{len(event_windows)} | blink segment: [{blink_start}, {blink_end}] | event window: [{event['win_start']}, {event['win_end']}]",
                f"Visible batch: {visible_start + 1}-{min(visible_start + batch_size, event_len)} / {event_len} | Remaining in event: {remaining_in_event}",
                f"Remaining events: {remaining_events} | Remaining work: {remaining_total_frames}/{total_event_frames} frames ({remaining_pct:.1f}%)",
                "Mouse: hover=highlight, left click=cycle label, right click=next event",
                "Cycle: UNCERTAIN -> OPEN -> CLOSED -> PARTIAL -> UNCERTAIN",
                "Keys: a/d=slide 1, A/D=slide 5, j/l or p/n=prev/next event, Enter=next event, Home/End=first/last, s=save&quit, q/Esc=quit",
            ]
            display = _draw_hud(canvas, hud, x=10, y=22, line_h=22)

            cv.imshow(WINDOW_NAME, display)
            key = cv.waitKey(20) & 0xFFFF

            if key == 0xFFFF:
                continue

            if key in (ord("q"), 27):
                break

            if key in (ord("s"), ord("S")):
                if _confirm_save(len(event_windows), save_path):
                    save_now()
                    break
                continue

            if key in (ord("j"), ord("p")):
                goto_prev_event()
                continue

            if key in (ord("l"), ord("n"), 13):
                goto_next_event()
                continue

            if key == ord("a"):
                slide_left(1)
                continue

            if key == ord("d"):
                slide_right(event_len, 1)
                continue

            if key == ord("A"):
                slide_left(5)
                continue

            if key == ord("D"):
                slide_right(event_len, 5)
                continue

            if key == 65360:  # Home
                current_event_idx = 0
                visible_start = 0
                continue

            if key == 65367:  # End
                current_event_idx = len(event_windows) - 1
                visible_start = 0
                continue

    except KeyboardInterrupt:
        print("\n[Interrupted] Ctrl+C detected.")
    finally:
        cv.destroyAllWindows()

    return save_path


if __name__ == "__main__":
    annotator = "gorkem"
    base_dir = config.get_dataset_base_directory()
    subject_dir = f"{base_dir}/2025-08-06/subj_xxxx"

    exp_dirs = config.get_all_exp_directories_under_a_subject_directory(subject_dir)
    exp_dirs = list(reversed(exp_dirs))

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

            out_path = annotate_blink_phases(
                dataloader=dataloader,
                annotator=annotator,
                batch_size=5,
                pad_open=2,
            )

            anns = np.load(out_path)
            print(f"\nSaved phase annotations to:\n{out_path}")
            print("Unique labels in saved file:", np.unique(anns))

    except KeyboardInterrupt:
        print("\n[Interrupted] Ctrl+C detected — stopping batch.")
    finally:
        cv.destroyAllWindows()

    duration = (time.time() - start_time) / 60.0
    print(f"duration: {duration:.1f} minutes")