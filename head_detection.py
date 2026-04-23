import cv2
from ultralytics import YOLO
import os
import numpy as np
from collections import Counter
from tqdm import tqdm
from ultralytics.utils import LOGGER
from enum import Enum
LOGGER.setLevel("ERROR")

#------------------------------------------------------------

def get_latest_subdirectory_by_name(parent_directory):
    """
    Returns the name of the latest subdirectory based on lexicographical ordering (e.g., timestamp-named dirs).

    Args:
        parent_directory (str): Path to the directory containing timestamp-named subdirectories.

    Raises:
        Exception: If no subdirectories are found.

    Returns:
        str: Name of the latest subdirectory.
    """
    try:
        # List all entries and filter only directories
        subdirs = [d for d in os.listdir(parent_directory)
                   if os.path.isdir(os.path.join(parent_directory, d))]

        if not subdirs:
            raise Exception(f"No subdirectories found in '{parent_directory}'")

        # Sort subdirectories by name (assumes timestamp-naming)
        latest_subdir = max(subdirs)
        return latest_subdir

    except FileNotFoundError:
        raise Exception(f"The directory '{parent_directory}' does not exist.")

DETECTION_BOX_PADDING_MARGIN_WIDTH, DETECTION_BOX_PADDING_MARGIN_HEIGHT = 0.1, 0.1  # Increase detected box width/height by this margin to make sure eyes are included. (Note: 0.1 means increase by 10%, -0.1 means decrease)
CWD = ""
# CWD = os.path.join(CWD, get_latest_subdirectory_by_name(CWD))
RGB_IMAGES_PATH = f"{CWD}/rgb_video.mp4"
HEAD_TRACKED_VIDEO_OUT_PATH = f"{CWD}/head_tracked_output.mp4"
HEAD_CROPS_OUT_PATH = f"{CWD}/head_crops.mp4"
YOLO_PARAMS_PATH = "/home/kovan/USTA/src/robot_controller/robot_controller/models/YOLOv8l/yolov8n-face.pt"

#------------------------------------------------------------
def save_bboxes_to_npy(bbox_array: np.ndarray, output_path: str):
    """
    Saves the bounding boxes as a .npy file.

    Args:
        bbox_array (np.ndarray): Array in [x1, y1, x2, y2, id, frame_idx] format.
        output_path (str): Destination .npy file path.
    """
    np.save(output_path, bbox_array)
    print(f"Bounding boxes saved to: {output_path}")

def track_faces_from_video(video_path: str, 
                        output_video_path: str = HEAD_TRACKED_VIDEO_OUT_PATH,
                        model_path: str = YOLO_PARAMS_PATH, 
                        tracker_cfg: str = "bytetrack.yaml", 
                        device: str = "cuda"
                        ) -> np.ndarray:
    """
    Tracks faces in the video, and returns the bounding boxes and a video where the most frequent face ID is marked.

    Returns:
        (np.ndarray, str): [x1, y1, x2, y2, id, frame_idx] array + video path.
    """

    model = YOLO(model_path)
    model.to(device)

    # Get video properties
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # for tqdm
    cap.release()

    # Run tracking
    results = model.track(
        source=video_path,
        tracker=tracker_cfg,
        stream=True,
        persist=True,
        conf=0.3,
        device=device
    )

    # First pass: collect data and optionally draw
    frame_idx = 0
    all_bboxes = []
    all_ids = []
    frame_dict = {}  # {frame_idx: list of (box, id)}

    with tqdm(total=total_frames, desc="Tracking Faces", unit="frame") as pbar:
        for result in results:
            frame = result.orig_img.copy()
            boxes = result.boxes
            current_frame_data = []

            if boxes is not None and boxes.id is not None:
                ids = boxes.id.cpu().numpy()
                coords = boxes.xyxy.cpu().numpy()
                for box, id_ in zip(coords, ids):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    pad_x = DETECTION_BOX_PADDING_MARGIN_WIDTH * w
                    pad_y = DETECTION_BOX_PADDING_MARGIN_HEIGHT * h

                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(width, x2 + pad_x)
                    y2 = min(height, y2 + pad_y)

                    id_ = int(id_)

                    all_bboxes.append([x1, y1, x2, y2, id_, frame_idx])
                    all_ids.append(id_)
                    current_frame_data.append(([x1, y1, x2, y2], id_))

            frame_dict[frame_idx] = (frame, current_frame_data)
            frame_idx += 1
            pbar.update(1)

    # Determine the most frequent face ID
    if not all_ids:
        return np.empty((0, 6))

    most_common_id = Counter(all_ids).most_common(1)[0][0]

    # Redraw boxes only for the most common ID
    filtered_bboxes = []

    for frame_idx, (frame, detections) in frame_dict.items():
        for box, id_ in detections:
            if id_ == most_common_id:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {id_}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                filtered_bboxes.append([x1, y1, x2, y2, id_, frame_idx])

    return np.array(filtered_bboxes)

#------------------------------------------------------------
def crop_faces_with_bbox(bbox_array: np.ndarray,
                                    video_path: str,
                                    output_path: str,
                                    output_size: tuple,
                                    padding: bool = 0
                                    ) -> None:
    """
    Uses the face bounding boxes for a single ID to crop and save a new video of the cropped face regions.

    Args:
        bbox_array (np.ndarray): Bounding boxes in [x1, y1, x2, y2, id, frame_idx] format.
        video_path (str): Path to the original video.
        output_path (str): Path to the output cropped video.
        output_size (tuple): Output frame size, e.g., (224, 224).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    # Group bboxes by frame index
    bbox_dict = {}
    for bbox in bbox_array:
        idx = int(bbox[5])
        bbox_dict.setdefault(idx, []).append(bbox)

    frame_idx = 0
    with tqdm(total=total_frames, desc="Cropping Faces", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in bbox_dict:
                for bbox in bbox_dict[frame_idx]:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    face_crop = frame[y1:y2, x1:x2]
                    if padding:
                        face_crop = resize_with_padding(face_crop, output_size)
                    else:
                        face_crop = cv2.resize(face_crop, output_size)
                    out_writer.write(face_crop)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    out_writer.release()
    print(f"✅ Cropped face video saved to: {output_path}")

#------------------------------------------------------------

def resize_with_padding(image, target_size=(224, 224)):
    h, w = image.shape[:2]
    target_w, target_h = target_size

    if h == 0 or w == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))

    pad_w = target_w - resized.shape[1]
    pad_h = target_h - resized.shape[0]

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded

class BoundingBoxValidationException(Exception):
    pass

class HeadDetectionStatus(Enum):
    SUCCESSFUL = 1
    UNEXPECTED_ERROR = 2
    VALIDATION_FAILED = 3

def validate_bboxes(bbox_array: np.ndarray, video_path: str):
    """
    Validates that there is exactly one face tracked in every frame of the video.

    Args:
        bbox_array (np.ndarray): Bounding boxes in [x1, y1, x2, y2, id, frame_idx] format.
        video_path (str): Path to the input video.

    Raises:
        Exception: If number of frames with a detection does not match total frame count,
                   or if multiple IDs are present.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if bbox_array.shape[0] != total_frames:
        raise Exception(f"Validation failed: expected {total_frames} bboxes (one per frame), "
                        f"but got {bbox_array.shape[0]}")

    ids = np.unique(bbox_array[:, 4])
    if len(ids) != 1:
        raise Exception(f"Validation failed: expected exactly 1 unique face ID, but got {len(ids)}: {ids}")
    
    print("✅ Bounding box validation passed.")


def save_visualized_bboxes_video(bbox_array: np.ndarray,
                                  video_path: str,
                                  output_path: str,
                                  box_color=(0, 255, 0),
                                  box_thickness: int = 2,
                                  font_scale: float = 0.7,
                                  font_thickness: int = 2):
    """
    Draws the bounding boxes onto the original full video and saves a new visualized video.

    Args:
        bbox_array (np.ndarray): Bounding boxes in [x1, y1, x2, y2, id, frame_idx] format.
        video_path (str): Path to the original RGB video.
        output_path (str): Path to save the visualized video.
        box_color (tuple): Color for the bounding boxes.
        box_thickness (int): Thickness of the rectangle.
        font_scale (float): Font scale for text.
        font_thickness (int): Thickness of the text.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Group bboxes by frame index
    bbox_dict = {}
    for bbox in bbox_array:
        idx = int(bbox[5])
        bbox_dict.setdefault(idx, []).append(bbox)

    frame_idx = 0
    with tqdm(total=total_frames, desc="Saving Visualized Video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in bbox_dict:
                for bbox in bbox_dict[frame_idx]:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    id_ = int(bbox[4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
                    cv2.putText(frame, f"ID {id_}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, font_thickness)

            out_writer.write(frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    out_writer.release()
    print(f"✅ Visualized video with bounding boxes saved to: {output_path}")


#------------------------------------------------------------
def track_and_save_one_face_always_visible(video_path: str, output_path: str) -> tuple[HeadDetectionStatus, np.ndarray]:
    """
    Tracks faces in the video and saves the bounding boxes of the most frequently detected face.
    Args:
        video_path (str): Path to the input video.
        output_path (str): Path to save the bounding boxes as a .npy file.
    Returns:
        tuple[HeadDetectionStatus, np.ndarray]: Status of the operation and the bounding boxes array (or none if unexpected error has occurred).
    """
    try:
        bboxes = track_faces_from_video(video_path)
        validate_bboxes(bboxes, video_path)
        save_bboxes_to_npy(bboxes, output_path)
        return HeadDetectionStatus.SUCCESSFUL, bboxes
    except BoundingBoxValidationException as ve:
        print(f"Validation failed: {ve}")
        return HeadDetectionStatus.VALIDATION_FAILED, bboxes
    except Exception as e:
        print(f"Unexpected error for {video_path}: {e}")
        return HeadDetectionStatus.UNEXPECTED_ERROR, None
def main():
    # status, bboxes = track_and_save_one_face_always_visible()

    # # Optional: save cropped video
    # crop_faces_with_bbox(bbox_array=bboxes, video_path=RGB_IMAGES_PATH, output_path=HEAD_CROPS_OUT_PATH, output_size=(224, 224))

    # Optional: save full video with bboxes visualized:
    bboxes = np.load(os.path.join(CWD, "head_bboxes.npy"))
    save_visualized_bboxes_video(
        bbox_array=bboxes,
        video_path=RGB_IMAGES_PATH,
        output_path=os.path.join(CWD, "head_bboxes.mp4")
    )


if __name__ == "__main__":
    main()
