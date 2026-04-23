from re import sub
import numpy as np
import subprocess
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="./.env")

def get_cwd():
    """Return the working directory for the experiment from .env"""
    return os.getenv("CWD")

def get_experiment_type():
    """Return the type of experiment from .env"""
    return os.getenv("EXPERIMENT_TYPE")

def get_experiment_types(order="plot_error_by_experiment_type"):
    if order == "data_collection":
        return ["lighting_10", "lighting_25", "lighting_50", "lighting_100", "circular_movement", "head_pose_middle", "head_pose_left", "head_pose_right", "line_movement_slow", "line_movement_fast"]
    elif order == "plot_error_by_experiment_type":
        return ["lighting_10", "lighting_25", "lighting_50", "lighting_100", "head_pose_left", "head_pose_middle", "head_pose_right", "circular_movement", "line_movement_slow", "line_movement_fast"]
    else:
        raise ValueError("Unknown order: {order}")

def get_model():
    """Return the model name for visualization from .env"""
    return os.getenv("MODEL")

def experiment_has_movement():
    return get_experiment_type() in ["rectangular_wave_movement_fast", "rectangular_wave_movement_slow", "circular_movement", "line_movement_slow", "line_movement_fast"]

def is_experiment_type_rectangular_wave_movement():
    """
    Return whether the current experiment type is 'rectangular_wave_movement'.
    This is used to determine if the experiment has a specific movement pattern.
    """
    return get_experiment_type() in ["rectangular_wave_movement_fast", "rectangular_wave_movement_slow"]

def is_experiment_type_line_movement():
    """
    Return whether the current experiment type is 'line_movement'.
    This is used to determine if the experiment has a specific movement pattern.
    """
    return get_experiment_type() in ["line_movement_fast", "line_movement_slow"]

def is_experiment_type_lighting():
    """
    Return whether the current experiment type is 'lighting'.
    """
    return get_experiment_type() in ["lighting_10", "lighting_25", "lighting_50", "lighting_100"]

def get_head_eye_calib_path():
    """Return the path to the .npy file storing the eye position in the head frame."""
    return get_cwd() + "/eye_position_in_head_frame.npy"

def get_table_target_calib_dir():
    """Return the directory path containing target position calibration data in table frame."""
    return "/home/kovan/USTA/gaze_test/July-Experiments/table_target_calibration" # used to be "/home/kovan/USTA/gaze_test/table_target_calibration"

def get_record_time():
    """
    Return the time of actual data collection in seconds.
    For 'rectangular_wave_movement', it's extended to match movement duration.
    """
    if is_experiment_type_rectangular_wave_movement() or is_experiment_type_line_movement():
        return None
    elif get_experiment_type() == "circular_movement":
        return 10.0
    else:
        return 5.0 

def should_check_record_time():
    """
    Return whether to check the record time.
    For 'rectangular_wave_movement', it is not checked as the duration is dynamic.
    """
    return not is_experiment_type_rectangular_wave_movement() and not is_experiment_type_line_movement()

def get_wait_time():
    """
    Return the wait time between node startup and recording in seconds.
    """
    return 0.0 if experiment_has_movement() else 0.0 # must be 0 if experiment has movement, otherwise there can be some wait time (but for now there is no wait time for any experiment)

def get_point_variations():
    """
    Return a dictionary mapping experiment types to point label lists.
    Also includes special cases for horizontal and circular movements.
    """
    p1_9 = [("p" + str(i)) for i in range(1, 10)]
    pv = {
        "genel": p1_9,
        "head_pose_left": [("h" + str(i)) for i in [1, 2, 4, 5]],
        "head_pose_middle": [("h" + str(i)) for i in range(1, 7)],
        "head_pose_right": [("h" + str(i)) for i in [2, 3, 5, 6]],
    }
    pv["rectangular_wave_movement_fast"] = []
    pv["rectangular_wave_movement_slow"] = []
    pv["line_movement_fast"] = []
    pv["line_movement_slow"] = []
    pv["circular_movement"] = p1_9
    pv["lighting_10"] = p1_9
    pv["lighting_25"] = p1_9
    pv["lighting_50"] = p1_9
    pv["lighting_100"] = p1_9
    return pv

def get_points():
    """Return the list of point labels to be used for data collection in the current experiment."""
    return get_point_variations()[get_experiment_type()]

def get_all_exp_directories_under_a_subject_directory(subject_dir: str):
    exp_dirs = []

    lighting_variations = [10, 25, 50, 100]
    for variation in lighting_variations:
        exp_type = f"lighting_{variation}"
        for point in get_point_variations()[exp_type]:
            exp_dirs.append(f"{subject_dir}/{exp_type}/{point}")

    exp_type = "circular_movement"
    for point in get_point_variations()[exp_type]:
        exp_dirs.append(f"{subject_dir}/{exp_type}/{point}")

    head_pose_variations = ["left", "middle", "right"]
    for variation in head_pose_variations:
        exp_type = f"head_pose_{variation}"
        for point in get_point_variations()[exp_type]:
            exp_dirs.append(f"{subject_dir}/{exp_type}/{point}")

    line_movement_variations = ["line_movement_fast", "line_movement_slow"]
    for exp_type in line_movement_variations:
        for _type in get_line_movement_types():
            exp_dirs.append(f"{subject_dir}/{exp_type}/{_type}")

    return exp_dirs

def get_exp_directories_under_a_subject_directory(subject_dir: str, included_exp_types: list):
    base_dir = get_dataset_base_directory()
    if not subject_dir.startswith(base_dir):
        subject_dir = os.path.join(base_dir, subject_dir)

    exp_dirs = []
    
    for exp_type in included_exp_types:
        if exp_type.startswith("lighting_") or exp_type.startswith("head_pose_") or exp_type == "circular_movement":
            for point in get_point_variations()[exp_type]:
                exp_dirs.append(f"{subject_dir}/{exp_type}/{point}")
        elif exp_type in ["line_movement_fast", "line_movement_slow"]:
            for _type in get_line_movement_types():
                exp_dirs.append(f"{subject_dir}/{exp_type}/{_type}")
                
    return exp_dirs

def get_head_tracker():
    """Return the name of the rigid body used to track the head (ground truth)."""
    return "rigid_body_6"

def get_eye_device_tracker():
    """Return the name of the rigid body used to track the head (ground truth)."""
    return "rigid_body_5"

def get_camera_tracker():
    """
    Return the tracker source for camera pose.
    Options: 'rgb_camera_link' (UR5 + static TF), 'rigid_body_6' (OptiTrack), choose rgb_camera_link since it is more and accurate.
    """
    return "rgb_camera_link"

def get_assume_gaze_target_is_at_table_height():
    """
    Return whether to force the gaze target's Z position to match the table height.
    """
    return False

def get_load_table_pose_from_file():
    """
    Return whether to load the table pose from a file rather than using the TF tree.
    Useful for debugging.
    """
    return False

def get_table_pose_path():
    """Return the file path to the saved table pose (.npy file)."""
    return "/home/kovan/USTA/gaze_test/2025-05-06/fix-calib/genel/p1/1746528213565/table_pose.npy"

def get_table_tracker():
    """Return the name of the rigid body used to track the table."""
    return "rigid_body_3"

def get_table_dimensions():
    """Return the physical dimensions of the table in meters: [width, depth, height]."""
    return [1.6, 0.7, 0.72]

def get_rgb_resolution():
    """Return the resolution of the RGB camera as a tuple (width, height)."""
    return (1920, 1080)

def get_rgb_fps():
    """Return the frame rate (Hz) for RGB camera data collection."""
    return 30

def get_mocap_freq():
    """Return the frequency (Hz) of motion capture data collection."""
    return 100

def get_start_trigger():
    """Return the input method used to start recording: 'mouse' or 'keyboard'."""
    return 'mouse'

def get_target_period():
    """Return the time (in milliseconds) for which each target is displayed."""
    return 10.0

def get_camera_pose_period():
    """Return the duration (in milliseconds) for which the camera pose is considered valid."""
    return 10.0

def get_time_diff_max():
    """
    Return the maximum allowed time difference (in milliseconds) between synchronized data sources.
    Used by data loader, analyzer, and visualizer.
    """
    return 20.0

def get_play_sound_when_recording_is_ready():
    """Return whether to play a sound cue when recording is ready to start (to inform the subject)."""
    return True

def get_play_sound_when_recording_starts():
    """Return whether to play a sound cue when recording starts."""
    return True


def get_play_sound_when_recording_stops():
    """Return whether to play a sound cue when recording stops."""
    return True

def get_play_sound_when_recording_is_ready_at_new_controller():
    """Return whether to play a sound cue when recording is ready to start at the new controller node."""
    return False # for the "ready" sound, we will always use the data collector to play the sound.

def get_play_sound_when_recording_starts_at_new_controller():
    """Return whether to play a sound cue when recording starts at the new controller node."""
    return get_play_sound_when_recording_starts() and experiment_has_movement()

def get_play_sound_when_recording_stops_at_new_controller():
    """Return whether to play a sound cue when recording stops at the new controller node."""
    return get_play_sound_when_recording_stops() and experiment_has_movement()

def get_play_sound_when_recording_is_ready_at_data_collector():
    """Return whether to play a sound cue when recording is ready to start at the new controller node."""
    return get_play_sound_when_recording_is_ready() # for the "ready" sound, we will always use the data collector to play the sound.

def get_play_sound_when_recording_starts_at_data_collector():
    """Return whether to play a sound cue when recording starts at the data collector node."""
    return get_play_sound_when_recording_starts() and not experiment_has_movement()

def get_play_sound_when_recording_stops_at_data_collector():
    """Return whether to play a sound cue when recording stops at the data collector node."""
    return get_play_sound_when_recording_stops() and not experiment_has_movement()

def play_sound_when_recording_is_ready():
    """Play a sound when recording is ready to start to inform the subject."""
    subprocess.run(["aplay", "/home/kovan/USTA/src/robot_controller/robot_controller/gaze/sounds/ready.wav"])

def play_recording_started_sound():
    """
    Play a sound when recording starts.
    This is useful to indicate recording has started.
    """
    subprocess.run(["aplay", "/home/kovan/USTA/src/robot_controller/robot_controller/gaze/sounds/start.wav"])

def play_recording_stopped_sound():
    """
    Play a sound when recording stops.
    This is useful to indicate recording has stopped.
    """
    subprocess.run(["aplay", "/home/kovan/USTA/src/robot_controller/robot_controller/gaze/sounds/finish.wav"])

def get_joint_pos_dir():
    """The directory path containing joint position states to be used in experiments for the robot."""
    return "/home/kovan/USTA/src/robot_controller/robot_controller/gaze/joint_pos"

def get_joint_pos_path_for_center():
    """Return the file path to the joint position state for the center position."""
    return get_joint_pos_dir() + "/center.npy"

def get_head_pose_fixed_config():
    """
    Return the fixed head pose configuration.
    This is used to set the head pose to a specific orientation.
    """
    return {
        "roll": 0.0,  # in degrees, roll angle of the head,
        "pitch": 0.0,  # in degrees, pitch angle of the head
        "yaw": 25.0, # in degrees, yaw angle of the head,
    }

def get_circular_movement_config():
    """
    Return the configuration for circular movement experiments.
    Includes parameters like radius, center, and angular velocity.
    """
    RADIUS = 0.5 # Should be calculated as the distance from the end effector to the eye. hard-coded for simplicity as it is constants for these experiments
    DURATION = get_record_time() # duration of the circular movement will match the recording time
    PERIOD = DURATION
    ARC_ANGLE_DEG = 60
    GAZE_TARGET_TRACKER = "eye" # gaze target to center the circular movement around
    CAMERA_TRACKER = "rgb_camera_link" # camera tracker to execute the circular movement

    return {
        "radius": RADIUS,
        "period": PERIOD,
        "duration": DURATION,
        "arc_angle_deg": ARC_ANGLE_DEG,
        "gaze_target_tracker": GAZE_TARGET_TRACKER,
        "camera_tracker": CAMERA_TRACKER,
    }

def get_rectangular_wave_movement_config():
    """
    Return the configuration for rectangular wave movement experiments.
    """

    HORIZONTAL_DISTANCE = 0.25 # in meters, the max distance from the center to the left/right
    VERTICAL_DISTANCE = 0.15  # in meters, the max distance from the center upwards/downwards
    FAST_SPEED = 0.1 # m/s, speed of the fast movement
    SLOW_SPEED = FAST_SPEED / 2.0 # m/s, speed of the slow movement, half of fast movement speed
    NUM_OF_PERIODS_PER_SIDE = 2 # number of periods for the wave for each side (left/right)

    return {
        "horizontal_distance": HORIZONTAL_DISTANCE,
        "vertical_distance": VERTICAL_DISTANCE,
        "fast_speed": FAST_SPEED,
        "slow_speed": SLOW_SPEED,
        "num_of_periods_per_side": NUM_OF_PERIODS_PER_SIDE,
    }
    
def get_line_movement_config():
    """
    Return the configuration for line movement experiments.
    """

    HORIZONTAL_DISTANCE = 0.25 # in meters, the max distance from the center to the left/right
    VERTICAL_DISTANCE = 0.10  # in meters, the max distance from the center upwards/downwards
    FAST_SPEED = 0.1 # m/s, speed of the fast movement
    SLOW_SPEED = FAST_SPEED / 2.0 # m/s, speed of the slow movement, half of fast movement speed

    return {
        "horizontal_distance": HORIZONTAL_DISTANCE,
        "vertical_distance": VERTICAL_DISTANCE,
        "fast_speed": FAST_SPEED,
        "slow_speed": SLOW_SPEED,
    }
    
def get_line_movement_types():
    """Get the types of line movements in the order they are executed.
    """
    return ["horizontal", "vertical"]
    
def get_neutral_head_orientation_in_cam_frame() -> np.ndarray:
    m = np.zeros((3, 3), dtype=np.float64)
    m[0, 0] = -1.0
    m[2, 1] = 1.0
    m[1, 2] = 1.0
    return m

def get_neutral_head_orientation_in_world_frame() -> np.ndarray:
    m = np.zeros((3, 3), dtype=np.float64)
    m[0, 0] = 1.0
    m[2, 1] = 1.0
    m[1, 2] = -1.0
    return m

def get_neutral_eye_orientation_in_world_frame() -> np.ndarray:
    return get_neutral_head_orientation_in_world_frame()

def get_neutral_cam_orientation_in_world_frame() -> np.ndarray:
    m = np.zeros((3, 3), dtype=np.float64)
    m[0, 0] = -1.0
    m[1, 1] = -1.0
    m[2, 2] = 1.0
    return m

def get_neutral_cam_pose_in_world_frame() -> np.ndarray:
    m = np.zeros((4, 4), dtype=np.float64)
    m[:3,:3] = get_neutral_cam_orientation_in_world_frame()
    m[:3, 3] = np.array([1.31864, 0.06311, 1.11644]) # in meters, this is the neutral position we have used (empirically measured by the neutral_cam_pose_calculation.py script)
    m[3,3] = 1.0
    return m

def get_dataset_base_directory() -> str:
    return os.getenv("DATASET_BASE_DIR") # e.g. "/home/kovan/USTA/gaze_test/July-Experiments"

def get_dataset_subject_directories(rnd=False, n=20, seed=42):
    """Get the subject directories for the dataset.

    Args:
        rnd (bool, optional): Whether to randomize the selection of subjects (which is good for quick tests). Defaults to False.
        n (int, optional): The number of subjects to select. Defaults to 20.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.

    Returns:
        list: The subject directories for the dataset.
    """
    SUBJECT_DIRS = []

    def append_immediate_subdirectories(root_dir):
        for name in os.listdir(root_dir):
            full_path = os.path.join(root_dir, name)
            if os.path.isdir(full_path):
                SUBJECT_DIRS.append(os.path.abspath(full_path))


    BASE_DIR = get_dataset_base_directory()

    dates = [
        f"{BASE_DIR}/2025-07-28",
        f"{BASE_DIR}/2025-07-29",
        f"{BASE_DIR}/2025-07-30",
        f"{BASE_DIR}/2025-07-31",
        f"{BASE_DIR}/2025-08-01",
        f"{BASE_DIR}/2025-08-04",
        f"{BASE_DIR}/2025-08-05",
        f"{BASE_DIR}/2025-08-06",
    ]

    for date in dates:
        if os.path.isdir(date):
            append_immediate_subdirectories(date)
        else:
            print(f"{date} doesnt exists, skipping...")
       
        
    if rnd:
        import random
        def filter_random_subjects(subject_dirs):
            """
            Returns a filtered list of n random subject dirs (deterministic with seed).
            """
            unique_sorted = sorted(set(subject_dirs))  # stable order, no duplicates
            rng = random.Random(seed)
            return rng.sample(unique_sorted, k=min(n, len(unique_sorted)))
        SUBJECT_DIRS = filter_random_subjects(SUBJECT_DIRS)
    return SUBJECT_DIRS

def get_neutral_eye_position_per_subject_csv_path():
    return os.path.join(get_dataset_base_directory(), "neutral_eye_position_per_subject.csv")

def get_main_models_included_in_the_paper():
    # ids of main models
    model_ids = ["puregaze_rectification_unrectified","gazetr_rectification_unrectified","puregaze_gaze360_rectification_unrectified","gazetr_gaze360_rectification_unrectified","l2cs_padding0fixed_isrgb_False_resize_448","mcgaze_clip_size_7","gaze3d_clip_len_8"] #["puregaze", "puregaze_rectification_unrectified","gazetr", "gazetr_rectification_unrectified","puregaze_gaze360","puregaze_gaze360_rectification_unrectified", "gazetr_gaze360", "gazetr_gaze360_rectification_unrectified","l2cs_padding0fixed_isrgb_False_resize_448","mcgaze_clip_size_7", "mcgaze_clip_size_7_rectification_unrectified", "l2cs_padding0fixed_rectification_isrgb_False_resize_448_unrectified","gaze3d_clip_len_8", "gaze3d_clip_len_8_rectification_unrectified"]
    dislay_names = get_model_display_names()
    ret = {}
    for id in model_ids:
        ret[id] = dislay_names[id]
    return ret

def get_model_display_names():
    return {
        "puregaze": "PureGaze (ETH-X) w/ Crop",
        "puregaze_rectification_unrectified": "PureGaze (ETH-X) w/ Rect.",
        "gazetr": "GazeTR (ETH-X) w/ Crop",
        "gazetr_rectification_unrectified": "GazeTR (ETH-X) w/ Rect.",
        "puregaze_gaze360": "PureGaze (Gaze360) w/ Crop",
        "puregaze_gaze360_rectification_unrectified": "PureGaze (Gaze360) w/ Rect.",
        "gazetr_gaze360": "GazeTR (Gaze360) w/ Crop",
        "gazetr_gaze360_rectification_unrectified": "GazeTR (Gaze360) w/ Rect.",
        "l2cs_padding0fixed_isrgb_False_resize_448": "L2CS-Net (Gaze360) w/ Crop",
        "l2cs_padding0fixed_rectification_isrgb_False_resize_448_unrectified": "L2CS-Net (Gaze360) w/ Rect.",
        "mcgaze_clip_size_7": "MCGaze (Gaze360) w/ Crop",
        "mcgaze_clip_size_7_rectification_unrectified": "MCGaze (Gaze360) w/ Rect.",
        "gaze3d_clip_len_8": "GaT (Gaze360) w/ Crop",
        "gaze3d_clip_len_8_rectification_unrectified": "GaT (Gaze360) w/ Rect.",
    }

def display_model_name(model_id, model_display_names_dict=None, clean_display_name=False):
    if model_display_names_dict is None:
        model_display_names_dict = get_model_display_names()
    m = model_display_names_dict.get(model_id, model_id)
    if clean_display_name:
        m = m.split(" w/")[0].strip()
    return m

def get_currently_analyzed_models():
    return os.getenv("CURRENTLY_ANALYZED_MODELS").split(",")

def get_subject_directories_excluded_from_eval():
    return [
        "2025-07-31/subj_0021",
        "2025-07-31/subj_0017",
        "2025-08-01/subj_0028",
    ]

def is_subject_directory_excluded_from_eval(subject_dir: str) -> bool:
    base_dir = get_dataset_base_directory()
    if subject_dir.startswith(base_dir):
        subject_dir = os.path.relpath(subject_dir, get_dataset_base_directory())
    return subject_dir in get_subject_directories_excluded_from_eval()

def is_experiment_directory_excluded_from_eval(exp_dir: str) -> bool:
    """
    Check if a given experiment directory is in the excluded list for evaluation (for gazeh4ri). Allows filtering out bad data.
    """
    base_dir = get_dataset_base_directory()
    if exp_dir.startswith(base_dir):
        exp_dir = os.path.relpath(exp_dir, get_dataset_base_directory())

    # exp_dir is excluded if it is in the completely excluded subject dirs
    completely_exc_subject_dirs = get_subject_directories_excluded_from_eval()
    for completely_exc_subject_dir in completely_exc_subject_dirs:
        if exp_dir.startswith(completely_exc_subject_dir):
            return True

    # exp_dir is excluded if it is in the following head pose directories, as they are currently incorrect (can be fixed later, if need be)
    for subject_dir in ["2025-07-29/subj_0010", "2025-07-31/subj_0020"]:
        for exp_type in ["head_pose_left", "head_pose_right"]:
            if exp_dir.startswith(f"{subject_dir}/{exp_type}/"):
                return True

    return False

def is_experiment_directory_excluded_from_blink4hri(exp_dir: str) -> bool:
    """
    Check if a given experiment directory is in the excluded list for blink4hri. Allows filtering out bad data.
    """
    base_dir = get_dataset_base_directory()
    if exp_dir.startswith(base_dir):
        exp_dir = os.path.relpath(exp_dir, get_dataset_base_directory())

    extra_excluded_exp_dirs_for_blink4hri = [
        "2025-07-28/subj_0002/lighting_25/p9",
        "2025-07-29/subj_0004/lighting_10/p7",
        "2025-07-29/subj_0004/lighting_100/p7",
        "2025-07-29/subj_0004/circular_movement/p4",
        "2025-07-29/subj_0004/circular_movement/p7",
        "2025-07-29/subj_0010/circular_movement/p7",
        "2025-07-29/subj_0010/circular_movement/p9",
        "2025-07-29/subj_0007/circular_movement/p7",
        "2025-08-05/subj_0039/circular_movement/p7"
    ]
    for exc_exp_dir in extra_excluded_exp_dirs_for_blink4hri:
        if exp_dir.startswith(exc_exp_dir):
            return True

    return is_experiment_directory_excluded_from_eval(exp_dir) # automatically exclude exp_dirs excluded from gaze4hri