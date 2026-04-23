import rclpy
import numpy as np
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from cv_bridge import CvBridge
import cv2
import json
import os
import pyrealsense2 as rs
from robot_controller.cam_calib.realsense_calib import get_device_intrinsics
from robot_controller.util import quaternion_to_homogeneous_matrix
from playsound import playsound
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import robot_controller.gaze.config as config
import traceback

CWD = config.get_cwd()
EXPERIMENT_TYPE = config.get_experiment_type()
IS_EXPERIMENT_TYPE_RECTANGULAR_WAVE_MOVEMENT = config.is_experiment_type_rectangular_wave_movement()
IS_EXPERIMENT_TYPE_LINE_MOVEMENT = config.is_experiment_type_line_movement()
HEAD_EYE_CALIB_PATH = config.get_head_eye_calib_path()
TABLE_TARGET_CALIB_DIR = config.get_table_target_calib_dir()
RECORD_TIME = config.get_record_time()
WAIT_TIME = config.get_wait_time()
POINTS = config.get_points()
HEAD_TRACKER = config.get_head_tracker()
CAMERA_TRACKER = config.get_camera_tracker()
ASSUME_GAZE_TARGET_IS_AT_TABLE_HEIGHT = config.get_assume_gaze_target_is_at_table_height()
LOAD_TABLE_POSE_FROM_FILE = config.get_load_table_pose_from_file()
TABLE_POSE_PATH = config.get_table_pose_path()
TABLE_TRACKER = config.get_table_tracker()
TABLE_DIMENSIONS = config.get_table_dimensions()
RGB_RESOLUTION = config.get_rgb_resolution()
RGB_FPS = config.get_rgb_fps()
MOCAP_FREQ = config.get_mocap_freq()
START_TRIGGER = config.get_start_trigger()
PLAY_SOUND_WHEN_RECORDING_IS_READY = config.get_play_sound_when_recording_is_ready_at_data_collector()
PLAY_SOUND_WHEN_RECORDING_STARTS = config.get_play_sound_when_recording_starts_at_data_collector()
PLAY_SOUND_WHEN_RECORDING_STOPS = config.get_play_sound_when_recording_stops_at_data_collector()
LINE_MOVEMENT_TYPES = config.get_line_movement_types()

class GazeTestDataCollector(Node):

    def init_tf(self):
        self.tfBuffer = Buffer()
        self.tfListener = TransformListener(self.tfBuffer, self)
        can_transform = False
        
        while not can_transform:
            try:
                curr_ts = rclpy.time.Time()
                can_transform = self.tfBuffer.can_transform("base", "wrist_3_link", curr_ts) and self.tfBuffer.can_transform("world", "base", curr_ts) and self.tfBuffer.can_transform("world", HEAD_TRACKER, curr_ts) and self.tfBuffer.can_transform("world", CAMERA_TRACKER, curr_ts)
                if can_transform:
                    tf = self.tfBuffer.lookup_transform("world", "base_link", curr_ts).transform
                    tf_quat = np.array([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w])
                    tf_trans = np.array([tf.translation.x, tf.translation.y, tf.translation.z])
                    homog = quaternion_to_homogeneous_matrix(tf_quat, tf_trans)
                    #sometimes an error occurs in the tf tree, and the transform from world to base_link is identity. In that case, we need to reinitialize the tfBuffer and tfListener until the tf tree is fixed.
                    if np.equal(homog, np.eye(4)).all():
                        self.tfBuffer = Buffer()
                        self.tfListener = TransformListener(self.tfBuffer, self)
                        can_transform = False   
                #     gaze_target_tf = self.tfBuffer.lookup_transform("world", CAMERA_TRACKER, curr_ts).transform
                #     base_tf = self.tfBuffer.lookup_transform("world", "base", curr_ts).transform
                #     can_transform = gaze_target_tf.translation.x - base_tf.translation.x < 0.0
            except Exception as e:
                # time.sleep(0.1) @ToDo maybe sleep a bit 
                print(e)
                pass
            finally:
                if can_transform:
                    self.get_logger().info('TF tree received.')
                    break
                else:
                    rclpy.spin_once(self)

    def init_rgb_camera(self):
        """Initialize camera  (RealSense+OpenCV) pipeline
        """
        self.bridge = CvBridge()
        self.camera_intrinsics = get_device_intrinsics(stream=rs.stream.color, res=RGB_RESOLUTION, fps=RGB_FPS)
        self.rs_pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(rs.stream.color, RGB_RESOLUTION[0], RGB_RESOLUTION[1], rs.format.bgr8, RGB_FPS)
        profile = self.rs_pipeline.start(self.rs_config)
        
        # --- Reset all options to default ---
        color_sensor = profile.get_device().first_color_sensor()

        # Get list of supported options directly from the sensor
        for option in color_sensor.get_supported_options():
            try:
                option_range = color_sensor.get_option_range(option)
                color_sensor.set_option(option, option_range.default)
            except Exception as e:
                print(f"Skipping {option.name}: {e}")

        # Enable auto exposure & auto white balance (true factory-like behavior)
        try:
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            if color_sensor.supports(rs.option.enable_auto_white_balance):
                color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
        except Exception as e:
            print(f"Skipping auto modes: {e}")
            
        # We will set settings manually to make sure they are correct.
        # Set custom exposure (e.g., 156) and gain (e.g., 64)
        if color_sensor.supports(rs.option.exposure):
            color_sensor.set_option(rs.option.exposure, 250.0)
            
        if color_sensor.supports(rs.option.gain):
            color_sensor.set_option(rs.option.gain, 90.0)
    
        # Set white balance (Kelvin)
        if color_sensor.supports(rs.option.white_balance):
            color_sensor.set_option(rs.option.white_balance, 3000.0)

        # Set saturation
        if color_sensor.supports(rs.option.saturation):
            color_sensor.set_option(rs.option.saturation, 68.0)
            
        self.rs_settings = {}
        for option in color_sensor.get_supported_options():
            try:
                value = color_sensor.get_option(option)
                self.rs_settings[option.name] = value
            except Exception as e:
                print(f"Skipping {option.name}: {e}")
        
        self.rs_last_frame_ts = None # since we poll rgb frames with a frequency higher than the camera frequency, we need to keep track of the last frame timestamp to avoid processing the same frame multiple times.

    def __init__(self, point, stop_circular_movement_at_terminate=False, stop_line_movement_at_terminate=False, line_movement_type=None):
        """

        Args:
            point (str, optional): _description_. Set to None to use with experiment type in "rectangular_wave_movement" or "line_movement". otherwise, set to the point to be used for data collection. e.g. "p1", "p2", "h1", "h2" etc.

        Raises:
            FileNotFoundError: _description_
            ValueError: _description_
        """
        super().__init__('gaze_test_data_collector')
        self.get_logger().info('gaze test data collector Node Started')
                
        self.point = point # e.g. "p1", "p2", "h1", "h2" etc.
        self.stop_circular_movement_at_terminate = stop_circular_movement_at_terminate
        self.stop_line_movement_at_terminate = stop_line_movement_at_terminate
        self.line_movement_type = line_movement_type

        self.started_at = None  # the time at which the data collection process started (not necessarily the time at which the node started), see the is_record_time function for details of use)

        self.camera_to_base_matrix = np.eye(4) # initialize as 4x4 identity matrix. will be used to compute the camera pose
        self.head_poses = [] # haad poses, in world frame
        self.eye_positions = [] # eye positions, in world frame
        self.target_positions = [] # gaze target positions, in world frame
        self.camera_poses = [] # camera poses, in world frame
        self.rgb_images = []
        self.rgb_image_timestamps = []
        self.ur5_joint_states = [] # joint states of the UR5 robot, saved to do visualization in RVIZ later.
        self.ur5_base_pose = None # base pose of the UR5 robot, saved to do visualization in RVIZ later (since it is static in our current set up, it will be single homogeneous transformation matrix).
        self.table_pose = None # table pose, in world frame, saved to do visualization in RVIZ later (since it is static in our current set up, it will be single homogeneous transformation matrix).
        if LOAD_TABLE_POSE_FROM_FILE:
            if os.path.isfile(TABLE_POSE_PATH):
                self.table_pose = np.load(TABLE_POSE_PATH)
            else:
                raise FileNotFoundError(f"Table pose file not found: {TABLE_POSE_PATH}")
            
            if self.table_pose.shape != (4, 4):
                raise ValueError(f"Table pose shape is not (4, 4): {self.table_pose.shape}")
        self.eye_position_in_head_frame = self.load_head_eye_position_in_head_frame() # will be used to obtain eye position in world frame from head pose in world frame.
        # do not load target position in table frame if experiment type is horizontal movement, since for that experiment the gaze target is the camera itself (it is basically mutual gaze)
        if not IS_EXPERIMENT_TYPE_RECTANGULAR_WAVE_MOVEMENT and not IS_EXPERIMENT_TYPE_LINE_MOVEMENT:
            self.target_position_in_table_frame = self.load_target_position_in_table_frame() # will be used to obtain target position in world frame from table pose in world frame.
        self.init_tf()
        self.init_rgb_camera()
        # if experiment type is rectangular_wave_movement, we will not use the mouse click to start data collection, instead we will use the rectangular_wave_movement_start_recording topic to start data collection.
        if IS_EXPERIMENT_TYPE_RECTANGULAR_WAVE_MOVEMENT:
            self.rectangular_wave_movement_is_recording = False
            self.rectangular_wave_movement_start_recording_sub = self.create_subscription(Bool, 'gaze_test_rectangular_wave_movement_start_recording', self.rectangular_wave_movement_start_callback, 10)
        elif IS_EXPERIMENT_TYPE_LINE_MOVEMENT:
            self.line_movement_is_recording = False
            self.start_line_movement_pub = self.create_publisher(Bool, "gaze_test_start_line_movement", 10)
            self.line_movement_start_recording_sub = self.create_subscription(Bool, 'gaze_test_line_movement_start_recording', self.line_movement_start_recording_callback, 10)
            # wait for the start trigger, then publish the line_movement_subject_ready topic that will make the robot start doing the line movement.
            self.wait_for_start_trigger()
            self.start_line_movement_pub.publish(Bool(data=True))
        elif EXPERIMENT_TYPE == "circular_movement":
            self.start_circular_movement_pub = self.create_publisher(Bool, "gaze_test_start_circular_movement", 10)
            self.circular_movement_start_recording_sub = self.create_subscription(Bool, 'gaze_test_circular_movement_start_recording', self.circular_movement_start_recording_callback, 10)
            # wait for the start trigger, then publish the circular_movement_subject_ready topic that will make the robot start doing the circular movement.
            self.wait_for_start_trigger()
            self.start_circular_movement_pub.publish(Bool(data=True))
        else:
            self.wait_for_start_trigger()
            self.start()
        self.ur5_joint_state_sub = self.create_subscription(JointState, 'joint_states', self.ur5_joint_state_callback, 10)
        self.mocap_timer = self.create_timer(0.01, self.mocap_callback) # set a 100Hz timer for motion capture (optitrack) callback.
        self.rgb_timer = self.create_timer(0.01, self.capture_rgb_frame) # set a 100Hz timer even though the camera is 30Hz because capture_rgb_frame uses polling (and this high frequency polling is the best strategy, if we use blocking functions in capture_rgb_frame, we block other timers like mocap_timers).

    def wait_for_start_trigger(self):
        if PLAY_SOUND_WHEN_RECORDING_IS_READY:
            config.play_sound_when_recording_is_ready()

        if START_TRIGGER == 'mouse':
            self.get_logger().info("Please LEFT CLICK to start data collection when ready...")

            clicked = [False]

            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    clicked[0] = True

            # Create a full-screen OpenCV window
            window_name = "Click to Start Data Collection"
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback(window_name, mouse_callback)

            # Create a black screen
            screen_res = (1080, 1920, 3)  # You can adjust this if your resolution is different
            black_screen = np.zeros(screen_res, dtype=np.uint8)
            cv2.putText(black_screen, "Left click anywhere to start data collection", (100, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            # Wait until mouse click
            while not clicked[0]:
                cv2.imshow(window_name, black_screen)
                cv2.waitKey(100)

            cv2.destroyWindow(window_name)
            if not clicked[0]:
                raise ValueError("data collection failed, no mouse event has been received.")
        elif START_TRIGGER == 'keyboard':
            input("Press ENTER to start data collection when ready...")
        else:
            raise ValueError("Invalid START_TRIGGER value. Use 'mouse' or 'keyboard'.")
    
    def start(self):
        # the play sound function is deliberately called before creating the stop-recording timer, so that the time that passes during the sound playback does not affect the recording time.
        # In addition, since the recording will be stopped upon a subscription callback in the movement experiments , we cannot call any blocking functions here because if we lose time there will
        # be a time difference between the start of the recording and the start of the movements. Hence, for the movement experiments, we do not play the sound when recording starts on data collector,
        # instead, we play the sound just before the recording starts on the robot controller.
        if PLAY_SOUND_WHEN_RECORDING_STARTS:
            config.play_recording_started_sound()

        self.started_at = self.get_current_timestamp_ms()
        # create a one-off timer to stop the recording
        # (don't if experiment type is one of the movement types, since in that case the recording will be stopped upon a subscription callback to stop the recording just as the movement stops on the robot controller).
        if EXPERIMENT_TYPE != "circular_movement" and not IS_EXPERIMENT_TYPE_RECTANGULAR_WAVE_MOVEMENT and not IS_EXPERIMENT_TYPE_LINE_MOVEMENT:
            self.create_timer(WAIT_TIME + RECORD_TIME, self.stop_rec_callback) 

    def rectangular_wave_movement_start_callback(self, msg):
        if(msg.data):
            self.start()
            self.rectangular_wave_movement_is_recording = True
        else:
            self.rectangular_wave_movement_is_recording = False
            self.stop_rec_callback()
    
    def get_current_timestamp_ms(self, return_rclpy_time_object=False):
        """ Make sure to use this for consistent and accurate timestamps.
        Returns:
            rclpy.time.Time: returns the current ROS time, in milliseconds (with fractional ms). If return_rclpy_time_object is True, returns the rclpy.time.Time object, otherwise returns as float in milliseconds.
        """
        t = self.get_clock().now()
        if return_rclpy_time_object:
            return t
        else:
            return t.nanoseconds * 1e-6

    def is_record_time(self):
        if IS_EXPERIMENT_TYPE_RECTANGULAR_WAVE_MOVEMENT:
            return self.rectangular_wave_movement_is_recording
        elif IS_EXPERIMENT_TYPE_LINE_MOVEMENT:
            return self.line_movement_is_recording
        s = self.started_at / 1000.0 if self.started_at is not None else None # start time in seconds
        c = self.get_current_timestamp_ms() / 1000.0 # current time in seconds
        
        return (s is not None) and (s + WAIT_TIME <= c) and (c <= s + WAIT_TIME + RECORD_TIME)

    def appendWithTimestamp(self, v, data, timestamp=None):
        if not self.is_record_time():
            return
        timestamp = self.get_current_timestamp_ms()
        new_row = [timestamp, *v]
        data.append(new_row)
        
    def check_time_difference(self, data_name, data, expected_time_diff, margin=0.1):
        """_summary_

        Args:
            data_name (str): to print in the error message
            data (np.array): the data to check
            expected_time_diff (float): the expected time difference between two consecutive data points, in ms. For instance, if the data has 100 HZ frequency, then the time difference between two consecutive data points should be around 10ms.
            margin (float, optional): _description_. Defaults to 0.1.  margin of error as a ratio. For instance, 0.1 means 10% of the expected time difference. so if RGB_FPS is 30, the expected time difference is 33.33ms, so the margin of error is 3.33ms, so anything gexceeding 36.66ms will raise an error.

        Raises:
            ValueError: _description_
        """
        # check time difference to ensure data collection frequency is correct.
        if len (data) >= 2:
            c = data[-1] # current row's timestamp
            p = data[-2] # previous row's timestamp
            
            # if c and p are np.arrays, we will plug out the first element (the timestamp) to get the time difference.
            if type(c) != float:
                c = data[-1][0] # current row's timestamp
                p = data[-2][0] # previous row's timestamp
            limit = (1.0+margin) * expected_time_diff
            if c - p > limit: # if the time difference is greater than limit, then raise error.
                self.get_logger().warning(f"WARNING: Time difference between {data_name} is too large: {c - p:.2f} ms. The time difference is expected to be {expected_time_diff} ms, and it has a max limit of {limit} ms.")

    def mocap_callback(self):
        if not self.is_record_time():
            return
        try:
            # Part: head_poses and eye_positions
            head_pose = self.tfBuffer.lookup_transform("world", HEAD_TRACKER, rclpy.time.Time()).transform
            head_pose_quat = np.array([head_pose.rotation.x, head_pose.rotation.y, head_pose.rotation.z, head_pose.rotation.w])
            head_pose_trans = np.array([head_pose.translation.x, head_pose.translation.y, head_pose.translation.z])
            head_pose = quaternion_to_homogeneous_matrix(head_pose_quat, head_pose_trans)
            self.appendWithTimestamp(head_pose.flatten(), self.head_poses)
            
            # check time difference to ensure data collection frequency is correct (raises an error if the time difference is too large).
            self.check_time_difference(data_name="head poses (motion capture)", data=self.head_poses, expected_time_diff=1000.0/MOCAP_FREQ, margin=0.1)

            eye_pose_in_head_frame = np.eye(4)
            # eye and head have the same orientation, so we can use the same rotation matrix.
            # so, we will only set the translation vector.
            eye_pose_in_head_frame[0:3, 3] = self.eye_position_in_head_frame
            eye_pose_in_world_frame =  head_pose @ eye_pose_in_head_frame
            # extract the position from homogeneous transformation matrix
            eye_position = eye_pose_in_world_frame[0:3, 3] # in world frame
            self.appendWithTimestamp(eye_position, self.eye_positions)
            
            # Part: camera_poses
            camera = self.tfBuffer.lookup_transform("world", CAMERA_TRACKER, rclpy.time.Time()).transform
            camera_quat = np.array([camera.rotation.x, camera.rotation.y, camera.rotation.z, camera.rotation.w])
            camera_trans = np.array([camera.translation.x, camera.translation.y, camera.translation.z])
            camera_pose = quaternion_to_homogeneous_matrix(camera_quat, camera_trans)
            self.appendWithTimestamp(camera_pose.flatten(), self.camera_poses)            
            
            # since table_pose and ur5_base_pose is static in our current set up, save only once (each as a single homogeneous transformation matrix)
            if self.table_pose is None:
                table_pose = self.tfBuffer.lookup_transform("world", TABLE_TRACKER, rclpy.time.Time()).transform
                table_quat = np.array([table_pose.rotation.x, table_pose.rotation.y, table_pose.rotation.z, table_pose.rotation.w])
                table_trans = np.array([table_pose.translation.x, table_pose.translation.y, table_pose.translation.z])
                self.table_pose = quaternion_to_homogeneous_matrix(table_quat, table_trans)
                
            # Part: target_positions
            if IS_EXPERIMENT_TYPE_LINE_MOVEMENT or IS_EXPERIMENT_TYPE_RECTANGULAR_WAVE_MOVEMENT:
                # in this experiment, the gaze target is the camera itself (it is basically mutual gaze)
                gaze_target = camera_trans
            else:
                # does this work porperly:(self.target_position_in_table_frame shape is (3,))?
                # gaze_target = self.table_pose[0:3,0:3] @ self.target_position_in_table_frame + self.table_pose[0:3, 3] # in world frame
                target_pose_in_table_frame = np.eye(4)
                # target and table have the same orientation, so we can use the same rotation matrix.
                # so, we will only set the translation vector.
                target_pose_in_table_frame[0:3, 3] = self.target_position_in_table_frame
                target_pose_in_world_frame =  self.table_pose @ target_pose_in_table_frame
                # extract the position from homogeneous transformation matrix
                gaze_target = target_pose_in_world_frame[0:3, 3] # in world frame
                if ASSUME_GAZE_TARGET_IS_AT_TABLE_HEIGHT:
                    gaze_target[2] = self.table_pose[0:3,3][2] if self.table_pose is not None else TABLE_DIMENSIONS[2] # set the z-axis position to table height (use table pose if available, otherwise use the default table height)

            self.appendWithTimestamp(gaze_target, self.target_positions)
            
            # since ur5_base_pose is static in our current set up, save only once (as a single homogeneous transformation matrix)
            if self.ur5_base_pose is None:
                base = self.tfBuffer.lookup_transform("world", "base", rclpy.time.Time())
                self.ur5_base_pose = quaternion_to_homogeneous_matrix(
                    [base.transform.rotation.x, base.transform.rotation.y, base.transform.rotation.z, base.transform.rotation.w],
                    [base.transform.translation.x, base.transform.translation.y, base.transform.translation.z]
                )

        except Exception as e:
            print(e)
            self.get_logger().info("error in mocap_callback")

    def ur5_joint_state_callback(self, msg):
        if not self.is_record_time():
            return
        self.appendWithTimestamp(np.array([
                                            msg.position[5],
                                            msg.position[0],
                                            msg.position[1],
                                            msg.position[2],
                                            msg.position[3],
                                            msg.position[4]]), self.ur5_joint_states)

    def capture_rgb_frame(self):
        if not self.is_record_time():
            return
        try:
            # This callback is driven by a 10 ms ROS timer
            frames = self.rs_pipeline.poll_for_frames()        # non-blocking (polling)
            if not frames:
                return                                       # nothing new yet
            frame = frames.get_color_frame()
            ts = frame.get_timestamp()
            if ts == self.rs_last_frame_ts:
                return                       # return because you’ve already processed this frame
            # if self.rs_last_frame_ts is not None:
            #     print(f"realsense ts, diff: {ts}, {(ts - self.rs_last_frame_ts):.2f} ms")
            self.rs_last_frame_ts = ts
            cv_image = np.asanyarray(frame.get_data())
            resized_frame = cv2.resize(cv_image, (1920, 1080))
            self.rgb_images.append(resized_frame)
            self.rgb_image_timestamps.append(ts)
            
            # check time difference to ensure data collection frequency is correct (raises an error if the time difference is too large).
            # @ToDo: for some reason,
            self.check_time_difference(data_name="RGB images", data=self.rgb_image_timestamps, expected_time_diff=1000.0/RGB_FPS, margin=0.1)

        except Exception as e:
            self.get_logger().error(f"RealSense frame capture error: {e}")

    def load_head_eye_position_in_head_frame(self):
        exception_prefix = "Failed to load head-eye calibration data. You must do the calibration before data collection."
        path = HEAD_EYE_CALIB_PATH
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{exception_prefix} File not found: {path}")

        try:
            data = np.load(path)
        except Exception as e:
            raise ValueError(f"{exception_prefix} Failed to load .npy file: {e}")

        # Sanity check: ensure it’s a 1D vector of size 3
        if not isinstance(data, np.ndarray):
            raise TypeError(f"{exception_prefix} Loaded object is not a NumPy array: {type(data)}")

        if data.shape != (3,):
            raise ValueError(f"{exception_prefix} Expected shape (3,), got {data.shape}")

        return data
    
    def load_target_position_in_table_frame(self):
        exception_prefix = "Failed to load table-target calibration data."
        path = TABLE_TARGET_CALIB_DIR + "/" + self.point + ".npy"
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{exception_prefix} File not found: {path}")

        try:
            data = np.load(path)
        except Exception as e:
            raise ValueError(f"{exception_prefix} Failed to load .npy file: {e}")

        # Sanity check: ensure it’s a 1D vector of size 3
        if not isinstance(data, np.ndarray):
            raise TypeError(f"{exception_prefix} Loaded object is not a NumPy array: {type(data)}")

        if data.shape != (3,):
            raise ValueError(f"{exception_prefix} Expected shape (3,), got {data.shape}")

        return data

    def save_to_file(self):
        timestamp = int(self.get_current_timestamp_ms())
        if IS_EXPERIMENT_TYPE_LINE_MOVEMENT:
            save_dir = f"{CWD}/{EXPERIMENT_TYPE}/{self.line_movement_type}/{timestamp}"
        elif IS_EXPERIMENT_TYPE_RECTANGULAR_WAVE_MOVEMENT:
            save_dir = f"{CWD}/{EXPERIMENT_TYPE}/{timestamp}"
        else:
            save_dir = f"{CWD}/{EXPERIMENT_TYPE}/{self.point}/{timestamp}"

        # an exception will be thrown if the save directory already exists.
        os.makedirs(save_dir, exist_ok=False)

        np.save(save_dir + "/target_positions.npy", np.array(self.target_positions))
        np.save(save_dir + "/eye_positions.npy", np.array(self.eye_positions))
        np.save(save_dir + "/camera_poses.npy", np.array(self.camera_poses))
        np.save(save_dir + "/camera_intrinsics.npy", self.camera_intrinsics)
        np.save(save_dir + "/head_poses.npy", np.array(self.head_poses))
        np.save(save_dir + "/eye_position_in_head_frame.npy", np.array(self.eye_position_in_head_frame))
        np.save(save_dir + "/table_pose.npy", np.array(self.table_pose))
        np.save(save_dir + "/ur5_joint_states.npy", np.array(self.ur5_joint_states))
        np.save(save_dir + "/ur5_base_pose.npy", np.array(self.ur5_base_pose))
        np.save(save_dir + "/eye_position_in_head_frame.npy", np.array(self.eye_position_in_head_frame)) # save eye_position_in_head_frame (the calibration) to the save_dir, too. So that we can easily see what eye_position_in_head_frame vector was used for this data.
        np.save(save_dir + "/rgb_timestamps.npy", np.array(self.rgb_image_timestamps))
        rgb_video_path = save_dir + "/rgb_video.mp4"
        height, width = self.rgb_images[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'avc1' for H.264
        out = cv2.VideoWriter(rgb_video_path, fourcc, RGB_FPS, (width, height))

        for frame in self.rgb_images:
            out.write(frame)
        out.release()
        
        # --- Save current camera settings (like exposure, saturation, etc.) to file ---
        # Save to JSON file
        with open(os.path.join(save_dir, "rgb_camera_settings.json"), "w") as f:
            json.dump(self.rs_settings, f, indent=4)

        print(f"Data saved with the timestamp {timestamp}")

    
    def check_record_time(self, margin):
        """
        Checks the recording time. If the recording time deviates too much from the expected recording time, then logs to the console.
        """
        if config.should_check_record_time() is False:
            return
        observed_record_time = self.get_current_timestamp_ms() - self.started_at
        exptected_record_time = RECORD_TIME * 1000.0 # in ms
        limit = margin * exptected_record_time
        if abs(observed_record_time - exptected_record_time) > limit: # if the difference is greater than the limit, then raise error.
            self.get_logger().warning(f"WARNING: Recording time deviated too much from the expected recording time. Observed: {observed_record_time:.2f} ms, Expected: {exptected_record_time:.2f} ms.")
    
    def stop_rec_callback(self):
        self.check_record_time(margin=0.1)
        raise StopRecordingException()
    
    # run after stopping recording
    def terminate(self):
        # if this is the last point of the circular movement experiment, publish so that the robot controller does not start the next circular movement.
        if self.stop_circular_movement_at_terminate and EXPERIMENT_TYPE == "circular_movement":
            self.start_circular_movement_pub.publish(Bool(data=False))
        elif self.stop_line_movement_at_terminate and IS_EXPERIMENT_TYPE_LINE_MOVEMENT:
            self.start_line_movement_pub.publish(Bool(data=False))
        if PLAY_SOUND_WHEN_RECORDING_STOPS:
           config.play_recording_stopped_sound()
        self.save_to_file()
        self.rs_pipeline.stop()

    def line_movement_start_recording_callback(self, msg):
        if msg.data:
            self.start()
            self.line_movement_is_recording = True
        else:
            self.line_movement_is_recording = False
            self.stop_rec_callback()

    def circular_movement_start_recording_callback(self, msg):
        if msg.data:
            self.start()
        else:
            self.stop_rec_callback()

def run_point(point, stop_circular_movement_at_terminate=False, stop_line_movement_at_terminate=False, line_movement_type=None, args=None):
    rclpy.init(args=args)
    node = GazeTestDataCollector(
        point=point,stop_circular_movement_at_terminate=stop_circular_movement_at_terminate,stop_line_movement_at_terminate=stop_line_movement_at_terminate,line_movement_type=line_movement_type
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except StopRecordingException as e:
        print("SUCCESS - Recording stopped.")
    except Exception as e:
        print(e)
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        node.terminate()
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    if IS_EXPERIMENT_TYPE_RECTANGULAR_WAVE_MOVEMENT:
        print(f"Running data collection for {EXPERIMENT_TYPE}...")
        run_point(None, args)
        print(f"Data collection for {EXPERIMENT_TYPE} completed.")
    elif IS_EXPERIMENT_TYPE_LINE_MOVEMENT:
        i = 0
        l = len(LINE_MOVEMENT_TYPES)
        for i in range(0, l):
            _type = LINE_MOVEMENT_TYPES[i]
            print(f"Running data collection for {_type}...")
            run_point(point=None, stop_circular_movement_at_terminate=False, stop_line_movement_at_terminate=i == l-1, line_movement_type=_type, args=args)
            print(f"Data collection for {_type} completed.")
    else:
        i = 0
        l = len(POINTS)
        for point in POINTS:
            print(f"Running data collection for {point}...")
            # stop_circular_movement_at_terminate if experiment type is circular movement and this is the last point to be collected.
            run_point(point, stop_circular_movement_at_terminate=(EXPERIMENT_TYPE == "circular_movement" and i == l - 1), args=args)
            i += 1
            print(f"Data collection for {point} completed.")


class StopRecordingException(Exception):
    pass