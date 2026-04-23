import rclpy
import numpy as np
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import time
import cv2
import os
from playsound import playsound
from robot_controller.util import quaternion_to_homogeneous_matrix
import robot_controller.gaze.config as config


# WARNING: make sure to update get_table_target_calib_dir() in config.py to use the saved calibration


CWD = config.get_table_target_calib_dir()
TARGETNAME = "p1"
RECORD_TIME = 5 # the time of recording (actual data collection), in seconds
WAIT_TIME = 1 # The time between starting the node and starting the recording, in seconds
TABLE_TRACKER = "rigid_body_3"
TARGET_TRACKER = "rigid_body_5" 
START_TRIGGER = 'mouse' # use 'mouse' or 'keyboard'
PLAY_SOUND_WHEN_RECORDING_STARTS = True
PLAY_SOUND_WHEN_RECORDING_STOPS = True
DEBUG_PRINT = False

class GazeTableTargetCalibrator(Node):

    def init_tf(self):
        self.tfBuffer = Buffer()
        self.tfListener = TransformListener(self.tfBuffer, self)
        can_transform = False
        
        while not can_transform:
            try:
                can_transform = self.tfBuffer.can_transform("base", "wrist_3_link", rclpy.time.Time()) and self.tfBuffer.can_transform("world", "base", rclpy.time.Time()) and self.tfBuffer.can_transform(self.table_tracker, self.target_calibrator, rclpy.time.Time()) and self.tfBuffer.can_transform("world", self.table_tracker, rclpy.time.Time())
            except Exception as e:
                pass
            finally:
                if can_transform:
                    self.get_logger().info('TF tree received.')
                    break
                else:
                    rclpy.spin_once(self)

    def __init__(self):
        super().__init__('gaze_table_target_calibration_node')
        self.get_logger().info('gaze_table_target_calibration_node Started')

        self.started_at = None # the time at which the data collection process started (not necessarily the time at which the node started or not necessarily, see should_record function for details of use)
        self.target_positions_in_table_frame = [] # in table frame
    
        self.table_tracker = TABLE_TRACKER
        self.target_calibrator = TARGET_TRACKER
        
        self.init_tf()
        self.start_calibration()
        
        self.collect_calibation_data_timer = self.create_timer(0.01, self.collect_calibation_data)

    def start_calibration(self):
        if START_TRIGGER == 'mouse':
            self.get_logger().info("Please LEFT CLICK to start calibration when ready...")

            clicked = [False]

            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    clicked[0] = True

            # Create a full-screen OpenCV window
            window_name = "Click to Start Calibration"
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback(window_name, mouse_callback)

            # Create a black screen
            screen_res = (1080, 1920, 3)  # You can adjust this if your resolution is different
            black_screen = np.zeros(screen_res, dtype=np.uint8)
            cv2.putText(black_screen, "Left click anywhere to start calibration", (100, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            # Wait until mouse click
            while not clicked[0]:
                cv2.imshow(window_name, black_screen)
                cv2.waitKey(100)

            cv2.destroyWindow(window_name)
            if not clicked[0]:
              raise ValueError("calibration failed, no mouse event has been received.")
        elif START_TRIGGER == 'keyboard':
            input("Press ENTER to start Calibration when ready...")
        else:
            raise ValueError("Invalid START_TRIGGER value. Use 'mouse' or 'keyboard'.")
        
        self.get_logger().info("Calibration started, maintain subject's position throughout the calibration process (until a sound occurs).")
        
        while not self.tfBuffer.can_transform(self.table_tracker, self.target_calibrator, rclpy.time.Time()) or not self.tfBuffer.can_transform("world", self.table_tracker, rclpy.time.Time()):
            rclpy.spin_once(self)

        self.start()
        
    def start(self):
        self.started_at = time.time()
        
        if PLAY_SOUND_WHEN_RECORDING_STARTS:
            self.play_recording_started_sound()

        self.stop_rec_timer = self.create_timer(RECORD_TIME if WAIT_TIME == 0 else RECORD_TIME+WAIT_TIME, self.stop_rec_callback)

    def collect_calibation_data(self):
        if not self.should_record():
            return

        if DEBUG_PRINT:
            table_pose = self.tfBuffer.lookup_transform("world", self.table_tracker, rclpy.time.Time()).transform
            table_pose_quat = np.array([table_pose.rotation.x, table_pose.rotation.y, table_pose.rotation.z, table_pose.rotation.w])
            table_pose_trans = np.array([table_pose.translation.x, table_pose.translation.y, table_pose.translation.z])
            print(f"Table pose quat: {table_pose.rotation.x}, {table_pose.rotation.y}, {table_pose.rotation.z}, {table_pose.rotation.w}")
            print(f"Table pose trans: {table_pose.translation.x}, {table_pose.translation.y}, {table_pose.translation.z}")
            table_pose = quaternion_to_homogeneous_matrix(table_pose_quat, table_pose_trans)

            target_pose = self.tfBuffer.lookup_transform("world", self.target_calibrator, rclpy.time.Time()).transform
            target_pose_quat = np.array([target_pose.rotation.x, target_pose.rotation.y, target_pose.rotation.z, target_pose.rotation.w])
            target_pose_trans = np.array([target_pose.translation.x, target_pose.translation.y, target_pose.translation.z])
            print(f"Target pose quat: {target_pose.rotation.x}, {target_pose.rotation.y}, {target_pose.rotation.z}, {target_pose.rotation.w}")
            print(f"Target pose trans: {target_pose.translation.x}, {target_pose.translation.y}, {target_pose.translation.z}")
            target_pose = quaternion_to_homogeneous_matrix(target_pose_quat, target_pose_trans)
            calculated_tf = np.linalg.inv(table_pose) @ target_pose
            print(f"Calculated TF homogenous: {calculated_tf}")

        tf = self.tfBuffer.lookup_transform(self.table_tracker, self.target_calibrator, rclpy.time.Time()).transform
        if DEBUG_PRINT:
            tf_quat = np.array([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w])
            tf_trans = np.array([tf.translation.x, tf.translation.y, tf.translation.z])
            print(f"looked up TF homogenous: {quaternion_to_homogeneous_matrix(tf_quat, tf_trans)}")
        target_position_in_table_frame = np.array([tf.translation.x, tf.translation.y, tf.translation.z])
        self.target_positions_in_table_frame.append(target_position_in_table_frame)

    def should_record(self):
        """Whether the current time is within the recording time (should we record at current time)"""
        return self.started_at is not None and self.started_at + float(WAIT_TIME) <= time.time() and time.time() <= self.started_at + float(WAIT_TIME) + float(RECORD_TIME)
        
    def estimate_target_position_in_table_frame(self):
        if len(self.target_positions_in_table_frame) == 0:
            raise ValueError("No data collected for eye position estimation.")
        
        positions = np.array(self.target_positions_in_table_frame)

        # Discard the first 20% of samples — likely to be noisy
        discard_ratio = 0.2
        cutoff = int(len(positions) * discard_ratio)
        if cutoff >= len(positions):
            raise ValueError("Not enough data left after discarding initial samples.")
        
        filtered = positions[cutoff:]

        # Use median absolute deviation to remove outliers
        median = np.median(filtered, axis=0)
        mad = np.median(np.abs(filtered - median), axis=0)
        threshold = 3.0  # Tune this if needed

        non_outlier_mask = np.all(np.abs(filtered - median) <= threshold * (mad + 1e-6), axis=1)
        cleaned = filtered[non_outlier_mask]

        if len(cleaned) == 0:
            raise ValueError("All samples filtered out as outliers.")

        return np.mean(cleaned, axis=0)

    def save_to_file(self):
        save_path = os.path.join(CWD, f"{TARGETNAME}.npy")

        try:
            # Ensure directory exists
            os.makedirs(CWD, exist_ok=True)

            # Estimate vector
            vector = self.estimate_target_position_in_table_frame()

            # Sanity check
            if not isinstance(vector, np.ndarray) or vector.shape != (3,):
                raise ValueError(f"Estimated vector must be a NumPy array of shape (3,), got shape {vector.shape}")

            # Save to file
            np.save(save_path, vector)
            print(f"Data saved to file: {save_path}")

        except Exception as e:
            self.get_logger().error(f"Failed to save eye position vector: {e}")

    def stop_rec_callback(self):
        raise StopRecordingException()
    
    # run after stopping recording
    def terminate(self):
        if PLAY_SOUND_WHEN_RECORDING_STOPS:
           self.play_recording_stopped_sound()
        self.save_to_file()

    def play_recording_stopped_sound(self):
        playsound('/usr/share/sounds/freedesktop/stereo/complete.oga')  # or any short WAV/MP3 file

    def play_recording_started_sound(self):
        playsound('/usr/share/sounds/freedesktop/stereo/complete.oga')  # or any short WAV/MP3 file


def main(args=None):
    rclpy.init(args=args)
    node = GazeTableTargetCalibrator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except StopRecordingException:
        print("SUCCESS - Recording stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        node.terminate()
        if rclpy.ok():
            rclpy.shutdown()

class StopRecordingException(Exception):
    pass