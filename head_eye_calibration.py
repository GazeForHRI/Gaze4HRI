import rclpy
import numpy as np
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import time
import cv2
import os
from playsound import playsound
import robot_controller.gaze.config as config


# WARNING: make sure to update get_head_eye_calib_path() in config.py to use the saved calibration


CWD = config.get_cwd()
RECORD_TIME = 5 # the time of recording (actual data collection), in seconds
WAIT_TIME = 2 # The time between starting the node and starting the recording, in seconds
HEAD_TRACKER = config.get_head_tracker() # the head tracker object used for ground truth. currently only option is rigid_body_2 (yellow helmet)
EYE_CALIBRATOR = config.get_eye_device_tracker() # the eye calibration device object used for ground truth. currently only option is rigid_body_5
EYE_DEPTH = 15 # the depth of the eye calibration device from the head tracker, in mm
START_TRIGGER = 'mouse' # use 'mouse' or 'keyboard'
PLAY_SOUND_WHEN_RECORDING_STARTS = True
PLAY_SOUND_WHEN_RECORDING_STOPS = True

class GazeHeadEyeCalibrator(Node):

    def init_tf(self):
        self.tfBuffer = Buffer()
        self.tfListener = TransformListener(self.tfBuffer, self)
        can_transform = False
        
        while not can_transform:
            try:
                can_transform = self.tfBuffer.can_transform("base", "wrist_3_link", rclpy.time.Time()) and self.tfBuffer.can_transform("world", "base", rclpy.time.Time()) and self.tfBuffer.can_transform(self.head_tracker, self.eye_calibrator, rclpy.time.Time())
            except Exception as e:
                pass
            finally:
                if can_transform:
                    self.get_logger().info('TF tree received.')
                    break
                else:
                    rclpy.spin_once(self)

    def __init__(self):
        super().__init__('gaze_head_eye_calibration_node')
        self.get_logger().info('gaze_head_eye_calibration_node Started')

        self.started_at = None # the time at which the data collection process started (not necessarily the time at which the node started or not necessarily, see should_record function for details of use)
        self.eye_positions_in_head_frame = [] # in head frame
    
        self.head_tracker = HEAD_TRACKER
        self.eye_calibrator = EYE_CALIBRATOR
        
        self.init_tf()
        self.start_calibration()
        
        self.collect_calibation_data_timer = self.create_timer(0.01, self.collect_calibation_data)

    def start_calibration(self):
        if START_TRIGGER == 'mouse':
            self.get_logger().info("Please LEFT CLICK to start head-eye calibration when ready...")

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
              raise ValueError("Head-eye calibration failed, no mouse event has been received.")
        elif START_TRIGGER == 'keyboard':
            input("Press ENTER to start head-eye calibration when ready...")
        else:
            raise ValueError("Invalid START_TRIGGER value. Use 'mouse' or 'keyboard'.")
        
        self.get_logger().info("Head-eye calibration started, maintain subject's position throughout the calibration process (until a sound occurs).")
        
        while not self.tfBuffer.can_transform(self.head_tracker, self.eye_calibrator, rclpy.time.Time()) or not self.tfBuffer.can_transform("world", self.head_tracker, rclpy.time.Time()):
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

        tf = self.tfBuffer.lookup_transform(self.head_tracker, self.eye_calibrator, rclpy.time.Time())
        eye_position_in_head_frame = np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z]) + np.array([-EYE_DEPTH/1000.0, 0.0, 0.0])
        self.eye_positions_in_head_frame.append(eye_position_in_head_frame)

    def should_record(self):
        """Whether the current time is within the recording time (should we record at current time)"""
        return self.started_at is not None and self.started_at + float(WAIT_TIME) <= time.time() and time.time() <= self.started_at + float(WAIT_TIME) + float(RECORD_TIME)
        
    def estimate_eye_position_in_head_frame(self):
        if len(self.eye_positions_in_head_frame) == 0:
            raise ValueError("No data collected for eye position estimation.")
        
        positions = np.array(self.eye_positions_in_head_frame)

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
        save_path = os.path.join(CWD, "eye_position_in_head_frame.npy")

        try:
            # Ensure directory exists
            os.makedirs(CWD, exist_ok=True)

            # Estimate vector
            vector = self.estimate_eye_position_in_head_frame()

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
    node = GazeHeadEyeCalibrator()

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