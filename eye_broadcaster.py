import rclpy
import numpy as np
from rclpy.node import Node
from robot_controller.util import quaternion_to_homogeneous_matrix
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import robot_controller.gaze.config as config
import os

HELMET = config.get_head_tracker()
HEAD_EYE_CALIB_PATH = config.get_head_eye_calib_path()

class GazeTestDataCollector(Node):

    def load_head_eye_position_in_head_frame(self):
        exception_prefix = "Failed to load head-eye calibration data."
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
    
    def init_tf(self):
        self.tfBuffer = Buffer()
        self.tfListener = TransformListener(self.tfBuffer, self)
        can_transform = False
        
        while not can_transform:
            try:
                can_transform = self.tfBuffer.can_transform("world", HELMET, rclpy.time.Time())
            except Exception as e:
                print(e)
                pass
            finally:
                if can_transform:
                    self.get_logger().info('TF tree received.')
                    break
                else:
                    rclpy.spin_once(self)
    
    def __init__(self):

        super().__init__('gaze_test_data_collector')
        self.get_logger().info('eye_publisher Node Started')
                
        self.eye_position_in_head_frame = self.load_head_eye_position_in_head_frame() # will be used to obtain eye position in world frame from head pose in world frame.
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        self.init_tf()
        
        self.broadcast_timer = self.create_timer(0.01, self.eye_callback)

    def eye_callback(self):
        try:
            head_pose = self.tfBuffer.lookup_transform("world", HELMET, rclpy.time.Time()).transform
            head_pose_quat = np.array([head_pose.rotation.x, head_pose.rotation.y, head_pose.rotation.z, head_pose.rotation.w])
            head_pose_trans = np.array([head_pose.translation.x, head_pose.translation.y, head_pose.translation.z])
            head_pose = quaternion_to_homogeneous_matrix(head_pose_quat, head_pose_trans)

            eye_pose_in_head_frame = np.eye(4)
            # eye and head have the same orientation, so we can use the same rotation matrix.
            # so, we will only set the translation vector.
            eye_pose_in_head_frame[0:3, 3] = self.eye_position_in_head_frame
            eye_pose_in_world_frame =  head_pose @ eye_pose_in_head_frame
            
            transform = tf2_ros.TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = "world"
            transform.child_frame_id = "eye"
            transform.transform.translation.x = eye_pose_in_world_frame[0, 3]
            transform.transform.translation.y = eye_pose_in_world_frame[1, 3]
            transform.transform.translation.z = eye_pose_in_world_frame[2, 3]
            transform.transform.rotation.x = head_pose_quat[0]
            transform.transform.rotation.y = head_pose_quat[1]
            transform.transform.rotation.z = head_pose_quat[2]
            transform.transform.rotation.w = head_pose_quat[3]
            self.tf_broadcaster.sendTransform(transform)
                                        
        except Exception as e:
            print(e)
            self.get_logger().info("error in eye_callback")

def main(args=None):
    
    rclpy.init(args=args)
    node = GazeTestDataCollector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()