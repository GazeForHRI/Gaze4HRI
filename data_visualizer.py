import time
import rclpy
import numpy as np
from skspatial.objects import Line, Plane
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from robot_controller.util import order_points_counter_clockwise
from robot_controller.gaze.data_loader import GazeDataLoader
import robot_controller.gaze.config as config
import os
from robot_controller.gaze.data_matcher import match_irregular_to_regular, match_regular_to_regular
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

# Configuration
MODEL = config.get_model()
ROOT_DIR = config.get_cwd()
EXPERIMENT_TYPE = config.get_experiment_type()
IS_EXPERIMENT_TYPE_RECTANGULAR_WAVE = config.is_experiment_type_rectangular_wave_movement()
IS_EXPERIMENT_TYPE_LINE_MOVEMENT = config.is_experiment_type_line_movement()
TARGET_EQUALS_CAMERA = IS_EXPERIMENT_TYPE_LINE_MOVEMENT
LINE_MOVEMENT_TYPES = config.get_line_movement_types()
POINTS = config.get_points()
# MAX_DURATION = 2.0 if not config.experiment_has_movement() else None # in seconds, the maximum duration to visualize each point. Set to None if you want to visualize until the end. (by setting this to a low duration like 2 sec, we can quickly go over all points)
MAX_DURATION = 2.0 if False else None # in seconds, the maximum duration to visualize each point. Set to None if you want to visualize until the end. (by setting this to a low duration like 2 sec, we can quickly go over all points)
if MAX_DURATION is not None and MAX_DURATION < 2.0:
    raise ValueError("MAX_DURATION should be at least 2.0 seconds to visualize the data properly, otherwise some points will be mistakenly skipped.")
TARGET_PERIOD = config.get_target_period()
CAMERA_POSE_PERIOD = config.get_camera_pose_period()
TIME_DIFF_MAX = config.get_time_diff_max()
TABLE_TARGET_CALIB_DIR = config.get_table_target_calib_dir()

UR5_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
UR5_JOINT_POSITIONS = [
    -0.39349014, -1.86757626, -2.45295930, 1.17818038, 1.16635084, 3.13581395
]


# TEST FUNCTIONS (start of this section)
def create_120bpm_pose_array(length = 20, frequency = 100, starting_timestamp = 0.0):
    """You can use this function to test whether the visualization is working properly.
    if visualization code is correct, visualizing the following pose array should show a pose that is moving up or down with 120 bpm rhythm (move up/down twice a second).
    For example, in GazeRVIZ::__init__() function, you can set the head_poses to this array to visualize a moving circle as follows:
    self.head_poses = create_120bpm_pose_array(length=20, frequency=100, starting_timestamp=self.start_data_ts)
    
    Args:
        length (int, optional): _description_. Defaults to 20. in seconds
        frequency (int, optional): _description_. Defaults to 100. per second (Hz)
        starting_timestamp (float, optional): _description_. Defaults to 0.0. the first timestamp value of the array, in ms, in float64. make sure to set this to the first timestamp of the other data you are visualizing if you want to visualize this array along with other data.
    """
    interval = 1000 // frequency# in ms, calculated as 1000ms / frequency per second
    down = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    
    hs = np.zeros((length * frequency, 17))
    for i in range(0, length * 1000, interval):
        ind = i // interval
        h = np.eye(4)
        if i % 500 != 0:
            h[0:3, 3] = (hs[ind-1][1:].reshape(4, 4))[0:3, 3]
        else:
            h[0:3, 3] = down if i % 1000 == 0 else up
        hs[ind] = [starting_timestamp+i, *h.flatten()]
        
    return hs

def create_moving_circle_pose_array(length = 20, frequency = 100, starting_timestamp = 0.0, radius = 1.0):
    from math import sqrt
    """You can use this function to test whether the visualization is working properly.
    if visualization code is correct, visualizing the following pose array should show a pose that traverses a quarter circle with the given radius.
    For example, in GazeRVIZ::__init__() function, you can set the head_poses to this array to visualize a moving circle as follows:
    self.head_poses = create_moving_circle_pose_array(length=20, frequency=100, starting_timestamp=self.start_data_ts, radius=1.0)

    Args:
        length (int, optional): _description_. Defaults to 20. in seconds
        frequency (int, optional): _description_. Defaults to 100. per second (Hz)
        starting_timestamp (float, optional): _description_. Defaults to 0.0. the first timestamp value of the array, in ms, in float64. make sure to set this to the first timestamp of the other data you are visualizing if you want to visualize this array along with other data.
        radius (float, optional): _description_. Defaults to 1.0. radius of the circle in meters.
    """
    interval = 1000 // frequency# in ms, calculated as 1000ms / frequency per second
    
    hs = np.zeros((length * frequency, 17))
    for i in range(0, length * 1000, interval):
        h = np.eye(4)
        x = float((i % (radius*1000))) / 1000.0
        y = sqrt(radius**2 - x**2)
        h[0:3, 3] = np.array([x, y, 0.0])
        ind = i // interval
        hs[ind] = [starting_timestamp+i, *h.flatten()]
        
    return hs
# TEST FUNCTIONS (end of this section)


class GazeRVIZ(Node):
    def __init__(self, root_dir, experiment_type, point, table_target_calib_dir, target_period=10.0, camera_pose_period=10.0, time_diff_max=20.0, max_duration=None):
        super().__init__('Gaze_RVIZ_Offline')
        self.get_logger().info('Gaze RVIZ Offline Node Started')
        
        self.root_dir = root_dir
        self.experiment_type = experiment_type
        self.point = point
        self.table_target_calib_dir = table_target_calib_dir
        self.target_period = target_period
        self.camera_pose_period = camera_pose_period
        self.time_diff_max = time_diff_max
        self.max_duration = max_duration

        # Load saved data
        self.loader = GazeDataLoader(
            root_dir=self.root_dir,
            target_period=self.target_period,
            camera_pose_period=self.camera_pose_period,
            time_diff_max=self.time_diff_max
        )
        self.eye_data = self.loader.load_eye_positions()         # (N,4)
        try:
            self.gaze_data = self.loader.load_gaze_estimations(model=MODEL, frame="world")  # (N,4)
        except Exception as e:
            self.gaze_data = None
            print("GAZE ERROR: Failed to load gaze data. Will continue by visualizing other data only. Please check if it exists.")
            print(e)
        self.table_pose = self.loader.load_table_pose()       # (4,4)
        self.camera_poses = self.loader.load_camera_poses()     # (K,17)
        self.head_poses = self.loader.load_head_poses()         # (L,17)
        self.target_positions = self.loader.load_target_positions()         # (L,17)
        # JointState publisher for robot_state_publisher
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10) # override the joint_states topic to publish replayed joint states.
        self.ur5_joint_states = self.loader.load_ur5_joint_states()  # (M,7), 7=ts+6 joint states.
        self.js_ts = self.ur5_joint_states[:, 0].astype(np.float64)     # (M,)
        self.js_pos = self.ur5_joint_states[:, 1:7].astype(np.float64)  # (M, 6)
        self._last_js_idx = 0  # optional, to keep a monotonic index

        # def pretty_print_array(arr, precision=4, width=15):
        #     # Make sure it's a NumPy array
        #     arr = np.array(arr)

        #     # Build format string like "{:15.4f}"
        #     fmt = f"{{:{width}.{precision}f}}"

        #     # Convert each element to formatted string
        #     formatted = np.vectorize(lambda x: fmt.format(x))(arr)

        #     # Join row-wise to mimic NumPy's print
        #     for row in formatted:
        #         print(" ".join(row) if isinstance(row, (np.ndarray, list)) else row)
                
        
        # print("=== UR5 Joint States ===")
        # pretty_print_array(self.ur5_joint_states)
        # exit()

        # (Optional) publish the static JointState at start, and keep sending during playback
        self._last_joint_state_stamp = None


        # Record wall-clock and data start times
        self.start_wall_ms = time.time() * 1000.0
        self.start_data_ts = float(self.head_poses[0, 0])
        self.get_logger().info(
            f"Data starts at ts={self.start_data_ts:.2f} ms; wall start={self.start_wall_ms:.2f} ms"
        )        

        self.idx = 0
        # Publishers
        self.gaze_est_pub = self.create_publisher(MarkerArray, 'gaze_est_marker_topic', 10)
        self.gaze_gt_pub = self.create_publisher(MarkerArray, 'gaze_gt_marker_topic', 10)
        self.table_pub = self.create_publisher(MarkerArray, 'table_marker_topic', 10)
        self.camera_pub = self.create_publisher(MarkerArray, 'camera_marker_topic', 10)
        self.head_pub = self.create_publisher(MarkerArray, 'head_marker_topic', 10)

        # Timer for playback at ~100 Hz
        self.playback_timer = self.create_timer(0.01, self.playback_callback)
        
        # do not load target position in table frame if experiment type is horizontal movement, since the gaze target is the camera itself (it is basically mutual gaze)
        if not IS_EXPERIMENT_TYPE_RECTANGULAR_WAVE and not IS_EXPERIMENT_TYPE_LINE_MOVEMENT:
            self.target_position_in_table_frame = self.load_target_position_in_table_frame() # will be used to obtain target position in world frame from table pose in world frame.

    def _find_nearest_joint_index(self, t_ms: float) -> int:
        """
        Return index in self.js_ts whose timestamp is closest to t_ms.
        Assumes self.js_ts is strictly increasing (monotonic).
        """
        arr = self.js_ts
        i = np.searchsorted(arr, t_ms, side='left')

        if i <= 0:
            idx = 0
        elif i >= len(arr):
            idx = len(arr) - 1
        else:
            # Pick the closer neighbor
            before = arr[i - 1]
            after = arr[i]
            idx = i - 1 if (t_ms - before) <= (after - t_ms) else i

        # Optional: enforce non-decreasing index to avoid any jitter
        if idx < self._last_js_idx:
            idx = self._last_js_idx
        return idx

    def _publish_joint_state_for_ts(self, t_ms: float):
        idx = self._find_nearest_joint_index(t_ms)
        positions = self.js_pos[idx]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()  # fresh stamp for RSP/RViz
        msg.name = UR5_JOINT_NAMES
        msg.position = positions.tolist()
        # velocities/effort omitted
        self.joint_pub.publish(msg)
        self._last_js_idx = idx


    def compute_gaze_target(self, gd, gt, table_points):
        try:
            pts = order_points_counter_clockwise(table_points)
            center = np.mean(pts, axis=0)
            normal = np.cross(pts[0] - center, pts[1] - center)
            line = Line(point=gt, direction=gd)
            plane = Plane(point=pts[0], normal=normal)
            tar = np.array(plane.intersect_line(line))
            if np.dot(tar - gt, gd) <= 0.0:
                return False, None
            first = None
            for i in range(len(pts) - 1):
                c = np.cross(pts[i+1] - pts[i], tar - pts[i+1])
                if first is None:
                    first = c
                elif np.dot(first, c) < 0:
                    return False, tar
            return True, tar
        except ValueError:
            return False, None

    def get_table_points(self, pos, rot):
        w, b = 0.35, 0.8
        return np.array([
            rot.apply([ w,0, b]) + pos,
            rot.apply([-w,0, b]) + pos,
            rot.apply([-w,0,-b]) + pos,
            rot.apply([ w,0,-b]) + pos
        ])

    def publish_axes(self, mat, publisher, frame_id, ns):
        """
        Publish X (red), Y (green), Z (blue) axes and label at pose given by 4×4 matrix `mat`.
        """
        origin = mat[:3, 3]
        rot = R.from_matrix(mat[:3, :3])
        markers = MarkerArray()
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]  # RGB
        arrow_len = 0.2

        # Arrows
        for i, axis in enumerate(np.eye(3)):
            arrow = Marker()
            arrow.header.frame_id = frame_id
            arrow.header.stamp = self.get_clock().now().to_msg()
            arrow.ns = ns
            arrow.id = i
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            start = Point(x=origin[0], y=origin[1], z=origin[2])
            axis_vec = rot.apply(axis)
            axis_unit = axis_vec / np.linalg.norm(axis_vec)
            end = Point(
                x=origin[0] + axis_unit[0] * arrow_len,
                y=origin[1] + axis_unit[1] * arrow_len,
                z=origin[2] + axis_unit[2] * arrow_len
            )
            arrow.points.extend([start, end])
            arrow.scale.x = 0.02  # thicker shaft
            arrow.scale.y = 0.04  # larger head
            arrow.scale.z = 0.0
            r, g, b = colors[i]
            arrow.color.a = 1.0
            arrow.color.r = r
            arrow.color.g = g
            arrow.color.b = b
            markers.markers.append(arrow)

        # Text label
        label = Marker()
        label.header.frame_id = frame_id
        label.header.stamp = self.get_clock().now().to_msg()
        label.ns = ns + "_label"
        label.id = 3
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = origin[0]
        label.pose.position.y = origin[1]
        label.pose.position.z = origin[2] + arrow_len + 0.05
        label.scale.z = 0.1
        label.text = ns.capitalize()
        label.color.a = 1.0
        label.color.r = 1.0
        label.color.g = 1.0
        label.color.b = 1.0
        markers.markers.append(label)

        publisher.publish(markers)
        
    def publish_x_axis_vector(self, mat, publisher, frame_id, ns, label_text=None, color=(1.0, 1.0, 0.0)):
        """
        Publish a single vector (default X-axis) as an arrow, with optional label.
        `color` should be a tuple (R, G, B).
        """
        origin = mat[:3, 3]
        rot = R.from_matrix(mat[:3, :3])
        x_axis_vec = rot.apply([1.0, 0.0, 0.0])
        x_axis_unit = x_axis_vec / np.linalg.norm(x_axis_vec)
        arrow_len = 0.2

        marker_array = MarkerArray()

        # Arrow Marker
        arrow = Marker()
        arrow.header.frame_id = frame_id
        arrow.header.stamp = self.get_clock().now().to_msg()
        arrow.ns = ns
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD

        start = Point(x=origin[0], y=origin[1], z=origin[2])
        end = Point(
            x=origin[0] + x_axis_unit[0] * arrow_len,
            y=origin[1] + x_axis_unit[1] * arrow_len,
            z=origin[2] + x_axis_unit[2] * arrow_len
        )
        arrow.points.extend([start, end])
        arrow.scale.x = 0.02
        arrow.scale.y = 0.04
        arrow.scale.z = 0.0
        arrow.color.a = 1.0
        arrow.color.r = color[0]
        arrow.color.g = color[1]
        arrow.color.b = color[2]

        marker_array.markers.append(arrow)

        # Text Marker (Label)
        if label_text:
            text = Marker()
            text.header.frame_id = frame_id
            text.header.stamp = self.get_clock().now().to_msg()
            text.ns = ns + "_label"
            text.id = 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = end.x
            text.pose.position.y = end.y
            text.pose.position.z = end.z + 0.05
            text.scale.z = 0.08
            text.color.a = 1.0
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.text = label_text
            marker_array.markers.append(text)

        publisher.publish(marker_array)

    def publish_head_dir_and_sphere(
        self,
        mat,
        publisher,
        frame_id,
        ns,
        label_text=None,
        color=(1.0, 1.0, 0.0),
        eye_pos=None,
        default_radius=0.09
    ):
        """
        Publishes the head X-axis direction (arrow) + a sphere whose DIAMETER goes
        from head origin to eye position (center = midpoint(head, eye), radius = 0.5 * distance).
        If eye_pos is None, falls back to a default-radius sphere centered at head origin.
        """
        origin = mat[:3, 3]
        rot = R.from_matrix(mat[:3, :3])
        x_axis_vec = rot.apply([1.0, 0.0, 0.0])
        x_axis_unit = x_axis_vec / np.linalg.norm(x_axis_vec)
        arrow_len = 0.2

        # --- Sphere geometry based on head↔eye ---
        if eye_pos is not None:
            eye = np.asarray(eye_pos, dtype=float)
            head = origin
            d = np.linalg.norm(head - eye)
            radius = max(1e-4, 0.5 * d)   # radius = half the distance
            sphere_center = 0.5 * (head + eye)  # midpoint
        else:
            radius = float(default_radius)
            sphere_center = origin

        marker_array = MarkerArray()

        # ---- Arrow (head direction) ----
        arrow = Marker()
        arrow.header.frame_id = frame_id
        arrow.header.stamp = self.get_clock().now().to_msg()
        arrow.ns = ns
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD

        # Start at the sphere center (midpoint of head and eye)
        start = Point(
            x=float(sphere_center[0]),
            y=float(sphere_center[1]),
            z=float(sphere_center[2])
        )
        end = Point(
            x=float(sphere_center[0] + x_axis_unit[0] * arrow_len),
            y=float(sphere_center[1] + x_axis_unit[1] * arrow_len),
            z=float(sphere_center[2] + x_axis_unit[2] * arrow_len)
        )
        arrow.points.extend([start, end])
        arrow.scale.x = 0.02
        arrow.scale.y = 0.04
        arrow.scale.z = 0.0
        arrow.color.a = 1.0
        arrow.color.r = color[0]
        arrow.color.g = color[1]
        arrow.color.b = color[2]
        marker_array.markers.append(arrow)

        # ---- Optional label ----
        if label_text:
            text = Marker()
            text.header.frame_id = frame_id
            text.header.stamp = self.get_clock().now().to_msg()
            text.ns = ns + "_label"
            text.id = 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = end.x
            text.pose.position.y = end.y
            text.pose.position.z = end.z + 0.05
            text.scale.z = 0.08
            text.color.a = 1.0
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.text = label_text
            marker_array.markers.append(text)

        # ---- Sphere (center = midpoint, radius = 0.5 * distance) ----
        sphere = Marker()
        sphere.header.frame_id = frame_id
        sphere.header.stamp = self.get_clock().now().to_msg()
        sphere.ns = ns + "_sphere"
        sphere.id = 2
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = float(sphere_center[0])
        sphere.pose.position.y = float(sphere_center[1])
        sphere.pose.position.z = float(sphere_center[2])
        sphere.pose.orientation.x = 0.0
        sphere.pose.orientation.y = 0.0
        sphere.pose.orientation.z = 0.0
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = 2.0 * radius  # diameter
        sphere.scale.y = 2.0 * radius
        sphere.scale.z = 2.0 * radius
        sphere.color.a = 0.9
        sphere.color.r = 1.0
        sphere.color.g = 1.0
        sphere.color.b = 1.0
        marker_array.markers.append(sphere)

        publisher.publish(marker_array)


    def publish_camera_prism(self, cam_mat, publisher, frame_id, ns='camera_body'):
        """
        Publish a rectangular prism representing the RealSense D435i body, whose
        dimensions and placement are specified in the CAMERA frame.

        Given camera position (x,y,z), the prism ranges are:
            x: [x - 0.025, x]
            y: [y - 0.07,   y + 0.015]
            z: [z - 0.01,   z + 0.01]

        We convert the CAMERA-frame center offset into WORLD coords via cam_mat.
        """
        origin = cam_mat[:3, 3]                 # camera position in world
        rot = R.from_matrix(cam_mat[:3, :3])    # camera orientation in world

        # Prism sizes along camera axes (meters)
        size_x = 0.025        # from x-0.025 to x
        size_y = 0.085        # from y-0.085 to y
        size_z = 0.02         # from z-0.01 to z+0.01

        # Center of the prism in CAMERA frame (midpoint of the given ranges)
        # x_center = (-0.025 + 0.0)/2 = -0.0125
        # y_center = (-0.085 + 0.0)/2 = -0.0425
        # z_center = (-0.01 + 0.01)/2 = 0.0
        center_cam = np.array([-0.0125, -0.0425, 0.0], dtype=float)


        # Transform center to WORLD frame
        center_world = origin + rot.apply(center_cam)

        # Build the marker (CUBE aligned with camera axes using the camera rotation)
        m = Marker()
        m.header.frame_id = frame_id  # 'world'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD

        m.pose.position.x = float(center_world[0])
        m.pose.position.y = float(center_world[1])
        m.pose.position.z = float(center_world[2])

        qx, qy, qz, qw = rot.as_quat()
        m.pose.orientation.x = float(qx)
        m.pose.orientation.y = float(qy)
        m.pose.orientation.z = float(qz)
        m.pose.orientation.w = float(qw)

        # RViz cube scale is full edge lengths (not radii)
        m.scale.x = size_x
        m.scale.y = size_y
        m.scale.z = size_z

        # Light gray
        m.color.a = 1.0
        m.color.r = 0.8
        m.color.g = 0.8
        m.color.b = 0.8

        arr = MarkerArray()
        arr.markers.append(m)

        # ---- Text label ("Camera") or delete it if camera==target ----
        label = Marker()
        label.header.frame_id = frame_id
        label.header.stamp = self.get_clock().now().to_msg()
        label.ns = ns + "_label"
        label.id = 1
        label.type = Marker.TEXT_VIEW_FACING
        label.pose.position.x = float(center_world[0])
        label.pose.position.y = float(center_world[1])
        label.pose.position.z = float(center_world[2] + size_z/2.0 + 0.05)
        label.scale.z = 0.08
        label.color.a = 1.0
        label.color.r = 1.0
        label.color.g = 1.0
        label.color.b = 1.0
        label.text = "Camera"

        if TARGET_EQUALS_CAMERA:
            # ensure any previous "Camera" label is removed on this topic
            label.action = Marker.DELETE
            arr.markers.append(label)
        else:
            label.action = Marker.ADD
            arr.markers.append(label)

        publisher.publish(arr)


    def playback_callback(self):
        now_wall_ms = time.time() * 1000.0
        elapsed_ms = now_wall_ms - self.start_wall_ms
        data_boundary = self.start_data_ts + elapsed_ms

        array_for_ts = self.head_poses  # timeline driver

        # Build index maps safely
        if self.gaze_data is None:
            # No estimation stream: don't try to match; just iterate head_poses
            index_array_gt = np.arange(len(self.head_poses), dtype=int)
            index_array_est = None
        else:
            index_array_gt, index_array_est = match_regular_to_regular(
                tuple1=(self.head_poses, 10),
                tuple2=(self.gaze_data, 33.33),
                max_match_diff_ms=config.get_time_diff_max(),
                stability_tolerance_ms=2.0,
                stability_window_size=5
            )

        while (
            self.idx < len(array_for_ts)
            and array_for_ts[self.idx, 0] <= data_boundary
            and (self.max_duration is None or elapsed_ms < self.max_duration * 1000)
        ):
            indx_gt = int(index_array_gt[self.idx])

            ts = float(self.head_poses[indx_gt, 0])
            _, ex, ey, ez = self.eye_data[indx_gt]
            tail = np.array([ex, ey, ez], dtype=float)
            self.get_logger().info(
                f"Gaze idx={indx_gt}, Estimation idx = {index_array_est[self.idx] if index_array_est is not None else 'N/A'} ts={ts:.2f} ms"
            )
            self.update_table(ts)
            self._publish_joint_state_for_ts(ts)

            # Publish estimation only if present
            if self.gaze_data is not None:
                indx_est = int(index_array_est[self.idx])
                _, gx, gy, gz = self.gaze_data[indx_est]
                dir_vec = np.array([gx, gy, gz], dtype=float)
                n = np.linalg.norm(dir_vec)
                if n > 0:
                    dir_vec /= n
                    success, tar = self.compute_gaze_target(dir_vec, tail, self.table_points)
                    self.get_logger().info(f"Target: {tar if success else 'none'}")

                    gaze_markers = MarkerArray()
                    arrow = Marker()
                    arrow.header.frame_id = 'world'
                    arrow.header.stamp = self.get_clock().now().to_msg()
                    arrow.id = 1
                    arrow.type = Marker.ARROW
                    arrow.action = Marker.ADD
                    p0 = Point(x=tail[0], y=tail[1], z=tail[2])
                    p1 = Point(x=tail[0]+dir_vec[0], y=tail[1]+dir_vec[1], z=tail[2]+dir_vec[2])
                    arrow.points.extend([p0, p1])
                    arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.01, 0.02, 0.0
                    arrow.color.a, arrow.color.r, arrow.color.g, arrow.color.b = 1.0, 1.0, 0.0, 0.0
                    gaze_markers.markers.append(arrow)
                    self.gaze_est_pub.publish(gaze_markers)

            # Ground-truth arrow (eye -> true target)
            ts_gt, tx, ty, tz = self.target_positions[indx_gt]
            gt_markers = MarkerArray()

            gt_arrow = Marker()
            gt_arrow.header.frame_id = 'world'
            gt_arrow.header.stamp = self.get_clock().now().to_msg()
            gt_arrow.ns = 'ground_truth'
            gt_arrow.id = 0
            gt_arrow.type = Marker.ARROW
            gt_arrow.action = Marker.ADD
            p0 = Point(x=tail[0], y=tail[1], z=tail[2])
            p1 = Point(x=tx, y=ty, z=tz)
            gt_arrow.points.extend([p0, p1])
            gt_arrow.scale.x, gt_arrow.scale.y, gt_arrow.scale.z = 0.01, 0.02, 0.0
            gt_arrow.color.a, gt_arrow.color.r, gt_arrow.color.g, gt_arrow.color.b = 1.0, 0.0, 1.0, 0.0
            gt_markers.markers.append(gt_arrow)
            
            # ---- "Gaze" label at the midpoint of the GT vector ----
            mx = 0.5 * (tail[0] + tx)
            my = 0.5 * (tail[1] + ty)
            mz = 0.5 * (tail[2] + tz)

            gaze_label = Marker()
            gaze_label.header.frame_id = 'world'
            gaze_label.header.stamp = self.get_clock().now().to_msg()
            gaze_label.ns = 'ground_truth_label'
            gaze_label.id = 3                     # keep unique; 0 is arrow, 1 target label, 2 combo label
            gaze_label.type = Marker.TEXT_VIEW_FACING
            gaze_label.action = Marker.ADD
            gaze_label.pose.position.x = mx
            gaze_label.pose.position.y = my
            gaze_label.pose.position.z = mz + 0.07  # slight lift so it doesn't overlap the arrow
            gaze_label.scale.z = 0.08
            gaze_label.color.a = 1.0
            gaze_label.color.r = 1.0
            gaze_label.color.g = 1.0
            gaze_label.color.b = 1.0
            gaze_label.text = "Gaze"
            gt_markers.markers.append(gaze_label)

            # ---- Target / Camera=Target labels with cleanup ----
            if TARGET_EQUALS_CAMERA:
                # 1) DELETE any old "Target" label
                del_target = Marker()
                del_target.header.frame_id = 'world'
                del_target.header.stamp = self.get_clock().now().to_msg()
                del_target.ns = 'ground_truth_label'
                del_target.id = 1
                del_target.type = Marker.TEXT_VIEW_FACING
                del_target.action = Marker.DELETE
                gt_markers.markers.append(del_target)

                # 2) ADD/UPDATE the combined "Camera=\nTarget" label
                combo_label = Marker()
                combo_label.header.frame_id = 'world'
                combo_label.header.stamp = self.get_clock().now().to_msg()
                combo_label.ns = 'camera_target_label'
                combo_label.id = 2
                combo_label.type = Marker.TEXT_VIEW_FACING
                combo_label.action = Marker.ADD
                combo_label.pose.position.x = tx
                combo_label.pose.position.y = ty
                combo_label.pose.position.z = tz + 0.07
                combo_label.scale.z = 0.05
                combo_label.color.a = 1.0
                combo_label.color.r = 1.0
                combo_label.color.g = 1.0
                combo_label.color.b = 1.0
                combo_label.text = "Camera=Target"
                gt_markers.markers.append(combo_label)
            else:
                # 1) DELETE any old combined label
                del_combo = Marker()
                del_combo.header.frame_id = 'world'
                del_combo.header.stamp = self.get_clock().now().to_msg()
                del_combo.ns = 'camera_target_label'
                del_combo.id = 2
                del_combo.type = Marker.TEXT_VIEW_FACING
                del_combo.action = Marker.DELETE
                gt_markers.markers.append(del_combo)

                # 2) ADD/UPDATE the "Target" label
                gt_label = Marker()
                gt_label.header.frame_id = 'world'
                gt_label.header.stamp = self.get_clock().now().to_msg()
                gt_label.ns = 'ground_truth_label'
                gt_label.id = 1
                gt_label.type = Marker.TEXT_VIEW_FACING
                gt_label.action = Marker.ADD
                gt_label.pose.position.x = tx
                gt_label.pose.position.y = ty
                gt_label.pose.position.z = tz + 0.05
                gt_label.scale.z = 0.08
                gt_label.color.a = 1.0
                gt_label.color.r = 1.0
                gt_label.color.g = 1.0
                gt_label.color.b = 1.0
                gt_label.text = "Target"
                gt_markers.markers.append(gt_label)
            self.gaze_gt_pub.publish(gt_markers)

            # Camera pose (labelled X-axis vector)
            cam_idx = int(np.argmin(np.abs(self.camera_poses[:, 0] - ts)))
            cam_mat = self.camera_poses[cam_idx, 1:].reshape(4, 4)
            self.publish_camera_prism(cam_mat, self.camera_pub, 'world', ns='camera_body')
            head_idx = int(np.argmin(np.abs(self.head_poses[:, 0] - ts)))
            head_mat = self.head_poses[head_idx, 1:].reshape(4, 4)
            self.publish_head_dir_and_sphere(
                head_mat,
                self.head_pub,
                'world',
                'head',
                label_text='Head',
                eye_pos=tail,  # <-- eye position [ex, ey, ez]
                color=(0.0, 0.0, 1.0)  # blue
            )
            self.idx += 1

        if self.idx >= len(array_for_ts) or (self.max_duration is not None and elapsed_ms >= self.max_duration * 1000):
            raise StopVisualizingException()

    def update_table(self, data_ts=None):
        mat = self.table_pose
        rot = R.from_matrix(mat[:3, :3])
        pos = mat[:3, 3]
        self.table_points = self.get_table_points(pos, rot)
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = pos
        marker.scale.x, marker.scale.y, marker.scale.z = 0.7, 0.001, 1.6
        marker.color.a, marker.color.r, marker.color.g, marker.color.b = 1.0, 205.0 / 255.0, 170.0 / 255.0, 125.0 / 255.0
        quat = rot.as_quat()
        marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w = quat
        arr = MarkerArray()
        arr.markers.append(marker)
        self.table_pub.publish(arr)

    def load_target_position_in_table_frame(self):
        exception_prefix = "Failed to load table-target calibration data."
        path = self.table_target_calib_dir + "/" + self.point + ".npy"
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

    def _publish_joint_state(self):
        """
        Publish a JointState that matches the URDF joint names.
        robot_state_publisher will use this to compute TF for all links.
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()  # important for RSP
        msg.name = UR5_JOINT_NAMES
        msg.position = UR5_JOINT_POSITIONS
        # velocities/effort optional; leave empty
        self.joint_pub.publish(msg)
        self._last_joint_state_stamp = msg.header.stamp


def run_point(point, line_movement_type=None, args=None):
    rclpy.init(args=args)
    
    if IS_EXPERIMENT_TYPE_LINE_MOVEMENT:
        root_dir = f"{ROOT_DIR}/{EXPERIMENT_TYPE}/{line_movement_type}"
    elif IS_EXPERIMENT_TYPE_RECTANGULAR_WAVE:
        root_dir = f"{ROOT_DIR}/{EXPERIMENT_TYPE}"
    else:
        root_dir = f"{ROOT_DIR}/{EXPERIMENT_TYPE}/{point}"

    node = GazeRVIZ(root_dir=root_dir, experiment_type=EXPERIMENT_TYPE, point=point, table_target_calib_dir=TABLE_TARGET_CALIB_DIR, target_period=TARGET_PERIOD, camera_pose_period=CAMERA_POSE_PERIOD, time_diff_max=TIME_DIFF_MAX, max_duration=MAX_DURATION)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except StopVisualizingException as e:
        print("SUCCESS - Visualization stopped.")
    except Exception as e:
        print(e)
        print(f"An error occurred: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    if IS_EXPERIMENT_TYPE_RECTANGULAR_WAVE:
        run_point(point=None, args=args)
    elif IS_EXPERIMENT_TYPE_LINE_MOVEMENT:
        for line_movement_type in LINE_MOVEMENT_TYPES:
            print(f"Running data visualization for {line_movement_type} line movement...")
            run_point(point=None, line_movement_type=line_movement_type, args=args)
            print(f"Data visualization for {line_movement_type} line movement completed.")
    else:
        for point in POINTS:
            print(f"Running data visualization for {point}...")
            run_point(point=point, args=args)
            print(f"Data visualization for {point} completed.")
        
class StopVisualizingException(Exception):
    pass