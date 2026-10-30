"""ROS 2 adapter. Inference never runs in the subscription/executor thread."""
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                      QoSHistoryPolicy, QoSDurabilityPolicy)
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool

from .core import FusionConfig, LaneFusion, draw_result
from .geometry import CameraGeometry
from .runtime import LatestFrameRuntime, RuntimeConfig


class LaneFusionNode(Node):
    def __init__(self):
        super().__init__("lane_detection_fusion")
        share = Path(get_package_share_directory("lane_detection_fusion"))
        self.declare_parameter("profile", str(share / "config" / "profile.json"))
        self.declare_parameter("camera_topic", "/camera_raw_16_9")
        self.declare_parameter("drive_enabled", False)
        self.declare_parameter("debug_enabled", False)
        self.declare_parameter("debug_hz", 2.0)
        self.declare_parameter("debug_topic", "/lane_fusion/debug/compressed")
        profile_path = Path(self.get_parameter("profile").value).expanduser()
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if not isinstance(profile, dict):
            raise ValueError("profile must contain a JSON object")
        self.runtime_config = RuntimeConfig(**profile.get("runtime", {}))
        self._drive_enabled = self.get_parameter("drive_enabled").value
        self._debug_enabled = self.get_parameter("debug_enabled").value
        if not isinstance(self._drive_enabled, bool) or not isinstance(self._debug_enabled, bool):
            raise ValueError("drive_enabled/debug_enabled must be bool")
        self._debug_hz = float(self.get_parameter("debug_hz").value)
        if not math.isfinite(self._debug_hz) or not 0 < self._debug_hz <= 10:
            raise ValueError("debug_hz must be in (0, 10]")
        camera = profile.get("camera", {})
        calibration = camera.get("calibration_file")
        if not isinstance(calibration, str) or not calibration:
            raise ValueError("camera.calibration_file is required")
        calibration_path = Path(calibration).expanduser()
        if not calibration_path.is_absolute():
            options = [profile_path.parent / calibration_path, share / calibration_path]
            calibration_path = next((p for p in options if p.is_file()), options[0])
        self.geometry = CameraGeometry(
            str(calibration_path),
            input_mode=camera.get("input_mode", "raw_16_9"),
            process_width=camera.get("process_width", 640))
        self.fusion = LaneFusion(FusionConfig(**profile.get("vision", {})))
        self._worker_generation = None
        self._last_debug = -float("inf")
        output_topic = "/cmd_vel" if self._drive_enabled else "/lane_fusion/cmd_vel_preview"
        self.command_publisher = self.create_publisher(Twist, output_topic, 1)
        self.debug_publisher = self.create_publisher(
            CompressedImage, self.get_parameter("debug_topic").value, 1)
        self.runtime = LatestFrameRuntime(
            self._infer, self._publish_command, self.runtime_config,
            ros_now=lambda: self.get_clock().now().nanoseconds / 1e9)
        camera_qos = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST,
                                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                durability=QoSDurabilityPolicy.VOLATILE)
        self.camera_subscription = self.create_subscription(
            CompressedImage, self.get_parameter("camera_topic").value,
            self._camera, camera_qos)
        self.state_subscription = self.create_subscription(Bool, "/lane_state", self._state, 10)
        self.mode_subscription = self.create_subscription(Bool, "/lane_mode", self._mode, 10)
        self.intersection_subscription = self.create_subscription(
            Bool, "/intersection_state", self._intersection, 10)
        self.add_on_set_parameters_callback(self._parameters)
        self.runtime.set_enabled(False)
        self.get_logger().info(
            "output=%s; initial /lane_state=False; mode is a hint, not a color/half selector" % output_topic)

    def _parameters(self, parameters):
        # Changing geometry/control while a job is running requires a coherent
        # new profile/generation. Restart rather than partially apply settings.
        if parameters:
            return SetParametersResult(successful=False,
                                       reason="Restart node to apply profile or adapter parameters")
        return SetParametersResult(successful=True)

    def _camera(self, message):
        stamp = message.header.stamp
        stamp_s = float(stamp.sec) + float(stamp.nanosec) / 1e9
        self.runtime.submit(message, stamp_s)

    def _state(self, message):
        self.runtime.set_enabled(message.data)

    def _mode(self, message):
        # Preserve integration interface without forcing color or image half.
        self.runtime.set_mode_hint(message.data)

    def _intersection(self, message):
        self.runtime.set_intersection(message.data)

    def _infer(self, job):
        if self._worker_generation != job.generation:
            self.fusion.reset()
            self._worker_generation = job.generation
        message = job.payload
        image = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("CompressedImage could not be decoded")
        prepared = self.geometry.prepare(image)
        # Geometry.prepare returns the processing image; no second undistort or
        # implicit resize/crop is added by this adapter.
        result = self.fusion.process(prepared, job.received_monotonic)
        if self._debug_enabled and time.monotonic() - self._last_debug >= 1.0 / self._debug_hz:
            self._last_debug = time.monotonic()
            try:
                display = draw_result(prepared.copy(), result)
                ok, encoded = cv2.imencode(".jpg", display)
                if ok:
                    debug = CompressedImage()
                    debug.header = message.header
                    debug.format = "jpeg"
                    debug.data = encoded.tobytes()
                    self.debug_publisher.publish(debug)
            except Exception as exc:
                self.get_logger().warning("debug image skipped: %s" % exc)
        return result

    def _publish_command(self, command):
        message = Twist()
        message.linear.x = float(command.speed)
        message.angular.z = float(command.angular)
        self.command_publisher.publish(message)

    def close(self):
        if hasattr(self, "runtime"):
            self.runtime.close()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LaneFusionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
