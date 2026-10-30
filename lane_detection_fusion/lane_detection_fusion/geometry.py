"""Explicit camera input preparation; no metric ground projection is implied."""
import json
from pathlib import Path

import cv2
import numpy as np


def _numeric_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "uif" or not np.isfinite(array).all():
        raise ValueError("{} must contain finite numeric values".format(name))
    return array.astype(np.float64)


def validate_calibration(data):
    """Validate the native pinhole calibration; return independent K/D copies."""
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unsupported calibration JSON schema")
    if data.get("camera_model") != "opencv_pinhole":
        raise ValueError("Expected OpenCV pinhole distortion model")
    if data.get("image_size") != [1280, 720]:
        raise ValueError("Calibration must describe native 1280x720 images")
    matrix = _numeric_array(data.get("camera_matrix"), "camera_matrix")
    distortion = _numeric_array(data.get("dist_coeffs"), "dist_coeffs")
    if matrix.shape != (3, 3):
        raise ValueError("camera_matrix must be 3x3")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("Focal lengths must be positive")
    if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-12, rtol=0):
        raise ValueError("Invalid homogeneous camera-matrix row")
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValueError("Singular camera_matrix")
    if distortion.ndim not in (1, 2) or distortion.size not in (4, 5, 8, 12, 14):
        raise ValueError("Unsupported OpenCV distortion vector")
    if distortion.ndim == 2 and min(distortion.shape) != 1:
        raise ValueError("dist_coeffs must be a vector")
    return matrix.copy(), distortion.reshape(-1).copy()


class CameraGeometry:
    """Prepare native raw or explicitly already-rectified BGR images.

    raw_16_9 requires native 1280x720. Undistortion uses a full-size alpha=0
    canvas, without applying the publisher's ROI crop or 4:3 stretching.
    rectified skips distortion correction and locks its first observed size.
    The latter may be the legacy /camera 640x480 view; its corrected intrinsic
    matrix cannot be recovered from this raw calibration alone.
    """

    def __init__(self, calibration_path, input_mode="raw_16_9", process_width=640):
        if input_mode not in ("raw_16_9", "rectified"):
            raise ValueError("input_mode must be raw_16_9 or rectified")
        if isinstance(process_width, bool) or not isinstance(process_width, int) or process_width <= 0:
            raise ValueError("process_width must be a positive integer")
        self.calibration_path = str(Path(calibration_path).resolve())
        with open(self.calibration_path, "r", encoding="utf-8") as handle:
            self.calibration = json.load(handle)
        self.camera_matrix, self.dist_coeffs = validate_calibration(self.calibration)
        self.input_mode = input_mode
        self.process_width = process_width
        self.input_size = (1280, 720) if input_mode == "raw_16_9" else None
        self.output_size = None
        self.new_camera_matrix = None
        self.camera_matrix_processed = None
        self.valid_roi = None
        self._map1 = self._map2 = None
        if input_mode == "raw_16_9":
            self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, self.input_size, 0, self.input_size)
            if not np.isfinite(self.new_camera_matrix).all():
                raise ValueError("Non-finite rectified camera matrix")
            self.valid_roi = tuple(int(value) for value in roi)
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix,
                self.input_size, cv2.CV_16SC2)
        self.last_info = {
            "input_mode": input_mode,
            "input_size": list(self.input_size) if self.input_size else None,
            "output_size": None,
            "remap_applied": False,
            "roi_crop_applied": False,
            "ground_metric_calibration_available": False,
        }

    @property
    def metadata(self):
        return dict(self.last_info)

    def prepare(self, image):
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a nonempty HxWx3 BGR ndarray")
        if image.dtype != np.uint8 or image.shape[0] < 1 or image.shape[1] < 1:
            raise ValueError("Expected finite uint8 BGR pixels")
        size = (int(image.shape[1]), int(image.shape[0]))
        if self.input_size is not None and size != self.input_size:
            raise ValueError("Input size {} differs from expected {}; select the correct input mode".format(
                size, self.input_size))
        if self.input_size is None:
            self.input_size = size
        height = max(1, int(round(size[1] * float(self.process_width) / size[0])))
        output_size = (self.process_width, height)
        if self.input_mode == "raw_16_9":
            corrected = cv2.remap(image, self._map1, self._map2, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
            scale = np.diag([float(output_size[0]) / size[0],
                             float(output_size[1]) / size[1], 1.0])
            self.camera_matrix_processed = np.dot(scale, self.new_camera_matrix)
        else:
            corrected = image
        prepared = (corrected.copy() if size == output_size else
                    cv2.resize(corrected, output_size, interpolation=cv2.INTER_AREA))
        self.output_size = output_size
        self.last_info = {
            "input_mode": self.input_mode,
            "input_size": list(size),
            "output_size": list(output_size),
            "remap_applied": self.input_mode == "raw_16_9",
            "undistort_alpha": 0 if self.input_mode == "raw_16_9" else None,
            "valid_roi_not_cropped": list(self.valid_roi) if self.valid_roi is not None else None,
            "roi_crop_applied": False,
            "resize_policy": "aspect ratio preserved, integer-pixel rounding only",
            "processed_camera_matrix": self.camera_matrix_processed.tolist()
            if self.camera_matrix_processed is not None else None,
            "rectified_intrinsics_known": self.camera_matrix_processed is not None,
            "ground_metric_calibration_available": False,
        }
        return prepared
