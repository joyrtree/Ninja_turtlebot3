"""Geometry/input-contract checks; no ROS or camera hardware is used."""
import copy
import importlib.util
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lane_detection_fusion.geometry import CameraGeometry, validate_calibration


REPO = Path(__file__).resolve().parents[1]
CALIBRATION = REPO / "config/camera_calibration.json"


class GeometryTests(unittest.TestCase):
    def test_native_aspect_and_map_cache(self):
        geometry = CameraGeometry(CALIBRATION)
        image = np.zeros((720, 1280, 3), np.uint8)
        map_id = id(geometry._map1)
        self.assertEqual(geometry.prepare(image).shape, (360, 640, 3))
        self.assertEqual(geometry.prepare(image).shape, (360, 640, 3))
        self.assertEqual(id(geometry._map1), map_id)
        self.assertFalse(geometry.metadata["roi_crop_applied"])
        with self.assertRaises(ValueError):
            geometry.prepare(np.zeros((480, 640, 3), np.uint8))
        with self.assertRaises(ValueError):
            geometry.prepare(np.full((720, 1280, 3), np.nan))

    def test_rectified_never_remaps_and_shape_is_locked(self):
        geometry = CameraGeometry(CALIBRATION, input_mode="rectified")
        image = np.full((480, 640, 3), 127, np.uint8)
        with patch("lane_detection_fusion.geometry.cv2.remap", side_effect=AssertionError("double undistort")):
            np.testing.assert_array_equal(geometry.prepare(image), image)
        self.assertFalse(geometry.metadata["remap_applied"])
        self.assertIsNone(geometry.camera_matrix_processed)
        with self.assertRaises(ValueError):
            geometry.prepare(np.zeros((360, 640, 3), np.uint8))

    def test_bad_intrinsics_are_rejected(self):
        with open(str(CALIBRATION), encoding="utf-8") as handle:
            data = json.load(handle)
        invalid = copy.deepcopy(data)
        invalid["camera_matrix"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            validate_calibration(invalid)
        invalid = copy.deepcopy(data)
        invalid["camera_matrix"][2] = [1, 0, 1]
        with self.assertRaises(ValueError):
            validate_calibration(invalid)

    def test_pickle_other_globals_are_forbidden(self):
        spec = importlib.util.spec_from_file_location("calibration_importer", REPO / "tools/import_camera_calibration.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import io
        with self.assertRaises(pickle.UnpicklingError):
            module.RestrictedNumpyUnpickler(io.BytesIO(pickle.dumps(ValueError))).load()


if __name__ == "__main__":
    unittest.main()
