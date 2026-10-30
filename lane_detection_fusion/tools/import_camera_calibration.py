#!/usr/bin/env python3
"""Offline conversion of the pinned NumPy calibration pickle to runtime JSON.

No ROS/camera code is imported or executed. All pickle globals except NumPy's
array/dtype reconstruction primitives are refused. Runtime uses JSON only.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import pickle
import sys

import numpy as np


EXPECTED_SHA256 = "c22cd925234e39a8fa587a9d9076f73fb4edcfb2b8e8110b406568a3f75a1da8"
REPO = Path(__file__).resolve().parents[1]


class RestrictedNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy" and name == "ndarray":
            return np.ndarray
        if module == "numpy" and name == "dtype":
            return np.dtype
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name == "_reconstruct":
            try:
                return np._core.multiarray._reconstruct
            except AttributeError:
                return np.core.multiarray._reconstruct
        raise pickle.UnpicklingError("Forbidden pickle global: {}.{}".format(module, name))


def load_restricted(path, expected_sha256=EXPECTED_SHA256):
    payload = Path(path).read_bytes()
    if len(payload) > 1024 * 1024:
        raise ValueError("Calibration pickle exceeds 1 MiB limit")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Calibration pickle SHA-256 mismatch")
    data = RestrictedNumpyUnpickler(io.BytesIO(payload)).load()
    if not isinstance(data, dict):
        raise ValueError("Calibration pickle must contain a dictionary")
    return data, digest


def convert(source, destination, provenance_path=None):
    source = Path(source).resolve()
    data, digest = load_restricted(source)
    image_size = data.get("image_size")
    if not isinstance(image_size, (list, tuple)) or tuple(image_size) != (1280, 720):
        raise ValueError("Expected native calibration image_size (1280,720)")
    matrix = np.asarray(data.get("camera_matrix"))
    distortion = np.asarray(data.get("dist_coeffs"))
    if matrix.dtype.kind not in "uif" or distortion.dtype.kind not in "uif":
        raise ValueError("Calibration arrays must be numeric")
    rms = float(data.get("rms_reprojection_error"))
    count = data.get("valid_image_count")
    if not np.isfinite(rms) or rms < 0 or not isinstance(count, int) or count < 1:
        raise ValueError("Invalid calibration quality metadata")
    result = {
        "schema_version": 1,
        "camera_model": "opencv_pinhole",
        "image_size": [1280, 720],
        "camera_matrix": matrix.astype(float).tolist(),
        "dist_coeffs": distortion.astype(float).reshape(-1).tolist(),
        "rms_reprojection_error_px": rms,
        "valid_image_count": count,
        "source_pickle_sha256": digest,
        "source_calibration_image_directory": data.get("image_directory"),
        "ground_extrinsics_available": False,
    }
    sys.path.insert(0, str(REPO))
    from lane_detection_fusion.geometry import validate_calibration
    validate_calibration(result)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if provenance_path is not None:
        camera_dir = source.parents[1]
        hashes = {}
        for relative in ("HANDOFF.md", "camera_ros/publisher.py", "camera_ros/calibration.py"):
            file_path = camera_dir / relative
            if file_path.is_file():
                hashes[relative] = hashlib.sha256(file_path.read_bytes()).hexdigest()
        provenance = {
            "source_pickle": str(source),
            "source_pickle_sha256": digest,
            "expected_source_pickle_sha256": EXPECTED_SHA256,
            "restricted_allowed_globals": ["numpy.ndarray", "numpy.dtype",
                "numpy.core.multiarray._reconstruct", "numpy._core.multiarray._reconstruct"],
            "runtime_format": "JSON; no runtime pickle loading",
            "calibration_scope": "native 1280x720 intrinsics and lens distortion only",
            "valid_image_count_from_pickle": count,
            "rms_reprojection_error_px_from_pickle": rms,
            "source_files_sha256": hashes,
            "source_handoff_claims": {
                "calibrated_at_kst": "2026-10-27 23:11:23",
                "environment": "ROS2 Galactic / Python3.8.10 / OpenCV4.2.0",
                "checkerboard_inner_corners": [7, 9],
                "checkerboard_square_mm": 18.2,
                "capture_count": 78,
                "status": "document claims; not recaptured or independently recalibrated here",
            },
            "topic_contract": {
                "/camera_raw_16_9": "1280x720 native raw, no remap/crop/resize",
                "/camera_raw": "central 960x720 crop then 640x480, not native wide raw",
                "/camera": "already remapped, ROI cropped, resized to640x480; use rectified mode",
            },
            "new_raw_processing": "alpha0 full1280x720 remap, no ROI crop, aspect-preserving640x360",
            "ground_extrinsics_available": False,
            "checkerboard_rvecs_tvecs_exported_as_ground_extrinsics": False,
            "metric_bev_or_pure_pursuit_calibration_complete": False,
        }
        provenance_path = Path(provenance_path)
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=REPO / "config/camera_calibration.json")
    parser.add_argument("--provenance", type=Path, default=REPO / "evidence/camera_provenance.json")
    args = parser.parse_args()
    result = convert(args.source, args.output, args.provenance)
    print(json.dumps({"image_size": result["image_size"],
                      "valid_image_count": result["valid_image_count"],
                      "source_pickle_sha256": result["source_pickle_sha256"]}))


if __name__ == "__main__":
    main()
