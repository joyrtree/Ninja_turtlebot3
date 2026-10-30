#!/usr/bin/env python3
"""Run an unchanged LaneDetection callback against recorded compressed frames.

This is sequential, offline callback execution with ROS message/Node stubs, not
a ROS executor, timing simulation, or closed-loop robot simulation. Missing
state/mode/intersection topics are replaced by explicitly reported scenarios.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types
from collections import Counter

WORKSPACE = Path(__file__).resolve().parents[2]
LOCAL_DEPS = WORKSPACE / ".runtime" / "site-packages"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


class CapturedOutput:
    """Bounded sink so the source's per-frame print cannot fill memory/logs."""
    def __init__(self, limit=6000):
        self.limit = limit
        self.characters = 0
        self.preview = ""

    def write(self, value):
        self.characters += len(value)
        self.preview += value[:max(0, self.limit - len(self.preview))]
        return len(value)

    def flush(self):
        pass


class Bool:
    def __init__(self, data=False):
        self.data = bool(data)


class Vector3:
    def __init__(self):
        self.x = self.y = self.z = 0.0


class Twist:
    def __init__(self):
        self.linear = Vector3()
        self.angular = Vector3()


class CompressedImage:
    pass


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append({
            "linear_x": float(message.linear.x),
            "linear_y": float(message.linear.y),
            "linear_z": float(message.linear.z),
            "angular_x": float(message.angular.x),
            "angular_y": float(message.angular.y),
            "angular_z": float(message.angular.z),
        })


class FakeLogger:
    def __init__(self):
        self.counts = Counter()
        self.recent = []

    def _log(self, level, message):
        self.counts[level] += 1
        self.recent.append({"level": level, "message": str(message)})
        self.recent = self.recent[-20:]

    def info(self, message):
        self._log("info", message)

    def error(self, message):
        self._log("error", message)

    def warning(self, message):
        self._log("warning", message)

    warn = warning


class FakeNode:
    def __init__(self, name):
        self._fake_name = name
        self._fake_logger = FakeLogger()

    def create_subscription(self, message_type, topic, callback, qos):
        return types.SimpleNamespace(topic=topic, callback=callback)

    def create_publisher(self, message_type, topic, qos):
        return FakePublisher()

    def get_logger(self):
        return self._fake_logger

    def destroy_node(self):
        pass


@contextlib.contextmanager
def ros_stubs_and_headless_gui():
    """Patch only imports and GUI calls; restore the process on exit."""
    replacements = {}
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda *a, **k: None
    rclpy.shutdown = lambda *a, **k: None
    rclpy.spin = lambda *a, **k: None
    node_module = types.ModuleType("rclpy.node")
    node_module.Node = FakeNode
    rclpy.node = node_module
    replacements.update({"rclpy": rclpy, "rclpy.node": node_module})
    for package, name, cls in [
        ("std_msgs", "Bool", Bool),
        ("geometry_msgs", "Twist", Twist),
        ("sensor_msgs", "CompressedImage", CompressedImage),
    ]:
        parent = types.ModuleType(package)
        child = types.ModuleType(package + ".msg")
        setattr(child, name, cls)
        parent.msg = child
        replacements[package] = parent
        replacements[package + ".msg"] = child
    original_modules = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    gui_names = ["imshow", "waitKey", "destroyAllWindows", "namedWindow",
                 "resizeWindow", "moveWindow"]
    original_gui = {name: getattr(cv2, name, None) for name in gui_names}
    for name in gui_names:
        setattr(cv2, name, (lambda *a, **k: -1) if name == "waitKey"
                else (lambda *a, **k: None))
    try:
        yield
    finally:
        for name, original in original_gui.items():
            if original is None:
                delattr(cv2, name)
            else:
                setattr(cv2, name, original)
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class InstrumentedScenario:
    def __init__(self, module, mode, output_sink):
        self.mode = mode
        self.node = module.LaneDetection()
        self.current = {}
        self.last_command = None
        self.last_command_record_ns = None
        self.last_command_sensor_ns = None
        self.frame_metrics = []
        self.reason_samples = Counter()
        self.regular_samples = 0
        self.samples = []
        self.output_sink = output_sink
        with contextlib.redirect_stdout(output_sink), contextlib.redirect_stderr(output_sink):
            self.node.laneModeSubscriber_callback(Bool(mode == "yellow"))
            self.node.stateSubscriber_callback(Bool(True))
            self.node.intersectionStateSubscriber_callback(Bool(False))
        original_center = self.node.findLaneCenter
        original_perspective = self.node.perspectiveTransformation
        original_many_white = self.node.many_white

        def center_wrapper(image):
            result = original_center(image)
            mask, center = result
            # Analyze the returned mask without changing it or the return tuple.
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_NONE)
            ratio = float(np.count_nonzero(mask) / mask.size)
            area_eligible = sum(cv2.contourArea(c) >= 100 for c in contours)
            # Mirror only the source's count eligibility for an observation flag.
            # The source itself is still responsible for its returned center.
            count_gate_pass = np.count_nonzero(mask) < mask.size // 4
            accepted = area_eligible if count_gate_pass else 0
            self.current.update({
                "center": int(center),
                "mask": mask.copy(),
                "mask_nonzero_ratio": ratio,
                "contour_count": len(contours),
                "area_eligible_contour_count": int(area_eligible),
                "accepted_contour_count": int(accepted),
                "has_accepted_contour": int(accepted > 0),
                "center_fallback": int(accepted == 0),
            })
            return result

        def perspective_wrapper(image):
            self.current["decoded_shape"] = tuple(image.shape) if image is not None else None
            result = original_perspective(image)
            self.current["warped"] = result.copy()
            return result

        def many_white_wrapper(image):
            result = original_many_white(image)
            self.current.setdefault("many_white_calls", []).append({
                "shape": list(image.shape), "value": float(result)})
            return result

        self.node.findLaneCenter = center_wrapper
        self.node.perspectiveTransformation = perspective_wrapper
        self.node.many_white = many_white_wrapper

    def process(self, message, record_ns, sensor_ns):
        self.current = {}
        publisher = self.node.velocityPublisher
        before = len(publisher.messages)
        printed_before = self.output_sink.characters
        errors_before = self.node._fake_logger.counts["error"]
        start = time.perf_counter()
        exception = ""
        try:
            with contextlib.redirect_stdout(self.output_sink), contextlib.redirect_stderr(self.output_sink):
                self.node.videoSubscriber_callback(message)
        except Exception as exc:
            # A runtime exception is a result, never a silent dropped CSV row.
            exception = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - start) * 1000
        new_messages = publisher.messages[before:]
        if new_messages:
            self.last_command = new_messages[-1]
            self.last_command_record_ns = record_ns
            self.last_command_sensor_ns = sensor_ns
        calls = self.current.get("many_white_calls", [])
        gate = calls[0]["value"] if calls else None
        row = {key: self.current.get(key, "") for key in [
            "center", "mask_nonzero_ratio", "contour_count",
            "area_eligible_contour_count", "accepted_contour_count",
            "has_accepted_contour", "center_fallback"]}
        row.update({
            "mode": self.mode, "assumed_state": True,
            "assumed_is_intersection": False,
            "new_command_count": len(new_messages),
            "no_publish": int(not new_messages),
            "new_linear_x": new_messages[-1]["linear_x"] if new_messages else "",
            "new_angular_z": new_messages[-1]["angular_z"] if new_messages else "",
            "last_linear_x": self.last_command["linear_x"] if self.last_command else "",
            "last_angular_z": self.last_command["angular_z"] if self.last_command else "",
            "last_command_age_record_ms": ((record_ns - self.last_command_record_ns) / 1e6
                                            if self.last_command_record_ns is not None else ""),
            "last_command_age_sensor_ms": ((sensor_ns - self.last_command_sensor_ns) / 1e6
                                            if sensor_ns is not None and self.last_command_sensor_ns is not None else ""),
            "first_many_white_value": gate if gate is not None else "",
            "first_many_white_shape": "x".join(map(str, calls[0]["shape"])) if calls else "",
            "many_white_gate_over_0_1": int(gate > 0.1) if gate is not None else "",
            "many_white_calls_json": json.dumps(calls, separators=(",", ":")),
            "callback_ms_including_instrumentation": elapsed_ms,
            "source_output_characters": self.output_sink.characters - printed_before,
            "source_error_logs": self.node._fake_logger.counts["error"] - errors_before,
            "exception": exception,
        })
        self.frame_metrics.append(row)
        return row

    def maybe_save_sample(self, out, raw_image, camera_index, row):
        if raw_image is None:
            return
        reasons = []
        if row["exception"]:
            reasons.append("exception")
        if row["no_publish"]:
            reasons.append("no_publish")
        if row["has_accepted_contour"] == 0:
            reasons.append("no_accepted_contour")
        if row["many_white_gate_over_0_1"] == 1:
            reasons.append("source_white_gate")
        reason = next((r for r in reasons if self.reason_samples[r] < 4), None)
        if reason:
            self.reason_samples[reason] += 1
        elif camera_index % 150 == 0 and self.regular_samples < 8:
            reason = "regular"
            self.regular_samples += 1
        else:
            return
        image_dir = out / "candidate_frames"
        image_dir.mkdir(exist_ok=True)
        stem = f"{self.mode}_{camera_index:06d}_{reason}"
        raw_path = image_dir / f"{stem}_raw.jpg"
        save_image(raw_path, raw_image)
        warped = self.current.get("warped")
        mask = self.current.get("mask")
        if warped is not None:
            # Draw on our copy; never edit the image consumed by original code.
            display = warped.copy()
            if row["center"] != "":
                cv2.circle(display, (int(row["center"]), 60), 5, (0, 0, 255), -1)
            cv2.putText(display, f"{self.mode} center={row['center']} pub={row['new_command_count']}",
                        (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 0), 1)
            save_image(image_dir / f"{stem}_warped.jpg", display)
        if mask is not None:
            save_image(image_dir / f"{stem}_mask.png", mask)
        self.samples.append({"camera_index": camera_index, "reason": reason,
                             "raw_path": str(raw_path.relative_to(out)),
                             "record_time_ns": row["record_time_ns"],
                             "center": row["center"]})


def save_image(path, image):
    extension = path.suffix
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    path.write_bytes(encoded.tobytes())


def sensor_time_ns(message):
    try:
        stamp = message.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError):
        return None


def image_statistics(image):
    if image is None:
        return {"decode_ok": 0, "width": "", "height": "", "matches_source_640x480": "",
                "gray_mean": "", "gray_p95": "", "value_p99": "",
                "any_channel_ge_250_ratio": "", "all_channels_ge_245_ratio": ""}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    value = np.max(image, axis=2)
    return {
        "decode_ok": 1, "width": image.shape[1], "height": image.shape[0],
        "matches_source_640x480": int(image.shape[:2] == (480, 640)),
        "gray_mean": float(np.mean(gray)), "gray_p95": float(np.percentile(gray, 95)),
        "value_p99": float(np.percentile(value, 99)),
        "any_channel_ge_250_ratio": float(np.mean(value >= 250)),
        "all_channels_ge_245_ratio": float(np.mean(np.min(image, axis=2) >= 245)),
    }


def quantiles(values):
    vals = np.array([float(v) for v in values if v != "" and v is not None], dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return None
    return {"count": len(vals), "min": float(vals.min()), "median": float(np.median(vals)),
            "mean": float(vals.mean()), "p95": float(np.percentile(vals, 95)),
            "max": float(vals.max())}


def aggregate(scenario):
    rows = scenario.frame_metrics
    n = len(rows)
    return {
        "frames": n,
        "has_accepted_contour_frames": sum(r["has_accepted_contour"] == 1 for r in rows),
        "center_fallback_frames": sum(r["center_fallback"] == 1 for r in rows),
        "no_publish_frames": sum(r["no_publish"] for r in rows),
        "published_commands": sum(r["new_command_count"] for r in rows),
        "gate_over_0_1_frames": sum(r["many_white_gate_over_0_1"] == 1 for r in rows),
        "exception_frames": sum(bool(r["exception"]) for r in rows),
        "source_error_logs": sum(r["source_error_logs"] for r in rows),
        "mask_nonzero_ratio": quantiles(r["mask_nonzero_ratio"] for r in rows),
        "center": quantiles(r["center"] for r in rows),
        "new_linear_x": quantiles(r["new_linear_x"] for r in rows),
        "new_angular_z": quantiles(r["new_angular_z"] for r in rows),
        "last_command_age_record_ms": quantiles(r["last_command_age_record_ms"] for r in rows),
        "callback_ms_including_instrumentation": quantiles(
            r["callback_ms_including_instrumentation"] for r in rows),
        "logger_counts": dict(scenario.node._fake_logger.counts),
        "recent_source_logs": scenario.node._fake_logger.recent,
        "samples": scenario.samples,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", choices=["both", "yellow", "white"], default="both")
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every Nth camera message; default records every frame.")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Maximum selected camera frames; 0 means all.")
    parser.add_argument("--camera-topic", default="/camera")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.stride < 1 or args.max_frames < 0:
        raise SystemExit("--stride must be >= 1 and --max-frames >= 0")
    source = args.source.resolve()
    bag = args.bag_dir.resolve()
    out = args.out.resolve()
    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    for filename in ["frames.csv", "summary.json", "source_snapshot.py"]:
        if (out / filename).exists():
            raise SystemExit(f"Output already exists, choose a fresh --out: {out / filename}")
    (out / "source_snapshot.py").write_bytes(source_bytes)
    output_sink = CapturedOutput()
    modes = ["yellow", "white"] if args.mode == "both" else [args.mode]
    processed = 0
    camera_seen = 0
    camera_read = 0
    dimensions = Counter()
    decode_errors = 0
    recorded_cmd = None
    recorded_cmd_ns = None
    start_wall = time.perf_counter()
    sensor_deltas = []
    record_deltas = []
    previous_sensor = None
    previous_record = None
    summary = {
        "tool": "unchanged_source_offline_callback_replay",
        "source": str(source), "source_sha256_before": source_hash,
        "source_snapshot": "source_snapshot.py", "bag_dir": str(bag),
        "scenario_assumptions": {"state": True, "is_intersection": False,
                                  "lane_modes": modes,
                                  "status": "explicit active scenarios; not inferred historical operating modes"},
        "stride": args.stride, "max_frames": args.max_frames,
        "limits": [
            "ROS imports/Node/messages and GUI functions are stubbed; original callback and processing methods execute unchanged.",
            "No ROS executor, queue scheduling, transport, timer behavior, or real-time playback is simulated.",
            "Missing state/mode/intersection messages are explicit scenarios, not reconstructed historical state.",
            "Candidate commands do not change recorded future images; results do not prove closed-loop driving performance.",
            "Brightness and saturation columns are image proxies, not labels for glare, camera exposure, or detection correctness.",
            "No-publish frames report command age in the bag timeline; actual downstream timeout behavior is unknown.",
            "Contour eligibility mirrors source logic and is not ground-truth lane detection accuracy.",
            "Measured callback time includes instrumentation on this analysis machine, not robot runtime latency.",
            "At stride > 1, command age and command sequence apply only to the sampled callback sequence.",
        ],
        "opencv_version": cv2.__version__, "numpy_version": np.__version__,
        "runtime_equivalence": "Original robot cv2/numpy versions are not pinned or verified; this is code-path replay in the documented analysis runtime.",
    }
    with ros_stubs_and_headless_gui():
        spec = importlib.util.spec_from_file_location("lane_baseline_original", source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load source {source}")
        module = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(output_sink), contextlib.redirect_stderr(output_sink):
            spec.loader.exec_module(module)
        scenarios = [InstrumentedScenario(module, mode, output_sink) for mode in modes]
        literal_node = module.LaneDetection()
        literal_checked = False
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        with AnyReader([bag], default_typestore=typestore) as reader, \
                (out / "frames.csv").open("w", newline="", encoding="utf-8-sig") as csvfile:
            summary["topics"] = [{"topic": c.topic, "msgtype": c.msgtype,
                                   "msgcount": c.msgcount} for c in reader.connections]
            camera_connections = [c for c in reader.connections if c.topic == args.camera_topic]
            if not camera_connections:
                raise RuntimeError(f"No camera topic {args.camera_topic}; see available topics: {summary['topics']}")
            if any(c.msgtype != "sensor_msgs/msg/CompressedImage" for c in camera_connections):
                raise RuntimeError("This baseline expects sensor_msgs/msg/CompressedImage.")
            summary["camera_messages_available"] = sum(c.msgcount for c in camera_connections)
            connections = [c for c in reader.connections
                           if c.topic in (args.camera_topic, "/cmd_vel")]
            writer = None
            for connection, record_ns, raw in reader.messages(connections=connections):
                if connection.topic == "/cmd_vel":
                    message = reader.deserialize(raw, connection.msgtype)
                    recorded_cmd = {"linear_x": float(message.linear.x),
                                    "angular_z": float(message.angular.z)}
                    recorded_cmd_ns = record_ns
                    continue
                camera_index = camera_seen
                camera_seen += 1
                if camera_index % args.stride:
                    continue
                if args.max_frames and processed >= args.max_frames:
                    break
                camera_read += 1
                message = reader.deserialize(raw, connection.msgtype)
                if not literal_checked:
                    before_publish = len(literal_node.velocityPublisher.messages)
                    with contextlib.redirect_stdout(output_sink), contextlib.redirect_stderr(output_sink):
                        literal_node.videoSubscriber_callback(message)
                    summary["literal_default_state_check"] = {
                        "input": "first selected actual recorded CompressedImage",
                        "camera_index": camera_index, "record_time_ns": record_ns,
                        "state": literal_node.state,
                        "lane_mode": literal_node.lane_mode,
                        "is_intersection": literal_node.is_intersection,
                        "new_command_count": len(literal_node.velocityPublisher.messages) - before_publish,
                        "interpretation": "Default state=False returns immediately. Bag contains no trigger reconstruction here; active scenarios below are explicit assumptions.",
                    }
                    literal_checked = True
                sensor_ns = sensor_time_ns(message)
                image = cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
                image_stats = image_statistics(image)
                if image is None:
                    decode_errors += 1
                else:
                    dimensions[f"{image.shape[1]}x{image.shape[0]}"] += 1
                if previous_record is not None:
                    record_deltas.append((record_ns - previous_record) / 1e6)
                if sensor_ns is not None and previous_sensor is not None:
                    sensor_deltas.append((sensor_ns - previous_sensor) / 1e6)
                previous_record = record_ns
                previous_sensor = sensor_ns
                common = {
                    "camera_index": camera_index, "selected_frame_index": processed,
                    "record_time_ns": record_ns, "sensor_time_ns": sensor_ns if sensor_ns is not None else "",
                    "record_minus_sensor_ms": (record_ns - sensor_ns) / 1e6 if sensor_ns is not None else "",
                    "compressed_bytes": len(message.data),
                    "image_format": str(message.format),
                    "recorded_linear_x_last": recorded_cmd["linear_x"] if recorded_cmd else "",
                    "recorded_angular_z_last": recorded_cmd["angular_z"] if recorded_cmd else "",
                    "recorded_cmd_age_ms": (record_ns - recorded_cmd_ns) / 1e6 if recorded_cmd_ns is not None else "",
                    **image_stats,
                }
                for scenario in scenarios:
                    row = {**common, **scenario.process(message, record_ns, sensor_ns)}
                    if writer is None:
                        writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
                        writer.writeheader()
                    writer.writerow(row)
                    scenario.maybe_save_sample(out, image, camera_index, row)
                processed += 1
                if processed % 250 == 0:
                    csvfile.flush()
                    print(f"Processed {processed} camera frames ({len(modes)} scenario(s))", flush=True)
        summary["scenarios"] = {scenario.mode: aggregate(scenario) for scenario in scenarios}
    after_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    summary.update({
        "source_sha256_after": after_hash, "source_unchanged": after_hash == source_hash,
        "processed_camera_frames": processed, "camera_messages_iterated": camera_seen,
        "selected_camera_messages_read": camera_read, "decode_errors": decode_errors,
        "image_dimensions": dict(dimensions),
        "source_assumed_dimensions": "640x480; source projection coordinates retained unchanged",
        "input_matches_source_dimensions": list(dimensions) == ["640x480"] if dimensions else False,
        "selected_record_delta_ms": quantiles(record_deltas),
        "selected_sensor_delta_ms": quantiles(sensor_deltas),
        "nonpositive_selected_sensor_deltas": sum(v <= 0 for v in sensor_deltas),
        "source_output_characters": output_sink.characters,
        "source_output_preview": output_sink.preview,
        "elapsed_wall_seconds": time.perf_counter() - start_wall,
    })
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not summary["source_unchanged"]:
        raise RuntimeError("Source hash changed while replay was running.")
    print(json.dumps({"out": str(out), "frames": processed,
                      "source_unchanged": summary["source_unchanged"],
                      "dimensions": dict(dimensions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
