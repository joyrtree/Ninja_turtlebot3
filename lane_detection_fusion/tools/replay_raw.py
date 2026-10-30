"""Sequential offline replay of recorded camera input, without ROS or motors."""
from pathlib import Path
import argparse
import collections
import csv
from dataclasses import asdict
import hashlib
import json
import sqlite3
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT.parent / ".runtime/site-packages"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from lane_detection_fusion.core import FusionConfig, LaneFusion, draw_result
from lane_detection_fusion.geometry import CameraGeometry


def decode_compressed(blob):
    if len(blob) < 16 or blob[:2] not in (b"\x00\x01", b"\x00\x00"):
        raise ValueError("Expected standard CDR1 CompressedImage")
    order = "<" if blob[1] == 1 else ">"
    sec, ns = struct.unpack_from(order + "iI", blob, 4)
    pos = 12
    strings = []
    for _ in range(2):
        pos = 4 + ((pos - 4 + 3) // 4) * 4
        length = struct.unpack_from(order + "I", blob, pos)[0]
        pos += 4
        if length < 1 or pos + length > len(blob) or blob[pos + length - 1] != 0:
            raise ValueError("Invalid CDR string")
        strings.append(blob[pos:pos + length - 1].decode("utf-8"))
        pos += length
    pos = 4 + ((pos - 4 + 3) // 4) * 4
    length = struct.unpack_from(order + "I", blob, pos)[0]
    pos += 4
    if pos + length > len(blob):
        raise ValueError("Truncated CompressedImage payload")
    return sec * 1000000000 + ns, strings[1], blob[pos:pos + length]


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {}
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=ROOT / "config/profile.json")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.stride < 1 or args.max_frames < 0:
        raise SystemExit("Invalid frame selection")
    files = list(args.bag_dir.glob("*.db3"))
    if len(files) != 1:
        raise SystemExit("Expected one SQLite bag storage file")
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    cam = profile["camera"]
    calib = Path(cam["calibration_file"])
    if not calib.is_absolute():
        calib = args.profile.resolve().parent / calib
    geometry = CameraGeometry(calib, input_mode=cam["input_mode"], process_width=cam["process_width"])
    fusion = LaneFusion(FusionConfig(**profile.get("vision", {})))
    source_hashes = {name: hashlib.sha256((ROOT / "lane_detection_fusion" / name).read_bytes()).hexdigest()
                     for name in ("core.py", "geometry.py")}
    cv2.setNumThreads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "frames.csv").exists():
        raise SystemExit("Choose a fresh output folder")
    con = sqlite3.connect(files[0].resolve().as_uri() + "?mode=ro", uri=True)
    con.execute("PRAGMA query_only=ON")
    topics = con.execute("SELECT id,name,type FROM topics").fetchall()
    cameras = [t for t in topics if t[1] == "/camera_raw_16_9" and t[2] == "sensor_msgs/msg/CompressedImage"]
    if len(cameras) != 1:
        raise SystemExit("This evaluation expects the valid BAG_RAW /camera_raw_16_9 input")
    origin = con.execute("SELECT min(timestamp) FROM messages WHERE topic_id=?", (cameras[0][0],)).fetchone()[0]
    count = con.execute("SELECT count(*) FROM messages WHERE topic_id=?", (cameras[0][0],)).fetchone()[0]
    frequencies = collections.Counter()
    timings = collections.defaultdict(list)
    examples = []
    previous_state = None
    n = 0
    encoder = None
    next_video_s = 0.0
    video_frames = 0
    started = time.perf_counter()
    if args.video:
        import imageio_ffmpeg
        encoder = imageio_ffmpeg.write_frames(str(args.out / "replay.mp4"), (1280, 720),
            fps=8, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=16,
            ffmpeg_log_level="error", output_params=["-preset", "veryfast", "-crf", "24", "-threads", "2"])
        encoder.send(None)
    try:
        with (args.out / "frames.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["camera_index", "record_ns", "header_ns", "time_s", "state",
                "quality_proxy", "speed_preview", "angular_preview", "reason", "curve_candidates",
                "lateral_error", "heading_error", "decode_ms", "geometry_ms", "core_ms", "total_processing_ms"])
            writer.writeheader()
            for index, (_, receipt_ns, blob) in enumerate(con.execute(
                "SELECT id,timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id", (cameras[0][0],))):
                if index % args.stride:
                    continue
                if args.max_frames and n >= args.max_frames:
                    break
                ts = (receipt_ns - origin) / 1e9
                begin = time.perf_counter()
                header, fmt, payload = decode_compressed(blob)
                raw = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if raw is None:
                    raise ValueError("JPEG decode failed at camera {}".format(index))
                after_decode = time.perf_counter()
                image = geometry.prepare(raw)
                after_geometry = time.perf_counter()
                # Replay uses recorded observation intervals, never the current wall clock.
                result = fusion.process(image, ts)
                done = time.perf_counter()
                stage = dict(decode_ms=(after_decode - begin) * 1000,
                    geometry_ms=(after_geometry - after_decode) * 1000,
                    core_ms=(done - after_geometry) * 1000, total_processing_ms=(done - begin) * 1000)
                for key, value in stage.items():
                    timings[key].append(value)
                if not np.isfinite([result.speed, result.angular, result.confidence]).all():
                    raise ValueError("Nonfinite output at camera {}".format(index))
                frequencies[result.state] += 1
                writer.writerow(dict(camera_index=index, record_ns=receipt_ns, header_ns=header,
                    time_s=ts, state=result.state, quality_proxy=result.confidence,
                    speed_preview=result.speed, angular_preview=result.angular,
                    reason=result.debug.get("reason", ""), curve_candidates=result.debug.get("curve_candidates", ""),
                    lateral_error=result.debug.get("lateral_error", ""), heading_error=result.debug.get("heading_error", ""), **stage))
                select_still = (n % 400 == 0 or result.state != previous_state) and len(examples) < 28
                select_video = args.video and ts >= next_video_s
                if select_still or select_video:
                    annotated = draw_result(image, result)
                    canvas = np.full((720, 1280, 3), (25, 22, 18), np.uint8)
                    left = cv2.resize(raw, (640, 360))
                    rh, rw = annotated.shape[:2]
                    right = cv2.resize(annotated, (640, int(rh * 640 / rw)))
                    canvas[75:435, :640] = left
                    canvas[75:75 + min(480, right.shape[0]), 640:] = right[:480]
                    lines = [
                        "BAG_RAW {} | frame {} | t={:.3f}s".format(args.bag_dir.name, index, ts),
                        "LEFT: native input     RIGHT: supplied camera calibration + fusion v1",
                        "state={}   quality proxy={:.3f}".format(result.state, result.confidence),
                        "PREVIEW v={:.3f}m/s  w={:+.3f}rad/s | normalized image control".format(result.speed, result.angular),
                        "processing {:.1f}ms (analysis PC; NOT robot latency)".format(stage["total_processing_ms"]),
                        "Offline recorded input. No ROS executor, no motors, no closed-loop driving claim.",
                    ]
                    for line, y in zip(lines, [27, 58, 535, 575, 615, 670]):
                        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, .62, (235, 235, 235), 1, cv2.LINE_AA)
                    if select_still:
                        name = "frame_{:06d}_t{:07.3f}.jpg".format(index, ts)
                        ok, data = cv2.imencode(".jpg", canvas)
                        if not ok:
                            raise ValueError("Still encode failed")
                        (args.out / name).write_bytes(data.tobytes())
                        examples.append({"camera_index": index, "time_s": ts, "state": result.state, "path": name})
                    if select_video:
                        while next_video_s <= ts:
                            encoder.send(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                            video_frames += 1
                            next_video_s += 1.0 / 8
                previous_state = result.state
                n += 1
                if n % 500 == 0:
                    f.flush()
                    print("{}: processed {}/{}".format(args.bag_dir.name, n, count), flush=True)
    finally:
        con.close()
        if encoder is not None:
            encoder.close()
    summary = dict(bag_dir=str(args.bag_dir.resolve()), camera_topic=cameras[0][1],
        available_camera_frames=count, processed_frames=n, stride=args.stride,
        complete=(args.stride == 1 and n == count), states=dict(frequencies),
        processing_ms={key: stats(values) for key, values in timings.items()},
        profile=profile, resolved_vision_config=asdict(fusion.config), source_sha256=source_hashes,
        calibration_sha256=hashlib.sha256(calib.read_bytes()).hexdigest(),
        opencv=cv2.__version__, numpy=np.__version__, video_frames=video_frames, examples=examples,
        wall_seconds=time.perf_counter() - started,
        limits=["State counts/quality are proxies, not labeled detection accuracy.",
                "Open-loop recorded observations do not change with generated commands.",
                "Processing timings are analysis-PC timings excluding video rendering and I/O, not robot latency.",
                "No metric ground homography was supplied; controller is normalized image P/heading, not metric Pure Pursuit.",
                "Camera intrinsics/undistortion are from the supplied camera_ros snapshot; mounting and ground geometry remain unverified."])
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "frames": n, "states": dict(frequencies), "complete": summary["complete"]}))


if __name__ == "__main__":
    main()
