#!/usr/bin/env python3
"""BAG_RAW-only original white-mode callback baseline, without ROS or motors.

Reconstructs camera_ros publisher geometric steps from raw JPEG observations.
It cannot recreate the publisher's pre-JPEG sensor pixels bit-for-bit.
"""
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
if (WORKSPACE / ".runtime/site-packages").is_dir():
    sys.path.insert(0, str(WORKSPACE / ".runtime/site-packages"))
import cv2
import numpy as np
from rosbags.typesys import Stores, get_typestore


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_bag(label, bag_dir, out, source, helper, calibration):
    database = bag_dir / (label + "_0.db3")
    db = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    topics = [dict(id=r[0], name=r[1], type=r[2]) for r in db.execute("SELECT id,name,type FROM topics")]
    camera = next(t for t in topics if t["name"] == "/camera_raw_16_9")
    if camera["type"] != "sensor_msgs/msg/CompressedImage":
        raise ValueError("Expected native wide CompressedImage")
    if any(t["name"] == "/camera" for t in topics):
        raise ValueError("This experiment expects /camera absent and explicit reconstruction")
    first = db.execute("SELECT min(timestamp) FROM messages WHERE topic_id=?", (camera["id"],)).fetchone()[0]
    count = db.execute("SELECT count(*) FROM messages WHERE topic_id=?", (camera["id"],)).fetchone()[0]
    matrix = np.asarray(calibration["camera_matrix"], np.float64)
    distortion = np.asarray(calibration["dist_coeffs"], np.float64)
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (1280,720), 0, (1280,720))
    map1, map2 = cv2.initUndistortRectifyMap(matrix, distortion, None, new_matrix, (1280,720), cv2.CV_16SC2)
    x,y,w,h = [int(v) for v in roi]
    if w < 1 or h < 1:
        raise ValueError("Invalid publisher reconstruction ROI")
    out.mkdir(parents=True, exist_ok=True)
    store = get_typestore(Stores.ROS2_HUMBLE)
    sink = helper.CapturedOutput()
    window = (140.0,155.0) if label == "normal" else (64.0,92.0)
    started = time.perf_counter()
    samples = []
    with helper.ros_stubs_and_headless_gui():
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            module = load_module("original_white_raw_" + label, source)
            scenario = helper.InstrumentedScenario(module, "white", sink)
        with (out / "frames.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = None
            for index, (record_ns, blob) in enumerate(db.execute(
                    "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY id", (camera["id"],))):
                message = store.deserialize_cdr(blob, camera["type"])
                raw = cv2.imdecode(np.asarray(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if raw is None or raw.shape != (720,1280,3):
                    raise ValueError("Native1280x720 decode failed at frame {}".format(index))
                remapped = cv2.remap(raw, map1, map2, cv2.INTER_LINEAR)
                reconstructed = cv2.resize(remapped[y:y+h,x:x+w], (640,480))
                ok, jpeg = cv2.imencode(".jpg", reconstructed)
                if not ok:
                    raise ValueError("Reconstructed JPEG encode failed")
                derived = helper.CompressedImage()
                derived.data = jpeg.tobytes()
                derived.format = "jpeg"
                derived.header = message.header
                sensor_ns = int(message.header.stamp.sec) * 10**9 + int(message.header.stamp.nanosec)
                row = scenario.process(derived, record_ns, sensor_ns)
                row.update(camera_index=index, record_time_ns=record_ns, sensor_time_ns=sensor_ns,
                           t_camera_s=(record_ns-first)/1e9, reconstructed_width=640,
                           reconstructed_height=480, native_width=1280, native_height=720)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                want_sample = index == 0 or (window[0] <= row["t_camera_s"] <= window[1]
                                             and len(samples) < 3
                                             and (not samples or row["t_camera_s"]-samples[-1]["t_camera_s"] > 5))
                if want_sample:
                    stem = "frame_{:06d}".format(index)
                    for suffix, image in [("reconstructed_camera.jpg", reconstructed),
                                          ("source_mask.png", scenario.current.get("mask"))]:
                        if image is not None:
                            good, encoded = cv2.imencode(Path(suffix).suffix, image)
                            if good:
                                (out / (stem + "_" + suffix)).write_bytes(encoded.tobytes())
                    samples.append(dict(camera_index=index,t_camera_s=row["t_camera_s"],stem=stem))
                if (index+1) % 500 == 0:
                    print(json.dumps(dict(label=label, processed=index+1,total=count)), flush=True)
    db.close()
    rows = scenario.frame_metrics
    def summarize(subset):
        original_rows = scenario.frame_metrics
        scenario.frame_metrics = subset
        result = helper.aggregate(scenario)
        scenario.frame_metrics = original_rows
        result["fallback_with_new_command_frames"] = sum(
            r["center_fallback"] == 1 and r["new_command_count"] > 0 for r in subset)
        result["center_equals_640_frames"] = sum(r["center"] == 640 for r in subset)
        return result
    subset = [r for r in rows if window[0] <= r["t_camera_s"] <= window[1]]
    result = {
        "bag_label": label,
        "bag_dir": str(bag_dir.resolve()),
        "database": str(database.resolve()),
        "topics": topics,
        "first_camera_record_ns": first,
        "camera_message_count": count,
        "processed_frames": len(rows),
        "source_sha256": sha(source),
        "source_unchanged_after_run": sha(source) == sha(out.parent / "source_snapshot.py"),
        "input_path": "rawJPEG decode -> alpha0 native remap -> validROI crop -> resize640x480 -> JPEG encode -> original callback",
        "publisher_roi_xywh": [x,y,w,h],
        "scenario": {"lane_mode":False,"state":True,"is_intersection":False,
                     "status":"assumed active white-mode scenario; historical control state not recorded"},
        "scope": "all camera frames, stride1",
        "all_frames": summarize(rows),
        "requested_contiguous_window_s": list(window),
        "requested_window": summarize(subset),
        "samples": samples,
        "elapsed_analysis_wall_s": time.perf_counter()-started,
        "analysis_wall_time_is_robot_latency": False,
    }
    (out / "summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(label=label,frames=len(rows),fallback=result["all_frames"]["center_fallback_frames"],
                          no_publish=result["all_frames"]["no_publish_frames"],exceptions=result["all_frames"]["exception_frames"])),flush=True)
    return result


def write_report(results, path, source, helper, calibration_path):
    lines = ["# BAG_RAW: 원본 흰선 모드의 독립 오프라인 기준 결과", "",
             "이번 실행은 새 `inputs/bags_raw/bag_raw/{normal,reverse}`만 사용했다. 이전640×480 bag의 통계는 사용하지 않았다.", "",
             "## 입력 재구성과 실행 범위", "",
             "기록에는 `/camera_raw_16_9`가 있고 `/camera`는 없다. native1280×720 JPEG를 디코딩해 camera_ros publisher와 같은 기하 순서인 `getOptimalNewCameraMatrix(alpha=0)` → native remap → validROI crop →640×480 resize를 적용하고 JPEG로 다시 인코딩했다. 이를 변경하지 않은 원본 `LaneDetection` callback에 입력했다.", "",
             "**이것은 실제 `/camera`의 비트 정확한 복원이 아니다.** 실제 publisher는 JPEG 인코딩 전의 카메라 프레임에서 `/camera`를 만들지만 여기서는 이미 손실 압축된 raw JPEG에서 출발하므로 추가 JPEG 손실이 있다. 현재 분석 환경의 OpenCV와 원래 장비 버전 차이도 있다.", "",
             "ROS imports·Node·publisher와 GUI만 가짜 객체로 대체했다. 원본 인식·제어 함수는 그대로 실행했다. mode=False(white), state=True, intersection=False를 명시했으며 기록 당시의 상태를 복구했다고 주장하지 않는다. ROS executor·전송·큐·타이머·실차 폐루프 재현은 아니다.", "",
             "## 실제 처리 횟수", "",
             "| BAG_RAW | 전체 카메라 프레임 | center=640 fallback | fallback 중 새 명령 계산 | 새 명령 없음(return) | 예외 |",
             "|---|---:|---:|---:|---:|---:|"]
    for result in results:
        a=result["all_frames"]
        lines.append("| {} | {} | {} | {} | {} | {} |".format(result["bag_label"],result["processed_frames"],a["center_fallback_frames"],a["fallback_with_new_command_frames"],a["no_publish_frames"],a["exception_frames"]))
    lines += ["", "| 확인한 연속 구간: 첫camera 기록 기준 초 | 프레임 | fallback | 새 명령 없음 |", "|---|---:|---:|---:|"]
    for result in results:
        a=result["requested_window"]
        lines.append("| {} {}–{}초 | {} | {} | {} |".format(result["bag_label"],*result["requested_contiguous_window_s"],a["frames"],a["center_fallback_frames"],a["no_publish_frames"]))
    lines += ["", "원본은 유효 성분을 얻지 못하면 white 중심값을640으로 대체한다. 별도의 차선 없음 상태가 아니므로 이후 분기가 막지 않으면 이 가상 중심값으로도 선속도·각속도를 계산한다. `many_white` 조건으로 callback이 일찍 return하면 그 프레임에는 새 명령이 발행되지 않는다. 실제 하위 구동부가 이전 명령을 언제까지 유지하는지는 이 실행으로 확인하지 않았다.", "",
              "## 해석 제한", "",
              "- fallback·return 횟수는 코드 경로의 관측값이다. 독립 수동 정답이 없으므로 인식 정확도·오답률·주행 실패율로 부르지 않는다.",
              "- 이 입력도 수동 주행 기록이며, 새 계산 명령이 저장된 미래 영상을 바꾸지 않는다. 완주·실제 복귀·직진 진동 개선을 증명하지 않는다.",
              "- fusion 경로는 전체1280×720 rectification 후640×360을 유지한다. 이 baseline은 기존 publisher의 ROI crop과640×480 변형을 재구성한다. **동일 픽셀·동일 시야·동일 좌표계의 직접 수치 비교가 아니다.** callback 시간도 분석 PC 계측이며 로봇 끝단 지연이 아니다.",
              "- 표의 시간은 각 bag 첫camera `messages.timestamp`를0초로 했다. 센서 header와 기록시각의 차이를 실기 지연으로 해석하지 않는다.", "",
              "## 재현 자료", "",
              "- `evidence/baseline_raw/{normal,reverse}/frames.csv`: 전 프레임·기록/센서 시각·fallback·mask 관측·새 계산 명령·return·예외.",
              "- 같은 폴더의 `summary.json`: 전 프레임과 요청한 연속 구간의 집계, 출처·실행 가정.",
              "- `evidence/baseline_raw/manifest.json`, `source_snapshot.py`: 원본 코드·설정·도구 해시와 분석 환경.",
              "- `tools/compare_baseline_raw.py`로 같은 BAG_RAW 입력에서 재실행할 수 있다.", ""]
    path.write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    workspace_source=WORKSPACE/"inputs/source/turtlebot3_autorace_2026-master/lane_detection/lane_detection/detection.py"
    workspace_helper=WORKSPACE/"lane_lab/scripts/replay_baseline.py"
    bundled_evidence=REPO/"evidence/baseline_raw"
    default_source=workspace_source if workspace_source.is_file() else bundled_evidence/"source_snapshot.py"
    default_helper=workspace_helper if workspace_helper.is_file() else bundled_evidence/"replay_helper.py"
    parser.add_argument("--bag-root",type=Path,default=WORKSPACE/"inputs/bags_raw/bag_raw")
    parser.add_argument("--source",type=Path,default=default_source,
                        help="Original source; workspace copy if available, otherwise bundled source_snapshot.py")
    parser.add_argument("--helper",type=Path,default=default_helper,
                        help="Callback stubs/instrumentation; workspace copy if available, otherwise bundled replay_helper.py")
    parser.add_argument("--calibration",type=Path,default=REPO/"config/camera_calibration.json")
    args=parser.parse_args()
    for required in (args.source,args.helper,args.calibration):
        if not required.is_file():
            parser.error("Required file not found: {}".format(required))
    cv2.setNumThreads(1)
    helper=load_module("baseline_raw_callback_helper",args.helper)
    calibration=json.loads(args.calibration.read_text(encoding="utf-8"))
    if calibration.get("image_size") != [1280,720]:
        raise ValueError("Native1280x720 calibration required")
    out=REPO/"evidence/baseline_raw";out.mkdir(parents=True,exist_ok=True)
    (out/"source_snapshot.py").write_bytes(args.source.read_bytes())
    manifest={"source":str(args.source.resolve()),"source_sha256":sha(args.source),
              "helper":str(args.helper.resolve()),"helper_sha256":sha(args.helper),
              "tool_sha256":sha(__file__),"calibration_sha256":sha(args.calibration),
              "camera_calibration_source_pickle_sha256":calibration.get("source_pickle_sha256"),
              "python":sys.version,"opencv":cv2.__version__,"numpy":np.__version__,
              "opencv_threads":1,"ros_runtime_reproduced":False,"closed_loop_verified":False}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    results=[]
    for label in ("normal","reverse"):
        results.append(run_bag(label,args.bag_root/label,out/label,args.source,helper,calibration))
    write_report(results,REPO/"evidence/BASELINE_RAW.md",args.source,helper,args.calibration)
    print(json.dumps({"completed":[r["bag_label"] for r in results],"report":"evidence/BASELINE_RAW.md"}),flush=True)


if __name__=="__main__":
    main()
