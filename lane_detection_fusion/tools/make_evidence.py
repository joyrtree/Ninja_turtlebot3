"""Make selected recorded-input previews and a compact meeting contact sheet."""
from pathlib import Path
import csv
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT.parent / ".runtime/site-packages"
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))
import cv2
import numpy as np
import imageio_ffmpeg


def save_jpeg(path, image):
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("JPEG encode failed: " + str(path))
    path.write_bytes(encoded.tobytes())


def main():
    out = ROOT / "evidence"
    tiles = []
    for label, times in (("normal", [0, 40, 90, 140, 145, 150]),
                         ("reverse", [0, 40, 64, 70, 78, 86])):
        folder = out / ("raw_" + label + "_v1")
        with (folder / "frames.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        capture = cv2.VideoCapture(str(folder / "replay.mp4"))
        for t in times:
            capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Video preview missing at %s %s" % (label, t))
            # Replay stores the first observed frame at/after each 8 Hz grid time.
            closest = next((row for row in rows if float(row["time_s"]) >= t), rows[-1])
            tile = np.full((240, 426, 3), 22, np.uint8)
            tile[:240 - 35, :] = cv2.resize(frame[75:435, 640:1280], (426, 205))
            cv2.putText(tile, "%s %.2fs: %s" % (label, float(closest["time_s"]), closest["state"]),
                        (6, 230), cv2.FONT_HERSHEY_SIMPLEX, .57, (240, 240, 240), 1, cv2.LINE_AA)
            tiles.append(tile)
            if t in (145, 78):
                save_jpeg(out / ("zigzag_" + label + "_" + str(t) + ".jpg"), frame)
        capture.release()
        start, duration = (140, 15) if label == "normal" else (64, 28)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
            "-y", "-ss", str(start), "-i", str(folder / "replay.mp4"), "-t", str(duration),
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", "2",
            str(out / ("zigzag_" + label + ".mp4"))], check=True)
    sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    save_jpeg(out / "contact_sheet.jpg", sheet)
    print(json.dumps({"contact_sheet": str(out / "contact_sheet.jpg"), "tiles": len(tiles)}))


if __name__ == "__main__":
    main()
