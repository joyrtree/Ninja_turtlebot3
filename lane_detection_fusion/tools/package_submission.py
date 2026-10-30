"""Package source and review evidence; never include input bags or dependencies."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "deliverables_20261030"


def selected_files():
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file():
            continue
        if any(part in (".git", "__pycache__", "build", "dist", "replays") or
               part.endswith(".egg-info") or part.startswith("smoke_") for part in relative.parts):
            continue
        if path.suffix in (".pyc", ".db3", ".bag", ".pkl", ".pt", ".onnx"):
            continue
        yield path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    manifest_path = ROOT / "evidence/SUBMISSION_MANIFEST.json"
    entries = {path.relative_to(ROOT).as_posix(): {"bytes": path.stat().st_size, "sha256": sha(path)}
               for path in selected_files() if path != manifest_path}
    manifest_path.write_text(json.dumps({"package": ROOT.name, "files": entries,
        "scope": "Source, calibration JSON and offline BAG_RAW evidence; no input bag, ROS or driving execution claim."},
        ensure_ascii=False, indent=2), encoding="utf-8")
    for label, evidence in (("submission", True), ("code", False)):
        target = OUT / ("lane_detection_fusion_" + label + "_20261030.zip")
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
            for path in selected_files():
                relative = path.relative_to(ROOT)
                if (not evidence and relative.parts[0] == "evidence" and
                    path.suffix not in (".md", ".py") and
                    path.name not in ("camera_provenance.json", "local_checks.txt", "manifest.json")):
                    continue
                archive.write(path, (Path(ROOT.name) / relative).as_posix())
        with zipfile.ZipFile(target) as archive:
            bad = archive.testzip()
            if bad:
                raise RuntimeError("ZIP integrity failure: " + bad)
            count = len(archive.namelist())
            assert "lane_detection_fusion/config/camera_calibration.json" in archive.namelist()
            assert "lane_detection_fusion/lane_detection_fusion/node.py" in archive.namelist()
        digest = sha(target)
        target.with_suffix(".sha256.txt").write_text(digest + "  " + target.name + "\n", encoding="ascii")
        print(json.dumps({"zip": str(target), "bytes": target.stat().st_size, "files": count,
                          "sha256": digest, "integrity": "passed"}))


if __name__ == "__main__":
    main()
