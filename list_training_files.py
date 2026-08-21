"""Scan training-relevant folders and write dashboard/files.json (ISL)."""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = BASE_DIR / "dashboard"
SAMPLE_DIR = BASE_DIR / "SampleVideos"
EXTRACTED_DIR = BASE_DIR / "extracted"


def list_files():
    videos = []
    if SAMPLE_DIR.exists():
        for f in sorted(SAMPLE_DIR.rglob("*.mp4")):
            videos.append({"name": f.name, "size_mb": round(f.stat().st_size / (1024 * 1024), 2)})

    extracted = []
    if EXTRACTED_DIR.exists():
        for f in sorted(EXTRACTED_DIR.glob("*.npy")):
            extracted.append({"name": f.name, "size_kb": round(f.stat().st_size / 1024, 1)})

    return {"videos": videos, "extracted": extracted}


def write_files_json():
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    (DASHBOARD_DIR / "files.json").write_text(json.dumps(list_files(), indent=2))


if __name__ == "__main__":
    write_files_json()
    print("Wrote dashboard/files.json")
