"""Hardened batch extractor for real training data (ISL).

Unlike extract.py (a bare-minimum pipeline smoke test), this is built to
survive messy real-world source data: dropped hand tracking, variable frame
rates, mixed video quality across public datasets and self-recorded clips.
Raw per-clip extraction is always kept as source of truth; a manifest
summarizes per-clip detection quality so bad clips get flagged for review
instead of silently blended into the training set.

Input convention -- one subfolder per label, video or image files inside:
    TrainingData/
      hello/
        clip1.mp4
        clip2.mp4
      a/                  # static-pose labels (e.g. fingerspelling) too
        img1.jpg

Output: one .npz per clip under <output_dir>/<label>/, containing:
    features   (T, 450) float32  -- 225 normalized position + 225 velocity
    presence   (T, 3)   float32  -- [pose, left_hand, right_hand] present-or-interpolated
    detected   (T, 3)   float32  -- same, but RAW (interpolated frames count as False)
    timestamps_ms (T,)  float64

Plus dataset_manifest.json / .csv in the project root, one row per clip.
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
HAND_MODEL = MODELS_DIR / "hand_landmarker.task"
POSE_MODEL = MODELS_DIR / "pose_landmarker_lite.task"

DEFAULT_INPUT_DIR = BASE_DIR / "TrainingData"
DEFAULT_OUTPUT_DIR = BASE_DIR / "training_extracted"
MANIFEST_JSON = BASE_DIR / "dataset_manifest.json"
MANIFEST_CSV = BASE_DIR / "dataset_manifest.csv"

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

SHORT_GAP_MAX_FRAMES = 5        # gaps this long or shorter get linearly interpolated
LOW_DETECTION_THRESHOLD = 0.5   # below this fraction detected -> quality flag
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def initialize_landmarkers():
    # Lower confidence thresholds than the bare-minimum extractor on purpose:
    # our own gap-fill + quality-flagging downstream handles noisy/marginal
    # detections more intelligently than just discarding them at the source.
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return (
        vision.HandLandmarker.create_from_options(hand_options),
        vision.PoseLandmarker.create_from_options(pose_options),
    )


def get_hand_label(hand_result, index):
    handedness = getattr(hand_result, "handedness", None)
    if handedness and index < len(handedness) and handedness[index]:
        label = getattr(handedness[index][0], "label", None)
        if label:
            return label.lower()
    return None


def landmarks_to_array(raw_landmarks, count):
    if not raw_landmarks:
        return None
    arr = np.full((count, 3), np.nan, dtype=np.float32)
    for i, lm in enumerate(raw_landmarks[:count]):
        arr[i] = [lm.x, lm.y, lm.z]
    return arr


def normalize(points, origin, scale):
    if points is None:
        return None
    return (points - origin) / scale


def get_body_normalization(pose_points, fallback_origin, fallback_scale):
    if pose_points is None or np.isnan(pose_points[LEFT_SHOULDER]).any() or np.isnan(pose_points[RIGHT_SHOULDER]).any():
        return fallback_origin, fallback_scale
    left = pose_points[LEFT_SHOULDER]
    right = pose_points[RIGHT_SHOULDER]
    origin = (left + right) / 2.0
    scale = float(np.linalg.norm(left - right))
    if scale < 1e-6:
        return fallback_origin, fallback_scale
    return origin, scale


def detect_frame(frame_bgr, hand_landmarker, pose_landmarker, timestamp_ms, origin, scale):
    """Run both landmarkers on one frame, normalize against the running body
    origin/scale. Returns (pose_frame, left_frame, right_frame, origin, scale)
    -- each *_frame is NaN-filled where that part wasn't detected."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    pose_points = landmarks_to_array(pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None, 33)
    origin, scale = get_body_normalization(pose_points, origin, scale)
    pose_norm = normalize(pose_points, origin, scale)

    left_points = right_points = None
    for hand_idx, hand_landmarks in enumerate(hand_result.hand_landmarks[:2]):
        side_label = get_hand_label(hand_result, hand_idx)
        points = landmarks_to_array(hand_landmarks, 21)
        if side_label == "left" or (side_label is None and hand_idx == 0):
            left_points = points
        else:
            right_points = points

    left_norm = normalize(left_points, origin, scale)
    right_norm = normalize(right_points, origin, scale)

    pose_frame = pose_norm if pose_norm is not None else np.full((33, 3), np.nan, dtype=np.float32)
    left_frame = left_norm if left_norm is not None else np.full((21, 3), np.nan, dtype=np.float32)
    right_frame = right_norm if right_norm is not None else np.full((21, 3), np.nan, dtype=np.float32)

    return pose_frame, left_frame, right_frame, origin, scale


def extract_raw_sequence(video_path, hand_landmarker, pose_landmarker):
    """Pass 1: decode every frame, run MediaPipe, normalize per-frame.
    Returns arrays that may contain NaN where a body part wasn't detected --
    gap-filling happens afterward, once the whole clip is visible at once."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval_ms = 1000.0 / fps if fps > 0 else 33.3

    pose_seq, left_seq, right_seq, timestamps_ms = [], [], [], []
    origin, scale = np.zeros(3, dtype=np.float32), 1.0
    last_ts = -1.0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Prefer the container's own per-frame timestamp; some codecs/containers
        # report it unreliably (or not at all), so fall back to FPS-derived
        # timing, and always enforce strictly-increasing timestamps since
        # MediaPipe VIDEO mode hard-requires that.
        reported_ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        computed_ts = frame_idx * frame_interval_ms
        ts = reported_ts if reported_ts > last_ts else computed_ts
        ts = max(ts, last_ts + 1)
        last_ts = ts
        timestamps_ms.append(ts)

        pose_frame, left_frame, right_frame, origin, scale = detect_frame(
            frame, hand_landmarker, pose_landmarker, int(ts), origin, scale
        )
        pose_seq.append(pose_frame)
        left_seq.append(left_frame)
        right_seq.append(right_frame)
        frame_idx += 1

    cap.release()
    return (
        np.stack(pose_seq), np.stack(left_seq), np.stack(right_seq),
        np.array(timestamps_ms, dtype=np.float64), fps,
    )


def extract_raw_from_image(image_path, hand_landmarker, pose_landmarker, repeat_frames=5):
    """Static poses (fingerspelling, hardcoded words) get one real detection,
    replicated into a short pseudo-sequence so the output shape matches video
    clips -- keeps every downstream step (resampling, model input) uniform."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")

    pose_frame, left_frame, right_frame, _, _ = detect_frame(
        frame, hand_landmarker, pose_landmarker, 0, np.zeros(3, dtype=np.float32), 1.0
    )
    pose_seq = np.repeat(pose_frame[None], repeat_frames, axis=0)
    left_seq = np.repeat(left_frame[None], repeat_frames, axis=0)
    right_seq = np.repeat(right_frame[None], repeat_frames, axis=0)
    timestamps_ms = np.arange(repeat_frames, dtype=np.float64) * (1000.0 / 30)
    return pose_seq, left_seq, right_seq, timestamps_ms, 30.0


def fill_gaps(sequence, short_gap_max):
    """sequence: (T, K, 3) with NaN for undetected frames.
    Returns (filled, was_detected, is_present):
      filled       -- NaN gaps replaced (interpolated or zeroed, see below)
      was_detected -- (T,) bool, True only for genuinely-detected frames (raw)
      is_present   -- (T,) bool, True for detected OR short-gap-interpolated
                       frames -- this is the "usable signal" channel.
    Short NaN runs (<= short_gap_max, with real data on both sides) are
    linearly interpolated -- a tracking flicker, not a real absence. Longer
    runs, or runs touching the start/end of the clip (no far side to
    interpolate from), are left at zero: a real absence, not a glitch, and
    faking a position there would teach the model something false."""
    T = sequence.shape[0]
    flat = sequence.reshape(T, -1)
    was_detected = ~np.isnan(flat).any(axis=1)
    is_present = was_detected.copy()
    filled = np.where(np.isnan(flat), 0.0, flat)

    i = 0
    while i < T:
        if was_detected[i]:
            i += 1
            continue
        j = i
        while j < T and not was_detected[j]:
            j += 1
        gap_len = j - i
        if gap_len <= short_gap_max and i > 0 and j < T:
            start_vec, end_vec = filled[i - 1], filled[j]
            for k in range(gap_len):
                t = (k + 1) / (gap_len + 1)
                filled[i + k] = (1 - t) * start_vec + t * end_vec
            is_present[i:j] = True
        i = j

    return filled.reshape(sequence.shape).astype(np.float32), was_detected, is_present


def compute_velocity(position_flat, is_present):
    """Frame-to-frame delta, with one fix: the frame where a part *reappears*
    after a real (non-interpolated) absence gets its velocity zeroed. Without
    this, "hand re-entered frame at a new spot" reads as a huge fake motion
    spike -- it's not movement, it's tracking coming back."""
    velocity = np.zeros_like(position_flat)
    velocity[1:] = position_flat[1:] - position_flat[:-1]
    reappeared = is_present[1:] & ~is_present[:-1]
    velocity[1:][reappeared] = 0.0
    return velocity


def assemble_features(pose_raw, left_raw, right_raw, timestamps_ms, fps):
    pose_filled, pose_detected, pose_present = fill_gaps(pose_raw, SHORT_GAP_MAX_FRAMES)
    left_filled, left_detected, left_present = fill_gaps(left_raw, SHORT_GAP_MAX_FRAMES)
    right_filled, right_detected, right_present = fill_gaps(right_raw, SHORT_GAP_MAX_FRAMES)

    T = pose_filled.shape[0]
    pose_flat = pose_filled.reshape(T, -1)
    left_flat = left_filled.reshape(T, -1)
    right_flat = right_filled.reshape(T, -1)

    position = np.concatenate([pose_flat, left_flat, right_flat], axis=1)  # (T, 225)
    pose_vel = compute_velocity(pose_flat, pose_present)
    left_vel = compute_velocity(left_flat, left_present)
    right_vel = compute_velocity(right_flat, right_present)
    velocity = np.concatenate([pose_vel, left_vel, right_vel], axis=1)  # (T, 225)

    features = np.concatenate([position, velocity], axis=1).astype(np.float32)  # (T, 450)
    presence = np.stack([pose_present, left_present, right_present], axis=1).astype(np.float32)
    detected = np.stack([pose_detected, left_detected, right_detected], axis=1).astype(np.float32)

    stats = {
        "frame_count": int(T),
        "fps_source": float(fps),
        "pct_pose_detected": float(pose_detected.mean()) if T else 0.0,
        "pct_left_hand_detected": float(left_detected.mean()) if T else 0.0,
        "pct_right_hand_detected": float(right_detected.mean()) if T else 0.0,
    }
    return features, presence, detected, timestamps_ms, stats


def compute_quality_flag(stats):
    hand_coverage = max(stats["pct_left_hand_detected"], stats["pct_right_hand_detected"])
    if stats["frame_count"] < 5:
        return "too_short"
    if hand_coverage < LOW_DETECTION_THRESHOLD:
        return "low_hand_detection"
    if stats["pct_pose_detected"] < LOW_DETECTION_THRESHOLD:
        return "low_pose_detection"
    return None


def save_clip(output_path, features, presence, detected, timestamps_ms):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, features=features, presence=presence,
        detected=detected, timestamps_ms=timestamps_ms,
    )


def discover_clips(input_dir):
    clips = []
    for label_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        label = label_dir.name
        for f in sorted(label_dir.rglob("*")):
            if f.is_file() and (f.suffix.lower() in VIDEO_EXTS or f.suffix.lower() in IMAGE_EXTS):
                clips.append((label, f))
    return clips


def append_manifest(records):
    existing = []
    if MANIFEST_JSON.exists():
        try:
            existing = json.loads(MANIFEST_JSON.read_text())
        except json.JSONDecodeError:
            existing = []
    by_id = {r["clip_id"]: r for r in existing}
    for r in records:
        by_id[r["clip_id"]] = r
    merged = list(by_id.values())
    MANIFEST_JSON.write_text(json.dumps(merged, indent=2))

    if merged:
        fieldnames = sorted({k for r in merged for k in r.keys()})
        with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged)


def build_dataset(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    clips = discover_clips(input_dir)
    if not clips:
        print(f"No clips found under {input_dir} (expected <label>/<file> structure)")
        return []

    labels = sorted({label for label, _ in clips})
    print(f"Found {len(clips)} clips across {len(labels)} labels: {', '.join(labels)}", flush=True)

    # A fresh landmarker pair per clip, not shared across the whole batch:
    # MediaPipe VIDEO mode tracks timestamps globally per landmarker instance,
    # so reusing one pair across clips whose timestamps each restart near 0ms
    # collides with the previous clip's ending timestamp and hard-crashes with
    # "Input timestamp must be monotonically increasing." A little re-init
    # overhead per clip buys correctness regardless of batch size/order.
    records = []
    for i, (label, clip_path) in enumerate(clips, 1):
        print(f"[{i}/{len(clips)}] {label}/{clip_path.name}", flush=True)
        is_image = clip_path.suffix.lower() in IMAGE_EXTS
        clip_id = f"{label}/{clip_path.stem}"
        hand_landmarker, pose_landmarker = initialize_landmarkers()
        try:
            if is_image:
                raw = extract_raw_from_image(clip_path, hand_landmarker, pose_landmarker)
            else:
                raw = extract_raw_sequence(clip_path, hand_landmarker, pose_landmarker)

            features, presence, detected, timestamps_ms, stats = assemble_features(*raw)

            out_path = output_dir / label / f"{clip_path.stem}.npz"
            save_clip(out_path, features, presence, detected, timestamps_ms)

            quality_flag = compute_quality_flag(stats)
            records.append({
                "clip_id": clip_id, "label": label, "source_path": str(clip_path),
                "npz_path": str(out_path), "source_type": "image" if is_image else "video",
                "extracted_at": datetime.now().isoformat(timespec="seconds"),
                "quality_flag": quality_flag, "error": None, **stats,
            })
            if quality_flag:
                print(f"  ⚠ flagged: {quality_flag}", flush=True)
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
            records.append({
                "clip_id": clip_id, "label": label, "source_path": str(clip_path),
                "npz_path": None, "source_type": "image" if is_image else "video",
                "extracted_at": datetime.now().isoformat(timespec="seconds"),
                "quality_flag": "extraction_failed", "error": str(exc),
            })
        finally:
            hand_landmarker.close()
            pose_landmarker.close()

    append_manifest(records)

    flagged = [r for r in records if r.get("quality_flag")]
    print(f"\nDone: {len(records)} clips processed, {len(flagged)} flagged for review.", flush=True)
    for r in flagged:
        print(f"  ⚠ {r['clip_id']}: {r['quality_flag']}", flush=True)

    return records


if __name__ == "__main__":
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT_DIR
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR
    if not input_dir.exists():
        print(f"Input folder not found: {input_dir}")
        print("Usage: python build_dataset.py [input_dir] [output_dir]")
        print("Expected structure: <input_dir>/<label>/<clip files>")
        sys.exit(1)
    build_dataset(input_dir, output_dir)
