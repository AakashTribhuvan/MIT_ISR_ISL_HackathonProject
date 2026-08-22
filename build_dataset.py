"""Hardened batch extractor for real training data (ISL).

This is built to
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
TEST_MODE = True  # Process up to five sentence videos per action for template matching.
TEST_LABELS = ["hello"]  # Set to None to include every sentence action.
SHOW_TEST_PREVIEW = False  # Enable only when visually inspecting extraction on a few clips.
TEST_INPUT_DIR = BASE_DIR / "Videos_Sentence_Level"
TEST_OUTPUT_DIR = BASE_DIR / "hello_test_extracted"
TEST_VIDEO_LIMIT = 5
MANIFEST_JSON = BASE_DIR / "dataset_manifest.json"
MANIFEST_CSV = BASE_DIR / "dataset_manifest.csv"

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

SHORT_GAP_MAX_FRAMES = 5        # gaps this long or shorter get linearly interpolated
LOW_DETECTION_THRESHOLD = 0.5   # below this fraction detected -> quality flag
HAND_DETECTION_CONFIDENCE = 0.35
HAND_PRESENCE_CONFIDENCE = 0.35
HAND_TRACKING_CONFIDENCE = 0.35
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
LIGHTING_CLIP_LIMIT = 3.0
DASHBOARD_STATUS_PATH = BASE_DIR / "dashboard" / "status.json"

POSE_CONNECTIONS = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def compensate_lighting(frame_bgr):
    """Improve local contrast while preserving the frame's color information."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=LIGHTING_CLIP_LIMIT, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def write_progress(status, phase, processed, total, current=None):
    DASHBOARD_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    percent = round(processed / total * 100, 1) if total else 100
    DASHBOARD_STATUS_PATH.write_text(json.dumps({
        "status": status, "phase": phase, "processed": processed,
        "total": total, "percent": percent, "current": current,
    }), encoding="utf-8")


def prepare_detection_frame(frame_bgr):
    # Preserve source resolution for landmark extraction; only preview views
    # are resized later for display.
    return compensate_lighting(frame_bgr)


def draw_preview_landmarks(frame, landmarks, connections, color):
    height, width = frame.shape[:2]
    points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
    for point in points:
        cv2.circle(frame, point, 4, color, -1)
    for start, end in connections:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], color, 2)
    return points


def resize_preview(frame, max_dim=640):
    height, width = frame.shape[:2]
    scale = min(1.0, max_dim / max(height, width))
    if scale == 1.0:
        return frame.copy()
    return cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def draw_hand_mask(mask, hand_landmarks):
    height, width = mask.shape[:2]
    points = np.array(
        [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks],
        dtype=np.int32,
    )
    if len(points) < 21:
        return

    # Fill the palm, then draw the individual finger bones so the mask keeps
    # the hand's branching shape instead of becoming one large blob.
    palm = cv2.convexHull(points[[0, 5, 9, 13, 17]])
    cv2.fillConvexPoly(mask, palm, 255)
    for start, end in HAND_CONNECTIONS:
        cv2.line(mask, tuple(points[start]), tuple(points[end]), 255, 11, cv2.LINE_AA)
    for point in points:
        cv2.circle(mask, tuple(point), 7, 255, -1, cv2.LINE_AA)


def draw_body_context(frame, pose_result, color=(150, 150, 150)):
    if not pose_result.pose_landmarks:
        return
    pose = pose_result.pose_landmarks[0]
    visible_indices = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
    visible = [pose[index] for index in visible_indices]
    index_map = {original: index for index, original in enumerate(visible_indices)}
    connections = [
        (index_map[start], index_map[end])
        for start, end in POSE_CONNECTIONS
        if start in index_map and end in index_map
    ]
    draw_preview_landmarks(frame, visible, connections, color)

    # Shoulder midpoint is the upper-body reference used by normalization.
    height, width = frame.shape[:2]
    left, right = pose[11], pose[12]
    midpoint = (int(((left.x + right.x) / 2) * width), int(((left.y + right.y) / 2) * height))
    cv2.drawMarker(frame, midpoint, (0, 220, 255), cv2.MARKER_CROSS, 16, 2)


def draw_contrast_outline(output, frame, hand_landmarks):
    """Draw local contrast edges only inside the detected hand silhouette."""
    height, width = frame.shape[:2]
    points = np.array(
        [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks],
        dtype=np.int32,
    )
    if len(points) < 21:
        return

    x, y, box_width, box_height = cv2.boundingRect(points)
    padding = max(8, int(max(box_width, box_height) * 0.08))
    left, top = max(0, x - padding), max(0, y - padding)
    right = min(width, x + box_width + padding)
    bottom = min(height, y + box_height + padding)
    roi = frame[top:bottom, left:right]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray)
    edges = cv2.Canny(gray, 35, 100)

    region = np.zeros(edges.shape, dtype=np.uint8)
    local_points = points - np.array([left, top])
    palm = cv2.convexHull(local_points[[0, 5, 9, 13, 17]])
    cv2.fillConvexPoly(region, palm, 255)
    for start, end in HAND_CONNECTIONS:
        cv2.line(region, tuple(local_points[start]), tuple(local_points[end]), 255, 13, cv2.LINE_AA)
    for point in local_points:
        cv2.circle(region, tuple(point), 8, 255, -1, cv2.LINE_AA)
    region = cv2.dilate(region, np.ones((3, 3), dtype=np.uint8))
    edges = cv2.bitwise_and(edges, region)
    output[top:bottom, left:right][edges > 0] = (0, 255, 255)


def show_test_preview(frame, pose_result, hand_result, frame_idx, total_frames):
    """Show five diagnostic views only during TEST_MODE."""
    detection_view = resize_preview(prepare_detection_frame(frame))
    media_pipe_view = detection_view.copy()
    lighting_view = resize_preview(compensate_lighting(frame))
    depth_view = np.zeros_like(detection_view)
    mask_view = np.zeros(detection_view.shape[:2], dtype=np.uint8)
    contrast_view = np.zeros_like(detection_view)

    if pose_result.pose_landmarks:
        pose = pose_result.pose_landmarks[0]
        visible_indices = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
        visible = [pose[index] for index in visible_indices]
        index_map = {original: index for index, original in enumerate(visible_indices)}
        connections = [
            (index_map[start], index_map[end])
            for start, end in POSE_CONNECTIONS
            if start in index_map and end in index_map
        ]
        draw_preview_landmarks(media_pipe_view, visible, connections, (50, 220, 255))

    draw_body_context(lighting_view, pose_result, (50, 220, 255))
    draw_body_context(depth_view, pose_result)
    draw_body_context(contrast_view, pose_result)

    for hand in hand_result.hand_landmarks[:2]:
        points = np.array(draw_preview_landmarks(media_pipe_view, hand, HAND_CONNECTIONS, (70, 255, 120)), dtype=np.int32)
        if len(points) < 3:
            continue
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask_view, hull, 255)
        depth_values = np.array([landmark.z for landmark in hand], dtype=np.float32)
        low, high = float(depth_values.min()), float(depth_values.max())
        spread = max(high - low, 1e-6)
        for point, depth in zip(points, depth_values):
            amount = float((depth - low) / spread)
            color = (0, int(255 * amount), 255)
            cv2.circle(depth_view, tuple(point), 7, color, -1)
        cv2.polylines(depth_view, [hull], True, (0, 220, 255), 2)
        draw_hand_mask(mask_view, hand)
        draw_contrast_outline(contrast_view, detection_view, hand)

    mask_view = cv2.cvtColor(mask_view, cv2.COLOR_GRAY2BGR)
    draw_body_context(mask_view, pose_result)
    label = f"Frame {frame_idx}/{total_frames} | pose {len(pose_result.pose_landmarks[0]) if pose_result.pose_landmarks else 0}/33 | hands {len(hand_result.hand_landmarks[:2])}/2"
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask_view = cv2.morphologyEx(mask_view, cv2.MORPH_CLOSE, kernel)
    mask_view = cv2.morphologyEx(mask_view, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    for view, title in (
        (media_pipe_view, "MediaPipe landmarks"),
        (lighting_view, "Lighting compensated"),
        (depth_view, "Hand depth estimate"),
        (contrast_view, "Contrast hand outline"),
        (mask_view, "Detailed binary hand mask"),
    ):
        cv2.putText(view, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(view, label, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.imshow("Mudra - MediaPipe landmarks", media_pipe_view)
    cv2.imshow("Mudra - hand depth", depth_view)
    cv2.imshow("Mudra - contrast outline", contrast_view)
    cv2.imshow("Mudra - hand mask", mask_view)
    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
        raise KeyboardInterrupt


def initialize_landmarkers():
    # Lower confidence thresholds than the bare-minimum extractor on purpose:
    # our own gap-fill + quality-flagging downstream handles noisy/marginal
    # detections more intelligently than just discarding them at the source.
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
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
    detection_frame = prepare_detection_frame(frame_bgr)
    rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
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

    return pose_frame, left_frame, right_frame, origin, scale, pose_result, hand_result


def extract_raw_sequence(video_path, hand_landmarker, pose_landmarker, preview_callback=None):
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

        pose_frame, left_frame, right_frame, origin, scale, pose_result, hand_result = detect_frame(
            frame, hand_landmarker, pose_landmarker, int(ts), origin, scale
        )
        if preview_callback is not None:
            preview_callback(frame, pose_result, hand_result, frame_idx + 1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        pose_seq.append(pose_frame)
        left_seq.append(left_frame)
        right_seq.append(right_frame)
        frame_idx += 1

    cap.release()
    return (
        np.stack(pose_seq), np.stack(left_seq), np.stack(right_seq),
        np.array(timestamps_ms, dtype=np.float64), fps,
    )


def extract_raw_from_image(image_path, hand_landmarker, pose_landmarker, timestamp_ms=0, repeat_frames=5):
    """Static poses (fingerspelling, hardcoded words) get one real detection,
    replicated into a short pseudo-sequence so the output shape matches video
    clips -- keeps every downstream step (resampling, model input) uniform.
    timestamp_ms is a parameter (not hardcoded 0) so callers batching many
    images through one shared landmarker pair can pass a strictly increasing
    value per image -- required by MediaPipe VIDEO mode, see process_image_clips."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")

    pose_frame, left_frame, right_frame, _, _, _, _ = detect_frame(
        frame, hand_landmarker, pose_landmarker, timestamp_ms, np.zeros(3, dtype=np.float32), 1.0
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


def load_processed_clip_ids():
    """Clips already in the manifest with a real .npz on disk and no error --
    safe to skip on a re-run unless --force. New clips only extend the
    dataset instead of reprocessing everything from scratch every time."""
    if not MANIFEST_JSON.exists():
        return set()
    try:
        manifest = json.loads(MANIFEST_JSON.read_text())
    except json.JSONDecodeError:
        return set()
    return {
        r["clip_id"] for r in manifest
        if r.get("npz_path") and Path(r["npz_path"]).exists() and not r.get("error")
    }


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


def make_record(clip_id, label, clip_path, is_image, npz_path=None, stats=None, quality_flag=None, error=None):
    return {
        "clip_id": clip_id, "label": label, "source_path": str(clip_path),
        "npz_path": str(npz_path) if npz_path else None,
        "source_type": "image" if is_image else "video",
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
        "quality_flag": quality_flag, "error": error,
        **(stats or {}),
    }


def process_video_clips(video_clips, output_dir, preview_callback=None, progress_callback=None):
    """A fresh landmarker pair per clip, not shared across the batch:
    MediaPipe VIDEO mode tracks timestamps globally per landmarker instance,
    so reusing one pair across clips whose timestamps each restart near 0ms
    collides with the previous clip's ending timestamp and hard-crashes with
    "Input timestamp must be monotonically increasing." A little re-init
    overhead per clip buys correctness regardless of batch size/order --
    fine for tens/hundreds of videos, see process_image_clips for the
    thousands-of-images case where that overhead is no longer fine."""
    records = []
    for i, (label, clip_path) in enumerate(video_clips, 1):
        clip_id = f"{label}/{clip_path.stem}"
        print(f"[video {i}/{len(video_clips)}] {label}/{clip_path.name}", flush=True)
        hand_landmarker, pose_landmarker = initialize_landmarkers()
        try:
            raw = extract_raw_sequence(clip_path, hand_landmarker, pose_landmarker, preview_callback)
            features, presence, detected, timestamps_ms, stats = assemble_features(*raw)
            out_path = output_dir / label / f"{clip_path.stem}.npz"
            save_clip(out_path, features, presence, detected, timestamps_ms)
            quality_flag = compute_quality_flag(stats)
            records.append(make_record(clip_id, label, clip_path, False, out_path, stats, quality_flag))
            if quality_flag:
                print(f"  ⚠ flagged: {quality_flag}", flush=True)
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
            records.append(make_record(clip_id, label, clip_path, False, error=str(exc), quality_flag="extraction_failed"))
        finally:
            hand_landmarker.close()
            pose_landmarker.close()
        if progress_callback:
            progress_callback(i, f"video: {label}/{clip_path.name}")
    return records


def process_image_clips(image_clips, output_dir, progress_callback=None, progress_offset=0):
    """One shared landmarker pair for the WHOLE image batch, not per-clip.
    Safe here (unlike for videos) because each image is an independent
    single-frame detection with no real temporal relationship to the next --
    we just need a strictly increasing timestamp per call, tracked with a
    running counter, to satisfy MediaPipe VIDEO mode's requirement. This is
    what makes processing thousands of images (e.g. a public alphabet
    dataset) take minutes instead of hours -- per-clip re-init dominated
    runtime when each "clip" is a single frame."""
    if not image_clips:
        return []

    print(f"Processing {len(image_clips)} images with a shared landmarker pair (fast path)...", flush=True)
    hand_landmarker, pose_landmarker = initialize_landmarkers()
    records = []
    ts = 0
    try:
        for i, (label, clip_path) in enumerate(image_clips, 1):
            clip_id = f"{label}/{clip_path.stem}"
            if i % 200 == 0 or i == len(image_clips):
                print(f"  [image {i}/{len(image_clips)}]", flush=True)
            try:
                raw = extract_raw_from_image(clip_path, hand_landmarker, pose_landmarker, timestamp_ms=ts)
                ts += 33
                features, presence, detected, timestamps_ms, stats = assemble_features(*raw)
                out_path = output_dir / label / f"{clip_path.stem}.npz"
                save_clip(out_path, features, presence, detected, timestamps_ms)
                quality_flag = compute_quality_flag(stats)
                records.append(make_record(clip_id, label, clip_path, True, out_path, stats, quality_flag))
            except Exception as exc:
                ts += 33
                records.append(make_record(clip_id, label, clip_path, True, error=str(exc), quality_flag="extraction_failed"))
    finally:
        hand_landmarker.close()
        pose_landmarker.close()
        if progress_callback:
            progress_callback(progress_offset + i, f"image: {label}/{clip_path.name}")
    return records


def build_dataset(input_dir, output_dir, force=False):
    if TEST_MODE:
        input_dir = TEST_INPUT_DIR
        output_dir = TEST_OUTPUT_DIR
        all_clips = discover_clips(input_dir)
        clips = []
        labels = sorted({label for label, _ in all_clips})
        if TEST_LABELS is not None:
            labels = [label for label in labels if label in TEST_LABELS]
        for label in labels:
            label_videos = [
                path for clip_label, path in all_clips
                if clip_label == label and path.suffix.lower() in VIDEO_EXTS
            ]
            clips.extend((label, path) for path in label_videos[:TEST_VIDEO_LIMIT])
        if not clips:
            print(f"No test videos found under {input_dir}")
            return []
        print(f"TEST_MODE: processing up to {TEST_VIDEO_LIMIT} videos per action: {len(clips)} total", flush=True)
        try:
            preview_callback = show_test_preview if SHOW_TEST_PREVIEW else None
            write_progress("running", "starting", 0, len(clips))
            progress = lambda processed, current: write_progress("running", "extracting", processed, len(clips), current)
            records = process_video_clips(
                clips, output_dir, preview_callback=preview_callback,
                progress_callback=progress,
            )
            write_progress("done", "complete", len(records), len(clips))
        finally:
            cv2.destroyAllWindows()
        print(f"TEST_MODE complete: outputs written to {output_dir}; manifest unchanged", flush=True)
        return records

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    clips = discover_clips(input_dir)
    if not clips:
        print(f"No clips found under {input_dir} (expected <label>/<file> structure)")
        return []

    labels = sorted({label for label, _ in clips})
    print(f"Found {len(clips)} clips across {len(labels)} labels: {', '.join(labels)}", flush=True)

    already_processed = set() if force else load_processed_clip_ids()
    if already_processed:
        print(f"{len(already_processed)} clips already processed, will skip (use --force to redo)", flush=True)

    remaining = [(l, p) for l, p in clips if f"{l}/{p.stem}" not in already_processed]
    video_clips = [(l, p) for l, p in remaining if p.suffix.lower() in VIDEO_EXTS]
    image_clips = [(l, p) for l, p in remaining if p.suffix.lower() in IMAGE_EXTS]

    total_remaining = len(video_clips) + len(image_clips)
    write_progress("running", "starting", 0, total_remaining)
    progress = lambda processed, current: write_progress("running", "extracting", processed, total_remaining, current)
    records = process_video_clips(video_clips, output_dir, progress_callback=progress)
    records += process_image_clips(image_clips, output_dir, progress_callback=progress, progress_offset=len(video_clips))
    append_manifest(records)
    write_progress("done", "complete", len(records), total_remaining)

    flagged = [r for r in records if r.get("quality_flag")]
    print(f"\nDone: {len(records)} clips processed, {len(flagged)} flagged for review.", flush=True)
    if len(flagged) <= 30:
        for r in flagged:
            print(f"  ⚠ {r['clip_id']}: {r['quality_flag']}", flush=True)
    else:
        print(f"  ({len(flagged)} flagged clips -- see dataset_manifest.csv for the full list)", flush=True)

    return records


if __name__ == "__main__":
    force = "--force" in sys.argv
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    input_dir = Path(positional[0]) if len(positional) > 0 else DEFAULT_INPUT_DIR
    output_dir = Path(positional[1]) if len(positional) > 1 else DEFAULT_OUTPUT_DIR
    if not TEST_MODE and not input_dir.exists():
        print(f"Input folder not found: {input_dir}")
        print("Usage: python build_dataset.py [input_dir] [output_dir] [--force]")
        print("Expected structure: <input_dir>/<label>/<clip files>")
        sys.exit(1)
    build_dataset(input_dir, output_dir, force=force)
