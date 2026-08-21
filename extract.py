"""Bare-minimum video -> landmark sequence extractor (ISL). Streams a live
skeleton preview to the local web dashboard (dashboard/preview.jpg) by default;
pass --native-window for a local OpenCV window instead/as well."""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from list_training_files import write_files_json

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
HAND_MODEL = MODELS_DIR / "hand_landmarker.task"
POSE_MODEL = MODELS_DIR / "pose_landmarker_lite.task"

DEFAULT_VIDEO = BASE_DIR / "SampleVideos" / "SampleVideo.mp4"
DEFAULT_OUTPUT_DIR = BASE_DIR / "extracted"

DASHBOARD_DIR = BASE_DIR / "dashboard"
STATUS_PATH = DASHBOARD_DIR / "status.json"
PREVIEW_PATH = DASHBOARD_DIR / "preview.jpg"

DISPLAY_MAX_DIM = 720
PREVIEW_JPEG_QUALITY = 80

ARM_CONNECTIONS = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def to_px(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)


def draw_skeleton(frame, pose_landmarks, hand_landmarks_list, width, height):
    if pose_landmarks:
        for a, b in ARM_CONNECTIONS:
            cv2.line(frame, to_px(pose_landmarks[a], width, height), to_px(pose_landmarks[b], width, height), (255, 0, 0), 3)
        for idx in (11, 12, 13, 14, 15, 16):
            cv2.circle(frame, to_px(pose_landmarks[idx], width, height), 6, (0, 255, 0), -1)

    for hand_landmarks in hand_landmarks_list:
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, to_px(hand_landmarks[a], width, height), to_px(hand_landmarks[b], width, height), (255, 0, 0), 2)
        for lm in hand_landmarks:
            cv2.circle(frame, to_px(lm, width, height), 4, (0, 255, 0), -1)


def write_status(**fields):
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(fields, indent=2))


def write_preview(frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
    if not ok:
        return
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = PREVIEW_PATH.with_suffix(".tmp")
    try:
        tmp_path.write_bytes(buf.tobytes())
        tmp_path.replace(PREVIEW_PATH)
    except OSError:
        # Dashboard may hold the file open mid-read on Windows; a dropped
        # preview frame is harmless, unlike losing the whole extraction run.
        pass


def extract(video_path, output_path, web_preview=True, native_window=False):
    hand_landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval_ms = int(1000 / fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing {video_path.name}: {total_frames} frames at {fps:.1f} fps", flush=True)

    start_time = time.time()
    write_status(
        status="running", video=video_path.name, frame=0, total_frames=total_frames,
        percent=0.0, elapsed_sec=0.0, processing_fps=0.0,
        output_path=str(output_path), output_shape=None, error=None,
    )

    frames = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        height, width, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = frame_idx * frame_interval_ms

        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        pose_vector = np.zeros(33 * 3, dtype=np.float32)
        raw_pose = None
        if pose_result.pose_landmarks:
            raw_pose = pose_result.pose_landmarks[0]
            for i, lm in enumerate(raw_pose):
                pose_vector[i * 3:i * 3 + 3] = [lm.x, lm.y, lm.z]

        left_vector = np.zeros(21 * 3, dtype=np.float32)
        right_vector = np.zeros(21 * 3, dtype=np.float32)
        raw_hands = hand_result.hand_landmarks[:2]
        for hand_idx, hand_landmarks in enumerate(raw_hands):
            target = left_vector if hand_idx == 0 else right_vector
            for j, lm in enumerate(hand_landmarks):
                target[j * 3:j * 3 + 3] = [lm.x, lm.y, lm.z]

        frames.append(np.concatenate([pose_vector, left_vector, right_vector]))
        frame_idx += 1

        elapsed = time.time() - start_time
        proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0

        if web_preview or native_window:
            display_frame = frame
            scale = min(1.0, DISPLAY_MAX_DIM / max(width, height))
            if scale < 1.0:
                display_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            display_height, display_width = display_frame.shape[:2]

            draw_skeleton(display_frame, raw_pose, raw_hands, display_width, display_height)
            cv2.putText(display_frame, f"Frame {frame_idx}/{total_frames}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Processing FPS: {proc_fps:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            if web_preview and frame_idx % 2 == 0:
                write_preview(display_frame)

            if native_window:
                cv2.imshow("Mudra Extraction", display_frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

        if frame_idx % 10 == 0 or frame_idx == total_frames:
            percent = (frame_idx / total_frames * 100) if total_frames else 0.0
            print(f"  frame {frame_idx}/{total_frames}", flush=True)
            write_status(
                status="running", video=video_path.name, frame=frame_idx, total_frames=total_frames,
                percent=round(percent, 1), elapsed_sec=round(elapsed, 1), processing_fps=round(proc_fps, 1),
                output_path=str(output_path), output_shape=None, error=None,
            )

    cap.release()
    if native_window:
        cv2.destroyAllWindows()
    hand_landmarker.close()
    pose_landmarker.close()

    sequence = np.stack(frames).astype(np.float32) if frames else np.empty((0, 225), dtype=np.float32)
    np.save(output_path, sequence)
    print(f"Saved {sequence.shape} to {output_path}")

    elapsed = time.time() - start_time
    write_status(
        status="done", video=video_path.name, frame=frame_idx, total_frames=total_frames,
        percent=100.0, elapsed_sec=round(elapsed, 1),
        processing_fps=round(frame_idx / elapsed, 1) if elapsed > 0 else 0.0,
        output_path=str(output_path), output_shape=list(sequence.shape), error=None,
    )
    write_files_json()


if __name__ == "__main__":
    web_preview = "--no-preview" not in sys.argv
    native_window = "--native-window" in sys.argv
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]

    video = Path(positional[0]) if len(positional) > 0 else DEFAULT_VIDEO
    if not video.exists():
        print(f"Video not found: {video}")
        print("Usage: python extract.py <video.mp4> [output.npy] [--native-window] [--no-preview]")
        sys.exit(1)

    if len(positional) > 1:
        out = Path(positional[1])
    else:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = DEFAULT_OUTPUT_DIR / video.with_suffix(".npy").name

    try:
        extract(video, out, web_preview=web_preview, native_window=native_window)
    except Exception as exc:
        write_status(
            status="error", video=video.name, frame=None, total_frames=None,
            percent=None, elapsed_sec=None, processing_fps=None,
            output_path=str(out), output_shape=None, error=str(exc),
        )
        raise
