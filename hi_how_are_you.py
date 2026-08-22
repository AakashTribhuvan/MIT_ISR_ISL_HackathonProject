"""Focused single-action ISL extraction and webcam matching experiment."""

import argparse
import time
import winsound
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from build_dataset import (
    HAND_MODEL,
    POSE_MODEL,
    VIDEO_EXTS,
    assemble_features,
    extract_raw_sequence,
    initialize_landmarkers,
    get_body_normalization,
    get_hand_label,
    landmarks_to_array,
    normalize,
)
from sign_matcher import SignTemplateMatcher

BASE_DIR = Path(__file__).resolve().parent
ACTION = "hi how are you"
SOURCE_DIR = BASE_DIR / "Videos_Sentence_Level" / ACTION
OUTPUT_DIR = BASE_DIR / "hi_how_are_you_templates"
MAX_VIDEOS = 5
WINDOW = 45
CAPTURE_SECONDS = 5
# Calibrated from the five real templates: their pairwise distances range
# from roughly 0.64 to 1.15, so 0.45 rejected genuine examples.
MAX_TEMPLATE_DISTANCE = 1.25


def extract_templates():
    videos = sorted(path for path in SOURCE_DIR.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTS)
    videos = videos[:MAX_VIDEOS]
    if not videos:
        raise FileNotFoundError(f"No videos found in {SOURCE_DIR}")

    action_output_dir = OUTPUT_DIR / ACTION
    action_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {len(videos)} '{ACTION}' videos")
    for index, video_path in enumerate(videos, 1):
        print(f"[{index}/{len(videos)}] {video_path.name}", flush=True)
        hand_landmarker, pose_landmarker = initialize_landmarkers()
        try:
            raw = extract_raw_sequence(video_path, hand_landmarker, pose_landmarker)
            features, presence, detected, timestamps, stats = assemble_features(*raw)
            output_path = action_output_dir / f"{video_path.stem}.npz"
            np.savez_compressed(
                output_path,
                features=features,
                presence=presence,
                detected=detected,
                timestamps_ms=timestamps,
            )
            print(
                f"  saved {features.shape}; hands detected "
                f"left={stats['pct_left_hand_detected']:.1%}, "
                f"right={stats['pct_right_hand_detected']:.1%}",
                flush=True,
            )
        finally:
            hand_landmarker.close()
            pose_landmarker.close()
    return len(videos)


def initialize_webcam_landmarkers():
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return (
        vision.HandLandmarker.create_from_options(hand_options),
        vision.PoseLandmarker.create_from_options(pose_options),
    )


def webcam_match():
    matcher = SignTemplateMatcher(OUTPUT_DIR)
    if not matcher.available:
        raise RuntimeError("No templates found. Run with --extract first.")
    print(f"Loaded {len(matcher.templates.get(ACTION, []))} templates for '{ACTION}'")

    hand_landmarker, pose_landmarker = initialize_webcam_landmarkers()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam")

    previous_pose = np.zeros((33, 3), dtype=np.float32)
    previous_left = np.zeros((21, 3), dtype=np.float32)
    previous_right = np.zeros((21, 3), dtype=np.float32)
    previous_position = None
    origin, scale = np.zeros(3, dtype=np.float32), 1.0
    capture_frames = []
    capture_started = None
    result = "..."
    distance = None
    frame_index = 0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = int(frame_index * 1000 / 30)
            pose_result = pose_landmarker.detect_for_video(image, timestamp)
            hand_result = hand_landmarker.detect_for_video(image, timestamp)

            pose_points = landmarks_to_array(pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None, 33)
            origin, scale = get_body_normalization(pose_points, origin, scale)
            pose = normalize(pose_points, origin, scale)
            pose_present = pose is not None
            pose = pose if pose_present else previous_pose

            left_points = right_points = None
            for hand_index, hand in enumerate(hand_result.hand_landmarks[:2]):
                side = get_hand_label(hand_result, hand_index)
                points = landmarks_to_array(hand, 21)
                if side == "left" or (side is None and hand_index == 0):
                    left_points = points
                else:
                    right_points = points
            left = normalize(left_points, origin, scale)
            right = normalize(right_points, origin, scale)
            left_present, right_present = left is not None, right is not None
            left = left if left_present else previous_left
            right = right if right_present else previous_right
            previous_pose, previous_left, previous_right = pose, left, right

            position = np.concatenate([pose.ravel(), left.ravel(), right.ravel()])
            velocity = np.zeros(225, dtype=np.float32) if previous_position is None else position - previous_position
            previous_position = position
            features = np.concatenate([position, velocity]).astype(np.float32)
            if capture_started is not None:
                capture_frames.append(features.copy())
                if time.monotonic() - capture_started >= CAPTURE_SECONDS:
                    winsound.Beep(880, 180)
                    candidate, candidate_distance = matcher.match(np.stack(capture_frames))
                    distance = candidate_distance
                    result = candidate if candidate_distance <= MAX_TEMPLATE_DISTANCE else "..."
                    print(
                        f"Capture complete: candidate={candidate!r}, "
                        f"distance={candidate_distance:.3f}, result={result!r}",
                        flush=True,
                    )
                    capture_frames = []
                    capture_started = None

            display = cv2.flip(frame, 1)
            cv2.putText(display, f"Match: {result}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 220, 255), 2)
            state = f"CAPTURING {max(0.0, CAPTURE_SECONDS - (time.monotonic() - capture_started)):.1f}s" if capture_started is not None else "PRESS SPACE TO START"
            cv2.putText(display, f"State: {state}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            if distance is not None:
                cv2.putText(display, f"Distance: {distance:.3f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, .6, (200, 220, 220), 1)
            cv2.putText(display, "q / Esc to quit", (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            cv2.imshow("ISL - hi how are you matcher", display)
            frame_index += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" ") and capture_started is None:
                winsound.Beep(660, 180)
                capture_started = time.monotonic()
                capture_frames = []
                result = "..."
                distance = None
    finally:
        camera.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        pose_landmarker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true", help="Extract up to five action templates")
    parser.add_argument("--live", action="store_true", help="Run webcam matching")
    args = parser.parse_args()
    if not args.extract and not args.live:
        args.extract = True
        args.live = True
    if args.extract:
        extract_templates()
    if args.live:
        webcam_match()


if __name__ == "__main__":
    main()
