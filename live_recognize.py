"""Live webcam ISL recognition -- proof of concept (ISL).

Dual-layered: runs TWO classifiers side by side every frame, not one.
- Static-pose layer (static_pose_classifier.pkl): classifies the CURRENT
  frame alone -- a held pose (alphabet, hardcoded words) is fully described
  by one frame, so there's no reason to make it wait for a window to fill.
  Reacts instantly.
- Dynamic-sign layer (sign_classifier.pkl): classifies a rolling 45-frame
  window, for motion-based signs where a single frame isn't enough.
Both are shown at once; the static model is optional (skipped with a
message if not trained yet) so this still runs with just the dynamic model.

No start/stop segmentation yet (that's workflow.md Phase 4) -- this is the
fastest path to answering "does recognition actually work," not the final UX.

Detection runs on the RAW (unflipped) camera frame, matching how training
videos/images were processed -- flipping before detection would visually
mirror hands and can confuse MediaPipe's left/right handedness classifier.
The frame is only flipped for the on-screen mirror-style display.

Controls: q / Esc to quit.
"""

import pickle
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from build_dataset import (
    HAND_MODEL, POSE_MODEL,
    landmarks_to_array, normalize, get_body_normalization, get_hand_label,
)
from sign_matcher import SignTemplateMatcher

BASE_DIR = Path(__file__).resolve().parent
DYNAMIC_MODEL_PATH = BASE_DIR / "sign_classifier.pkl"
STATIC_MODEL_PATH = BASE_DIR / "static_pose_classifier.pkl"
TEMPLATE_DIR = BASE_DIR / "hello_test_extracted"

DYNAMIC_CONFIDENCE_THRESHOLD = 0.15  # 61 classes -> chance is ~1.6%
STATIC_CONFIDENCE_THRESHOLD = 0.30   # 26 classes -> chance is ~3.8%, and single-frame is noisier
PREDICT_DYNAMIC_EVERY_N_FRAMES = 5
PREDICT_STATIC_EVERY_N_FRAMES = 3
MOTION_START_THRESHOLD = 0.025
MOTION_END_THRESHOLD = 0.012
START_FRAMES = 3
END_FRAMES = 8
PRE_ROLL_FRAMES = 8
TEMPLATE_MAX_DISTANCE = 0.45


def initialize_landmarkers():
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35, min_tracking_confidence=0.35,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1, min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5, min_tracking_confidence=0.5,
    )
    return (
        vision.HandLandmarker.create_from_options(hand_options),
        vision.PoseLandmarker.create_from_options(pose_options),
    )


def load_model(path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def main(max_frames=None):
    dynamic_bundle = load_model(DYNAMIC_MODEL_PATH)
    dynamic_clf = dynamic_bundle["model"] if dynamic_bundle else None
    window = dynamic_bundle["window"] if dynamic_bundle else 45
    if dynamic_clf is None:
        print("Dynamic model not found; running template matching only.")

    static_bundle = load_model(STATIC_MODEL_PATH)
    static_clf = static_bundle["model"] if static_bundle else None
    if static_clf is None:
        print(f"No static-pose model at {STATIC_MODEL_PATH} -- running with dynamic-sign layer only. "
              f"Run train_static_model.py to add the static layer.")

    matcher = SignTemplateMatcher(TEMPLATE_DIR)
    print(f"Loaded {len(matcher.labels)} template labels from {TEMPLATE_DIR}" if matcher.available
          else f"No templates found at {TEMPLATE_DIR}; run build_dataset.py first.")

    hand_landmarker, pose_landmarker = initialize_landmarkers()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    feature_buffer = deque(maxlen=window)
    presence_buffer = deque(maxlen=window)
    pre_roll = deque(maxlen=PRE_ROLL_FRAMES)
    action_buffer = []
    signing = False
    motion_frames = 0
    quiet_frames = 0

    prev_pose = np.zeros((33, 3), dtype=np.float32)
    prev_left = np.zeros((21, 3), dtype=np.float32)
    prev_right = np.zeros((21, 3), dtype=np.float32)
    prev_position = None
    origin, scale = np.zeros(3, dtype=np.float32), 1.0

    frame_idx = 0
    dynamic_label, dynamic_conf = "...", 0.0
    static_label, static_conf = "...", 0.0
    template_label, template_distance = "...", None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(frame_idx * (1000 / 30))

            pose_result = pose_landmarker.detect_for_video(mp_image, ts)
            hand_result = hand_landmarker.detect_for_video(mp_image, ts)

            pose_points = landmarks_to_array(pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None, 33)
            origin, scale = get_body_normalization(pose_points, origin, scale)
            pose_norm = normalize(pose_points, origin, scale)
            pose_present = pose_norm is not None
            pose_vec = pose_norm if pose_present else prev_pose
            prev_pose = pose_vec

            left_points = right_points = None
            for hand_idx, hand_landmarks in enumerate(hand_result.hand_landmarks[:2]):
                side = get_hand_label(hand_result, hand_idx)
                points = landmarks_to_array(hand_landmarks, 21)
                if side == "left" or (side is None and hand_idx == 0):
                    left_points = points
                else:
                    right_points = points
            left_norm = normalize(left_points, origin, scale)
            right_norm = normalize(right_points, origin, scale)
            left_present = left_norm is not None
            right_present = right_norm is not None
            left_vec = left_norm if left_present else prev_left
            right_vec = right_norm if right_present else prev_right
            prev_left, prev_right = left_vec, right_vec

            position = np.concatenate([pose_vec.flatten(), left_vec.flatten(), right_vec.flatten()])
            velocity = np.zeros_like(position) if prev_position is None else position - prev_position
            prev_position = position
            features = np.concatenate([position, velocity]).astype(np.float32)
            presence_now = np.array([float(pose_present), float(left_present), float(right_present)], dtype=np.float32)

            feature_buffer.append(features)
            presence_buffer.append(presence_now)

            hand_velocity = features[225 + 99:225 + 225]
            motion_energy = float(np.linalg.norm(hand_velocity))
            hand_present = bool(presence_now[1] or presence_now[2])
            pre_roll.append(features.copy())
            if not signing:
                motion_frames = motion_frames + 1 if hand_present and motion_energy >= MOTION_START_THRESHOLD else 0
                if motion_frames >= START_FRAMES:
                    signing = True
                    action_buffer = list(pre_roll)
                    quiet_frames = 0
            else:
                action_buffer.append(features.copy())
                quiet_frames = quiet_frames + 1 if motion_energy <= MOTION_END_THRESHOLD else 0
                if quiet_frames >= END_FRAMES:
                    if matcher.available and len(action_buffer) >= 8:
                        candidate, distance = matcher.match(np.stack(action_buffer))
                        if candidate and distance <= TEMPLATE_MAX_DISTANCE:
                            template_label, template_distance = candidate, distance
                        else:
                            template_label, template_distance = "...", distance
                    signing = False
                    motion_frames = 0
                    quiet_frames = 0
                    action_buffer = []

            # Static layer: current frame only, no buffer needed -- reacts every frame (throttled for display stability).
            if static_clf is not None and frame_idx % PREDICT_STATIC_EVERY_N_FRAMES == 0:
                static_input = np.concatenate([position, presence_now]).astype(np.float32)
                probs = static_clf.predict_proba([static_input])[0]
                top_idx = int(np.argmax(probs))
                static_label = static_clf.classes_[top_idx]
                static_conf = float(probs[top_idx])

            # Dynamic layer: needs a full window of history first.
            if dynamic_clf is not None and len(feature_buffer) == window and frame_idx % PREDICT_DYNAMIC_EVERY_N_FRAMES == 0:
                feat_arr = np.stack(feature_buffer)
                pres_arr = np.stack(presence_buffer)
                summary = np.concatenate([feat_arr.mean(axis=0), feat_arr.std(axis=0), pres_arr.mean(axis=0)])
                probs = dynamic_clf.predict_proba([summary])[0]
                top_idx = int(np.argmax(probs))
                dynamic_label = dynamic_clf.classes_[top_idx]
                dynamic_conf = float(probs[top_idx])

            display_frame = cv2.flip(frame, 1)  # mirror only for viewing, detection already ran on the original

            static_text = f"Letter: {static_label} ({static_conf:.0%})" if static_conf >= STATIC_CONFIDENCE_THRESHOLD else "Letter: ..."
            dynamic_text = f"Word: {dynamic_label} ({dynamic_conf:.0%})" if dynamic_conf >= DYNAMIC_CONFIDENCE_THRESHOLD else "Word: ..."
            template_text = f"Match: {template_label}" if template_label != "..." else "Match: ..."
            cv2.putText(display_frame, static_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(display_frame, dynamic_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
            cv2.putText(display_frame, template_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 180, 0), 2)
            state_text = "SIGNING" if signing else "IDLE"
            cv2.putText(display_frame, state_text, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "q/Esc to quit", (20, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Mudra Live Recognition (proof of concept)", display_frame)

            frame_idx += 1
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
            if max_frames is not None and frame_idx >= max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        pose_landmarker.close()


if __name__ == "__main__":
    main()
