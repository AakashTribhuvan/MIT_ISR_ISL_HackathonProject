"""HTTP adapter for the trained static-pose classifier.

The browser sends one image per request. MediaPipe extracts the same 225
normalized position features used during training, then the pickle model
returns the most likely static-pose label and confidence.
"""

import json
import pickle
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from build_dataset import (
    HAND_MODEL,
    POSE_MODEL,
    get_body_normalization,
    get_hand_label,
    landmarks_to_array,
    normalize,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "static_pose_classifier.pkl"
HOST = "127.0.0.1"
PORT = 8765
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def initialize_landmarkers():
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return (
        vision.HandLandmarker.create_from_options(hand_options),
        vision.PoseLandmarker.create_from_options(pose_options),
    )


def extract_static_features(image_bytes, hand_landmarker, pose_landmarker):
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The request body is not a readable image")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    pose_result = pose_landmarker.detect(mp_image)
    hand_result = hand_landmarker.detect(mp_image)

    pose_points = landmarks_to_array(
        pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None, 33
    )
    origin, scale = get_body_normalization(
        pose_points, np.zeros(3, dtype=np.float32), 1.0
    )
    pose_norm = normalize(pose_points, origin, scale)

    left_points = right_points = None
    for hand_index, hand_landmarks in enumerate(hand_result.hand_landmarks[:2]):
        side = get_hand_label(hand_result, hand_index)
        points = landmarks_to_array(hand_landmarks, 21)
        if side == "left" or (side is None and hand_index == 0):
            left_points = points
        else:
            right_points = points

    left_norm = normalize(left_points, origin, scale)
    right_norm = normalize(right_points, origin, scale)
    pose_vector = pose_norm if pose_norm is not None else np.zeros((33, 3), dtype=np.float32)
    left_vector = left_norm if left_norm is not None else np.zeros((21, 3), dtype=np.float32)
    right_vector = right_norm if right_norm is not None else np.zeros((21, 3), dtype=np.float32)
    position = np.concatenate([
        pose_vector.flatten(), left_vector.flatten(), right_vector.flatten(),
    ])
    presence = np.array([
        float(pose_norm is not None),
        float(left_norm is not None),
        float(right_norm is not None),
    ])
    return np.concatenate([position, presence]).astype(np.float32)


class StaticModelHandler(BaseHTTPRequestHandler):
    server_version = "MudraStaticModel/1.0"

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if urlparse(self.path).path != "/predict":
            self.send_json({"error": "Not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_IMAGE_BYTES:
            self.send_json({"error": "Send one image up to 10 MB"}, 413)
            return
        image_bytes = self.rfile.read(length)
        try:
            features = extract_static_features(
                image_bytes, self.server.hand_landmarker, self.server.pose_landmarker
            )
            probabilities = self.server.classifier.predict_proba([features])[0]
            top_index = int(np.argmax(probabilities))
            self.send_json({
                "label": str(self.server.classifier.classes_[top_index]),
                "confidence": float(probabilities[top_index]),
                "detected": {
                    "pose": bool(features[225]),
                    "left_hand": bool(features[226]),
                    "right_hand": bool(features[227]),
                },
            })
        except (ValueError, cv2.error) as error:
            self.send_json({"error": str(error)}, 400)

    def log_message(self, format, *args):
        return


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH.name}; run train_static_model.py first"
        )
    if not HAND_MODEL.exists() or not POSE_MODEL.exists():
        raise FileNotFoundError("Missing MediaPipe files in models/")

    with MODEL_PATH.open("rb") as model_file:
        bundle = pickle.load(model_file)
    hand_landmarker, pose_landmarker = initialize_landmarkers()
    server = ThreadingHTTPServer((HOST, PORT), StaticModelHandler)
    server.classifier = bundle["model"]
    server.hand_landmarker = hand_landmarker
    server.pose_landmarker = pose_landmarker
    print(f"Static model API: http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        hand_landmarker.close()
        pose_landmarker.close()


if __name__ == "__main__":
    main()