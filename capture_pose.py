"""Webcam capture tool for hardcoded static poses (ISL alphabet, custom words).

Saves raw frames as labeled images under TrainingData/<label>/, the same
folder-per-label convention build_dataset.py already expects -- run
build_dataset.py afterward to extract these and fold them into the dataset
like any other clip, then train_model.py to retrain.

Default labels: a-z (ISL fingerspelling alphabet), one label per keyboard
key. Hold a pose, press its letter key to capture the CURRENT frame under
that label; capture a few times per letter (slightly different angle/
position each time) for a bit of natural variation. q/Esc to quit.

Not alphabet-only -- pass any single-character-per-label string to capture()
for other hardcoded static words, as long as each has a distinct key.
"""

import time
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = BASE_DIR / "TrainingData"

DEFAULT_LABELS = "abcdefghijklmnopqrstuvwxyz"


def main(labels=DEFAULT_LABELS):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    valid_keys = {ord(ch) for ch in labels}
    counts = {
        ch: len(list((TRAINING_DIR / ch).glob("*.jpg"))) if (TRAINING_DIR / ch).exists() else 0
        for ch in labels
    }

    print("Hold a pose, press its letter key to capture. q/Esc to quit.")
    print(f"Labels: {', '.join(labels)}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            display = cv2.flip(frame, 1)  # mirror for viewing only; saved frame stays unflipped
            cv2.putText(display, "Press a letter key to capture that pose (q/Esc to quit)",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, f"Captured so far: {sum(counts.values())}",
                        (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.imshow("Mudra Pose Capture", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in valid_keys:
                label = chr(key)
                label_dir = TRAINING_DIR / label
                label_dir.mkdir(parents=True, exist_ok=True)
                counts[label] += 1
                out_path = label_dir / f"{label}_{int(time.time() * 1000)}.jpg"
                cv2.imwrite(str(out_path), frame)
                print(f"Captured '{label}' -> {out_path.name} (total for '{label}': {counts[label]})")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
