"""First feasibility model: RandomForest over windowed summary features (ISL).

Quick, dependency-light baseline to check the extraction pipeline actually
carries recognizable signal -- not the final production model (that's a
sequence model exported to the browser, per workflow.md Phase 3). Train/test
split here is at the WINDOW level, and windows from the same source clip are
highly correlated (we only have one video per label right now) -- so the
reported "held-out" accuracy is a sanity check, not a real generalization
estimate. The real test is live webcam recognition on a performance the
model has never seen.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BASE_DIR / "dataset_manifest.json"
MODEL_PATH = BASE_DIR / "sign_classifier.pkl"

WINDOW = 45
STRIDE = 15


def window_starts(total_frames, window, stride):
    if total_frames <= window:
        return [0]
    starts = list(range(0, total_frames - window + 1, stride))
    if starts[-1] != total_frames - window:
        starts.append(total_frames - window)
    return starts


def summarize_window(features, presence):
    if features.shape[0] < WINDOW:
        pad = WINDOW - features.shape[0]
        features = np.pad(features, ((0, pad), (0, 0)), mode="edge")
        presence = np.pad(presence, ((0, pad), (0, 0)), mode="edge")
    return np.concatenate([features.mean(axis=0), features.std(axis=0), presence.mean(axis=0)])


def load_windows():
    manifest = json.loads(MANIFEST_PATH.read_text())
    X, y = [], []
    skipped = 0
    for record in manifest:
        # Video-sourced clips only -- static image-sourced poses (alphabet,
        # hardcoded words) belong to train_static_model.py's single-frame
        # classifier instead. Keeping the two training sets disjoint is the
        # whole point of the dual-layer design; mixing them in dilutes both.
        if record.get("source_type") != "video" or record.get("quality_flag") or not record.get("npz_path"):
            skipped += 1
            continue
        data = np.load(record["npz_path"])
        features, presence = data["features"], data["presence"]
        for start in window_starts(features.shape[0], WINDOW, STRIDE):
            end = start + WINDOW
            X.append(summarize_window(features[start:end], presence[start:end]))
            y.append(record["label"])
    if skipped:
        print(f"Skipped {skipped} flagged/failed clips")
    return np.array(X, dtype=np.float32), np.array(y)


def main():
    X, y = load_windows()
    n_classes = len(set(y))
    print(f"Loaded {X.shape[0]} windows, {X.shape[1]} features/window, {n_classes} classes")
    print(f"(random-chance baseline for {n_classes} classes: {1 / n_classes:.1%})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Train accuracy: {train_acc:.1%}")
    print(f"Held-out WINDOW accuracy: {test_acc:.1%}")
    print("(caveat: held-out windows come from the same source clips as training windows -- ")
    print(" this checks the pipeline carries signal, not real-world generalization)")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf, "window": WINDOW, "stride": STRIDE}, f)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
