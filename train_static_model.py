"""Static-pose classifier: single-frame RandomForest, trained on image-sourced
clips only (ISL fingerspelling alphabet + any other hardcoded static poses).

Runs ALONGSIDE the dynamic windowed model (train_model.py), not instead of
it -- a held pose is fully described by one frame, so there's no reason to
make it wait for a rolling window to fill the way a motion-based sign does.
See live_recognize.py for how both models are combined at inference time.
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
MODEL_PATH = BASE_DIR / "static_pose_classifier.pkl"


def load_static_samples():
    manifest = json.loads(MANIFEST_PATH.read_text())
    X, y = [], []
    skipped = 0
    for record in manifest:
        if record.get("source_type") != "image" or record.get("quality_flag") or not record.get("npz_path"):
            skipped += 1
            continue
        data = np.load(record["npz_path"])
        features, presence = data["features"], data["presence"]
        # All repeated frames are identical for an image source -- frame 0 is enough.
        position = features[0, :225]
        pres = presence[0]
        X.append(np.concatenate([position, pres]))
        y.append(record["label"])
    if skipped:
        print(f"Skipped {skipped} non-image/flagged/failed clips")
    return np.array(X, dtype=np.float32), np.array(y)


def main():
    X, y = load_static_samples()
    if X.shape[0] == 0:
        print("No image-sourced clips found in the manifest -- run build_dataset.py on image data first.")
        return

    n_classes = len(set(y))
    print(f"Loaded {X.shape[0]} static-pose samples, {X.shape[1]} features, {n_classes} classes")
    print(f"(random-chance baseline for {n_classes} classes: {1 / n_classes:.1%})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Train accuracy: {train_acc:.1%}")
    print(f"Held-out accuracy: {test_acc:.1%}")
    print("(each image here is an independent sample, so unlike the dynamic model's")
    print(" windowed-from-one-clip caveat, this IS a meaningful generalization check)")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf}, f)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
