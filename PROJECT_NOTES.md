# MudraAI — Project Notes

Source reviewed: `D:\MudraAI-main\README.md`, `D:\MudraAI-main\landmark_extractor.py`

## What it is

Data pipeline for Indian Sign Language (ISL) dynamic gesture recognition. Extracts pose + hand landmarks from video using MediaPipe, builds normalized feature sequences for training a recognition model.

## Pipeline (landmark_extractor.py)

1. Input: folder of `.mp4` files or a single `.mp4`
2. MediaPipe Pose (33 landmarks) + Hand Landmarker (21 landmarks x 2 hands)
3. Normalize landmarks relative to shoulder midpoint / shoulder distance
4. Smooth landmarks over time (exponential smoothing, alpha=0.7)
5. Fill missing hand detections from last known value, zero out after 3 missed frames
6. Feature vector per frame: 99 (pose) + 63 (left hand) + 63 (right hand) = 225
7. Append 225-value velocity vector (frame-to-frame delta) = 450 total per frame
8. Save `.npy` sequence `(T, 450)` + metadata JSON (`source_video`, `relative_path`, `shape`, `frame_count`, `feature_size`)
9. Optional `--label` writes to `labels.json`

## Current file inventory (as of 2026-08-21)

Present in `D:\MudraAI-main`:

Referenced by README but **not yet present**:
- `v1/hand_tracker.py` — real-time webcam capture + recording
- `v1/models/hand_landmarker.task`, `v1/models/pose_landmarker_lite.task` — MediaPipe model files (required for extractor to run at all)
- `training_data/` — video inputs / extracted output folder

## Quick extraction test

Set `TEST_MODE = True` near the top of `build_dataset.py` and run `python build_dataset.py`. It processes up to five videos per action under `Videos_Sentence_Level/`, writes labeled templates to `sentence_test_extracted/`, and does not update the real dataset manifest. `SHOW_TEST_PREVIEW = False` keeps this batch fast; enable it only for visual inspection. Set `TEST_MODE = False` for the full `TrainingData/` dataset.

## Hybrid live recognition

`live_recognize.py` now combines the static alphabet classifier, dynamic word classifier, idle/signing segmentation, and nearest matching against normalized templates under `training_extracted/<label>/*.npz`. Run the full dataset builder first to populate that template library.

## Known issues / gaps

- `DEFAULT_INPUT_PATH` in `landmark_extractor.py` (line 15) is hardcoded to `D:/Coding/ISL/training_data/Sample Videos`, which doesn't match this project's actual location. Needs updating or always pass input path explicitly.
- Script will fail immediately without the `v1/models/*.task` files — need to source/download these before running anything.
- `v1/hand_tracker.py` doesn't exist yet — no real-time capture capability currently.

## Tech stack

-  **Language:** Python 3
- **Computer vision / video I/O:** OpenCV (`cv2`) — reads `.mp4` frames via `VideoCapture`, color-space conversion (BGR→RGB), draws skeleton overlays (`cv2.line`, `cv2.circle`, `cv2.putText`), live preview window (`cv2.imshow`)
- **Landmark detection:** MediaPipe Tasks API (`mediapipe`, `mediapipe.tasks.python`, `mediapipe.tasks.python.vision`)
  - `vision.HandLandmarker` — 21 landmarks/hand, up to 2 hands, `VIDEO` running mode
  - `vision.PoseLandmarker` — 33 body landmarks, `VIDEO` running mode
  - Both driven by pretrained `.task` model files (`hand_landmarker.task`, `pose_landmarker_lite.task`), loaded via `python.BaseOptions`
- **Numerical processing:** NumPy (`numpy`) — feature vectors, smoothing math, normalization, velocity deltas, `.npy` serialization
- **Data format / persistence:** `.npy` (NumPy binary arrays) for sequences, JSON (`json` module) for metadata and labels
- **CLI:** `argparse` for `landmark_extractor.py` command-line options
- **Filesystem:** `pathlib.Path` throughout for path handling
- **Timing:** `time` module (imported; used for frame timestamp bookkeeping alongside FPS-derived intervals)

Install:
```
pip install opencv-python numpy mediapipe
```

Not yet in the repo but implied by the README for `v1/hand_tracker.py`: same stack (OpenCV for webcam capture via `VideoCapture(0)`, MediaPipe for live detection, NumPy for buffering/saving sequences), plus keyboard input handling (`cv2.waitKey`) for the `s` (save) / `r` (reset) / `Esc` (exit) controls.

## Detailed workflow — how `landmark_extractor.py` achieves this

**1. Setup**
- Resolves `v1/models/hand_landmarker.task` and `v1/models/pose_landmarker_lite.task` paths relative to the script.
- `initialize_landmarkers()` builds a `HandLandmarker` (2 hands, VIDEO mode, 0.7 confidence thresholds for detection/presence/tracking) and a `PoseLandmarker` (1 pose, same mode/thresholds) from those model files.

**2. Input discovery**
- `discover_videos()` accepts either a single `.mp4` file or a folder, and recursively globs (`rglob("*.mp4")`) all videos in a folder, sorted for deterministic ordering.
- CLI options (`argparse`): input path (positional, defaults to a hardcoded path — see Known issues), `--output-dir`, `--max-frames`, `--skip-frames`, `--no-display`, `--label`.

**3. Per-video processing loop (`extract_sequence_from_video`)**
For each video:
- Opens it with `cv2.VideoCapture`, reads FPS to compute a per-frame timestamp interval (MediaPipe VIDEO mode requires monotonically increasing timestamps).
- Maintains running state across frames: a `LandmarkSmoother` for pose and one each for left/right hand, previous feature vectors (for gap-filling), a body normalization origin/scale, and per-hand "missing frame" counters.
- Loops frame-by-frame (optionally skipping every Nth frame via `--skip-frames` for speed):
  1. Convert frame BGR→RGB, wrap as an `mp.Image`.
  2. Run `hand_landmarker.detect_for_video()` and `pose_landmarker.detect_for_video()` at the computed timestamp.
  3. **Pose:** if detected, smooth it (`LandmarkSmoother`, EMA with `alpha=0.7`, i.e. `smoothed = 0.7*current + 0.3*previous`), compute body normalization (`get_body_normalization`: origin = shoulder midpoint, scale = distance between shoulders), normalize all pose points `(point - origin) / scale`, and build a flat 99-value pose vector (`build_pose_vector`). Also draws the arm skeleton on the preview frame.
  4. **Hands:** for each detected hand, determine left/right via MediaPipe's handedness label (falls back to detection index if unavailable), smooth it the same EMA way, normalize using the *same* body origin/scale as the pose (so hands and body share one coordinate frame), and build a flat 63-value vector per hand (`build_hand_vector`). Draws hand skeleton (21-point connections) on the preview frame.
  5. **Missing-hand handling:** if a hand isn't detected this frame, its last known vector is reused; after `MAX_MISSING_HAND_FRAMES` (3) consecutive misses, that hand's vector is zeroed out instead of carrying stale data forever.
  6. **Feature assembly:** concatenate pose(99) + left hand(63) + right hand(63) = 225-value position vector (`build_feature_vector`).
  7. **Velocity:** subtract the previous frame's 225-value position vector from the current one to get a 225-value velocity vector (zero on the first frame); concatenate → 450-value final feature vector for the frame.
  8. Append to the video's `sequence_frames` list. If `show_video` is on, overlay frame number / feature count and render the annotated frame; `q`/`Esc` aborts early.
- After all frames: stack into a single `(T, 450)` NumPy array (`T` = frames processed).

**4. Saving (`save_sequence`)**
- Writes the array to `<output_dir>/<video_stem>.npy`.
- Writes a sibling `<video_stem>.json` with `source_video`, `relative_path` (relative to the input root), `shape`, `frame_count`, `feature_size`.
- Creates `<output_dir>/labels/` (currently just ensured to exist, not populated per-video here).
- If `--label` was passed, upserts `{video_stem: label}` into `<output_dir>/labels.json` (single shared file across all processed videos in that run).

**5. Multi-video timestamp continuity**
- `main()` runs the loop above once per discovered video, advancing a running `timestamp_offset_ms` by each video's duration (+1000ms buffer) so MediaPipe's VIDEO-mode timestamp requirement (strictly increasing) is respected even though each video is a logically separate capture.
- Landmarkers are closed in a `finally` block regardless of success/failure.

**End result:** a folder of `(T, 450)` `.npy` sequences + JSON metadata + an optional `labels.json`, ready to be loaded as training data for a downstream sign-classification model (that model itself isn't in this repo yet).

## Requirements

Python 3, OpenCV (`opencv-python`), NumPy, MediaPipe.

```
pip install opencv-python numpy mediapipe
```

---

## Environment check (2026-08-21)

- Active `python` on PATH: **3.13.9** (Anaconda), at `C:\Users\Shamita\anaconda3\python.exe`. A separate Python 3.14 install also exists via the `py` launcher but is not what `python`/`pip` resolve to.
- `numpy` already installed: **2.3.5**
- `opencv-python` and `mediapipe` were **not installed** — resolved via `pip install --dry-run` against Python 3.13.9 with no conflicts: `mediapipe==1.0.1`, `opencv-python==5.0.0.93` (mediapipe pulls in `opencv-contrib-python`, `absl-py`, `sounddevice`, `flatbuffers` automatically).
- Conclusion: the full stack (Python 3.13.9 + numpy 2.3.5 + opencv-python 5.0.0.93 + mediapipe 1.0.1) resolves coherently — no version conflicts. Not yet actually `pip install`-ed; run `pip install -r requirements.txt` in `D:\ISL` when ready.
- Still missing (blocks actually running extraction): `models/hand_landmarker.task` and `models/pose_landmarker_lite.task` — need to be downloaded into `D:\ISL\models\`.

## Canonical extractor (`D:\ISL\build_dataset.py`)

Folder-based dataset extractor with an optional five-video preview mode:

- No smoothing, no shoulder-based normalization, no velocity features, no display/drawing, no CLI flags beyond input/output path.
- Pose (33x3=99) + hand 1 (21x3=63) + hand 2 (21x3=63) = **225 features/frame**, raw MediaPipe coordinates.
- Hand order is just detection order (index 0 = "left" slot, index 1 = "right" slot) — not handedness-checked, unlike the original.
- Test mode: set `TEST_MODE = True` and run `run_extract.bat` to skip the first video and process the next five from `Videos_Sentence_Level/` with the diagnostic preview windows.
- Full mode: set `TEST_MODE = False` and run `run_extract.bat` to process `TrainingData/`.

## Tasks

<!-- Add tasks here — ask Claude to read this file to pick them up. -->
