# Mudra — Workflow

Execution plan for: **"Smart Sign Language Learning & Real-Time Translation App for Deaf & Mute Students."**
Real-time CV system that recognises ISL signs and translates them to text/speech, plus a learning-mode component. Runs on a standard laptop/phone camera.

This file is the step-by-step roadmap. `PROJECT_NOTES.md` stays as the technical log of what's actually been built/tested so far — this file is where we plan what's next and check things off.

---

## 0. How to read this file

- Phases are roughly sequential (each depends on the one before), but not rigidly — we'll adjust as we go.
- `[ ]` = not started, `[~]` = in progress, `[x]` = done. Update as we go, or tell me to and I will.
- **⚠ Decision** callouts are places I'm flagging a choice rather than silently picking one — read these and tell me which way to go before we build on top of them.
- Phases 1–4 are the actual core objective from the problem statement (video in → recognised sign → text → speech, in real time). Phases 5–6 (avatar, learning mode) are the ambitious/creative extension on top. If time runs short, 1–4 alone is a complete, demoable product.

---

## 1. Guiding architecture decisions

Reasoning through the hard parts up front, before we write more code:

**Landmark-based, not pixel-based.** We already extract MediaPipe pose+hand landmarks instead of feeding raw video frames to a model. This is the right call — it's ~1000x less data per frame (225 floats vs a full image), trains fast on modest hardware, and generalizes across skin tone/clothing/background instead of memorizing them. Keep this as the foundation for everything downstream.

**Inference should run in the browser, not on a server.** MediaPipe has a Tasks-for-Web build (WASM) that runs pose/hand landmark detection directly in JS, in-browser, from `getUserMedia` — no native app, no upload-to-server round trip. If we export the trained classifier to **TensorFlow.js** or **ONNX Runtime Web**, the *entire* pipeline (camera → landmarks → recognized sign) can run client-side. That's what actually satisfies "works on mobiles, laptops, any device" — a server-side model would mean hosting/scaling costs and network latency killing the "real-time" requirement, especially on mobile data.
> ⚠ **Decision:** confirm we're building for browser-only inference, not a Python backend serving predictions. I'm assuming browser-only unless you say otherwise — it's the only approach that's actually low-latency and free to run on any device.

**Isolated-sign recognition first, continuous segmentation later.** ISL is a continuous stream in real use — there's no natural "frame 1 = start of sign." Trying to solve start/stop detection and classification at the same time on day one is how projects stall. MVP plan:
1. Train the classifier on **trimmed, single-sign clips** (one label per clip) — much simpler, and this is how nearly all practical sign-recognition demos are actually built.
2. For live use, segment the continuous camera stream into candidate windows using a simple **idle/rest-pose detector** (hand landmarks absent, or motion magnitude below a threshold, for N frames = "not signing"). Feed only the windows between idle periods to the classifier.
3. This means we need an explicit **"idle" / "no sign" class** in the training data — silence in audio terms. Without it the model has no way to know when *not* to output a word.

**Feature vector: use the fuller, normalized version for training.** The canonical `build_dataset.py` pipeline stores normalized position and velocity features:
- Raw coordinates mean the model has to relearn "where the person is standing" for every sign — shoulder-relative normalization (already implemented once, in the original `landmark_extractor.py`) removes that noise.
- Velocity features (frame-to-frame delta) matter a lot for *dynamic* signs — many ISL signs are defined by motion, not just a static handshape. Without velocity, the model is working from position alone, which is like trying to lip-read from a single photo.
This 450-dim normalized+velocity feature set is the actual training input because ISL is motion-heavy.

**Variable length is a real problem.** Videos have different frame counts/FPS. Sequence models need either fixed-length input (resample every clip to N frames via interpolation) or padding + masking. Resampling to a fixed length (e.g. 45 frames) is simpler to implement and export to TF.js, so that's the default plan — flag if you'd rather do padding/masking.

**Signer diversity is the single biggest data risk.** If every training clip is the same one person doing each sign once, the model will overfit to *that person's* hand size, speed, and style, and fail on anyone else — including you, testing it live. This is worth being explicit about early since it's expensive to fix later (re-recording). See Phase 1 data-quality checklist.

---

## 2. Phase 1 — Data Pipeline Hardening

*Goal: a dataset we can actually trust to train on, plus a manifest that tracks everything.*

- [x] Build the canonical extractor (`build_dataset.py`) with shoulder-relative normalization + velocity features (450-dim/frame). **Done 2026-08-21** — see "Hardened extractor" log below for the full design and test results.
- [x] Add per-frame **detection-confidence / hand-presence flags** to saved metadata — `presence` (detected-or-interpolated) and `detected` (raw) arrays, per part (pose/left/right), saved alongside every clip.
- [x] Store an explicit **timestamp array per frame** (ms) — real per-frame timestamps with a monotonic-safety fallback, not an implied frame index.
- [ ] Add a **fixed-length resampling utility**: interpolate any sequence to N frames (e.g. 45) for model-ready batches, while keeping the original variable-length `.npz` as the source of truth (never destroy raw extracted data). *(Deliberately deferred — this is a training-prep step, not an extraction step; see design log below.)*
- [x] Build a **dataset manifest** (`dataset_manifest.json` + `.csv`) — one row per clip: source path, extracted `.npz` path, label, fps, frame count, per-part detection %, quality flag, timestamp. Re-running on the same clip overwrites its row (keyed by `label/filename`) instead of duplicating. Signer ID and train/val/test split columns still need adding once we have multi-signer data.
- [ ] **Data quality checklist** to actually follow while recording:
  - Multiple signers per label (target: at least 2-3 different people per sign if at all possible — flag this to whoever's recording).
  - Multiple takes per sign per signer (natural variation in speed/framing).
  - Explicit **idle/no-sign clips** recorded (person standing normally, not signing) — this is the negative class that makes real-time segmentation possible.
  - Consistent framing (upper body + hands in frame, reasonable lighting) — doesn't need to be identical, but avoid extremes.
- [x] Add a **single-image capture path** for hardcoded static poses/words not in standard ISL vocab — `build_dataset.py` detects `.jpg/.png/.bmp` alongside video files and replicates the single detected frame into a short pseudo-sequence, same 450-dim schema as video clips.
- [x] Decide on a **label taxonomy** — one subfolder per label under `TrainingData/`, folder name = label. `build_dataset.py` walks this structure directly.
- [ ] **Split strategy: signer-disjoint, not clip-random.** Validation/test data should come from a signer not seen in training whenever possible — random-clip splitting will look good on paper and fail in front of the demo camera, because the model just memorized your training signer.

### Hardened extractor design log (2026-08-21)

Built `build_dataset.py` — the canonical folder-per-label batch extractor with a five-video diagnostic preview mode. Design decisions:

- **Two-pass processing.** Pass 1 decodes the whole clip and records raw MediaPipe detections per frame, including gaps (as NaN, not guessed values). Pass 2 sees the whole sequence at once and fills gaps intelligently: runs ≤5 frames with real data on both sides get linearly interpolated (a tracking flicker); longer runs, or runs touching the very start/end of a clip, are left at zero — a real absence, not a glitch, and faking a position there would teach the model something false. Only possible because this is offline batch processing, not the live camera pipeline — we get to see the future frames before deciding how to fill the past.
- **Presence tracked separately from raw detection.** Every clip's `.npz` stores `detected` (true only for frames MediaPipe actually saw) alongside `presence` (detected OR short-gap-interpolated) — so downstream code can tell "this hand was really there" from "we smoothed over a 2-frame blip," instead of conflating them.
- **Velocity reappearance fix.** A hand reappearing after a real (non-interpolated) absence would otherwise show a huge fake velocity spike (jumping from "wherever it disappeared" to "wherever it's now visible" reads as motion). That frame's velocity is zeroed instead.
- **Real per-frame timestamps**, not an assumed constant-FPS formula — pulled from the video container when available, with a monotonic-safety clamp (MediaPipe VIDEO mode hard-requires strictly increasing timestamps).
- **A fresh MediaPipe landmarker pair per clip**, not one shared across the whole batch. First real test run caught this the hard way — a shared landmarker instance carries its timestamp state *across* clips, so clip 2 restarting near 0ms collided with clip 1 ending near 57000ms and crashed with "Input timestamp must be monotonically increasing." Re-initializing per clip costs a little overhead but is correct regardless of batch size or clip order.
- **Speed-invariance (different signing speeds) is explicitly NOT handled here** — that's a training-time concern (time-warp augmentation on resampled sequences), not an extraction-time one. Extraction's job is to faithfully record what happened; the model should learn to be speed-robust from varied training data, not have speed normalized away before it ever sees it.
- **Quality gate**: every clip gets `pct_pose_detected` / `pct_left_hand_detected` / `pct_right_hand_detected`; if the better-detected hand is below 50% or the clip is under 5 frames, it's flagged in the manifest (`low_hand_detection`, `low_pose_detection`, `too_short`) rather than silently blended into training data.

**Validated with a real end-to-end test**, not just unit tests: ran `build_dataset.py` against the existing sample video plus a synthetically degraded copy (25% resolution, Gaussian blur, added noise, via a throwaway script). Result — degraded clip: pose still detected 92.8% of frames, but hands only 21.4%/4.1% → correctly flagged `low_hand_detection`. Clean clip: pose 87.7%, left hand 51.7%, right hand 15.4% → not flagged, but worth a flag *for us* anyway: right-hand detection was low even on our "good" footage, likely because the sign in that clip is mostly one-handed or the hand often drops out of frame — a real, useful thing the quality system surfaced on its first real run, not a pipeline defect. `test_gap_fill.py` (6 pure-numpy unit tests, no video/MediaPipe needed) covers the interpolation/velocity edge cases directly and all pass — run it any time the gap-fill logic changes.

**Feature schema**: `.npz` per clip = `features` (T, 450) [225 position + 225 velocity], `presence` (T, 3), `detected` (T, 3), `timestamps_ms` (T,). Position/velocity layout is [pose(99), left(63), right(63)], doubled for velocity.

---

## 3. Phase 2 — Website: Training Data Studio

*Goal: everyone involved can manage training data through the browser, not the filesystem. Nice-to-have for the final product, but great for the demo.*

- [ ] New dashboard tab: **file selector** — checkboxes over the manifest to include/exclude specific clips from the next training run (writes selection into a training-config file, doesn't delete anything).
- [ ] Manifest table view: label, signer, duration, split, quality flags, sortable/filterable.
- [ ] **Capture UI**: record a new clip (or snap a single image for a hardcoded pose) directly from the browser webcam, tag it with a label + signer name, auto-runs it through the extractor.
- [ ] Basic dataset stats: sample count per label, signer coverage per label (surfaces the diversity problem from Phase 1 visually — e.g. flag labels with only 1 signer).

---

## 4. Phase 3 — Model Training

*Goal: a small, fast classifier that runs client-side.*

- [x] **Feasibility baseline** (`train_model.py`, 2026-08-21): RandomForest over windowed summary features (mean+std of the 450-dim position+velocity vector across a 45-frame sliding window, plus per-part detection-presence rate — 903 features/window). Deliberately the simplest thing that could work, to prove the data carries signal before investing in a sequence model. **Result: 89% held-out window accuracy vs. 1.6% random-chance baseline for 61 classes** (584 windows generated from the 61 single-take clips via sliding-window sampling, since we currently have one video per label). Caveat logged prominently in the script's own output: held-out windows come from the *same* source clips as training windows, so this is a "does the pipeline carry signal" sanity check, not a real generalization score — true validation needs multiple independent takes/signers per label (see Phase 1's signer-disjoint split item, still open) or a live human performance the model never saw (see Phase 4 below).
- [ ] **Real architecture** (once more data lands): fixed-length landmark sequence → small **GRU/LSTM** or **1D-CNN/TCN** → softmax over `[labels..., idle]`. The RandomForest baseline proved the concept; a sequence model should do meaningfully better once there's enough independent data to justify it, and is also what actually exports cleanly to the browser (a pickled sklearn model does not).
- [ ] Train/val/test split pulled from the manifest's signer-disjoint split (Phase 1) — blocked on more signers actually being added.
- [ ] **Export to TensorFlow.js (or ONNX + onnxruntime-web)** — makes the model usable in a website per your requirement. `sign_classifier.pkl` (the current RandomForest) is a Python-only artifact and does NOT export to the browser — this is a real gap between today's proof-of-concept and the target architecture from Section 1, not yet closed.
- [ ] Evaluate: accuracy + confusion matrix per class, and real inference latency in-browser once exported.

---

### Dual-layer models + static alphabet layer (2026-08-21)

Live webcam test on the 61-word dynamic model showed ~no recognition — expected, and explained at the time: all 61 training clips came from one external source (one performer/camera/lighting), so the earlier 89% "held-out" number was a same-source sanity check, not real generalization. Rather than wait on more diverse word data, added a second, independent model layer specifically for **static poses** (alphabet, any hardcoded held pose), which sidesteps the motion/generalization problem differently:

- User supplied a public ISL alphabet image dataset (12,637 images, 26 letters, `TrainingData/<letter>/`, moved in from an `Alphabets/dataset_ISL/` folder).
- **Performance fix required first**: processing 12,637 individual images through the existing per-clip-fresh-landmarker path (needed for video correctness) would have taken hours. Added a genuinely safe optimization: images can share ONE landmarker pair across the whole batch (unlike videos), since each image is an independent single-frame detection — just needs a monotonically increasing timestamp counter across the run. Cut this from an estimated 3-7 hours to a few minutes. `build_dataset.py` now splits video vs. image clips and routes them through `process_video_clips` (fresh landmarker/clip) vs. `process_image_clips` (shared landmarker) respectively.
- Also added skip-if-already-processed (`load_processed_clip_ids`, `--force` to override) so re-running doesn't redo the 61 word clips every time new data is added — a real recurring need now that data arrives incrementally.
- **Quality gate caught something real**: 10,085 of 12,637 images (80%) flagged, mostly `low_hand_detection`. Checked an actual flagged image directly rather than assuming a bug — it's a legitimate small/low-res, tightly-cropped photo; MediaPipe genuinely struggles on the smaller "augmented" variants in this dataset. Correct behavior, not a bug. All 26 letters still have usable clean data after filtering (23-173 images/letter).
- **New `train_static_model.py`**: single-frame RandomForest (228 features: 225 position + 3 presence, no velocity — a held pose has none) trained only on the 2,552 clean image-sourced clips. **55.4% held-out accuracy vs. 3.8% chance baseline (26 classes)** — and unlike the dynamic model's caveat, this IS a real generalization check (independent images, not windowed sub-samples of one clip).
- **`live_recognize.py` is now dual-layered**: runs the static single-frame classifier AND the dynamic windowed classifier simultaneously every frame, displaying both ("Letter: X (NN%)" / "Word: Y (NN%)"), independently thresholded. Static model is optional at load time (skips gracefully if not trained) so the script still works with just the dynamic layer.
- Not yet live-tested by an actual person — that's the next step, same as before.

## 5. Phase 4 — Real-Time Recognition (Sign → Text → Speech)

*Goal: point a phone/laptop camera at someone signing, see text appear, hear it spoken. This is the core deliverable.*

- [x] **Python proof-of-concept** (`live_recognize.py`, 2026-08-21): webcam → same per-frame landmark/normalization logic as `build_dataset.py` → rolling 45-frame window → RandomForest prediction plus low-latency idle/signing segmentation and template matching. Still needs real-person accuracy testing. Run `python live_recognize.py`, perform a few trained words/phrases, q/Esc to quit.
- [x] Add hybrid MVP behavior: static alphabet prediction, dynamic model prediction, low-latency idle/signing segmentation, and nearest matching against normalized extracted templates. The template layer activates when `training_extracted/<label>/*.npz` exists.
- [ ] Browser-based live capture: MediaPipe Tasks Web (WASM) extracting landmarks directly from `getUserMedia`, same feature schema as training data (Phase 1) — critical that live features match training features exactly (normalization, velocity, frame rate) or the model will silently perform badly. *(Today's proof-of-concept runs in Python/OpenCV instead — fastest way to validate the idea; browser port is separate work once the model itself is proven and re-architected as a sequence model, see Phase 3.)*
- [x] Sliding-window buffer + idle-state segmentation (Section 1) to detect gesture start/stop live. *(Initial threshold-based MVP; thresholds need tuning with real webcam recordings.)*
- [ ] Run the exported classifier (TF.js/ONNX.js) client-side per segmented window → recognized sign.
- [ ] Assemble recognized signs into running text — needs simple debounce/confirmation logic so a held pose doesn't fire the same word 10 times.
- [ ] **Text-to-Speech** — options and tradeoffs:
  | Option | Pros | Cons |
  |---|---|---|
  | Web Speech API (`speechSynthesis`) | Free, built into every browser, zero network hop, lowest latency | Robotic voice, quality varies by OS/browser |
  | Murf API | High quality, natural voices | Paid, cloud round-trip = added latency, needs API key/backend call |
  | pyttsx3 | Offline, free | Python-only — needs a local server bridge to reach a website, adds complexity for no real benefit here |
  > ⚠ **Decision:** recommend **Web Speech API for the working MVP** (zero setup, zero latency, in-browser) and treat Murf as a "polish later" upgrade if voice quality matters for the final demo. Say the word if you'd rather commit to Murf from the start.

---

## 6. Phase 5 — Avatar: Speech/Text → Sign (stretch goal)

*Goal: reverse pipeline — spoken/typed English drives a 3D avatar performing ISL. Build after 1–4 work end-to-end.*

- [ ] Speech-to-text: Web Speech API (`SpeechRecognition`, browser-native, free) as the default; Whisper only if we need better accuracy and are OK with the server cost/latency.
- [ ] **Text → sign-sequence mapping**: dictionary of word → animation. This is itself a mini content-authoring problem — someone has to define what animation plays for each vocab word.
- [ ] **Reuse our own extracted landmark data to drive the avatar**, instead of separately authoring animations in Mixamo for every sign. We already have hand/pose landmark sequences per label from Phase 1 — retarget those directly onto avatar bones via Three.js. This avoids a second, entirely separate animation-production pipeline and keeps "sign data" as one single asset that feeds both recognition *and* avatar playback.
- [ ] 3D avatar: Mixamo-rigged humanoid + Three.js (or react-three-fiber) for bone animation/playback.
- [ ] Sentence assembly: timing/blending between consecutive signs so playback doesn't look like a slideshow.

---

## 7. Phase 6 — Duolingo-Style Learning Mode

*Goal: ties recognition + avatar together into the "learning" half of the problem statement.*

- [ ] Lesson content model: vocabulary list, each entry linked to a recognition label (Phase 4) and an avatar animation (Phase 5).
- [ ] **Mode A (receptive):** avatar performs a sign → user types what they think it means → checked against the answer.
- [ ] **Mode B (expressive):** app speaks/displays a word → user performs the sign on camera → live recognition (Phase 4) validates it.
- [ ] Progress tracking / scoring. Spaced repetition is a nice stretch but not essential for a working demo.

---

## 8. Phase 7 — Frontend Polish

- [ ] Keep the current interface **functionality-first** while we build (tabs: Live Translate, Training Studio, Learning Mode, Avatar Demo) — don't invest design effort here yet.
- [ ] Once you hand over the Figma design, restyle around the existing functional structure rather than rebuilding it.

---

## 9. Open decisions summary (things flagged above, collected here for quick review)

1. Browser-only client-side inference vs. a Python backend serving predictions — assuming browser-only.
2. Normalized + velocity (450-dim) feature set for real training data vs. staying with the current raw bare-minimum (225-dim) — recommending the fuller feature set.
3. Fixed-length resampling vs. padding+masking for variable-length sequences — recommending fixed-length resampling for simplicity.
4. TTS engine for the MVP: Web Speech API vs. Murf vs. pyttsx3 — recommending Web Speech API first, Murf as later polish.

---

## 10. Immediate next step

Everything else depends on data quality, so **Phase 1** is the logical starting point. Say the word (and weigh in on the decisions above) and we'll start there.
