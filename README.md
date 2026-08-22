# Mudra static image model

This repository contains an ISL static-pose classifier. The trained model is
`static_pose_classifier.pkl`; it is a Python scikit-learn pickle and should be
called through the included HTTP adapter rather than loaded in browser code.

## Start the API

From the repository root, install the pinned dependencies and start the local
server:

```powershell
python -m pip install -r requirements.txt
python static_model_server.py
```

The API listens on `http://127.0.0.1:8765`. The server requires these files:

- `static_pose_classifier.pkl` (created by `python train_static_model.py`)
- `models/hand_landmarker.task`
- `models/pose_landmarker_lite.task`

The model artifact is intentionally not expected to be committed to Git. Share
it separately or retrain it from the dataset on the target machine.

## Frontend request

Send a `POST` request to `/predict` with the image bytes as the request body.
The endpoint accepts JPEG, PNG, and other formats supported by OpenCV, up to 10
MB. A file upload is not required; this works directly with a camera frame or
`canvas.toBlob()` result.

```javascript
async function predictStaticPose(blob) {
  const response = await fetch("http://127.0.0.1:8765/predict", {
    method: "POST",
    headers: { "Content-Type": blob.type || "image/jpeg" },
    body: blob,
  });
  if (!response.ok) {
    throw new Error((await response.json()).error || "Prediction failed");
  }
  return response.json();
}

const result = await predictStaticPose(imageBlob);
console.log(result.label, result.confidence);
```

Successful response:

```json
{
  "label": "a",
  "confidence": 0.97,
  "detected": {
    "pose": true,
    "left_hand": true,
    "right_hand": false
  }
}
```

`confidence` is the classifier's top-class probability, from `0` to `1`.
The UI should choose its own display threshold; the desktop proof of concept
uses `0.30`. Treat low-confidence results as unknown instead of showing them as
certain predictions. The `detected` fields indicate whether MediaPipe found
each body part and are useful for prompting the user to reposition.

## Browser camera example

```javascript
const canvas = document.createElement("canvas");
const context = canvas.getContext("2d");

async function predictVideoFrame(video) {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  context.drawImage(video, 0, 0);
  const blob = await new Promise((resolve) =>
    canvas.toBlob(resolve, "image/jpeg", 0.85)
  );
  return predictStaticPose(blob);
}
```

Call this at a modest interval such as 3 to 10 frames per second. Do not send
every camera frame unless the frontend deliberately adds throttling and drops
requests that are still in flight.

## Important model behavior

- This is a **static** classifier: each request represents one held pose.
- The API runs MediaPipe pose and hand landmark extraction before prediction.
- It expects the same normalized 225-value position vector used during
  training, plus three presence values.
- It does not recognize motion-based words or sentences. Those use the dynamic
  model and are a separate integration.
- The API currently allows cross-origin requests for local development. Put it
  behind the website's own backend or configure a restricted origin before
  deploying publicly.