"""Resample extracted variable-length clips to a fixed frame count."""

import argparse
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "training_extracted"
DEFAULT_OUTPUT_DIR = BASE_DIR / "training_resampled"
DEFAULT_FRAMES = 45


def resample_array(values, target_frames):
    values = np.asarray(values)
    if values.shape[0] == 0:
        return np.zeros((target_frames, *values.shape[1:]), dtype=values.dtype)
    if values.shape[0] == target_frames:
        return values.copy()
    source_positions = np.linspace(0.0, 1.0, values.shape[0])
    target_positions = np.linspace(0.0, 1.0, target_frames)
    flat = values.reshape(values.shape[0], -1).astype(np.float32)
    result = np.empty((target_frames, flat.shape[1]), dtype=np.float32)
    for column in range(flat.shape[1]):
        result[:, column] = np.interp(target_positions, source_positions, flat[:, column])
    return result.reshape((target_frames, *values.shape[1:])).astype(values.dtype, copy=False)


def resample_clip(source_path, target_path, target_frames):
    data = np.load(source_path)
    source_length = data["features"].shape[0]
    target_path.parent.mkdir(parents=True, exist_ok=True)

    indices = np.rint(np.linspace(0, max(0, source_length - 1), target_frames)).astype(int)
    features = resample_array(data["features"], target_frames)
    presence = data["presence"][indices]
    detected = data["detected"][indices]
    timestamps_ms = resample_array(data["timestamps_ms"], target_frames)
    np.savez_compressed(
        target_path,
        features=features,
        presence=presence,
        detected=detected,
        timestamps_ms=timestamps_ms,
    )


def resample_directory(input_dir, output_dir, target_frames):
    sources = sorted(Path(input_dir).rglob("*.npz"))
    for source_path in sources:
        relative_path = source_path.relative_to(input_dir)
        target_path = Path(output_dir) / relative_path
        resample_clip(source_path, target_path, target_frames)
    print(f"Resampled {len(sources)} clips to {target_frames} frames in {output_dir}")
    return len(sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("output_dir", nargs="?", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be at least 1")
    if not args.input_dir.exists():
        parser.error(f"Input folder not found: {args.input_dir}")
    resample_directory(args.input_dir, args.output_dir, args.frames)


if __name__ == "__main__":
    main()
