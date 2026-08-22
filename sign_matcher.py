"""Fast normalized template matching for extracted ISL sign sequences."""

from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_TARGET_FRAMES = 45


def resample_sequence(sequence, target_frames=DEFAULT_TARGET_FRAMES):
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.shape[0] == target_frames:
        return sequence
    if sequence.shape[0] == 0:
        return np.zeros((target_frames, sequence.shape[1]), dtype=np.float32)
    source = np.linspace(0.0, 1.0, sequence.shape[0])
    target = np.linspace(0.0, 1.0, target_frames)
    result = np.empty((target_frames, sequence.shape[1]), dtype=np.float32)
    for column in range(sequence.shape[1]):
        result[:, column] = np.interp(target, source, sequence[:, column])
    return result


def template_distance(left, right):
    left = resample_sequence(left)
    right = resample_sequence(right)
    # Position is more stable than velocity across signers; both still matter.
    position_distance = np.mean((left[:, :225] - right[:, :225]) ** 2)
    velocity_distance = np.mean((left[:, 225:] - right[:, 225:]) ** 2)
    return float(np.sqrt(position_distance) + 0.5 * np.sqrt(velocity_distance))


class SignTemplateMatcher:
    def __init__(self, template_dir, max_templates_per_label=8):
        self.template_dir = Path(template_dir)
        self.templates = defaultdict(list)
        for path in sorted(self.template_dir.rglob("*.npz")):
            label = path.parent.name
            if len(self.templates[label]) >= max_templates_per_label:
                continue
            try:
                data = np.load(path)
                self.templates[label].append(np.asarray(data["features"], dtype=np.float32))
            except (OSError, KeyError, ValueError):
                continue

    @property
    def available(self):
        return bool(self.templates)

    @property
    def labels(self):
        return tuple(sorted(self.templates))

    def match(self, sequence):
        best_label, best_distance = None, float("inf")
        for label, templates in self.templates.items():
            label_distance = min(template_distance(sequence, template) for template in templates)
            if label_distance < best_distance:
                best_label, best_distance = label, label_distance
        return best_label, best_distance
