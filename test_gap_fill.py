"""Correctness checks for build_dataset.fill_gaps / compute_velocity.
No video/MediaPipe dependency -- run directly: python test_gap_fill.py"""

import numpy as np

from build_dataset import fill_gaps, compute_velocity, SHORT_GAP_MAX_FRAMES


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


def test_short_gap_interpolated():
    seq = np.array([[[0.0]], [[np.nan]], [[np.nan]], [[np.nan]], [[4.0]]])
    filled, detected, present = fill_gaps(seq, short_gap_max=SHORT_GAP_MAX_FRAMES)
    ok = True
    ok &= check("short gap: endpoints preserved", filled[0, 0, 0] == 0.0 and filled[4, 0, 0] == 4.0)
    ok &= check("short gap: linear interpolation", np.allclose(filled[:, 0, 0], [0, 1, 2, 3, 4]))
    ok &= check("short gap: interpolated frames marked present", present[1] and present[2] and present[3])
    ok &= check("short gap: interpolated frames NOT marked as raw-detected", not detected[1] and not detected[2] and not detected[3])
    return ok


def test_long_gap_left_zeroed():
    gap_len = SHORT_GAP_MAX_FRAMES + 3
    seq = np.full((gap_len + 2, 1, 1), np.nan)
    seq[0, 0, 0] = 5.0
    seq[-1, 0, 0] = 9.0
    filled, detected, present = fill_gaps(seq, short_gap_max=SHORT_GAP_MAX_FRAMES)
    ok = True
    ok &= check("long gap: middle left at zero, not fabricated", np.allclose(filled[1:-1, 0, 0], 0.0))
    ok &= check("long gap: middle NOT marked present", not present[1:-1].any())
    return ok


def test_gap_touching_start_not_interpolated():
    # No "before" frame to interpolate from -> stays zero/absent even though short.
    seq = np.array([[[np.nan]], [[np.nan]], [[5.0]]])
    filled, detected, present = fill_gaps(seq, short_gap_max=SHORT_GAP_MAX_FRAMES)
    return check("start gap: no lookback -> left zero/absent", filled[0, 0, 0] == 0.0 and not present[0])


def test_no_gap_passthrough():
    seq = np.array([[[1.0]], [[2.0]], [[3.0]]])
    filled, detected, present = fill_gaps(seq, short_gap_max=SHORT_GAP_MAX_FRAMES)
    return check("no gap: values untouched, all detected/present", np.allclose(filled[:, 0, 0], [1, 2, 3]) and detected.all() and present.all())


def test_velocity_reappearance_masked():
    # is_present: [True, False, True] -- frame 2 "reappears" after an absence;
    # its raw delta would be a fake jump and should be suppressed instead.
    position = np.array([[0.0], [0.0], [100.0]])
    is_present = np.array([True, False, True])
    velocity = compute_velocity(position, is_present)
    return check("velocity: reappearance spike suppressed", velocity[2, 0] == 0.0)


def test_velocity_normal_motion_preserved():
    position = np.array([[0.0], [1.0], [3.0]])
    is_present = np.array([True, True, True])
    velocity = compute_velocity(position, is_present)
    return check("velocity: normal frame-to-frame motion preserved", np.allclose(velocity[:, 0], [0, 1, 2]))


if __name__ == "__main__":
    results = [
        test_short_gap_interpolated(),
        test_long_gap_left_zeroed(),
        test_gap_touching_start_not_interpolated(),
        test_no_gap_passthrough(),
        test_velocity_reappearance_masked(),
        test_velocity_normal_motion_preserved(),
    ]
    print(f"\n{sum(results)}/{len(results)} tests passed")
    if not all(results):
        raise SystemExit(1)
