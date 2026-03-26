# dicUtils/tracking.py
#
# Single-point Lucas-Kanade tracker.
# Torch is imported lazily inside create_lk_tracker() only,
# so importing this module does NOT trigger torch DLL loading.

import cv2
import numpy as np
from dicUtils.io import _load_gray   # single source of truth


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

class LKParams:
    def __init__(self, win_size: int, max_level: int, iters: int):
        self.win_size  = win_size
        self.max_level = max_level
        self.iters     = iters
        self.lk_params = dict(
            winSize  = (win_size, win_size),
            maxLevel = max_level,
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                iters,
                0.03,
            ),
        )


def create_lk_tracker(win_size: int, max_level: int, iters: int) -> LKParams:
    """Drop-in replacement for the old create_cuda_lk()."""
    try:
        import torch   # lazy import — only triggered by main.py, not main_dense.py
        if torch.cuda.is_available():
            print(f"[Tracker] CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("[Tracker] CUDA not available — running on CPU.")
    except Exception:
        print("[Tracker] Torch not available — running on CPU.")
    return LKParams(win_size, max_level, iters)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def track_sequence(
    params: LKParams,
    frame_paths: list,
    prev_pts: np.ndarray,
    frame_step: int,
    display: bool = False,
) -> np.ndarray:
    """
    Track a single seed point using cv2.calcOpticalFlowPyrLK.
    """
    trajectory    = []
    prev_np       = _load_gray(frame_paths[0])
    height, width = prev_np.shape
    current_pts   = prev_pts.reshape(-1, 1, 2).astype(np.float32)

    for idx in range(frame_step, len(frame_paths), frame_step):

        curr_np = _load_gray(frame_paths[idx])

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_np,
            curr_np,
            current_pts,
            None,
            **params.lk_params,
        )

        if next_pts is None or status is None:
            print("Tracking failed (LK returned None).")
            break

        good_new = next_pts[status.flatten() == 1]

        if len(good_new) == 0:
            print("Tracking lost (no valid points).")
            break

        x, y = good_new[0].flatten()[:2]

        if not (0 <= x < width and 0 <= y < height):
            print(f"Seed left frame at Frame {idx:04d} | x={x:.3f}, y={y:.3f}")
            break

        trajectory.append((x, y))
        print(f"Frame {idx:04d} | x={x:.3f}, y={y:.3f}")

        if display:
            disp = curr_np.copy()
            cv2.circle(disp, (int(x), int(y)), 6, 255, -1)
            cv2.imshow("Tracking", disp)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        current_pts = good_new.reshape(-1, 1, 2).astype(np.float32)
        prev_np     = curr_np

    if display:
        cv2.destroyAllWindows()

    return np.array(trajectory)