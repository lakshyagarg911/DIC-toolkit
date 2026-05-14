# dicUtils/dic_runner.py
#
# DIC computation thread + field reshaping utilities.
# Bridges dic_engine.py with the existing streaming store and player.

import numpy as np
import torch
from PyQt5.QtCore import QThread, pyqtSignal

from dicUtils.io import _load_gray
from dicUtils.dic_engine import (
    compute_dic_frame, make_roi_grid, get_device,
    SUBSET_RADIUS, ZNCC_THRESHOLD, GRID_STEP
)

# ── DIC field catalogue ───────────────────────────────────────────────────────
# These keys are produced by DIC (as opposed to optical flow keys)

DIC_FIELD_CATALOGUE = {
    "dic_u":    ("DIC Displacement u",   "pixels",  True),
    "dic_v":    ("DIC Displacement v",   "pixels",  True),
    "dic_exx":  ("DIC Strain e_xx",      "px/px",   True),
    "dic_eyy":  ("DIC Strain e_yy",      "px/px",   True),
    "dic_exy":  ("DIC Strain e_xy",      "px/px",   True),
    "dic_zncc": ("DIC Correlation ZNCC", "[-1,1]",  True),
    "dic_ux":   ("DIC du/dx",            "px/px",   False),
    "dic_uy":   ("DIC du/dy",            "px/px",   False),
    "dic_vx":   ("DIC dv/dx",            "px/px",   False),
    "dic_vy":   ("DIC dv/dy",            "px/px",   False),
}


def scatter_to_grid(values: np.ndarray,
                    points: np.ndarray,
                    bbox: tuple,
                    fill: float = np.nan) -> np.ndarray:
    """
    Scatter point values back onto a 2D grid (ROI bounding box size).

    values  : (N,) float32
    points  : (N, 2) float32 — (x, y) in image coords
    bbox    : (x0, y0, w, h)
    fill    : value for pixels with no data point

    Returns: (h, w) float32 array
    """
    x0, y0, w, h = bbox
    grid = np.full((h, w), fill, dtype=np.float32)
    xs = (points[:, 0] - x0).astype(np.int32)
    ys = (points[:, 1] - y0).astype(np.int32)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    grid[ys[valid], xs[valid]] = values[valid]
    return grid


# ── DIC Compute Thread ────────────────────────────────────────────────────────

class DICComputeThread(QThread):
    """
    Runs subset-based DIC (Newton-Raphson, affine, ZNCC) on GPU.
    Emits frame_ready with per-frame field grids — written to HDF5
    by the main thread (same pattern as AddFieldsThread).

    Reference strategy: incremental (each frame vs previous).
    """
    progress    = pyqtSignal(int, int)
    frame_ready = pyqtSignal(int, dict)   # frame_idx, {key: (h,w) array}
    done        = pyqtSignal()
    error       = pyqtSignal(str)

    def __init__(self, frame_paths, polygon, mask, bbox,
                 frame_step, fps, selected_dic_keys, h5_path,
                 display_keys=None, R=SUBSET_RADIUS, grid_step=GRID_STEP):
        super().__init__()
        self.frame_paths       = frame_paths
        self.polygon           = polygon
        self.mask              = mask
        self.bbox              = bbox
        self.frame_step        = frame_step
        self.fps               = fps
        self.selected_dic_keys = selected_dic_keys
        self.h5_path           = h5_path
        self.display_keys      = display_keys or selected_dic_keys
        self.R                 = R
        self.grid_step         = grid_step

    @classmethod
    def from_config(cls, frame_paths, polygon, mask, bbox,
                    frame_step, fps, selected_dic_keys, h5_path,
                    display_keys=None, cfg=None):
        """Create thread using parameters from configuration.json."""
        cfg       = cfg or {}
        R         = cfg.get("dic_subset_radius", SUBSET_RADIUS)
        grid_step = cfg.get("dic_grid_step",     GRID_STEP)
        return cls(frame_paths, polygon, mask, bbox,
                   frame_step, fps, selected_dic_keys, h5_path,
                   display_keys=display_keys, R=R, grid_step=grid_step)

    def run(self):
        try:
            device = get_device()
            paths  = self.frame_paths
            total  = (len(paths) - 1) // self.frame_step

            # Generate analysis point grid
            pts = make_roi_grid(self.mask, self.bbox, self.R, self.grid_step)
            print(f"[DIC] {len(pts)} analysis points  "
                  f"R={self.R}  step={self.grid_step}  subset={(2*self.R+1)}x{(2*self.R+1)}")

            prev_np = _load_gray(paths[0])

            for i, idx in enumerate(
                    range(self.frame_step, len(paths), self.frame_step)):

                curr_np = _load_gray(paths[idx])

                # Run DIC
                result = compute_dic_frame(prev_np, curr_np, pts, device, self.R)

                # Apply ZNCC threshold — mask failed points as NaN
                bad = result["zncc"] < ZNCC_THRESHOLD
                for key in result:
                    if key not in ("zncc", "converged", "iters"):
                        result[key][bad] = np.nan

                # Scatter point arrays onto 2D grids
                fields = {}
                for key in self.selected_dic_keys:
                    engine_key = key.replace("dic_", "")
                    if engine_key in result:
                        grid = scatter_to_grid(
                            result[engine_key], pts, self.bbox)
                        fields[key] = grid

                self.frame_ready.emit(idx, fields)
                self.progress.emit(i + 1, total)
                prev_np = curr_np   # incremental reference

            self.done.emit()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))