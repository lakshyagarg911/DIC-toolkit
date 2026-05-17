"""
rg_dic.py
---------
Reliability-Guided DIC (RG-DIC) engine.

Implements the algorithm from Blaber et al. (2015) §"Reliability Guided DIC"
and Fig. 4–5.  The computation proceeds from a seed point outward, using a
min-heap (priority queue) keyed by the ZNSSD correlation cost (CLS).  This
ensures that good (low-error) points are always processed before their
neighbours, so that the initial guess supplied to the IC-GN optimizer is as
accurate as possible.

Output arrays use NaN to indicate points that could not be analysed
(outside the ROI, boundary effects, or correlation cost above the cutoff).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .bspline import BSplineInterpolator, circular_subset, image_gradient
from .ncc import ncc_initial_guess
from .icgn import SubsetData, precompute_subset, run_icgn


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class DICParams:
    """All user-configurable parameters for a DIC analysis."""
    subset_radius: int   = 20       # pixels
    subset_spacing: int  = 5        # pixels (grid step)
    strain_window: int   = 10       # pixels (half-width for strain fitting)
    max_iter: int        = 50       # IC-GN max iterations per subset
    conv_tol: float      = 1e-4     # IC-GN convergence threshold
    corr_cutoff: float   = 0.8      # reject points with CLS > this value
    search_radius: int   = 30       # NCC search half-extent (pixels)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DICResult:
    """Full displacement field for one reference→current image pair."""
    # Displacement fields (NaN where not analysed)
    u:     np.ndarray   # x-displacement (pixels)
    v:     np.ndarray   # y-displacement (pixels)
    # Deformation gradient components
    du_dx: np.ndarray
    du_dy: np.ndarray
    dv_dx: np.ndarray
    dv_dy: np.ndarray
    # Correlation coefficient map (CLS; lower = better)
    corr:  np.ndarray
    # Mask of analysed points
    analyzed: np.ndarray   # bool
    # Grid of subset centres that were attempted
    grid_x: np.ndarray
    grid_y: np.ndarray


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def run_rg_dic(
    ref_image: np.ndarray,
    cur_image: np.ndarray,
    roi_mask: np.ndarray,
    params: DICParams,
    seed_xy: Optional[tuple[int, int]] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    cancel_flag: Optional[list[bool]] = None,
) -> DICResult:
    """
    Run Reliability-Guided DIC on a single reference → current image pair.

    Parameters
    ----------
    ref_image : (H, W) float64
        Reference (undeformed) greyscale image, normalised to [0, 1] or full
        dynamic range — both work.
    cur_image : (H, W) float64
        Deformed greyscale image.
    roi_mask : (H, W) bool
        True = include pixel in analysis.
    params : DICParams
        Algorithm parameters.
    seed_xy : (x, y) int tuple, optional
        Explicit seed point.  Defaults to the centroid of the ROI.
    progress_cb : callable(fraction: float, message: str), optional
        Called periodically with progress in [0, 1] and a status string.
    cancel_flag : list[bool], optional
        A one-element list; set cancel_flag[0] = True from another thread
        to request cancellation.

    Returns
    -------
    DICResult
    """
    H, W = ref_image.shape
    if cancel_flag is None:
        cancel_flag = [False]

    # ---- 1. Build grid of subset centres ---------------------------------
    grid_x, grid_y = _build_grid(roi_mask, params.subset_radius, params.subset_spacing)
    n_total = len(grid_x)

    if n_total == 0:
        raise ValueError("No valid subset centres within the ROI.  "
                         "Try reducing subset_radius or subset_spacing.")

    # ---- 2. Precompute interpolator and image gradients ------------------
    _report(progress_cb, 0.0, "Precomputing B-spline coefficients…")
    cur_interp = BSplineInterpolator(cur_image.astype(np.float64))
    grad_x, grad_y = image_gradient(ref_image)
    ref_f64 = ref_image.astype(np.float64)

    # Circular subset pixel offsets (shared by all subsets)
    dx_sub, dy_sub = circular_subset(params.subset_radius)

    # ---- 3. Allocate output arrays ---------------------------------------
    shape = (H, W)
    u_field     = np.full(shape, np.nan, dtype=np.float64)
    v_field     = np.full(shape, np.nan, dtype=np.float64)
    du_dx_f     = np.full(shape, np.nan, dtype=np.float64)
    du_dy_f     = np.full(shape, np.nan, dtype=np.float64)
    dv_dx_f     = np.full(shape, np.nan, dtype=np.float64)
    dv_dy_f     = np.full(shape, np.nan, dtype=np.float64)
    corr_field  = np.full(shape, np.nan, dtype=np.float64)
    analyzed    = np.zeros(shape, dtype=bool)

    # Map from (x, y) → grid index for quick lookup
    grid_set: dict[tuple[int, int], bool] = {
        (int(grid_x[i]), int(grid_y[i])): True for i in range(n_total)
    }
    # Lookup: which grid index corresponds to (x, y)?
    grid_map: dict[tuple[int, int], int] = {
        (int(grid_x[i]), int(grid_y[i])): i for i in range(n_total)
    }
    in_queue  = np.zeros(n_total, dtype=bool)   # currently in heap
    done      = np.zeros(n_total, dtype=bool)   # fully processed

    # ---- 4. Select seed point -------------------------------------------
    if seed_xy is None:
        ys_roi, xs_roi = np.where(roi_mask)
        seed_xy = (int(xs_roi.mean()), int(ys_roi.mean()))

    seed_x, seed_y = _snap_to_grid(seed_xy[0], seed_xy[1], grid_x, grid_y)
    seed_idx = grid_map.get((seed_x, seed_y), 0)

    # ---- 5. Analyse seed with NCC + IC-GN --------------------------------
    _report(progress_cb, 0.01, "Analysing seed point…")

    u0, v0, _ncc_score = ncc_initial_guess(
        ref_f64, cur_image, seed_x, seed_y,
        params.subset_radius, params.search_radius,
    )
    p_seed = np.array([u0, v0, 0.0, 0.0, 0.0, 0.0])

    seed_subset = precompute_subset(ref_f64, grad_x, grad_y,
                                    seed_x, seed_y, dx_sub, dy_sub)
    p_seed, cls_seed, _ = run_icgn(cur_interp, seed_subset, p_seed,
                                    params.max_iter, params.conv_tol)

    _store(u_field, v_field, du_dx_f, du_dy_f, dv_dx_f, dv_dy_f,
           corr_field, analyzed, seed_x, seed_y, p_seed, cls_seed)
    done[seed_idx] = True

    # ---- 6. Priority queue (min-heap keyed by CLS) ----------------------
    # Heap elements: (CLS, grid_index, p)
    heap: list[tuple[float, int, np.ndarray]] = []
    heapq.heappush(heap, (cls_seed, seed_idx, p_seed.copy()))
    in_queue[seed_idx] = True

    n_done = 1
    step = params.subset_spacing

    # ---- 7. RG-DIC main loop --------------------------------------------
    while heap and not cancel_flag[0]:
        cls_parent, parent_idx, p_parent = heapq.heappop(heap)

        px = int(grid_x[parent_idx])
        py = int(grid_y[parent_idx])

        # 4-connected neighbours on the subset grid
        neighbours = [
            (px + step, py),
            (px - step, py),
            (px, py + step),
            (px, py - step),
        ]

        for nx, ny in neighbours:
            key = (nx, ny)
            if key not in grid_map:
                continue
            nb_idx = grid_map[key]
            if done[nb_idx]:
                continue

            # Precompute subset for this neighbour
            nb_subset = precompute_subset(ref_f64, grad_x, grad_y,
                                          nx, ny, dx_sub, dy_sub)

            # Use parent deformation as initial guess
            p_init = p_parent.copy()

            p_opt, cls_opt, _ = run_icgn(cur_interp, nb_subset, p_init,
                                          params.max_iter, params.conv_tol)

            done[nb_idx] = True
            n_done += 1

            if cls_opt <= params.corr_cutoff:
                _store(u_field, v_field, du_dx_f, du_dy_f, dv_dx_f, dv_dy_f,
                       corr_field, analyzed, nx, ny, p_opt, cls_opt)

                if not in_queue[nb_idx]:
                    heapq.heappush(heap, (cls_opt, nb_idx, p_opt.copy()))
                    in_queue[nb_idx] = True

        # Progress report every ~1% of total points
        if n_done % max(1, n_total // 100) == 0:
            frac = min(0.05 + 0.93 * n_done / n_total, 0.98)
            _report(progress_cb, frac,
                    f"Analysing subsets… {n_done}/{n_total}")

    _report(progress_cb, 1.0, "Done.")

    return DICResult(
        u=u_field, v=v_field,
        du_dx=du_dx_f, du_dy=du_dy_f,
        dv_dx=dv_dx_f, dv_dy=dv_dy_f,
        corr=corr_field, analyzed=analyzed,
        grid_x=grid_x, grid_y=grid_y,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_grid(
    roi_mask: np.ndarray,
    subset_radius: int,
    spacing: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a regular grid of subset centre candidates that lie within the
    ROI and are at least `subset_radius` pixels away from any edge.
    """
    H, W = roi_mask.shape
    r = subset_radius

    # Erode ROI by subset_radius to ensure no subset extends outside image
    ys = np.arange(r, H - r, spacing, dtype=int)
    xs = np.arange(r, W - r, spacing, dtype=int)
    gx, gy = np.meshgrid(xs, ys)
    gx = gx.ravel()
    gy = gy.ravel()

    # Keep only centres where the entire circular subset is within the ROI
    # Quick approximation: check the centre pixel only
    in_roi = roi_mask[gy, gx]
    return gx[in_roi], gy[in_roi]


def _snap_to_grid(
    x: int, y: int,
    grid_x: np.ndarray, grid_y: np.ndarray,
) -> tuple[int, int]:
    """Find the grid point closest to (x, y)."""
    dist2 = (grid_x - x) ** 2 + (grid_y - y) ** 2
    idx = int(np.argmin(dist2))
    return int(grid_x[idx]), int(grid_y[idx])


def _store(
    u_f, v_f, du_dx_f, du_dy_f, dv_dx_f, dv_dy_f,
    corr_f, analyzed,
    cx: int, cy: int,
    p: np.ndarray, cls: float,
) -> None:
    """Write results for subset centre (cx, cy) into the output arrays."""
    u_f[cy, cx]     = p[0]
    v_f[cy, cx]     = p[1]
    du_dx_f[cy, cx] = p[2]
    du_dy_f[cy, cx] = p[3]
    dv_dx_f[cy, cx] = p[4]
    dv_dy_f[cy, cx] = p[5]
    corr_f[cy, cx]  = cls
    analyzed[cy, cx] = True


def _report(
    cb: Optional[Callable[[float, str], None]],
    frac: float, msg: str,
) -> None:
    if cb is not None:
        try:
            cb(frac, msg)
        except Exception:
            pass
