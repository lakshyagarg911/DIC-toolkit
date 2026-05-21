"""
strain.py
---------
Green-Lagrangian strain computation from DIC displacement fields.

Implements the strain-window least-squares plane-fit algorithm described in
Blaber et al. (2015) §"Computation of Strains" (eqs. 13–18) and the
deviatoric effective strain.

The displacement gradients (∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y) are obtained by
fitting the plane

    u(x, y) = a₀ + a₁·x + a₂·y
    v(x, y) = b₀ + b₁·x + b₂·y

over a window of radius `strain_window` centred at each analysed point.
The gradient components are a₁ = ∂u/∂x, a₂ = ∂u/∂y, b₁ = ∂v/∂x,
b₂ = ∂v/∂y.

For efficiency, the numerator sums (e.g., Σ u·Δx) are computed as 2-D
convolutions using scipy.ndimage.convolve.  This avoids a Python loop over
all analysed points and scales as O(N²·W²) → O(N²·log N) where N is the
image dimension and W is the window size.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve


# ---------------------------------------------------------------------------
# Main strain computation
# ---------------------------------------------------------------------------

def compute_strains(
    u: np.ndarray,
    v: np.ndarray,
    valid_mask: np.ndarray,
    strain_window: int,
) -> dict[str, np.ndarray]:
    """
    Compute Green-Lagrangian strains and effective strain from displacement
    fields.

    Parameters
    ----------
    u, v : (H, W) float arrays
        x- and y-displacement fields (NaN where not analysed).
    valid_mask : (H, W) bool array
        True at pixels where u, v are valid (non-NaN analysed points).
    strain_window : int
        Half-width of the square strain window (pixels).  The actual window
        size is (2·strain_window + 1)² pixels.

    Returns
    -------
    dict with keys:
        'Exx', 'Exy', 'Eyy'  — Green-Lagrangian strain components
        'Eeff'               — von-Mises effective strain
        'du_dx', 'du_dy', 'dv_dx', 'dv_dy'  — displacement gradient fields
    """
    r = int(strain_window)

    # Build coordinate kernels for the strain window
    y_kern, x_kern = np.mgrid[-r: r + 1, -r: r + 1]   # (size × size) grids
    x_kern = x_kern.astype(np.float64)
    y_kern = y_kern.astype(np.float64)
    x2_kern = x_kern ** 2
    y2_kern = y_kern ** 2
    ones_kern = np.ones_like(x_kern)

    # ---- Replace NaN with 0 for convolution --------------------------------
    valid = valid_mask & ~np.isnan(u) & ~np.isnan(v)
    u_z = np.where(valid, u, 0.0)
    v_z = np.where(valid, v, 0.0)
    count = valid.astype(np.float64)

    conv_kw = dict(mode="constant", cval=0.0)

    # Numerators: Σ u·Δx, Σ u·Δy, etc.
    sum_u_x = convolve(u_z, x_kern, **conv_kw)
    sum_u_y = convolve(u_z, y_kern, **conv_kw)
    sum_v_x = convolve(v_z, x_kern, **conv_kw)
    sum_v_y = convolve(v_z, y_kern, **conv_kw)

    # Effective denominators: Σ Δx² and Σ Δy² over *valid* pixels in window
    eff_x2 = convolve(count, x2_kern, **conv_kw)
    eff_y2 = convolve(count, y2_kern, **conv_kw)

    # Number of valid pixels per window (for minimum data requirement)
    n_pts = convolve(count, ones_kern, **conv_kw)
    # Need ≥6 points to robustly fit the 3-param plane (not a dense-pixel formula)
    min_pts = 6

    # Denominator safe-guards
    enough = (n_pts >= min_pts) & (eff_x2 > 1e-12) & (eff_y2 > 1e-12)

    safe_x2 = np.where(eff_x2 > 1e-12, eff_x2, 1.0)
    safe_y2 = np.where(eff_y2 > 1e-12, eff_y2, 1.0)

    # Displacement gradient fields
    du_dx = np.where(enough, sum_u_x / safe_x2, np.nan)
    du_dy = np.where(enough, sum_u_y / safe_y2, np.nan)
    dv_dx = np.where(enough, sum_v_x / safe_x2, np.nan)
    dv_dy = np.where(enough, sum_v_y / safe_y2, np.nan)

    # Mask to valid region
    du_dx[~valid_mask] = np.nan
    du_dy[~valid_mask] = np.nan
    dv_dx[~valid_mask] = np.nan
    dv_dy[~valid_mask] = np.nan

    # ---- Green-Lagrangian strains ----------------------------------------
    # Exx = ∂u/∂x + ½[(∂u/∂x)² + (∂v/∂x)²]   (eq. 13)
    # Eyy = ∂v/∂y + ½[(∂u/∂y)² + (∂v/∂y)²]   (eq. 15)
    # Exy = ½[∂u/∂y + ∂v/∂x + ∂u/∂x·∂u/∂y + ∂v/∂x·∂v/∂y]  (eq. 14)
    Exx = du_dx + 0.5 * (du_dx ** 2 + dv_dx ** 2)
    Eyy = dv_dy + 0.5 * (du_dy ** 2 + dv_dy ** 2)
    Exy = 0.5 * (du_dy + dv_dx + du_dx * du_dy + dv_dx * dv_dy)

    # ---- Effective (von-Mises) strain ------------------------------------
    # For general 2D (plane stress/strain, unknown Ezz):
    #   Eeff² = (2/3)[Exx² + Eyy² + 2·Exy²  − Exx·Eyy]
    # For incompressible materials (Ezz = −Exx − Eyy):
    #   Eeff = sqrt(2/3 · eᵢⱼeᵢⱼ) with deviatoric components:
    #   exx = (2Exx − Eyy) / 3,  eyy = (2Eyy − Exx)/3,  exy = Exy
    # We use the incompressible form (standard in ductile metals):
    with np.errstate(invalid="ignore"):
        Eeff = np.sqrt(
            np.maximum(
                (2.0 / 3.0) * (Exx ** 2 + Eyy ** 2 + 2.0 * Exy ** 2
                                - Exx * Eyy),
                0.0,
            )
        )

    return {
        "Exx":   Exx,
        "Exy":   Exy,
        "Eyy":   Eyy,
        "Eeff":  Eeff,
        "du_dx": du_dx,
        "du_dy": du_dy,
        "dv_dx": dv_dx,
        "dv_dy": dv_dy,
    }


# ---------------------------------------------------------------------------
# Utility: interpolate sparse grid to full-resolution field
# ---------------------------------------------------------------------------

def interpolate_to_full(
    sparse: np.ndarray,
    valid: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """
    Fill NaN gaps in a sparse displacement/strain field using 2-D scattered
    interpolation.

    Parameters
    ----------
    sparse : (H, W) float array with NaNs
    valid : (H, W) bool
    method : 'linear' | 'nearest'

    Returns
    -------
    filled : (H, W) float array (NaN where extrapolation is unreliable)
    """
    from scipy.interpolate import griddata

    H, W = sparse.shape
    ys, xs = np.where(valid & ~np.isnan(sparse))
    if len(xs) < 4:
        return sparse.copy()

    values = sparse[ys, xs]
    yi, xi = np.mgrid[0:H, 0:W]

    filled = griddata(
        (xs, ys), values,
        (xi, yi),
        method=method,
        fill_value=np.nan,
    )
    return filled
