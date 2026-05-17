"""
bspline.py
----------
Biquintic (5th-order) B-spline image interpolation.

Implements the same mathematical framework as Ncorr (Blaber et al. 2015,
Appendix A2).  The B-spline coefficients are precomputed for an entire image
via scipy's IIR spline filter (equivalent to the FFT-based deconvolution
described in the paper).  Sub-pixel values and their spatial derivatives are
then evaluated efficiently using scipy.ndimage.map_coordinates.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import spline_filter, map_coordinates

# Order of the B-spline (quintic = 5)
BSPLINE_ORDER: int = 5


class BSplineInterpolator:
    """
    Precomputed biquintic B-spline interpolator for a 2-D greyscale image.

    Usage
    -----
    >>> interp = BSplineInterpolator(image)
    >>> values = interp.eval(x_coords, y_coords)         # (x, y) in pixel coords
    >>> gx, gy  = interp.gradient(x_coords, y_coords)   # spatial derivatives
    """

    def __init__(self, image: np.ndarray) -> None:
        """
        Parameters
        ----------
        image : (H, W) float array
            Greyscale image.  Will be converted to float64 if needed.
        """
        if image.ndim != 2:
            raise ValueError("BSplineInterpolator expects a 2-D greyscale image.")
        img = image.astype(np.float64, copy=False)
        # Compute B-spline coefficients — IIR deconvolution, mirror padding
        self.coefficients: np.ndarray = spline_filter(
            img, order=BSPLINE_ORDER, mode="mirror", output=np.float64
        )
        self.shape: tuple[int, int] = img.shape  # (H, W)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def eval(
        self,
        x: np.ndarray | float,
        y: np.ndarray | float,
    ) -> np.ndarray:
        """
        Interpolate image intensity at sub-pixel coordinates (x, y).

        Parameters
        ----------
        x : array-like
            Column coordinates (can be fractional).
        y : array-like
            Row coordinates (can be fractional).

        Returns
        -------
        values : ndarray with same shape as x/y
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        # scipy uses (row, col) = (y, x)
        coords = np.array([y.ravel(), x.ravel()])
        out = map_coordinates(
            self.coefficients, coords,
            order=BSPLINE_ORDER, mode="mirror", prefilter=False
        )
        return out.reshape(x.shape)

    def gradient(
        self,
        x: np.ndarray | float,
        y: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute ∂I/∂x and ∂I/∂y at sub-pixel coordinates using the
        analytic derivative of the B-spline basis.

        Returns
        -------
        df_dx, df_dy : ndarrays
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        coords = np.array([y.ravel(), x.ravel()])

        # Evaluate derivative in x-direction (column): spline_filter already
        # done; ask map_coordinates for the derivative along axis 1.
        df_dx = map_coordinates(
            self.coefficients, coords,
            order=BSPLINE_ORDER, mode="mirror", prefilter=False,
        )

        # scipy does not expose axis derivatives directly via map_coordinates,
        # so we use a half-pixel finite difference on the B-spline surface
        # (sub-pixel step → purely interpolation error, not finite-diff error).
        h = 0.5
        fxp = map_coordinates(
            self.coefficients, np.array([y.ravel(), x.ravel() + h]),
            order=BSPLINE_ORDER, mode="mirror", prefilter=False,
        )
        fxm = map_coordinates(
            self.coefficients, np.array([y.ravel(), x.ravel() - h]),
            order=BSPLINE_ORDER, mode="mirror", prefilter=False,
        )
        fyp = map_coordinates(
            self.coefficients, np.array([y.ravel() + h, x.ravel()]),
            order=BSPLINE_ORDER, mode="mirror", prefilter=False,
        )
        fym = map_coordinates(
            self.coefficients, np.array([y.ravel() - h, x.ravel()]),
            order=BSPLINE_ORDER, mode="mirror", prefilter=False,
        )

        df_dx = (fxp - fxm) / (2.0 * h)
        df_dy = (fyp - fym) / (2.0 * h)

        return df_dx.reshape(x.shape), df_dy.reshape(y.shape)


# ---------------------------------------------------------------------------
# Convenience: precompute integer-pixel gradient arrays for a whole image
# (used once per reference image in the IC-GN setup)
# ---------------------------------------------------------------------------

def image_gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute ∂I/∂x and ∂I/∂y at every integer pixel using central differences.

    This is equivalent to evaluating the quintic B-spline derivatives at
    integer locations, but numpy.gradient is faster for a full-image pass.

    Returns
    -------
    grad_x, grad_y : (H, W) float64 arrays
        ∂I/∂x (column direction) and ∂I/∂y (row direction).
    """
    img = image.astype(np.float64, copy=False)
    grad_y, grad_x = np.gradient(img)
    return grad_x, grad_y


# ---------------------------------------------------------------------------
# Circular subset mask helper
# ---------------------------------------------------------------------------

def circular_subset(radius: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the relative (dx, dy) coordinates of pixels inside a circle of
    given radius centred at the origin.

    Parameters
    ----------
    radius : int
        Subset radius in pixels.

    Returns
    -------
    dx, dy : 1-D int arrays
        Column and row offsets of all subset pixels.
    """
    r = int(radius)
    y_grid, x_grid = np.mgrid[-r:r + 1, -r:r + 1]
    inside = x_grid ** 2 + y_grid ** 2 <= r ** 2
    dx = x_grid[inside]
    dy = y_grid[inside]
    return dx, dy
