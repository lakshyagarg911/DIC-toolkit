# dicUtils/dic_engine.py
#
# Subset-based DIC engine — matches ncorr algorithm exactly:
#
#   Interpolation : Biquintic B-spline (order 5) via FFT deconvolution
#                   (ncorr sec 3.6-3.7, Thevenaz et al.)
#   Optimizer     : Inverse Compositional Gauss-Newton (IC-GN)
#                   Hessian computed ONCE from reference — never re-evaluated
#                   (ncorr sec 3.4-3.5, Baker & Matthews 2004)
#   Initial guess : NCC search window (ncorr sec 3.1)
#   Strains       : Least-squares plane fit (Savitzky-Golay differentiator)
#                   on u/v displacement fields — ncorr sec 5, eq.49-54
#   Grid          : Regular grid with configurable spacing (ncorr "spacing")
#
# Why IC-GN is fast: the 6x6 Gauss-Newton Hessian H = Σ(∇f·∂W/∂p)ᵀ(∇f·∂W/∂p)
# depends only on the REFERENCE image and subset shape — it is constant across
# all iterations. So it is computed once in precompute_icgn(), Cholesky-
# decomposed once, and only the gradient ∇CLS changes each iteration.
# Each iteration: warp+sample current image → compute residual → solve 6×6 →
# compositional update. This is the exact same amount of work ncorr does.

import torch
import torch.nn.functional as F
import numpy as np
import math

# ── Constants ──────────────────────────────────────────────────────────────────

SUBSET_RADIUS  = 21      # R  — subset = (2R+1)² pixels  (ncorr default 20-30)
IC_MAX_ITER    = 50      # IC-GN max iterations
IC_CONVERGENCE = 1e-3    # ||Δp|| convergence threshold (ncorr uses ~1e-3)
SEARCH_RADIUS  = 10      # NCC search half-window
ZNCC_THRESHOLD = -1     # minimum acceptable correlation
BATCH_POINTS   = 256     # points per GPU batch
GRID_STEP      = 3       # subset spacing in pixels (ncorr "spacing" param)
STRAIN_WINDOW  = 15      # half-window for strain plane fit (ncorr default)


# ── Device selection ───────────────────────────────────────────────────────────

def get_device():
    # dev = torch.device("cpu")
    # return dev
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[DIC] GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[DIC] Running on CPU")
    return dev


# ── Quintic B-spline kernel (ncorr eq.39, fig.12) ─────────────────────────────

def _b5(t: torch.Tensor) -> torch.Tensor:
    """
    Quintic B-spline basis β⁵(t) — piecewise polynomial (Unser 1999 Table 1).
    t: any shape tensor. Returns same shape.
    """
    a = t.abs()
    w = torch.zeros_like(a)

    m1 = a < 1.0
    a1 = a[m1]
    w[m1] = (11.0/20.0 - a1**2 * (0.5 - a1**2/4.0)
             - a1**3/12.0 * (a1**2 - 5.0))

    m2 = (a >= 1.0) & (a < 2.0)
    a2 = a[m2]
    w[m2] = (17.0/40.0 + a2 * (5.0/8.0 + a2 * (
             -7.0/4.0 + a2 * (5.0/4.0 + a2 * (-3.0/8.0 + a2/24.0)))))

    m3 = (a >= 2.0) & (a < 3.0)
    a3 = a[m3]
    w[m3] = (3.0 - a3)**5 / 120.0

    return w


def _db5(t: torch.Tensor) -> torch.Tensor:
    """
    Derivative dβ⁵/dt of quintic B-spline (ncorr eq.47-48).
    """
    a   = t.abs()
    sgn = t.sign()
    dw  = torch.zeros_like(t)

    m1 = a < 1.0
    a1 = a[m1]
    dw[m1] = sgn[m1] * (-a1 + a1**3 - 5.0*a1**2/12.0 * a1 + 5.0*a1**4/12.0 -
                         a1 * (1.0 - a1**2/2.0))
    # simpler exact form:
    dw[m1] = sgn[m1] * a1 * (-1.0 + a1**2 * (1.0 - a1/4.0) * 5.0/6.0 - a1/6.0)

    m2 = (a >= 1.0) & (a < 2.0)
    a2 = a[m2]
    dw[m2] = sgn[m2] * (5.0/8.0 + a2 * (-7.0/2.0 + a2 * (
             15.0/4.0 + a2 * (-3.0/2.0 + a2/4.8))))

    m3 = (a >= 2.0) & (a < 3.0)
    a3 = a[m3]
    dw[m3] = -sgn[m3] * (3.0 - a3)**4 / 24.0

    return dw


# ── Biquintic B-spline coefficients via FFT deconvolution (ncorr sec 3.6-3.7) ─

def bspline5_coeffs(img: torch.Tensor, pad: int = 30) -> torch.Tensor:
    """
    Compute biquintic B-spline coefficients.

    Exact ncorr implementation:
      1. Pad with border replication
      2. FFT-deconvolve each row with quintic kernel
      3. FFT-deconvolve each column with quintic kernel

    Uses float64 for numerical stability (matching ncorr's MATLAB double).
    """
    device = img.device
    H, W   = img.shape
    img64  = img.double()

    # Pad
    padded = F.pad(img64.unsqueeze(0).unsqueeze(0),
                   (pad, pad, pad, pad), mode="replicate"
                   ).squeeze(0).squeeze(0)
    Ph, Pw = padded.shape

    # Quintic kernel evaluated at integer offsets, zero-padded to image size
    # Sampled at {-2,-1,0,1,2} (β⁵(3)=0 so only 5 nonzero taps)
    def _make_kernel_1d(L: int) -> torch.Tensor:
        k = torch.zeros(L, device=device, dtype=torch.float64)
        offsets = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float64, device=device)
        vals    = _b5(offsets.float()).double()
        k[0]    = vals[2]           # β⁵(0)
        k[1]    = vals[3]           # β⁵(1)
        k[2]    = vals[4]           # β⁵(2)
        k[-1]   = vals[1]           # β⁵(-1)  circular
        k[-2]   = vals[0]           # β⁵(-2)
        return k

    # Row deconvolution
    kern_r = _make_kernel_1d(Pw)
    Kr     = torch.fft.rfft(kern_r.unsqueeze(0), dim=1)          # (1, Pw//2+1)
    Gr     = torch.fft.rfft(padded, dim=1)                       # (Ph, Pw//2+1)
    C_rows = torch.fft.irfft(Gr / Kr, n=Pw, dim=1)              # (Ph, Pw)

    # Column deconvolution
    kern_c = _make_kernel_1d(Ph)
    Kc     = torch.fft.rfft(kern_c.unsqueeze(1), dim=0)          # (Ph//2+1, 1)
    Gc     = torch.fft.rfft(C_rows, dim=0)                       # (Ph//2+1, Pw)
    C_full = torch.fft.irfft(Gc / Kc, n=Ph, dim=0)              # (Ph, Pw)

    return C_full[pad:pad+H, pad:pad+W].float()


# ── Biquintic evaluation: value + gradient at subpixel locations ───────────────

def _bspline5_eval(coeff: torch.Tensor,
                   xs: torch.Tensor,
                   ys: torch.Tensor,
                   want_grad: bool = False):
    """
    Evaluate biquintic B-spline and optionally its x/y gradients.
    Matches ncorr eq.44 (value) and eq.47-48 (gradients).

    coeff      : (H, W) float32
    xs, ys     : (...) float32 coordinates
    want_grad  : if True, also return (df/dx, df/dy) of same shape

    Returns:
        val        : (...) float32
        (gx, gy)   : (...) float32 each — only if want_grad=True
    """
    H, W   = coeff.shape
    device = coeff.device
    shape  = xs.shape
    N      = xs.numel()

    xf = xs.reshape(-1).clamp(2.0, W - 3.0)
    yf = ys.reshape(-1).clamp(2.0, H - 3.0)

    xi = xf.long()
    yi = yf.long()
    dx = xf - xi.float()   # Δx ∈ [0,1)
    dy = yf - yi.float()   # Δy ∈ [0,1)

    # Tap offsets -2,-1,0,1,2,3
    off = torch.arange(-2, 4, device=device, dtype=torch.float32)  # (6,)

    # Weights for each tap: β⁵(Δ - offset) for Δ ∈ [0,1) and offsets -2..3
    # shape: (N, 6)
    dx_off = dx.unsqueeze(1) - off.unsqueeze(0)   # (N, 6): Δx - k for k=-2..3
    dy_off = dy.unsqueeze(1) - off.unsqueeze(0)   # (N, 6)

    wx  = _b5(dx_off)    # (N, 6)
    wy  = _b5(dy_off)    # (N, 6)
    if want_grad:
        dwx = _db5(dx_off)   # (N, 6)
        dwy = _db5(dy_off)   # (N, 6)

    # Integer tap indices (N, 6)
    ix_taps = (xi.unsqueeze(1) + off.long().unsqueeze(0)).clamp(0, W - 1)
    iy_taps = (yi.unsqueeze(1) + off.long().unsqueeze(0)).clamp(0, H - 1)

    coeff_flat = coeff.reshape(-1)

    # Gather all 6×6=36 coefficient values: (N, 6, 6)
    # coeff_vals[n, ky, kx] = coeff[iy_taps[n,ky], ix_taps[n,kx]]
    iy_exp = iy_taps.unsqueeze(2).expand(N, 6, 6)   # (N, 6, 6)
    ix_exp = ix_taps.unsqueeze(1).expand(N, 6, 6)   # (N, 6, 6)
    c_vals = coeff_flat[(iy_exp * W + ix_exp).reshape(-1)].reshape(N, 6, 6)

    # Outer product of weights: (N,6,1) * (N,1,6) → (N,6,6)
    W_2d = wy.unsqueeze(2) * wx.unsqueeze(1)   # (N, 6, 6)
    val  = (W_2d * c_vals).sum(dim=(1, 2))      # (N,)

    if want_grad:
        Wx_2d = wy.unsqueeze(2) * dwx.unsqueeze(1)   # (N,6,6) — df/dx weights
        Wy_2d = dwy.unsqueeze(2) * wx.unsqueeze(1)   # (N,6,6) — df/dy weights
        gx_out = (Wx_2d * c_vals).sum(dim=(1, 2))
        gy_out = (Wy_2d * c_vals).sum(dim=(1, 2))
        return val.reshape(shape), gx_out.reshape(shape), gy_out.reshape(shape)

    return val.reshape(shape)


# ── IC-GN precomputation: steepest descent + Hessian (ncorr steps 2-4) ────────

def precompute_icgn(ref_img: torch.Tensor,
                    ref_coeff: torch.Tensor,
                    points: torch.Tensor,
                    R: int,
                    device: torch.device):
    """
    Precompute all quantities that depend only on the reference image.
    Called ONCE per frame pair. This is ncorr's steps 2-4.

    Returns:
        f_norm  : (N, P)    zero-mean normalised reference subsets
        f_std   : (N,)      reference subset std
        sd      : (N, P, 6) steepest descent images
        H_chol  : (N, 6, 6) Cholesky factor of precomputed Hessian
        valid   : (N,) bool
        dx_g, dy_g : (P,) subset offset grids
    """
    Hi, Wi = ref_img.shape
    P = (2 * R + 1) ** 2
    N = points.shape[0]

    r = torch.arange(-R, R + 1, device=device, dtype=torch.float32)
    dy_grid, dx_grid = torch.meshgrid(r, r, indexing="ij")
    dx_g = dx_grid.reshape(-1)   # (P,)
    dy_g = dy_grid.reshape(-1)

    px = points[:, 0]
    py = points[:, 1]

    # Points need 3px margin for quintic taps (-2..3)
    margin = float(R + 3)
    valid  = (px >= margin) & (px < Wi - margin) & \
             (py >= margin) & (py < Hi - margin)

    # Reference pixel values at integer locations (direct lookup, no interpolation)
    xi = (px.unsqueeze(1) + dx_g.unsqueeze(0)).long().clamp(0, Wi - 1)  # (N,P)
    yi = (py.unsqueeze(1) + dy_g.unsqueeze(0)).long().clamp(0, Hi - 1)
    f_vals = ref_img.reshape(-1)[yi * Wi + xi]                           # (N,P)

    f_mean = f_vals.mean(dim=1, keepdim=True)
    f_raw  = f_vals - f_mean
    f_std  = f_raw.norm(dim=1).clamp(min=1e-12)
    f_norm = f_raw / f_std.unsqueeze(1)

    # Gradients of reference image via biquintic derivative (eq.47-48)
    # At integer positions, evaluate the B-spline derivative kernel
    # Process in batches to avoid building (N*P, 6, 6) at once
    NP = N * P
    xi_f = xi.reshape(NP).float()
    yi_f = yi.reshape(NP).float()

    _, gx_flat, gy_flat = _bspline5_eval(ref_coeff, xi_f, yi_f, want_grad=True)
    gx = gx_flat.reshape(N, P)
    gy = gy_flat.reshape(N, P)

    # Steepest descent images (eq.31-36):
    # sd[b,p,k] = gx[b,p]*dWx_k + gy[b,p]*dWy_k
    # dWx = [1,0,dx,dy,0,0], dWy = [0,1,0,0,dx,dy]
    sd = torch.stack([
        gx,
        gy,
        gx * dx_g.unsqueeze(0),
        gx * dy_g.unsqueeze(0),
        gy * dx_g.unsqueeze(0),
        gy * dy_g.unsqueeze(0),
    ], dim=2)  # (N, P, 6)

    # Gauss-Newton Hessian (eq.26): H = Σ_p sd_p^T sd_p / (P * f_std²)
    H_mat = torch.bmm(sd.transpose(1, 2), sd)
    H_mat = H_mat / (P * f_std ** 2).clamp(min=1e-12).unsqueeze(-1).unsqueeze(-1)

    # Cholesky decompose once — reused every iteration
    try:
        H_chol = torch.linalg.cholesky(H_mat)
    except Exception:
        # Regularise and retry
        H_mat  = H_mat + 1e-8 * torch.eye(6, device=device).unsqueeze(0)
        H_chol = torch.linalg.cholesky(H_mat)

    return f_norm, f_std, sd, H_chol, valid, dx_g, dy_g


# ── NCC initial guess (ncorr sec 3.1) ──────────────────────────────────────────

# def ncc_initial_guess(ref_img: torch.Tensor,
#                       cur_img: torch.Tensor,
#                       points: torch.Tensor,
#                       R: int,
#                       search_r: int,
#                       device: torch.device,
#                       batch_size: int = 256) -> tuple:
#     """
#     Integer-pixel initial guess via normalised cross-correlation (ncorr sec 3.1).
#     Searches a (2*search_r+1)² window for each subset. Adaptive OOM protection.
#     """
#     H, W = cur_img.shape
#     N    = points.shape[0]
#     P    = (2 * R + 1) ** 2
#
#     r = torch.arange(-R, R + 1, device=device, dtype=torch.float32)
#     dy_s, dx_s = torch.meshgrid(r, r, indexing="ij")
#     dx_s = dx_s.reshape(-1); dy_s = dy_s.reshape(-1)
#
#     sr = torch.arange(-search_r, search_r + 1, device=device, dtype=torch.float32)
#     sdy, sdx = torch.meshgrid(sr, sr, indexing="ij")
#     sdx = sdx.reshape(-1); sdy = sdy.reshape(-1)
#     M   = sdx.shape[0]
#
#     u0 = torch.zeros(N, device=device)
#     v0 = torch.zeros(N, device=device)
#
#     img4 = cur_img.unsqueeze(0).unsqueeze(0)
#
#     if device.type == "cuda":
#         free, _ = torch.cuda.mem_get_info(device)
#         m_chunk = max(1, min(M, int(free * 0.45 / 4) // (batch_size * P * 2)))
#     else:
#         m_chunk = M
#
#     for b0 in range(0, N, batch_size):
#         b1 = min(b0 + batch_size, N)
#         nb = b1 - b0
#
#         px = points[b0:b1, 0]; py = points[b0:b1, 1]
#
#         xi = (px.unsqueeze(1) + dx_s.unsqueeze(0)).long().clamp(0, W - 1)
#         yi = (py.unsqueeze(1) + dy_s.unsqueeze(0)).long().clamp(0, H - 1)
#         f  = ref_img.reshape(-1)[yi * W + xi].float()
#         fm = f.mean(1, keepdim=True); fn = f - fm
#         fs = fn.norm(dim=1, keepdim=True).clamp(min=1e-12)  # ← FIX: specify dim=1
#         fn = fn / fs  # ← Now this works: (nb, P) / (nb, 1) → (nb, P)
#
#         best = torch.full((nb,), -2.0, device=device)
#         bu   = torch.zeros(nb, device=device)
#         bv   = torch.zeros(nb, device=device)
#
#         mc = m_chunk; ms = 0
#         while ms < M:
#             me  = min(ms + mc, M)
#             nc  = me - ms
#             sx  = sdx[ms:me]; sy = sdy[ms:me]
#
#             try:
#                 cx = (px.unsqueeze(0).unsqueeze(2) +
#                       sx.unsqueeze(1).unsqueeze(2) +
#                       dx_s.unsqueeze(0).unsqueeze(0))   # (nc,nb,P)
#                 cy = (py.unsqueeze(0).unsqueeze(2) +
#                       sy.unsqueeze(1).unsqueeze(2) +
#                       dy_s.unsqueeze(0).unsqueeze(0))
#
#                 grid = (torch.stack([cx/(W-1)*2-1, cy/(H-1)*2-1], dim=-1)
#                         .permute(1,0,2,3).reshape(1, nb*nc, P, 2))
#                 samp = (F.grid_sample(img4, grid, mode="bilinear",
#                                       padding_mode="border",
#                                       align_corners=True)
#                         .squeeze(0).squeeze(0).reshape(nb, nc, P))
#
#                 gm  = samp.mean(2, keepdim=True); gn = samp - gm
#                 gs  = gn.norm(dim=2, keepdim=True).clamp(min=1e-12)  # ← FIX: specify dim=2
#                 zncc = (fn.unsqueeze(1) * gn).sum(2) / (
#                     fs * gs.squeeze(2) + 1e-12)   # ← FIX: squeeze gs dimension
#
#                 bi  = zncc.argmax(1)
#                 imp = zncc[torch.arange(nb, device=device), bi] > best
#                 best = torch.where(imp, zncc[torch.arange(nb, device=device), bi], best)
#                 bu   = torch.where(imp, sx[bi], bu)
#                 bv   = torch.where(imp, sy[bi], bv)
#                 ms  += nc
#
#             except torch.cuda.OutOfMemoryError:
#                 torch.cuda.empty_cache()
#                 mc = max(1, mc // 2)
#                 print(f"[DIC] OOM in NCC — chunk → {mc}")
#
#         u0[b0:b1] = bu
#         v0[b0:b1] = bv
#
#     return u0, v0


def ncc_initial_guess(ref_img: torch.Tensor,
                      cur_img: torch.Tensor,
                      points: torch.Tensor,
                      R: int,
                      search_r: int,
                      device: torch.device,
                      batch_size: int = 1024) -> tuple:
    """
    Optimized Coarse-to-Fine NCC Search.
    Fixes the 'int object has no attribute unsqueeze' error by ensuring
    base displacement is always a tensor.
    """
    H, W = cur_img.shape
    N = points.shape[0]

    def _do_ncc_search(ref_t, cur_t, pts, radius, s_radius, current_u=None, current_v=None):
        h, w = cur_t.shape
        num_pts = pts.shape[0]
        p_area = (2 * radius + 1) ** 2

        # Define search offsets
        sr_range = torch.arange(-s_radius, s_radius + 1, device=device, dtype=torch.float32)
        sy, sx = torch.meshgrid(sr_range, sr_range, indexing="ij")
        sx, sy = sx.reshape(-1), sy.reshape(-1)
        num_searches = sx.shape[0]

        # Subset grid (relative to center)
        r_range = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
        ry, rx = torch.meshgrid(r_range, r_range, indexing="ij")
        rx, ry = rx.reshape(-1), ry.reshape(-1)

        final_u = torch.zeros(num_pts, device=device)
        final_v = torch.zeros(num_pts, device=device)

        # Optimization: Move image to 4D once
        img4d = cur_t.view(1, 1, h, w)

        for b0 in range(0, num_pts, batch_size):
            b1 = min(b0 + batch_size, num_pts)
            nb = b1 - b0

            px, py = pts[b0:b1, 0], pts[b0:b1, 1]

            # --- Reference subsets (F) ---
            xi = (px.unsqueeze(1) + rx.unsqueeze(0)).long().clamp(0, w - 1)
            yi = (py.unsqueeze(1) + ry.unsqueeze(0)).long().clamp(0, h - 1)
            f = ref_t.reshape(-1)[yi * w + xi]
            fn = f - f.mean(1, keepdim=True)
            fs = fn.norm(dim=1, keepdim=True).clamp(min=1e-12)
            fn_normed = fn / fs

            best_zncc = torch.full((nb,), -2.0, device=device)
            bu, bv = torch.zeros(nb, device=device), torch.zeros(nb, device=device)

            # --- FIX: Ensure base_u is always a tensor for unsqueeze ---
            base_u = current_u[b0:b1] if current_u is not None else torch.zeros(nb, device=device)
            base_v = current_v[b0:b1] if current_v is not None else torch.zeros(nb, device=device)

            for s in range(num_searches):
                # Candidate coords
                cx = px.unsqueeze(1) + base_u.unsqueeze(1) + sx[s] + rx.unsqueeze(0)
                cy = py.unsqueeze(1) + base_v.unsqueeze(1) + sy[s] + ry.unsqueeze(0)

                # Normalize coordinates to [-1, 1] for grid_sample
                grid = torch.stack([cx / (w - 1) * 2 - 1, cy / (h - 1) * 2 - 1], dim=-1)
                grid = grid.unsqueeze(0)  # (1, nb, p_area, 2)

                g = F.grid_sample(img4d, grid, mode='bilinear', padding_mode='border', align_corners=True)
                g = g.view(nb, p_area)

                gn = g - g.mean(1, keepdim=True)
                gs = gn.norm(dim=1, keepdim=True).clamp(min=1e-12)
                zncc = (fn_normed * (gn / gs)).sum(1)

                better = zncc > best_zncc
                best_zncc[better] = zncc[better]
                bu[better] = base_u[better] + sx[s]
                bv[better] = base_v[better] + sy[s]

            final_u[b0:b1], final_v[b0:b1] = bu, bv

        return final_u, final_v

    # 1. Coarse Pass (Downsample 4x)
    k = 4
    ref_low = F.avg_pool2d(ref_img.view(1, 1, H, W), kernel_size=k).squeeze()
    cur_low = F.avg_pool2d(cur_img.view(1, 1, H, W), kernel_size=k).squeeze()

    u_low, v_low = _do_ncc_search(ref_low, cur_low, points / k,
                                  radius=max(2, R // k),
                                  s_radius=max(2, search_r // k))

    # 2. Fine Pass (Refine ±1 pixel at full res)
    u0, v0 = _do_ncc_search(ref_img, cur_img, points,
                            radius=R,
                            s_radius=1,
                            current_u=u_low * k,
                            current_v=v_low * k)

    return u0, v0

# ── IC-GN iterations (ncorr sec 3.4-3.8, eq.20-28) ───────────────────────────

# dicUtils/dic_engine.py

# def icgn_refinement(f_norm: torch.Tensor,
#                     f_std: torch.Tensor,
#                     sd: torch.Tensor,
#                     H_chol: torch.Tensor,
#                     cur_coeff: torch.Tensor,
#                     points: torch.Tensor,
#                     u0: torch.Tensor,
#                     v0: torch.Tensor,
#                     R: int,
#                     dx_g: torch.Tensor,
#                     dy_g: torch.Tensor,
#                     device: torch.device,
#                     batch_size: int = BATCH_POINTS) -> tuple:
#     """
#     IC-GN refinement (ncorr sec 3.4). Per-iteration cost:
#       • Warp + biquintic sample of current image  (the only expensive part)
#       • Matrix-vector multiply for gradient (eq.23)
#       • Cholesky back-solve (6×6, trivial)
#       • Compositional warp update (eq.27-28)
#     Hessian already Cholesky-decomposed — never recomputed.
#     """
#     Hi, Wi = cur_coeff.shape
#     N = points.shape[0]
#
#     p_out = torch.zeros(N, 6, device=device)
#     p_out[:, 0] = u0;
#     p_out[:, 1] = v0
#     zncc_out = torch.full((N,), -1.0, device=device)
#     iters_out = torch.zeros(N, dtype=torch.int32, device=device)
#     conv_out = torch.zeros(N, dtype=torch.bool, device=device)
#
#     for b0 in range(0, N, batch_size):
#         b1 = min(b0 + batch_size, N)
#         nb = b1 - b0
#
#         px = points[b0:b1, 0];
#         py = points[b0:b1, 1]
#         fn = f_norm[b0:b1]  # (nb, P)
#         fs = f_std[b0:b1]  # (nb,)
#         sd_b = sd[b0:b1]  # (nb, P, 6)
#         L_b = H_chol[b0:b1]  # (nb, 6, 6) Cholesky factor
#
#         p = torch.zeros(nb, 6, device=device)
#         p[:, 0] = u0[b0:b1];
#         p[:, 1] = v0[b0:b1]
#
#         dx = dx_g.unsqueeze(0).expand(nb, -1)  # (nb, P)
#         dy = dy_g.unsqueeze(0).expand(nb, -1)
#
#         done = torch.zeros(nb, dtype=torch.bool, device=device)
#         iters = torch.zeros(nb, dtype=torch.int32, device=device)
#         last_zncc = torch.full((nb,), -1.0, device=device)
#
#         for _ in range(IC_MAX_ITER):
#             # Warp: affine shape function applied to current image
#             xc = (px.unsqueeze(1) + p[:, 0:1]
#                   + dx * (1.0 + p[:, 2:3]) + dy * p[:, 3:4])
#             yc = (py.unsqueeze(1) + p[:, 1:2]
#                   + dx * p[:, 4:5] + dy * (1.0 + p[:, 5:6]))
#             xc = xc.clamp(2.0, Wi - 3.0)
#             yc = yc.clamp(2.0, Hi - 3.0)
#
#             # Biquintic sample (ncorr eq.44)
#             g = _bspline5_eval(cur_coeff, xc, yc)  # (nb, P)
#
#             gm = g.mean(1, keepdim=True)  # (nb, 1)
#             gn = g - gm  # (nb, P)
#             gs = gn.norm(dim=1, keepdim=True).clamp(min=1e-12)  # ← FIX: dim=1, keepdim=True
#             zncc = (fn * gn).sum(1) / (fs * gs.squeeze(1) + 1e-12)  # ← FIX: squeeze gs
#
#             # Gradient ∇CLS (ncorr eq.23)
#             # coeff_g shape: (nb, P)
#             coeff_g = (fn / fs.unsqueeze(1) -  # (nb, P) / (nb, 1) → (nb, P)
#                        zncc.unsqueeze(1) * gn / gs  # (nb, 1) * (nb, P) / (nb, 1) → (nb, P)
#                        ) / gs  # (nb, P) / (nb, 1) → (nb, P)
#
#             grad = torch.einsum("bp,bpk->bk", coeff_g, sd_b)  # (nb,6)
#
#             # Solve H·Δp = grad via precomputed Cholesky
#             dp = torch.cholesky_solve(grad.unsqueeze(-1), L_b).squeeze(-1)
#             dp_norm = dp.norm(dim=1)  # ← FIX: specify dim=1
#
#             # Compositional warp update (ncorr eq.27-28)
#             # W_new = W_old ∘ W(Δp)^{-1}
#             du, dv = dp[:, 0], dp[:, 1]
#             dux, duy = dp[:, 2], dp[:, 3]
#             dvx, dvy = dp[:, 4], dp[:, 5]
#             det = (1.0 + dux) * (1.0 + dvy) - duy * dvx
#             det = det.clamp(min=1e-12)
#
#             # Inverse of affine warp W(Δp):
#             i_u = -(du * (1.0 + dvy) - dv * duy) / det
#             i_v = -(dv * (1.0 + dux) - du * dvx) / det
#             i_ux = (1.0 + dvy) / det - 1.0
#             i_uy = -duy / det
#             i_vx = -dvx / det
#             i_vy = (1.0 + dux) / det - 1.0
#
#             # Compose p_old ∘ W_inv(Δp)
#             uo = p[:, 0];
#             vo = p[:, 1]
#             uxo = p[:, 2];
#             uyo = p[:, 3];
#             vxo = p[:, 4];
#             vyo = p[:, 5]
#
#             p_new = torch.stack([
#                 uo + (1.0 + uxo) * i_u + uyo * i_v,
#                 vo + vxo * i_u + (1.0 + vyo) * i_v,
#                 (1.0 + uxo) * (1.0 + i_ux) + uyo * i_vx - 1.0,
#                 (1.0 + uxo) * i_uy + uyo * (1.0 + i_vy),
#                 vxo * (1.0 + i_ux) + (1.0 + vyo) * i_vx,
#                 vxo * i_uy + (1.0 + vyo) * (1.0 + i_vy) - 1.0,
#             ], dim=1)
#
#             mask = ~done
#             p = torch.where(mask.unsqueeze(1), p_new, p)
#             last_zncc = torch.where(mask, zncc, last_zncc)
#             iters = iters + mask.int()
#             done = done | (dp_norm < IC_CONVERGENCE)
#
#             if done.all():
#                 break
#
#         p_out[b0:b1] = p
#         zncc_out[b0:b1] = last_zncc
#         iters_out[b0:b1] = iters
#         conv_out[b0:b1] = done
#
#     return p_out, zncc_out, iters_out, conv_out

def icgn_refinement(f_norm: torch.Tensor,
                    f_std: torch.Tensor,
                    sd: torch.Tensor,
                    H_chol: torch.Tensor,
                    cur_coeff: torch.Tensor,
                    points: torch.Tensor,
                    u0: torch.Tensor,
                    v0: torch.Tensor,
                    R: int,
                    dx_g: torch.Tensor,
                    dy_g: torch.Tensor,
                    device: torch.device,
                    batch_size: int = 1024) -> tuple:
    """
    Optimized IC-GN Refinement.
    - Active Subset Masking: Only processes points that haven't converged.
    - Reduced Tensor Overhead: Minimizes allocations inside the IC_MAX_ITER loop.
    """
    Hi, Wi = cur_coeff.shape
    N = points.shape[0]

    # Pre-allocate output tensors
    p_out = torch.zeros((N, 6), device=device)
    p_out[:, 0], p_out[:, 1] = u0, v0
    zncc_out = torch.full((N,), -1.0, device=device)
    iters_out = torch.zeros(N, dtype=torch.int32, device=device)
    conv_out = torch.zeros(N, dtype=torch.bool, device=device)

    # Process in larger batches to saturate GPU
    for b0 in range(0, N, batch_size):
        b1 = min(b0 + batch_size, N)
        nb = b1 - b0

        # Local batch data
        curr_p = p_out[b0:b1].clone()
        fn = f_norm[b0:b1]  # (nb, P)
        fs_inv = 1.0 / f_std[b0:b1].unsqueeze(1)  # (nb, 1) pre-inverted
        sd_b = sd[b0:b1]  # (nb, P, 6)
        L_b = H_chol[b0:b1]  # (nb, 6, 6)

        px, py = points[b0:b1, 0:1], points[b0:b1, 1:2]

        # Mask for active (non-converged) points in this batch
        active = torch.ones(nb, dtype=torch.bool, device=device)

        for i in range(IC_MAX_ITER):
            if not active.any():
                break

            # 1. Warp current image coordinates (Optimized Affine)
            # Only compute for active subsets
            idx = torch.where(active)[0]
            sub_nb = idx.numel()

            # W(x, p) = [1+u_x  u_y  u] [x]
            #           [v_x  1+v_y  v] [y]
            #                           [1]
            xc = (px[idx] + curr_p[idx, 0:1] +
                  dx_g * (1.0 + curr_p[idx, 2:3]) + dy_g * curr_p[idx, 3:4])
            yc = (py[idx] + curr_p[idx, 1:2] +
                  dx_g * curr_p[idx, 4:5] + dy_g * (1.0 + curr_p[idx, 5:6]))

            # 2. Biquintic Interpolation
            g = _bspline5_eval(cur_coeff, xc, yc)  # (sub_nb, P)

            # 3. ZNCC and Gradient
            gm = g.mean(dim=1, keepdim=True)
            gn = g - gm
            gs = gn.norm(dim=1, keepdim=True).clamp(min=1e-12)
            gs_inv = 1.0 / gs

            zncc = (fn[idx] * (gn * gs_inv)).sum(dim=1)

            # Difference Image Gradient (eq. 23)
            coeff_g = (fn[idx] * fs_inv[idx] - zncc.unsqueeze(1) * gn * (gs_inv ** 2)) * gs_inv
            grad = torch.einsum("bp,bpk->bk", coeff_g, sd_b[idx])

            # 4. Solve for Step Δp
            dp = torch.cholesky_solve(grad.unsqueeze(-1), L_b[idx]).squeeze(-1)

            # 5. Compositional Update: W_new = W_old ∘ W(Δp)⁻¹
            # Pre-calc inverse of Δp warp
            det = (1.0 + dp[:, 2]) * (1.0 + dp[:, 5]) - dp[:, 3] * dp[:, 4]
            det_inv = 1.0 / det.clamp(min=1e-12)

            i_u = -(dp[:, 0] * (1.0 + dp[:, 5]) - dp[:, 1] * dp[:, 3]) * det_inv
            i_v = -(dp[:, 1] * (1.0 + dp[:, 2]) - dp[:, 0] * dp[:, 4]) * det_inv
            i_ux = (1.0 + dp[:, 5]) * det_inv - 1.0
            i_uy = -dp[:, 3] * det_inv
            i_vx = -dp[:, 4] * det_inv
            i_vy = (1.0 + dp[:, 2]) * det_inv - 1.0

            # Composition logic
            p_old = curr_p[idx]
            p_new = torch.empty_like(p_old)
            p_new[:, 0] = p_old[:, 0] + (1.0 + p_old[:, 2]) * i_u + p_old[:, 3] * i_v
            p_new[:, 1] = p_old[:, 1] + p_old[:, 4] * i_u + (1.0 + p_old[:, 5]) * i_v
            p_new[:, 2] = (1.0 + p_old[:, 2]) * (1.0 + i_ux) + p_old[:, 3] * i_vx - 1.0
            p_new[:, 3] = (1.0 + p_old[:, 2]) * i_uy + p_old[:, 3] * (1.0 + i_vy)
            p_new[:, 4] = p_old[:, 4] * (1.0 + i_ux) + (1.0 + p_old[:, 5]) * i_vx
            p_new[:, 5] = p_old[:, 4] * i_uy + (1.0 + p_old[:, 5]) * (1.0 + i_vy) - 1.0

            curr_p[idx] = p_new
            iters_out[b0 + idx] += 1

            # Convergence check
            converged = dp.norm(dim=1) < IC_CONVERGENCE
            active[idx[converged]] = False

            # Store final ZNCC for converged points
            zncc_out[b0 + idx[converged]] = zncc[converged]

        p_out[b0:b1] = curr_p
        conv_out[b0:b1] = ~active

    return p_out, zncc_out, iters_out, conv_out

# ── Strains via plane fit (ncorr sec 5, eq.49-54) ─────────────────────────────

def compute_strains_plane_fit(u, v, points, step=GRID_STEP, strain_window=STRAIN_WINDOW):
    """
    Refactored to PyTorch for GPU acceleration.
    Uses Batch Least Squares to fit planes to u and v fields simultaneously.
    """
    # Move to GPU if not already there
    u_t = torch.as_tensor(u, dtype=torch.float32, device=get_device())
    v_t = torch.as_tensor(v, dtype=torch.float32, device=get_device())
    pts_t = torch.as_tensor(points, dtype=torch.float32, device=get_device())

    N = pts_t.shape[0]
    sw = strain_window

    # 1. Neighborhood Extraction
    # We find neighbors using a distance-based mask (vectorized)
    dist_x = pts_t[:, 0].unsqueeze(1) - pts_t[:, 0].unsqueeze(0)
    dist_y = pts_t[:, 1].unsqueeze(1) - pts_t[:, 1].unsqueeze(0)

    mask = (dist_x.abs() <= sw) & (dist_y.abs() <= sw) & (~u_t.isnan().unsqueeze(0))

    # 2. Batch Plane Fit (u = a + bx + cy)
    # Note: For massive grids, we use a fixed-size window to keep memory linear
    exx = torch.full((N,), torch.nan, device=u_t.device)
    eyy = torch.full((N,), torch.nan, device=u_t.device)
    exy = torch.full((N,), torch.nan, device=u_t.device)

    for i in range(N):
        idx = torch.where(mask[i])[0]
        if len(idx) < 6: continue

        # Local coordinates relative to center point
        dx = dist_x[i, idx]
        dy = dist_y[i, idx]

        A = torch.stack([torch.ones_like(dx), dx, dy], dim=1)

        # Solve for u and v gradients in one go
        b_u = torch.linalg.lstsq(A, u_t[idx]).solution
        b_v = torch.linalg.lstsq(A, v_t[idx]).solution

        du_dx, du_dy = b_u[1], b_u[2]
        dv_dx, dv_dy = b_v[1], b_v[2]

        # Green-Lagrangian Strain
        exx[i] = du_dx + 0.5 * (du_dx ** 2 + dv_dx ** 2)
        eyy[i] = dv_dy + 0.5 * (du_dy ** 2 + dv_dy ** 2)
        exy[i] = 0.5 * (du_dy + dv_dx + du_dx * du_dy + dv_dx * dv_dy)

    return {"exx": exx.cpu().numpy(), "eyy": eyy.cpu().numpy(), "exy": exy.cpu().numpy()}

# ── Full DIC computation for one frame pair ────────────────────────────────────

def compute_dic_frame(ref_img_np: np.ndarray,
                      def_img_np: np.ndarray,
                      points_np: np.ndarray,
                      device: torch.device,
                      R: int = SUBSET_RADIUS,
                      bbox: tuple = None,
                      step: int = GRID_STEP) -> dict:
    """
    Compute IC-GN DIC for one frame pair (full ncorr algorithm).

    ref_img_np : (H,W) uint8 or float32 reference frame
    def_img_np : (H,W) uint8 or float32 deformed frame
    points_np  : (N,2) float32 analysis grid (x,y)
    device     : torch device
    R          : subset radius
    bbox       : (x0,y0,w,h) for strain plane fit (pass None to skip)
    step       : grid spacing used in make_roi_grid
    """
    def _norm(a):
        a = a.astype(np.float32)
        return a / 255.0 if a.max() > 1.0 else a

    ref_t = torch.from_numpy(_norm(ref_img_np)).to(device)
    def_t = torch.from_numpy(_norm(def_img_np)).to(device)
    pts_t = torch.from_numpy(points_np.astype(np.float32)).to(device)

    # 1. Biquintic B-spline coefficients (ncorr sec 3.6-3.7)
    print("[DIC] B-spline coefficients...")
    ref_coeff = bspline5_coeffs(ref_t)
    cur_coeff = bspline5_coeffs(def_t)

    # 2. Precompute steepest descent images + Hessian (ncorr steps 2-4, ONCE)
    print(f"[DIC] Precomputing IC-GN for {len(points_np)} points...")
    f_norm, f_std, sd, H_chol, valid, dx_g, dy_g = precompute_icgn(
        ref_t, ref_coeff, pts_t, R, device)

    # 3. NCC initial guess (ncorr sec 3.1)
    print("[DIC] NCC initial guess...")
    u0, v0 = ncc_initial_guess(ref_t, def_t, pts_t, R, SEARCH_RADIUS, device)

    # 4. IC-GN sub-pixel refinement (ncorr sec 3.4-3.8)
    print("[DIC] IC-GN iterations...")
    p_out, zncc, iters, converged = icgn_refinement(
        f_norm, f_std, sd, H_chol, cur_coeff,
        pts_t, u0, v0, R, dx_g, dy_g, device)

    zncc[~valid]      = -1.0
    converged[~valid] = False

    u_np  = p_out[:,0].cpu().numpy()
    v_np  = p_out[:,1].cpu().numpy()
    ux_np = p_out[:,2].cpu().numpy()
    uy_np = p_out[:,3].cpu().numpy()
    vx_np = p_out[:,4].cpu().numpy()
    vy_np = p_out[:,5].cpu().numpy()
    zncc_np = zncc.cpu().numpy()

    bad = (zncc_np < ZNCC_THRESHOLD) | ~valid.cpu().numpy()
    for a in [u_np, v_np, ux_np, uy_np, vx_np, vy_np]:
        a[bad] = np.nan

    # 5. Strains via plane fit (ncorr sec 5)
    strain = compute_strains_plane_fit(u_np, v_np, points_np, step, STRAIN_WINDOW)

    return {
        "u":         u_np,
        "v":         v_np,
        "ux":        ux_np,
        "uy":        uy_np,
        "vx":        vx_np,
        "vy":        vy_np,
        "exx":       strain["exx"],
        "eyy":       strain["eyy"],
        "exy":       strain["exy"],
        "zncc":      zncc_np,
        "converged": converged.cpu().numpy(),
        "iters":     iters.cpu().numpy(),
    }


# ── Point grid (ncorr "spacing" parameter) ────────────────────────────────────

def make_roi_grid(mask: np.ndarray, bbox: tuple,
                  R: int = SUBSET_RADIUS,
                  step: int = GRID_STEP) -> np.ndarray:
    """Regular grid, one point every `step` pixels. Matches ncorr spacing."""
    x0, y0, w, h = bbox
    H, W = mask.shape
    margin = R + 3   # quintic needs 3px clearance

    pts = []
    for iy in range(y0 + margin, y0 + h - margin, step):
        for ix in range(x0 + margin, x0 + w - margin, step):
            if mask[iy, ix] == 255:
                pts.append([ix, iy])

    if not pts:
        raise RuntimeError(
            f"No valid DIC points (R={R}, step={step}). "
            "Try smaller R/step or larger ROI.")

    return np.array(pts, dtype=np.float32)