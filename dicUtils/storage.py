# dicUtils/storage.py
#
# HDF5-based save/load for DIC results.
# Save  : compressed per-field arrays + metadata → .h5
# Export: full pixel-level data in npy or CSV format + mask image

import os
import json
import numpy as np
from datetime import datetime

try:
    import h5py
    HAS_H5 = True
except ImportError:
    HAS_H5 = False


# ── Save ─────────────────────────────────────────────────────────────────────

def save_results(
    path: str,
    results: list,
    selected_keys: list,
    polygon: np.ndarray,
    mask: np.ndarray,
    bbox: tuple,
    frame_step: int,
    fps: float,
    source_info: str = "",
    save_every_n: int = 1,
    custom_frames: list = None,
):
    """Save DIC results to HDF5."""
    if not HAS_H5:
        raise RuntimeError("h5py not installed. Run: pip install h5py")

    if custom_frames is not None:
        frame_set = set(custom_frames)
        to_save = [r for r in results if r[0] in frame_set]
    else:
        to_save = results[::save_every_n]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with h5py.File(path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["created"]       = datetime.now().isoformat()
        meta.attrs["source"]        = source_info
        meta.attrs["frame_step"]    = frame_step
        meta.attrs["fps"]           = fps
        meta.attrs["selected_keys"] = json.dumps(selected_keys)
        meta.attrs["bbox"]          = list(bbox)
        meta.attrs["save_every_n"]  = save_every_n

        roi_g = f.create_group("roi")
        roi_g.create_dataset("polygon", data=polygon,
                             compression="gzip", compression_opts=6)
        roi_g.create_dataset("mask", data=mask,
                             compression="gzip", compression_opts=9)

        frames_g     = f.create_group("frames")
        frame_indices = []

        for frame_idx, bg_np, fields in to_save:
            fg = frames_g.create_group(f"frame_{frame_idx:06d}")
            fg.attrs["frame_idx"] = frame_idx
            fg.create_dataset("bg", data=bg_np,
                              compression="gzip", compression_opts=6)
            for key, arr in fields.items():
                fg.create_dataset(key, data=arr.astype(np.float32),
                                  compression="gzip", compression_opts=6)
            frame_indices.append(frame_idx)

        meta.create_dataset("frame_indices",
                            data=np.array(frame_indices, dtype=np.int32))

    print(f"Saved {len(to_save)} frames to {path}  "
          f"({os.path.getsize(path)/1e6:.1f} MB)")


# ── Load ─────────────────────────────────────────────────────────────────────

def load_results(path: str):
    """Load DIC results from HDF5."""
    if not HAS_H5:
        raise RuntimeError("h5py not installed. Run: pip install h5py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved data at: {path}")

    with h5py.File(path, "r") as f:
        meta          = f["metadata"]
        frame_step    = int(meta.attrs["frame_step"])
        fps           = float(meta.attrs["fps"])
        selected_keys = json.loads(meta.attrs["selected_keys"])
        bbox          = tuple(int(v) for v in meta.attrs["bbox"])
        source_info   = str(meta.attrs.get("source", ""))
        # display_keys includes virtual keys (e.g. quiver) — falls back to
        # selected_keys for files saved before this field was added
        display_keys  = json.loads(
            meta.attrs.get("display_keys", meta.attrs["selected_keys"]))

        polygon = f["roi/polygon"][:]
        mask    = f["roi/mask"][:]

        results  = []
        frames_g = f["frames"]
        for name in sorted(frames_g.keys()):
            fg        = frames_g[name]
            frame_idx = int(fg.attrs["frame_idx"])
            bg_np     = fg["bg"][:]
            fields    = {k: fg[k][:] for k in selected_keys if k in fg}
            results.append((frame_idx, bg_np, fields))

    print(f"Loaded {len(results)} frames from {path}")
    print(f"  stored fields:  {selected_keys}")
    print(f"  display fields: {display_keys}")
    return results, selected_keys, display_keys, polygon, mask, bbox, frame_step, fps, source_info


# ── Mask image ────────────────────────────────────────────────────────────────

def export_mask_image(mask: np.ndarray, polygon: np.ndarray,
                      bg_np: np.ndarray, out_dir: str):
    """
    Save mask files to out_dir.

    mask_roi.png  (Ncorr-compatible)
        Strict binary grayscale PNG at full frame resolution.
        Pixel values are exactly 0 (outside ROI) or 255 (inside ROI).
        No compression artefacts, no colour channels.
        Load directly into Ncorr as the reference subset mask.

    mask_overlay.png
        The first grayscale frame with the ROI polygon drawn in green,
        interior filled with a semi-transparent green tint, and vertices
        numbered. For visual verification only — do not load into Ncorr.

    mask_polygon.txt
        Polygon vertex coordinates, one vertex per line as "x,y".
    """
    import cv2
    from PIL import Image as PILImage

    os.makedirs(out_dir, exist_ok=True)

    # 1. Ncorr-compatible binary mask
    #    Must be strict uint8, values exactly 0 or 255, grayscale PNG.
    #    cv2.imwrite with a single-channel uint8 array produces this.
    mask_strict = np.where(mask > 0, np.uint8(255), np.uint8(0))
    ncorr_path  = os.path.join(out_dir, "mask_roi.png")
    cv2.imwrite(ncorr_path, mask_strict)

    H, W = mask_strict.shape
    n_inside = int(np.sum(mask_strict > 0))
    print(f"  mask_roi.png     — {W}x{H} px  "
          f"{n_inside} pixels inside ROI  (Ncorr-compatible)")

    # 2. Visual overlay — for verification, not for Ncorr
    bgr        = cv2.cvtColor(bg_np, cv2.COLOR_GRAY2BGR)
    green_fill = np.zeros_like(bgr)
    green_fill[mask_strict == 255] = (0, 200, 80)
    overlay    = cv2.addWeighted(bgr, 0.72, green_fill, 0.28, 0)

    pts = polygon.reshape((-1, 1, 2))
    cv2.polylines(overlay, [pts], isClosed=True,
                  color=(0, 255, 100), thickness=2, lineType=cv2.LINE_AA)
    for i, (x, y) in enumerate(polygon):
        cv2.circle(overlay, (int(x), int(y)), 5, (0, 230, 255), -1)
        cv2.putText(overlay, str(i + 1), (int(x) + 7, int(y) - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1,
                    cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, "mask_overlay.png"), overlay)
    print(f"  mask_overlay.png — visual only, do not load into Ncorr")

    # 3. Polygon vertex text file
    poly_path = os.path.join(out_dir, "mask_polygon.txt")
    with open(poly_path, "w") as f:
        f.write("# ROI polygon vertices  (x, y)  image pixel coordinates\n")
        for i, (x, y) in enumerate(polygon):
            f.write(f"{int(x)},{int(y)}\n")
    print(f"  mask_polygon.txt — {len(polygon)} vertices")


# ── Export to npy ─────────────────────────────────────────────────────────────

def export_to_npy(h5_path: str, out_dir: str, selected_keys: list = None):
    """
    Export full pixel-level DIC data to NumPy .npy files.

    Output files
    ------------
    <field>.npy
        Shape (T, h, w) float32.
        T = number of frames, h/w = ROI bounding box height/width.
        One file per field (vx, vy, u, v, mag, exx, eyy, exy, edxx, edyy, edxy).

    frame_indices.npy
        Shape (T,) int32.
        The actual frame numbers corresponding to axis 0 of every field array.

    time_s.npy
        Shape (T,) float64.
        Wall-clock time in seconds for each frame: frame_index / fps.

    background_frames.npy
        Shape (T, H, W) uint8.
        Grayscale background frames at full resolution (not cropped to ROI).

    polygon.npy
        Shape (N, 2) int32.
        ROI polygon vertices in image pixel coordinates.

    mask.npy
        Shape (H, W) uint8.
        Full-frame binary mask: 255 inside polygon, 0 outside.

    mask_binary.png
    mask_overlay.png
    mask_polygon.txt
        See export_mask_image() for details.

    meta.npy
        Python dict saved as a 0-d object array. Load with:
            meta = np.load("meta.npy", allow_pickle=True).item()
        Keys: fps, frame_step, bbox (x,y,w,h), keys, source.
    """
    results, keys, polygon, mask, bbox, frame_step, fps, source = \
        load_results(h5_path)

    if selected_keys:
        keys = [k for k in selected_keys if k in keys]

    os.makedirs(out_dir, exist_ok=True)

    frame_indices = np.array([r[0] for r in results], dtype=np.int32)
    time_s        = (frame_indices / fps).astype(np.float64)

    np.save(os.path.join(out_dir, "frame_indices.npy"), frame_indices)
    np.save(os.path.join(out_dir, "time_s.npy"),        time_s)
    print(f"  frame_indices.npy — shape {frame_indices.shape}")
    print(f"  time_s.npy        — shape {time_s.shape}")

    # Background frames (full resolution)
    bgs = np.stack([r[1] for r in results], axis=0)   # (T, H, W)
    np.save(os.path.join(out_dir, "background_frames.npy"), bgs)
    print(f"  background_frames.npy — shape {bgs.shape}  uint8")
    first_frame = results[0][1]

    # Field arrays
    for key in keys:
        arrays = [r[2][key] for r in results if key in r[2]]
        if not arrays:
            continue
        stacked = np.stack(arrays, axis=0).astype(np.float32)   # (T, h, w)
        out_path = os.path.join(out_dir, f"{key}.npy")
        np.save(out_path, stacked)
        print(f"  {key}.npy — shape {stacked.shape}  float32")

    # ROI geometry
    np.save(os.path.join(out_dir, "polygon.npy"), polygon)
    np.save(os.path.join(out_dir, "mask.npy"),    mask)
    print(f"  polygon.npy — shape {polygon.shape}")
    print(f"  mask.npy    — shape {mask.shape}")

    # Mask images
    export_mask_image(mask, polygon, first_frame, out_dir)

    # Metadata dict
    meta = {
        "fps":        fps,
        "frame_step": frame_step,
        "bbox":       list(bbox),    # [x, y, w, h]
        "keys":       keys,
        "source":     source,
        "n_frames":   len(results),
        "roi_h":      int(bbox[3]),
        "roi_w":      int(bbox[2]),
    }
    np.save(os.path.join(out_dir, "meta.npy"), meta)
    print(f"  meta.npy — fps={fps}  frame_step={frame_step}  "
          f"bbox={bbox}  fields={keys}")

    print(f"\nnpy export complete → {out_dir}/")
    print(f"  Total frames exported: {len(results)}")
    print(f"  Fields exported: {keys}")


# ── Export to CSV ─────────────────────────────────────────────────────────────

def export_to_csv(h5_path: str, out_dir: str, selected_keys: list = None):
    """
    Export full pixel-level DIC data to CSV files — one CSV per field.

    Output files
    ------------
    <field>.csv
        Full pixel-level data for that field across all frames.
        Each row represents one pixel at one frame.

        Columns:
            frame_idx   : frame number
            time_s      : wall-clock time in seconds
            pixel_x     : x coordinate in the full image (not ROI-relative)
            pixel_y     : y coordinate in the full image (not ROI-relative)
            roi_x       : x coordinate relative to ROI bounding box origin
            roi_y       : y coordinate relative to ROI bounding box origin
            in_polygon  : 1 if pixel is inside the polygon, 0 if in bbox but outside
            <field>     : the field value at this pixel and frame

    meta.csv
        One row of metadata: fps, frame_step, bbox, n_frames, fields.

    mask_binary.png
    mask_overlay.png
    mask_polygon.txt
        ROI geometry images (see export_mask_image).

    Notes
    -----
    These files can be very large. A 200x300 ROI over 300 frames produces
    ~18 million rows per field. For large datasets, npy export is faster
    and more memory-efficient.
    """
    import csv

    results, keys, polygon, mask, bbox, frame_step, fps, source = \
        load_results(h5_path)

    if selected_keys:
        keys = [k for k in selected_keys if k in keys]

    os.makedirs(out_dir, exist_ok=True)

    x0, y0, w, h = bbox

    # Precompute pixel coordinate grids (ROI-relative)
    roi_ys, roi_xs = np.meshgrid(
        np.arange(h, dtype=np.int32),
        np.arange(w, dtype=np.int32),
        indexing="ij"
    )   # both (h, w)

    img_xs = (roi_xs + x0).astype(np.int32)   # full-image x coords
    img_ys = (roi_ys + y0).astype(np.int32)   # full-image y coords

    # in_polygon mask cropped to bbox
    in_poly = (mask[y0:y0+h, x0:x0+w] == 255).astype(np.uint8)   # (h, w)

    # Flatten for iteration
    flat_roi_x  = roi_xs.ravel()
    flat_roi_y  = roi_ys.ravel()
    flat_img_x  = img_xs.ravel()
    flat_img_y  = img_ys.ravel()
    flat_inpoly = in_poly.ravel()
    n_pixels    = len(flat_roi_x)

    frame_indices = np.array([r[0] for r in results], dtype=np.int32)

    for key in keys:
        csv_path = os.path.join(out_dir, f"{key}.csv")
        print(f"  Writing {key}.csv  "
              f"({len(results)} frames x {n_pixels} pixels = "
              f"{len(results)*n_pixels:,} rows)...")

        with open(csv_path, "w", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerow([
                "frame_idx", "time_s",
                "pixel_x", "pixel_y",
                "roi_x", "roi_y",
                "in_polygon",
                key
            ])

            for frame_idx, _, fields in results:
                if key not in fields:
                    continue
                field_arr = fields[key]           # (h, w)
                flat_vals = field_arr.ravel()     # (h*w,)
                t         = frame_idx / fps

                for i in range(n_pixels):
                    writer.writerow([
                        frame_idx,
                        round(t, 6),
                        int(flat_img_x[i]),
                        int(flat_img_y[i]),
                        int(flat_roi_x[i]),
                        int(flat_roi_y[i]),
                        int(flat_inpoly[i]),
                        round(float(flat_vals[i]), 8),
                    ])

        size_mb = os.path.getsize(csv_path) / 1e6
        print(f"    Done — {size_mb:.1f} MB")

    # Meta CSV
    meta_path = os.path.join(out_dir, "meta.csv")
    with open(meta_path, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=[
            "fps", "frame_step", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "n_frames", "roi_pixels", "fields", "source"
        ])
        writer.writeheader()
        writer.writerow({
            "fps":        fps,
            "frame_step": frame_step,
            "bbox_x":     x0, "bbox_y": y0,
            "bbox_w":     w,  "bbox_h": h,
            "n_frames":   len(results),
            "roi_pixels": int(np.sum(in_poly)),
            "fields":     "|".join(keys),
            "source":     source,
        })
    print(f"  meta.csv written")

    # Mask images
    first_frame = results[0][1]
    export_mask_image(mask, polygon, first_frame, out_dir)

    print(f"\nCSV export complete → {out_dir}/")
    print(f"  Fields: {keys}")
    print(f"  Frames: {len(results)}  |  ROI pixels: {int(np.sum(in_poly))}")
    print(f"  Note: CSV files contain full pixel-level data and may be large.")