# Dense DIC — Digital Image Correlation via Optical Flow

A Python tool for full-field displacement, velocity, and strain analysis using Farneback dense optical flow, designed for machining, deformation, and material testing experiments.

---

## Features

- **Dense full-field DIC** — displacement, velocity, strain, and strain rate fields over a user-defined polygon ROI
- **Single-point tracker** — Lucas-Kanade pyramidal optical flow for tracking a seed point through a sequence
- **Interactive PyQt5 UI** — polygon ROI editor, live video player, hover zoom, velocity quiver overlay
- **Multiple data sources** — load from video file, image folder, or previously saved `.h5` results
- **Streaming computation** — results written to HDF5 frame-by-frame; never loads more than 50 frames into RAM
- **Add Fields** — compute additional fields after evaluating initial results, without restarting
- **HDF5 save/export** — efficient compressed storage, re-loadable without recomputing; export to CSV or `.npy`
- **Fully configurable** — persistent settings panel for colormap, normalization, quiver style, playback speed, and more

---

## Requirements

### Python

Python 3.10 or newer.

### Dependencies

```bash
pip install opencv-python PyQt5 numpy matplotlib Pillow h5py tqdm pandas
```

| Package | Purpose |
|---|---|
| `opencv-python` | Farneback flow, LK tracker, image I/O |
| `PyQt5` | All GUI windows |
| `numpy` | Array math |
| `matplotlib` | Colormapping (Agg backend, off-screen only) |
| `Pillow` | TIFF loading, 16-bit support |
| `h5py` | HDF5 save/load |
| `tqdm` | Video extraction progress |
| `pandas` | Frame metadata CSV |

> **PyCharm users:** Go to Settings > Tools > Python Scientific and uncheck "Show plots in tool window" to allow interactive Qt windows.

---

## Project Structure

```
project/
|
+-- main.py                        # Entry point -- run this
+-- configuration.json             # Shared configuration
|
+-- dicUtils/
    +-- config.py                  # Loads configuration.json
    +-- io.py                      # Frame path discovery, image loading
    +-- roi.py                     # PyQt5 polygon ROI editor
    +-- dense.py                   # Farneback DIC engine + player window
    +-- stream.py                  # Streaming HDF5 store (50-frame window)
    +-- launcher.py                # Startup screen (video/folder/saved)
    +-- settings.py                # DICSettings dataclass + settings panel
    +-- storage.py                 # HDF5 save/load, CSV/npy export
    +-- tracking.py                # Lucas-Kanade single-point tracker
    +-- velocity.py                # Velocity computation and plotting
    +-- seed.py                    # Interactive seed point selector
    +-- video_frame_extractor.py   # Extract frames from video files
```

---

## Configuration

Edit `configuration.json` before running:

```json
{
  "frame_folder":    "frames",
  "frame_extension": ".tiff",
  "frame_step":      5,
  "fps":             50,
  "win_size":        31,
  "max_level":       2,
  "iterations":      30,
  "display_tracking": true
}
```

| Parameter | Effect |
|---|---|
| `frame_step` | Process every Nth frame — larger = faster, coarser time resolution |
| `fps` | Camera frame rate — used to convert displacements to velocities |
| `win_size` | LK window size (single-point tracker only) |
| `max_level` | LK pyramid levels — increase if tracked point moves many pixels between frames |
| `iterations` | LK iterations per pyramid level |

---

## Running

```bash
python main.py
```

### Step 1 — Launcher

Three source modes:

| Mode | When to use |
|---|---|
| Load from Video | Raw `.mp4`, `.avi`, or `.mov` file |
| Load from Folder | Existing image sequence in a folder |
| Load Saved Data | `.h5` file from a previous DIC run |

**Load from Video** — select the video file and output folder. Frames are extracted automatically when you click CONTINUE. Set FPS, save-every-N frames, and image format (TIFF recommended for lossless quality).

**Load from Folder** — select the folder containing your image sequence. Set FPS and file extension, then click CONTINUE.

> Naming requirement: images must be zero-padded and lexicographically sorted — e.g. `frame_000001.tiff`, `frame_000002.tiff`. Without zero-padding, `frame_10` sorts before `frame_2` and frames will be processed out of order.

**Load Saved Data** — select a `.h5` file from a previous run. Opens the player directly with no recomputation.

---

### Step 2 — ROI Editor

A polygon editor opens on the first frame.

| Action | Control |
|---|---|
| Add vertex | Left click on canvas |
| Move vertex | Click vertex, then drag |
| Insert vertex on edge | Click near an edge midpoint dot |
| Delete vertex | Right click (removes nearest within 30 px) |
| Undo | Z |
| Clear all | C |
| Zoom | Scroll wheel |
| Pan | Middle mouse drag |
| Confirm | Enter |
| Cancel | Escape |

Draw tightly around the region of interest. The side panel shows live vertex coordinates in image space.

---

### Step 3 — Field Selector

| Field | Unit | Default |
|---|---|---|
| Velocity Vx | pixels/s | Yes |
| Velocity Vy | pixels/s | Yes |
| Velocity Quiver | arrow overlay | No |
| Displacement u | pixels/frame | No |
| Displacement v | pixels/frame | No |
| Magnitude |d| | pixels/frame | No |
| Strain e_xx | px/px | No |
| Strain e_yy | px/px | No |
| Strain e_xy | px/px | No |
| Strain rate e_xx/dt | px/px/s | Yes |
| Strain rate e_yy/dt | px/px/s | Yes |
| Strain rate e_xy/dt | px/px/s | Yes |

Click COMPUTE & DISPLAY. Fields are computed frame-by-frame and written to a temporary HDF5 file as they are produced — no large memory allocation. The player opens when computation is complete.

**Velocity Quiver** is a virtual field — it is derived from Vx and Vy at display time and does not require additional computation. Selecting it will automatically include Vx and Vy in the compute pass if they are not already selected.

---

### Step 4 — DIC Player

Panels are arranged automatically: 1 column for 1 field, 2 columns for 2-4 fields, 3 columns for 5 or more.

#### Playback controls

| Control | Action |
|---|---|
| Space | Play / Pause |
| Left / Right arrow | Step one frame |
| Home / End | First / Last frame |
| Scroll wheel | Step through frames |
| Timeline slider | Jump to any frame |
| -10 / +10 buttons | Jump 10 frames |
| Speed buttons | 0.25x  0.5x  1x  2x  4x |

#### Hover inspection

Move the cursor over any panel to see the exact field value at that pixel in the info bar at the top of the window. A zoomed view of the local region appears in tooltip, popup, or side panel mode (configured in Settings).

Right-click any panel for:
- Save field as .npy — saves the current frame's field array to disk
- Copy value at cursor — copies the numeric value to clipboard

#### Colorbar

Each panel has a symmetric colorbar: blue is negative, green is zero, red is positive (jet colormap by default). The colorbar scale is computed from the data and held stable across all frames.

---

### Adding Fields After Evaluation

Click **Add Fields** in the top bar at any time during playback.

The field selector reopens showing:
- Fields marked `[computed]` in cyan — already in the HDF5 store. Toggling these is instant (no recomputation).
- Fields in white — will require a new Farneback pass to compute.

Only the newly requested fields are computed. Existing data in the store is not touched. After computation the grid rebuilds automatically and the current frame is refreshed.

---

## Settings

Click **Settings** in the top bar. Preferences are saved automatically to `~/.dic_settings.json` and persist between sessions.

| Setting | Description |
|---|---|
| Colormap | jet, plasma, RdBu, viridis, inferno, coolwarm, seismic, bwr, turbo |
| Overlay opacity | How opaque the colour overlay is (10-100%) |
| Colorbar normalization | percentile: 99th percentile of current frame / global_max: max across all frames |
| Percentile | Percentile cutoff for normalization (50-100%) |
| Hover zoom mode | tooltip: follows cursor / popup: fixed near cursor / panel: side panel |
| Hover zoom size | Radius of zoomed region in pixels |
| Quiver arrow style | scaled: arrow length proportional to magnitude / uniform: fixed length, colour encodes magnitude |
| Arrow grid spacing | Pixels between arrow origins |
| Default playback speed | 0.25x to 4x |
| Save every N | Default N shown in the save dialog |

---

## Saving and Exporting Data

Click **Save / Export** in the player.

### Save — HDF5

Saves the full DIC results as a compressed `.h5` file. Open it later via Load Saved Data in the launcher — no recomputation required.

Save modes:

| Mode | Description |
|---|---|
| All frames | Every computed frame |
| Every N frames | e.g. every 5th computed frame |
| Custom frame list | Comma-separated frame indices, e.g. 100,200,500,1000 |

### Export — Human-readable

| Format | Description |
|---|---|
| CSV | Per-frame summary statistics. One row per frame. |
| npy | One array file per field. |

> Save and Export are different operations. Save creates a reloadable DIC session file. Export creates files for use in other tools such as MATLAB, Excel, or Python scripts.

---

## Exported File Formats

Both CSV and npy export write full pixel-level data — every pixel value for every frame, not just summary statistics. The two formats contain equivalent information; choose based on what downstream tools you are using.

---

### HDF5 (.h5) — Save format

The `.h5` file is the primary storage format. It is compressed and reloadable.

```
dic_results.h5
|
+-- metadata/
|   +-- frame_step        (int scalar)
|   +-- fps               (float scalar)
|   +-- selected_keys     (JSON string: list of field key names)
|   +-- bbox              (int array [x, y, w, h])
|   +-- vmaxes            (JSON string: dict of field -> global max value)
|   +-- frame_indices     (int32 array: list of frame index numbers)
|   +-- created           (ISO timestamp string)
|   +-- source            (path to original image folder or video)
|
+-- roi/
|   +-- polygon           (int32 array, shape [N, 2]: polygon vertex coordinates)
|   +-- mask              (uint8 array, shape [H, W]: 255 inside polygon, 0 outside)
|
+-- frames/
    +-- frame_000005/
    |   +-- bg            (uint8 array, shape [H, W]: full-resolution grayscale frame)
    |   +-- vx            (float32 array, shape [h, w]: x-velocity in pixels/s)
    |   +-- vy            (float32 array, shape [h, w]: y-velocity in pixels/s)
    |   +-- edxx          (float32 array, shape [h, w]: strain rate xx in px/px/s)
    |   +-- ...           (one dataset per computed field)
    +-- frame_000010/
    +-- ...
```

All field arrays are cropped to the ROI bounding box dimensions `[h, w]`. The `bg` frame is full-resolution `[H, W]`. The `mask` is full-resolution and used to identify which bounding-box pixels belong to the polygon interior.

Reading in Python:

```python
import h5py, json, numpy as np

with h5py.File("dic_results.h5", "r") as f:
    fps        = f["metadata"].attrs["fps"]
    frame_step = f["metadata"].attrs["frame_step"]
    keys       = json.loads(f["metadata"].attrs["selected_keys"])
    bbox       = list(f["metadata"].attrs["bbox"])       # [x, y, w, h]
    indices    = f["metadata/frame_indices"][:]          # (T,) frame numbers

    polygon    = f["roi/polygon"][:]                     # (N, 2) int32
    mask       = f["roi/mask"][:]                        # (H, W) uint8

    # Read one field from one frame
    vx = f["frames/frame_000005/vx"][:]                 # (h, w) float32
```

---

### NumPy (.npy) Export format

Exports full pixel-level data. One `.npy` file per field, plus coordinate, geometry, and metadata files. All files are written to a single folder.

| File | Shape | dtype | Description |
|---|---|---|---|
| `vx.npy` | (T, h, w) | float32 | x-velocity for all frames, pixels/s |
| `vy.npy` | (T, h, w) | float32 | y-velocity for all frames, pixels/s |
| `edxx.npy` | (T, h, w) | float32 | strain rate xx for all frames |
| `...` | (T, h, w) | float32 | one file per selected field |
| `background_frames.npy` | (T, H, W) | uint8 | full-resolution grayscale frames |
| `frame_indices.npy` | (T,) | int32 | actual frame numbers for axis 0 |
| `time_s.npy` | (T,) | float64 | time in seconds: frame_index / fps |
| `polygon.npy` | (N, 2) | int32 | ROI polygon vertex coordinates |
| `mask.npy` | (H, W) | uint8 | full-frame binary mask |
| `mask_binary.png` | — | — | mask as a PNG image (white = inside) |
| `mask_overlay.png` | — | — | first frame with ROI polygon drawn in green |
| `mask_polygon.txt` | — | — | polygon vertices as plain text, one per line |
| `meta.npy` | dict | — | fps, frame_step, bbox, keys, source, n_frames |

`T` = number of exported frames, `h`/`w` = ROI bounding box height/width, `H`/`W` = full frame height/width.

Loading in Python:

```python
import numpy as np

vx            = np.load("vx.npy")              # (T, h, w) float32
frame_indices = np.load("frame_indices.npy")   # (T,) int32
time_s        = np.load("time_s.npy")          # (T,) float64
meta          = np.load("meta.npy", allow_pickle=True).item()

fps        = meta["fps"]
frame_step = meta["frame_step"]
bbox       = meta["bbox"]    # [x, y, w, h]
x0, y0, w, h = bbox

# Plot Vx at frame 10
import matplotlib.pyplot as plt
plt.imshow(vx[10], cmap="jet", extent=[x0, x0+w, y0+h, y0])
plt.colorbar(label="pixels/s")
plt.title(f"Vx  t = {time_s[10]:.3f} s")
plt.show()

# Apply polygon mask (pixels outside polygon set to NaN)
mask  = np.load("mask.npy")                    # (H, W)
roi_mask = mask[y0:y0+h, x0:x0+w].astype(bool)
vx_masked = vx.copy()
vx_masked[:, ~roi_mask] = np.nan
```

---

### CSV Export format

Exports full pixel-level data. One CSV file per field, written to a folder. Each file contains one row per pixel per frame.

**Column layout:**

| Column | Description |
|---|---|
| `frame_idx` | Frame number in the original sequence |
| `time_s` | Time in seconds: frame_idx / fps |
| `pixel_x` | x coordinate in the full image |
| `pixel_y` | y coordinate in the full image |
| `roi_x` | x coordinate relative to ROI bounding box origin |
| `roi_y` | y coordinate relative to ROI bounding box origin |
| `in_polygon` | 1 if inside the polygon, 0 if in bounding box but outside polygon |
| `<field>` | The field value at this pixel and frame |

Example rows from `vx.csv`:

```
frame_idx, time_s,  pixel_x, pixel_y, roi_x, roi_y, in_polygon, vx
5,         0.10000, 860,     424,     0,     0,     1,          2.3412
5,         0.10000, 861,     424,     1,     0,     1,          2.3187
5,         0.10000, 862,     424,     2,     0,     0,          0.0000
...
10,        0.20000, 860,     424,     0,     0,     1,          2.3891
```

The `in_polygon` column allows filtering to only pixels inside the exact polygon boundary rather than the bounding box rectangle.

Additional files in the export folder:

| File | Description |
|---|---|
| `meta.csv` | Single-row metadata: fps, frame_step, bbox, n_frames, roi_pixels, fields |
| `mask_binary.png` | Mask as PNG image |
| `mask_overlay.png` | First frame with ROI polygon drawn in green |
| `mask_polygon.txt` | Polygon vertices as plain text |

> CSV files contain the same pixel-level data as npy but are significantly larger on disk and slower to write. For a 254x133 ROI over 339 frames, each field CSV will be approximately 300-500 MB. Use npy for large datasets and CSV when you need compatibility with tools like Excel or MATLAB's `readtable`.

---

### Mask files

Three mask files are generated by both npy and CSV export.

`mask_binary.png` — pure black and white image, same dimensions as the full frame. White pixels (255) are inside the polygon ROI; black pixels (0) are outside.

`mask_overlay.png` — the first grayscale frame with the ROI polygon drawn in green. The polygon interior has a semi-transparent green tint. Vertex positions are numbered. Useful for visually verifying the ROI alignment with the actual specimen.

`mask_polygon.txt` — plain text file listing the polygon vertex coordinates, one per line in `x,y` format. Can be read directly into MATLAB, Excel, or any text parser.

---

## Farneback Tuning

Parameters in `dicUtils/dense.py`:

```python
_FB_DEFAULTS = dict(
    pyr_scale  = 0.5,   # Pyramid downscale per level
    levels     = 3,     # Pyramid depth
    winsize    = 15,    # Neighbourhood size -- analogue of DIC subset size
    iterations = 3,     # Iterations per level
    poly_n     = 5,     # Polynomial fit neighbourhood (5 or 7)
    poly_sigma = 1.2,   # Gaussian weighting sigma (use 1.5 if poly_n=7)
    flags      = 0,
)
```

| Problem | Fix |
|---|---|
| Noisy or speckled field | Increase winsize (try 21 or 25) |
| Missing fine spatial detail | Decrease winsize (try 11) |
| Large inter-frame motion not tracked | Increase levels (try 4 or 5) |
| Slow computation | Decrease levels or iterations |

---

## Math Reference

### Displacement (per frame)

```
u = flow[:,:,0] / frame_step       # pixels/frame  (x)
v = flow[:,:,1] / frame_step       # pixels/frame  (y)
```

### Velocity

```
Vx = u * fps                       # pixels/second
Vy = v * fps
```

### Strain

```
e_xx = du/dx
e_yy = dv/dy
e_xy = 0.5 * (du/dy + dv/dx)
```

### Strain Rate

```
e_xx/dt = e_xx * fps   [px/px/s]
e_yy/dt = e_yy * fps
e_xy/dt = e_xy * fps
```

---

## Tips

- **Texture:** Farneback needs surface texture. Smooth featureless surfaces give unreliable results. Apply a speckle pattern if needed.
- **Frame step:** Start at 1 and increase until fast enough. Too large a step causes tracking failure on fast motion.
- **ROI:** Keep tight around the region of interest. Large ROIs are slower and may include regions with different motion characteristics.
- **16-bit TIFFs:** Supported automatically. Images are downscaled to 8-bit (uint16 >> 8) before processing.
- **FPS accuracy:** Use the exact frame rate from your acquisition software. An incorrect FPS value scales all velocity and strain rate values proportionally.
- **Memory:** The player keeps at most 50 frames in RAM at a time. For sequences of thousands of frames this keeps memory usage flat regardless of sequence length.

---

## License

MIT