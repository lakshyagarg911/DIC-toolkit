# main.py — DIC application entry point
#
# IMPORTANT: torch must be imported before PyQt5 on Windows to avoid
# CUDA DLL initialisation conflicts. Do not reorder these imports.

import sys
import os
import json

# 1. Torch first — initialises CUDA DLLs before Qt touches them
try:
    import torch
    if torch.cuda.is_available():
        _ = torch.zeros(1).cuda()   # force CUDA context creation now
        print(f"[startup] CUDA ready: {torch.cuda.get_device_name(0)}")
    else:
        print("[startup] CUDA not available — DIC will run on CPU")
except ImportError:
    print("[startup] PyTorch not installed — DIC engine unavailable")

# 2. PyQt5 after torch
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt

from dicUtils.config import load_config
from dicUtils.settings import DICSettings
from dicUtils.launcher import LauncherWidget
from dicUtils.io import get_frame_paths, _load_gray
from dicUtils.roi import select_roi
from dicUtils.dense import run_dense_dic, DICPlayerWindow
from dicUtils.stream import StreamingDICStore, WINDOW_SIZE

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)


def _open_store_from_h5(h5_path: str) -> StreamingDICStore:
    """
    Open an existing HDF5 file as a StreamingDICStore.
    Reads metadata only — frame data is loaded on demand by get_frame().
    Never loads all results into RAM.
    """
    try:
        import h5py
    except ImportError:
        raise RuntimeError("h5py not installed. Run: pip install h5py")

    import numpy as np

    with h5py.File(h5_path, "r") as f:
        meta          = f["metadata"]
        frame_step    = int(meta.attrs["frame_step"])
        fps           = float(meta.attrs["fps"])
        selected_keys = json.loads(meta.attrs["selected_keys"])
        display_keys  = json.loads(
            meta.attrs.get("display_keys", meta.attrs["selected_keys"]))
        bbox          = tuple(int(v) for v in meta.attrs["bbox"])
        source_info   = str(meta.attrs.get("source", ""))
        frame_indices = list(f["metadata/frame_indices"][:])

        polygon = f["roi/polygon"][:]
        mask    = f["roi/mask"][:]

        # Build vmaxes by scanning all frame datasets (metadata only pass)
        vmaxes = {k: 1e-9 for k in selected_keys}
        vmaxes["_quiver_mag"] = 1e-9
        fgrp = f["frames"]
        for name in sorted(fgrp.keys()):
            fg = fgrp[name]
            for k in selected_keys:
                if k in fg:
                    v = float(np.abs(fg[k][:]).max())
                    vmaxes[k] = max(vmaxes.get(k, 1e-9), v)
            if "vx" in fg and "vy" in fg:
                qv = max(float(np.abs(fg["vx"][:]).max()),
                         float(np.abs(fg["vy"][:]).max()))
                vmaxes["_quiver_mag"] = max(vmaxes["_quiver_mag"], qv)

    # Build store without calling __init__ (file already exists)
    store = object.__new__(StreamingDICStore)
    store.h5_path        = h5_path
    store.selected_keys  = selected_keys
    store.display_keys   = display_keys
    store.polygon        = polygon
    store.mask           = mask
    store.bbox           = bbox
    store.frame_step     = frame_step
    store.fps            = fps
    store.window_size    = WINDOW_SIZE
    store._frame_indices = list(frame_indices)
    store._total         = len(frame_indices)
    store._vmaxes        = vmaxes
    store._cache         = {}
    store._cache_start   = -1
    store._cache_end     = -1
    store._write_h5      = None
    store._frames_grp    = None

    print(f"Opened h5 store: {len(frame_indices)} frames")
    print(f"  stored fields:  {selected_keys}")
    print(f"  display fields: {display_keys}")
    return store, source_info


def main():
    app = QApplication(sys.argv)

    try:
        cfg = load_config()
    except Exception:
        cfg = {}

    frame_step = cfg.get("frame_step", 5)
    settings   = DICSettings.load()

    # ── Launcher window ───────────────────────────────────────────────────────
    win = QMainWindow()
    win.setWindowTitle("Dense DIC — Launcher")
    win.resize(840, 560)
    win.setStyleSheet("QMainWindow { background: #0D1117; }")

    launcher = LauncherWidget(settings)
    win.setCentralWidget(launcher)

    _active_players = []

    def on_launch_folder(folder, fps, ext):
        win.hide()
        try:
            print(f"Scanning folder: {folder}")
            frame_paths = get_frame_paths(folder, ext)
            print(f"Found {len(frame_paths)} frames  |  FPS={fps}  |  step={frame_step}")

            print("Loading first frame...")
            first_frame = _load_gray(frame_paths[0])
            print(f"  Shape: {first_frame.shape}  dtype: {first_frame.dtype}")

            print("Opening ROI editor...")
            polygon, mask, bbox = select_roi(first_frame)
            print(f"  ROI confirmed — bbox={bbox}  vertices={len(polygon)}")

            print("Starting DIC pipeline...")
            player = run_dense_dic(
                frame_paths, polygon, mask, bbox,
                frame_step  = frame_step,
                fps         = fps,
                settings    = settings,
                source_info = folder,
                fb_params   = cfg,   # pass full config for DIC params
            )
            if player is not None:
                _active_players.append(player)
                player.destroyed.connect(lambda: win.show())
            else:
                win.show()

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\nERROR: {e}")
            win.show()

    def on_launch_saved(h5_path):
        win.hide()
        try:
            print(f"Loading saved data: {h5_path}")
            store, source = _open_store_from_h5(h5_path)

            player = DICPlayerWindow(
                store, store.display_keys,
                store.polygon, store.mask, store.bbox,
                store.fps, store.frame_step,
                settings, source_info=source,
            )
            _active_players.append(player)
            player.destroyed.connect(lambda: win.show())
            player.show()

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\nERROR: {e}")
            win.show()

    launcher.launch_folder.connect(on_launch_folder)
    launcher.launch_saved.connect(on_launch_saved)

    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()