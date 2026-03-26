# main.py — DIC application entry point

import torch
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt

from dicUtils.config import load_config
from dicUtils.settings import DICSettings
from dicUtils.launcher import LauncherWidget
from dicUtils.io import get_frame_paths, _load_gray
from dicUtils.roi import select_roi
from dicUtils.dense import run_dense_dic, DICPlayerWindow
from dicUtils.stream import StreamingDICStore, WINDOW_SIZE
from dicUtils.storage import load_results

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)


def _make_store_from_h5(h5_path, results, selected_keys,
                        polygon, mask, bbox, frame_step, fps,
                        display_keys=None):
    """Wrap a loaded .h5 in a StreamingDICStore — reads on demand."""
    import numpy as np

    store = object.__new__(StreamingDICStore)
    store.h5_path        = h5_path
    store.selected_keys  = selected_keys
    store.display_keys   = display_keys if display_keys is not None else list(selected_keys)
    store.polygon        = polygon
    store.mask           = mask
    store.bbox           = bbox
    store.frame_step     = frame_step
    store.fps            = fps
    store.window_size    = WINDOW_SIZE
    store._frame_indices = [r[0] for r in results]
    store._total         = len(results)
    store._cache         = {}
    store._cache_window  = (-1, -1)
    store._cache         = {}
    store._cache_start   = -1
    store._cache_end     = -1
    store._write_h5      = None   # already finalized
    store._frames_grp    = None

    store._vmaxes = {k: 1e-9 for k in selected_keys}
    store._vmaxes["_quiver_mag"] = 1e-9
    for _, _, fields in results:
        for k, arr in fields.items():
            v = float(np.abs(arr).max())
            store._vmaxes[k] = max(store._vmaxes.get(k, 1e-9), v)
        if "vx" in fields and "vy" in fields:
            qv = max(float(np.abs(fields["vx"]).max()),
                     float(np.abs(fields["vy"]).max()))
            store._vmaxes["_quiver_mag"] = max(
                store._vmaxes["_quiver_mag"], qv)
    return store


def main():
    # Single QApplication — created once, lives for the whole session
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

    # Keep references so windows aren't garbage collected
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
            results, selected_keys, display_keys, polygon, mask, bbox, \
                fs, fps, source = load_results(h5_path)

            store = _make_store_from_h5(
                h5_path, results, selected_keys,
                polygon, mask, bbox, fs, fps,
                display_keys=display_keys)
            del results   # free summary list

            player = DICPlayerWindow(
                store, display_keys, polygon, mask, bbox,
                fps, fs, settings, source_info=source,
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

    # Single event loop — never call app.exec_() anywhere else
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()