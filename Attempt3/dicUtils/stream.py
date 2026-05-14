# dicUtils/stream.py
#
# Streaming DIC store — writes results to HDF5 as computed,
# reads back a sliding window on demand (main thread only, no background threads).

import os
import json
import tempfile
import numpy as np

try:
    import h5py
    HAS_H5 = True
except ImportError:
    HAS_H5 = False

WINDOW_SIZE = 50   # frames kept in RAM at once


class StreamingDICStore:
    """
    Writes each computed frame to HDF5 immediately (ComputeThread).
    Player reads frames on demand via get_frame(pos) — main thread only.
    No background threads — avoids all HDF5 concurrency issues.
    """

    def __init__(self, h5_path: str, selected_keys: list,
                 polygon, mask, bbox, frame_step, fps,
                 window_size: int = WINDOW_SIZE,
                 display_keys: list = None):
        """
        selected_keys : keys physically stored in HDF5 (no virtual keys)
        display_keys  : full user selection including virtual keys like quiver
                        if None, defaults to selected_keys
        """
        if not HAS_H5:
            raise RuntimeError("h5py required. Run: pip install h5py")

        self.h5_path        = h5_path
        self.selected_keys  = selected_keys
        self.display_keys   = display_keys if display_keys is not None else list(selected_keys)
        self.polygon        = polygon
        self.mask           = mask
        self.bbox           = bbox
        self.frame_step     = frame_step
        self.fps            = fps
        self.window_size    = window_size

        self._frame_indices = []
        self._total         = 0
        self._vmaxes        = {k: 1e-9 for k in selected_keys}
        self._vmaxes["_quiver_mag"] = 1e-9

        # Simple in-memory window — no threading
        self._cache         = {}          # frame_idx → (bg_np, fields_dict)
        self._cache_start   = -1          # first position in cache
        self._cache_end     = -1          # last position in cache (exclusive)

        # Open HDF5 for writing (ComputeThread only)
        self._write_h5 = h5py.File(h5_path, "w")
        meta = self._write_h5.create_group("metadata")
        meta.attrs["frame_step"]     = frame_step
        meta.attrs["fps"]            = fps
        meta.attrs["selected_keys"]  = json.dumps(selected_keys)
        meta.attrs["bbox"]           = list(bbox)
        roi = self._write_h5.create_group("roi")
        roi.create_dataset("polygon", data=polygon,
                           compression="gzip", compression_opts=6)
        roi.create_dataset("mask",    data=mask,
                           compression="gzip", compression_opts=9)
        self._frames_grp = self._write_h5.create_group("frames")

    # ── Write (ComputeThread only) ────────────────────────────────────────────

    def append(self, frame_idx: int, bg_np: np.ndarray, fields: dict):
        """Write one frame to HDF5. Called from ComputeThread only."""
        fg = self._frames_grp.create_group(f"frame_{frame_idx:06d}")
        fg.attrs["frame_idx"] = frame_idx
        fg.create_dataset("bg", data=bg_np,
                          compression="gzip", compression_opts=4)
        for key, arr in fields.items():
            fg.create_dataset(key, data=arr.astype(np.float32),
                              compression="gzip", compression_opts=4)
        self._write_h5.flush()

        self._frame_indices.append(frame_idx)
        self._total += 1

        for key, arr in fields.items():
            v = float(np.abs(arr).max())
            self._vmaxes[key] = max(self._vmaxes.get(key, 1e-9), v)
        if "vx" in fields and "vy" in fields:
            qv = max(float(np.abs(fields["vx"]).max()),
                     float(np.abs(fields["vy"]).max()))
            self._vmaxes["_quiver_mag"] = max(self._vmaxes["_quiver_mag"], qv)

    def finalize(self):
        """Flush metadata and close the write handle. Call after all appends."""
        meta = self._write_h5["metadata"]
        meta.create_dataset("frame_indices",
                            data=np.array(self._frame_indices, dtype=np.int32))
        meta.attrs["vmaxes"]      = json.dumps(
            {k: float(v) for k, v in self._vmaxes.items()})
        meta.attrs["display_keys"] = json.dumps(self.display_keys)
        self._write_h5.flush()
        self._write_h5.close()
        self._write_h5  = None
        self._frames_grp = None

    # ── Read (main thread only, no concurrency) ───────────────────────────────

    def __len__(self):
        return self._total

    def frame_index_at(self, pos: int) -> int:
        return self._frame_indices[pos]

    def get_frame(self, pos: int):
        """
        Return (frame_idx, bg_np, fields_dict) at list position pos.
        Loads a window from HDF5 if pos is outside the current cache.
        Single-threaded — safe to call from Qt main thread only.
        """
        frame_idx = self._frame_indices[pos]

        if frame_idx in self._cache:
            return (frame_idx, *self._cache[frame_idx])

        # Load window around pos
        half  = self.window_size // 2
        start = max(0, pos - half)
        end   = min(self._total, start + self.window_size)
        start = max(0, end - self.window_size)

        new_cache = {}
        # Open fresh read handle — write handle is already closed after finalize
        with h5py.File(self.h5_path, "r") as f:
            fgrp = f["frames"]
            for p in range(start, end):
                fidx = self._frame_indices[p]
                name = f"frame_{fidx:06d}"
                if name not in fgrp:
                    continue
                fg    = fgrp[name]
                bg_np = fg["bg"][:]
                fields = {k: fg[k][:].copy()
                          for k in self.selected_keys if k in fg}
                new_cache[fidx] = (bg_np, fields)

        self._cache       = new_cache
        self._cache_start = start
        self._cache_end   = end

        if frame_idx in self._cache:
            return (frame_idx, *self._cache[frame_idx])

        raise RuntimeError(f"Frame {frame_idx} not found in HDF5 store.")


def make_temp_h5() -> str:
    """Return a path for the streaming temp HDF5 file."""
    fd, path = tempfile.mkstemp(suffix=".h5", prefix="dic_stream_")
    os.close(fd)
    os.unlink(path)
    return path