# dicUtils/io.py
import glob
import os
import numpy as np
from PIL import Image


def get_frame_paths(folder, extension):
    paths = sorted(
        glob.glob(os.path.join(folder, f"*{extension}"))
    )
    if len(paths) == 0:
        raise RuntimeError("No frames found.")
    return paths


def load_grayscale(path):
    """Load an image as a uint8 numpy array (H, W) in grayscale."""
    img = Image.open(path).convert("L")
    if img is None:
        raise RuntimeError(f"Failed to load {path}")
    return np.array(img, dtype=np.uint8)


def _load_gray(path: str) -> np.ndarray:
    """
    Load a grayscale uint8 numpy frame.
    Handles 16-bit TIFFs by shifting uint16 -> uint8.
    Single source of truth — used by tracking.py, dense.py, roi.py.
    """
    img = Image.open(path)
    if img.mode in ("I;16", "I"):
        arr = np.array(img, dtype=np.uint16)
        return (arr >> 8).astype(np.uint8)
    return np.array(img.convert("L"), dtype=np.uint8)