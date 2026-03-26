# dicUtils/video_frame_extractor.py
# Extracts frames from video as lossless grayscale images.

import os
import cv2
import pandas as pd
from typing import Optional
from tqdm import tqdm


def get_video_metadata_cv2(video_path: str) -> dict:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps         = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int  = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc      = "".join([chr((fourcc_int >> 8*i) & 0xFF) for i in range(4)])
    cap.release()
    return {
        "video_path": video_path, "fps": fps, "frame_count": frame_count,
        "width": width, "height": height, "codec_fourcc": fourcc,
        "duration_sec": frame_count/fps if fps else None,
        "file_size_bytes": os.path.getsize(video_path),
    }


def extract_frames_dic_lossless_cv2(
    video_path: str,
    out_dir: str = "frames",
    csv_path: str = "frames/metadata.csv",
    image_ext: str = "tiff",
    save_every_n: int = 1,
    start_frame: int = 1,
    end_frame: Optional[int] = None,
    png_compression: int = 9,
    verbose: bool = True,
) -> pd.DataFrame:

    if save_every_n < 1:
        raise ValueError("save_every_n must be >= 1")
    image_ext = image_ext.lower()
    if image_ext not in ["png", "tif", "tiff"]:
        raise ValueError("Use lossless formats only: png / tif / tiff")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    meta         = get_video_metadata_cv2(video_path)
    fps          = meta["fps"]
    total_frames = meta["frame_count"]
    end_frame    = end_frame or total_frames

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

    rows = []; saved_count = 0

    for frame_offset in tqdm(range(end_frame-start_frame+1),
                             desc="Extracting", disable=not verbose):
        ret, frame = cap.read()
        if not ret: break
        current = start_frame + frame_offset
        if (frame_offset % save_every_n) != 0:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        name = f"frame_{current:06d}.{image_ext}"
        path = os.path.join(out_dir, name)
        if image_ext == "png":
            cv2.imwrite(path, gray, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
        else:
            cv2.imwrite(path, gray)
        saved_count += 1
        ts = (current-1)/fps if fps else None
        rows.append({
            "original_frame_index": current, "frame_file": name,
            "frame_path": path, "timestamp_sec": ts, "fps": fps,
            "video_width": meta["width"], "video_height": meta["height"],
            "video_frame_count": meta["frame_count"],
            "video_duration_sec": meta["duration_sec"],
            "video_codec_fourcc": meta["codec_fourcc"],
            "video_file_size_bytes": meta["file_size_bytes"],
            "video_path": meta["video_path"],
        })

    cap.release()
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"\nExtracted {saved_count} frames → {out_dir}/")
    return df