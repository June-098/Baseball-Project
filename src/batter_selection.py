"""
Stage 2 — Batter Selection
Reads per-video `*_keypoints.csv` (or combined keypoints_raw.csv), picks the
largest-bbox track per frame as the batter, assigns segment_ids for multi-swing
clips, exports keypoints_batter.csv.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from config import DATA_DIR
from src.keypoints_io import keypoints_csv_path, merge_per_video_csvs, read_csv_safe, KEYPOINT_COLUMNS

# Split a new segment when the batter disappears for this many frames, even if
# ByteTrack later reuses the same track_id (two swings in one file).
SEGMENT_GAP_FRAMES = 15


def _debounce_track_ids(ids: np.ndarray) -> np.ndarray:
    """Replace 1-frame (and 2-frame) ID blips with the surrounding ID."""
    smoothed = ids.copy()
    n = len(ids)
    for i in range(1, n - 1):
        if smoothed[i] != smoothed[i - 1] and smoothed[i - 1] == smoothed[i + 1]:
            smoothed[i] = smoothed[i - 1]
    for i in range(1, n - 2):
        if (
            smoothed[i] != smoothed[i - 1]
            and smoothed[i] == smoothed[i + 1]
            and smoothed[i - 1] == smoothed[i + 2]
        ):
            smoothed[i] = smoothed[i - 1]
            smoothed[i + 1] = smoothed[i - 1]
    return smoothed


def _assign_segment_ids(frames: np.ndarray, track_ids: np.ndarray) -> np.ndarray:
    """
    Segment on a sustained track_id change, or a gap in frame numbers.

    A one-frame tracker swap no longer splits a swing. A long hole in detections
    (new pitch in a concatenated file) starts a new segment even if the ID matches.
    """
    if len(frames) == 0:
        return np.array([], dtype=int)
    ids = _debounce_track_ids(track_ids)
    seg = np.zeros(len(frames), dtype=int)
    for i in range(1, len(frames)):
        id_changed = ids[i] != ids[i - 1]
        gap = int(frames[i]) - int(frames[i - 1]) > SEGMENT_GAP_FRAMES
        seg[i] = seg[i - 1] + (1 if id_changed or gap else 0)
    return seg


def select_batter(keypoints_raw: pd.DataFrame) -> pd.DataFrame:
    df = keypoints_raw.copy()
    if df.empty:
        return df
    df["bbox_area"] = (df["bbox_x2"] - df["bbox_x1"]) * (df["bbox_y2"] - df["bbox_y1"])

    out_frames = []
    for video, vdf in df.groupby("video", sort=False):
        per_frame_track = (
            vdf.groupby(["frame", "track_id"])["bbox_area"]
            .first()
            .reset_index()
        )
        batter_per_frame = (
            per_frame_track.sort_values("bbox_area", ascending=False)
            .groupby("frame")
            .first()
            .reset_index()[["frame", "track_id"]]
            .sort_values("frame")
            .reset_index(drop=True)
        )
        n_people = per_frame_track.groupby("frame")["track_id"].nunique()
        multi = int((n_people > 1).sum())
        if multi:
            print(f"  {video}: {multi} frame(s) with multiple people — picked largest bbox")

        batter_per_frame["segment_id"] = _assign_segment_ids(
            batter_per_frame["frame"].to_numpy(),
            batter_per_frame["track_id"].to_numpy(),
        )

        merged = vdf.merge(batter_per_frame, on=["frame", "track_id"], how="inner")
        out_frames.append(merged)

    if not out_frames:
        return df.drop(columns=["bbox_area"])
    result = pd.concat(out_frames, ignore_index=True)
    return result.drop(columns=["bbox_area"])


def load_batter_keypoints(video_name: str) -> pd.DataFrame:
    """
    Load one video's batter keypoints from its per-video CSV and assign segment_ids.

    Returns an empty DataFrame if the file is missing, empty, or unreadable.
    """
    path = keypoints_csv_path(video_name)
    if not path.exists():
        return pd.DataFrame()
    try:
        raw = pd.read_csv(path)
    except (EmptyDataError, ParserError, OSError, ValueError):
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    return select_batter(raw)


def summarize_segments(keypoints_batter: pd.DataFrame) -> pd.DataFrame:
    return (
        keypoints_batter.groupby(["video", "segment_id"])
        .agg(
            track_id=("track_id", "first"),
            start_frame=("frame", "min"),
            end_frame=("frame", "max"),
            frame_count=("frame", "nunique"),
        )
        .reset_index()
    )


def run_batter_selection() -> pd.DataFrame:
    raw_path = DATA_DIR / "keypoints_raw.csv"
    if not raw_path.exists():
        print(f"{raw_path.name} missing — merging per-video CSVs first")
        merge_per_video_csvs()

    kp_df = read_csv_safe(raw_path, KEYPOINT_COLUMNS)
    if kp_df.empty:
        # Batched runs may have per-video files but a missing/empty combined file.
        print("Combined keypoints_raw.csv is empty — merging per-video CSVs")
        kp_df, _ = merge_per_video_csvs()
    print(f"Loaded {len(kp_df)} rows from keypoints_raw.csv")

    batter_df = select_batter(kp_df)
    if not batter_df.empty:
        print(summarize_segments(batter_df).to_string())

    out_path = DATA_DIR / "keypoints_batter.csv"
    batter_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(batter_df)} rows → {out_path}")
    return batter_df


if __name__ == "__main__":
    run_batter_selection()
