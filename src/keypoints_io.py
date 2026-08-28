"""
Torch-free CSV helpers for per-video pose outputs.

Per-video `*_keypoints.csv` files are the source of truth. Combined
`keypoints_raw.csv` is optional and should be built with merge_per_video_csvs().
"""
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from config import DATA_DIR

KEYPOINT_COLUMNS = [
    "video", "frame", "track_id",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "keypoint", "x", "y", "confidence",
]
BAT_COLUMNS = [
    "video", "frame", "bat_x1", "bat_y1", "bat_x2", "bat_y2", "bat_conf",
]


def keypoints_csv_path(video_name: str) -> Path:
    return DATA_DIR / f"{Path(video_name).stem}_keypoints.csv"


def is_valid_keypoints_csv(path: Path) -> bool:
    """True if `path` is a readable keypoints CSV with the expected columns.

    Header-only (zero detection rows) is valid — that video was processed and
    simply had no tracks. Empty/truncated/wrong-schema files are not, so resume
    logic will redo them instead of skipping forever.
    """
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        df = pd.read_csv(path, nrows=1)
        return all(c in df.columns for c in KEYPOINT_COLUMNS)
    except (EmptyDataError, ParserError, OSError, ValueError):
        return False


def read_csv_safe(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a CSV, returning an empty frame with `columns` if the file is unusable."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return pd.DataFrame(columns=columns)
        df = pd.read_csv(path)
        if df.empty and not all(c in df.columns for c in columns):
            return pd.DataFrame(columns=columns)
        return df
    except (EmptyDataError, ParserError, OSError, ValueError):
        return pd.DataFrame(columns=columns)


def write_keypoints_csv(path: Path, rows: list[dict]) -> None:
    """Always write the known header so later reads never hit EmptyDataError."""
    pd.DataFrame(rows, columns=KEYPOINT_COLUMNS).to_csv(path, index=False)


def write_bat_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows, columns=BAT_COLUMNS).to_csv(path, index=False)


def merge_per_video_csvs(write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge every per-video CSV in DATA_DIR into keypoints_raw.csv / bat_boxes_raw.csv.

    Skips unreadable files instead of aborting the whole merge.
    """
    kp_paths = sorted(DATA_DIR.glob("*_keypoints.csv"))
    bat_paths = sorted(DATA_DIR.glob("*_bat_boxes.csv"))

    kp_frames = [read_csv_safe(p, KEYPOINT_COLUMNS) for p in kp_paths]
    bat_frames = [read_csv_safe(p, BAT_COLUMNS) for p in bat_paths]
    kp_df = pd.concat(kp_frames, ignore_index=True) if kp_frames else pd.DataFrame(columns=KEYPOINT_COLUMNS)
    bat_df = pd.concat(bat_frames, ignore_index=True) if bat_frames else pd.DataFrame(columns=BAT_COLUMNS)

    if write:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        kp_df.to_csv(DATA_DIR / "keypoints_raw.csv", index=False)
        bat_df.to_csv(DATA_DIR / "bat_boxes_raw.csv", index=False)
        n_videos = kp_df["video"].nunique() if len(kp_df) and "video" in kp_df.columns else 0
        print(f"\nMerged {len(kp_paths)} per-video CSV(s) → keypoints_raw.csv "
              f"({len(kp_df)} rows, {n_videos} videos)")
    return kp_df, bat_df
