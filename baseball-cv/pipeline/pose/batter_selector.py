"""
pipeline/pose/batter_selector.py

Step 5 of the baseball-cv pipeline: select the batter's keypoints from
the raw multi-person pose data.

Input:  keypoints_raw.csv  (video, frame, track_id, bbox_x1, bbox_y1,
                             bbox_x2, bbox_y2, keypoint, x, y, confidence)
Output: keypoints_batter.csv (same columns + segment_id, filtered to the
                               batter track per frame)

Selection logic (validated 2026-06-12 against Chae_friend_Righty_Batting_V2):
- Per frame, the batter is the track with the largest bounding-box area.
  This was 100% accurate across 444 frames / 18 track_ids / up to 3
  concurrent people per frame.
- Confidence score is NOT used for selection -- non-batter tracks can have
  higher avg confidence than the batter (e.g. a sharp, close background
  person vs. a blurred mid-swing batter). Confidence stays in the output
  as a per-keypoint quality field for downstream smoothing only.
- Consecutive frames where the selected track_id stays the same are
  grouped into a segment_id. A new segment starts whenever the selected
  track_id changes -- this naturally splits a video containing multiple
  concatenated swings (e.g. id:1, then id:9, then id:19...) into separate
  batting events with no extra config.

This module is decoupled from bat detection -- it only needs
keypoints_raw.csv, so it can be re-run cheaply without re-running GPU
inference. bat_boxes_raw.csv remains useful downstream (e.g. for
bat-speed metrics) but is no longer part of batter selection.
"""

import pandas as pd


def select_batter(keypoints_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Select the batter's track per frame (largest bbox) and assign a
    segment_id to group consecutive frames belonging to the same swing.

    Parameters
    ----------
    keypoints_raw : pd.DataFrame
        Columns: video, frame, track_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
        keypoint, x, y, confidence

    Returns
    -------
    pd.DataFrame
        Same columns as input, plus `segment_id`, filtered to only the
        selected batter track per frame.
    """
    df = keypoints_raw.copy()
    df["bbox_area"] = (df["bbox_x2"] - df["bbox_x1"]) * (df["bbox_y2"] - df["bbox_y1"])

    out_frames = []

    for video, vdf in df.groupby("video", sort=False):
        # One row per (frame, track_id) with its bbox area, so we don't
        # duplicate across the 17 keypoint rows when comparing sizes.
        per_frame_track = (
            vdf.groupby(["frame", "track_id"])["bbox_area"]
            .first()
            .reset_index()
        )

        # Pick the largest-bbox track_id per frame.
        batter_per_frame = (
            per_frame_track.sort_values("bbox_area", ascending=False)
            .groupby("frame")
            .first()
            .reset_index()[["frame", "track_id"]]
            .sort_values("frame")
            .reset_index(drop=True)
        )

        # Assign segment_id: increments whenever the selected track_id
        # changes from the previous frame. Starts at 0.
        batter_per_frame["segment_id"] = (
            batter_per_frame["track_id"] != batter_per_frame["track_id"].shift()
        ).cumsum() - 1

        # Filter keypoints_raw to only the (frame, track_id) pairs chosen
        # above, then attach segment_id.
        merged = vdf.merge(
            batter_per_frame, on=["frame", "track_id"], how="inner"
        )
        out_frames.append(merged)

    result = pd.concat(out_frames, ignore_index=True)
    return result.drop(columns=["bbox_area"])


def summarize_segments(keypoints_batter: pd.DataFrame) -> pd.DataFrame:
    """
    QA helper: one row per (video, segment_id) showing the track_id,
    frame range, and frame count -- use this to sanity-check that segment
    boundaries line up with actual swings.
    """
    summary = (
        keypoints_batter.groupby(["video", "segment_id"])
        .agg(
            track_id=("track_id", "first"),
            start_frame=("frame", "min"),
            end_frame=("frame", "max"),
            frame_count=("frame", "nunique"),
        )
        .reset_index()
    )
    return summary


if __name__ == "__main__":
    import sys

    in_path = sys.argv[1] if len(sys.argv) > 1 else "keypoints_raw.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "keypoints_batter.csv"

    raw = pd.read_csv(in_path)
    batter = select_batter(raw)
    batter.to_csv(out_path, index=False)

    print(f"Wrote {len(batter)} rows to {out_path}")
    print()
    print("Segment summary:")
    print(summarize_segments(batter).to_string(index=False))
