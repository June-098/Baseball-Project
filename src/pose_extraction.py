"""
Stage 1 — Pose Extraction (+ optional Bat Detection)
Runs YOLO pose model with ByteTrack on all videos in Batting Videos/, exporting:
  <DATA_DIR>/keypoints_raw.csv   — all person keypoints, all tracks
  <DATA_DIR>/bat_boxes_raw.csv   — highest-confidence bat box per frame (only if enabled)
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Bat detection is disabled for now — it doubles model load + per-frame inference
# cost for a signal we're not using yet. Flip to True to bring it back; no other
# code needs to change since _extract_video branches on this flag.
ENABLE_BAT_DETECTION = False

# PyTorch 2.12 on Windows fails to load c10.dll (WinError 1114) because it doesn't
# add its own lib dir to the DLL search path first. Fix: add it via PATH + add_dll_directory,
# then preload c10.dll so it's already in memory when torch's loader runs.
if sys.platform == "win32":
    import ctypes
    _torch_lib = Path(sys.executable).parent / "Lib" / "site-packages" / "torch" / "lib"
    if _torch_lib.is_dir():
        os.environ["PATH"] = str(_torch_lib) + ";" + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(_torch_lib))
        try:
            ctypes.CDLL(str(_torch_lib / "c10.dll"))
        except OSError:
            pass

from itertools import zip_longest

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from config import BATTING_VIDEOS_DIR, DATA_DIR, KEYPOINT_NAMES, list_videos
from src.keypoints_io import (
    KEYPOINT_COLUMNS,
    is_valid_keypoints_csv,
    merge_per_video_csvs,
    read_csv_safe,
    write_bat_csv,
    write_keypoints_csv,
)


def _select_device() -> str:
    """Pick the fastest available backend: CUDA > Apple Silicon MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _validate_paths() -> None:
    if not BATTING_VIDEOS_DIR.exists():
        raise RuntimeError(
            f"Batting Videos folder not found: {BATTING_VIDEOS_DIR}\n"
            "Make sure Google Drive is mounted and the path in config.py is correct."
        )
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _extract_video(
    vid: str,
    pose_model: YOLO,
    bat_model: "YOLO | None",
    bat_class_id: "int | None",
    device: str,
) -> tuple[list[dict], list[dict]]:
    video_path = str(BATTING_VIDEOS_DIR / vid)
    print(f"\nProcessing: {video_path}")

    # stream=True is the key RAM fix: without it, Ultralytics runs inference on
    # every frame of the video up front and holds the full list of Results
    # (each carrying a copy of the original frame) in memory simultaneously —
    # for a ~1000-frame clip that's several GB. With stream=True the call
    # returns a generator instead, so only one frame's Results is ever
    # resident at a time, no matter how long the video or how many videos are
    # processed in the run. imgsz caps the side Ultralytics resizes to for
    # inference — the internal working tensor stays small even on 4K/slow-mo
    # source footage (see the efficiency notes below for why this matters).
    # persist=False starts a fresh ByteTrack state for this clip. persist=True
    # reuses the previous video's tracks/IDs, which bleeds associations across files.
    # Within this one .track() call the tracker still persists across frames.
    pose_results = pose_model.track(
        video_path, save=False, show=False,
        persist=False, tracker="bytetrack.yaml",
        stream=True, device=device, imgsz=640, verbose=False,
    )

    keypoint_rows: list[dict] = []
    bat_rows: list[dict] = []
    untracked_frames = 0
    pose_only_tail = 0

    if ENABLE_BAT_DETECTION:
        bat_results = bat_model.predict(
            video_path, save=False, show=False,
            classes=[bat_class_id],
            stream=True, device=device, imgsz=640, verbose=False,
        )
        # zip_longest: a shorter bat stream must not silently drop remaining pose frames.
        frame_iter = enumerate(zip_longest(pose_results, bat_results, fillvalue=None))
    else:
        frame_iter = ((frame_idx, (pr, None)) for frame_idx, pr in enumerate(pose_results))

    for frame_idx, (pr, br) in frame_iter:
        if pr is None:
            pose_only_tail += 1
            continue

        # --- keypoints ---
        if pr.keypoints is not None and pr.boxes is not None and len(pr.boxes) > 0:
            kps       = pr.keypoints.xy.cpu().numpy()        # (N, 17, 2)
            raw_conf  = pr.keypoints.conf                     # may be None
            confs     = raw_conf.cpu().numpy() if raw_conf is not None else np.zeros(kps.shape[:2])
            boxes     = pr.boxes.xyxy.cpu().numpy()
            n = min(len(boxes), kps.shape[0])
            if pr.boxes.id is not None:
                track_ids = pr.boxes.id.cpu().numpy().astype(int)
            else:
                # ByteTrack sometimes has no IDs on the first frames; still emit rows
                # so this frame isn't a hole in the time series.
                track_ids = np.arange(n)
                untracked_frames += 1

            for i in range(n):
                track_id = int(track_ids[i]) if i < len(track_ids) else i
                x1, y1, x2, y2 = boxes[i]
                for kp_idx, kp_name in enumerate(KEYPOINT_NAMES):
                    x, y = kps[i, kp_idx]
                    keypoint_rows.append({
                        "video":      vid,
                        "frame":      frame_idx,
                        "track_id":   track_id,
                        "bbox_x1":    float(x1),
                        "bbox_y1":    float(y1),
                        "bbox_x2":    float(x2),
                        "bbox_y2":    float(y2),
                        "keypoint":   kp_name,
                        "x":          float(x),
                        "y":          float(y),
                        "confidence": float(confs[i, kp_idx]),
                    })

        # --- bat box (only populated when ENABLE_BAT_DETECTION is True) ---
        if br is not None and br.boxes is not None and len(br.boxes) > 0:
            best = int(br.boxes.conf.argmax().item())
            bx1, by1, bx2, by2 = br.boxes.xyxy.cpu().numpy()[best]
            bat_rows.append({
                "video":    vid,
                "frame":    frame_idx,
                "bat_x1":  float(bx1),
                "bat_y1":  float(by1),
                "bat_x2":  float(bx2),
                "bat_y2":  float(by2),
                "bat_conf": float(br.boxes.conf.cpu().numpy()[best]),
            })

    if untracked_frames:
        print(f"  NOTE: {untracked_frames} frame(s) had detections but no ByteTrack IDs "
              f"— used ephemeral ids so the frame isn't dropped")
    if pose_only_tail:
        print(f"  NOTE: bat stream ended {pose_only_tail} frame(s) before pose "
              f"(pose frames kept, no bat boxes on the tail)")

    return keypoint_rows, bat_rows


def load_video_keypoints(video_name: str) -> pd.DataFrame:
    """
    Load one video's keypoints from its per-video CSV — the memory-safe way to get
    keypoints for a specific clip without touching the multi-hundred-MB merge.
    """
    path = DATA_DIR / f"{Path(video_name).stem}_keypoints.csv"
    if not path.exists():
        raise FileNotFoundError(f"No keypoints CSV for {video_name} (expected {path})")
    return read_csv_safe(path, KEYPOINT_COLUMNS)


def run_pose_extraction(start: int = None, end: int = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    start/end slice into the sorted video list (end exclusive), so a driver
    script can process the folder in fixed-size batches, each in its own
    subprocess. Left as None, this processes every video in one run.

    NOTE: when a slice is given, the per-video CSVs are the real output. The
    combined keypoints_raw.csv is NOT written — see the comment at the end of
    this function for why.
    """
    _validate_paths()

    # Sorted so the same start/end indices always refer to the same videos,
    # no matter which process (or how many separate subprocess calls) lists
    # the directory.
    all_video_files = list_videos(BATTING_VIDEOS_DIR)
    if not all_video_files:
        raise RuntimeError(f"No video files found in {BATTING_VIDEOS_DIR}")

    video_files = all_video_files[start:end]
    if not video_files:
        raise RuntimeError(f"No videos in slice [{start}:{end}] of {len(all_video_files)} found")
    print(f"Found {len(all_video_files)} video(s) total; processing slice "
          f"[{start or 0}:{end if end is not None else len(all_video_files)}] "
          f"= {len(video_files)} video(s): {video_files}")

    device = _select_device()
    print(f"Using device: {device}")

    pose_model = YOLO("yolo26m-pose.pt")
    bat_model, bat_class_id = None, None
    if ENABLE_BAT_DETECTION:
        bat_model    = YOLO("yolo26m.pt")
        bat_class_id = next(k for k, v in bat_model.names.items() if v == "baseball bat")
        print(f"Bat class id: {bat_class_id}")
    else:
        print("Bat detection disabled (ENABLE_BAT_DETECTION = False) — pose only.")

    # Per-video CSVs are written as we go (see below) and re-read here at the end
    # to build the combined files. This keeps peak memory bounded to a single
    # video's rows at any one time — important now that Batting Videos/ holds
    # 180+ clips instead of a handful, since a single in-memory list spanning
    # every video would grow unbounded exactly like the frame-buffering issue
    # stream=True fixes above.
    kp_paths:  list[Path] = []
    bat_paths: list[Path] = []
    failed: list[str] = []

    for vid in video_files:
        vid_stem = Path(vid).stem
        kp_path  = DATA_DIR / f"{vid_stem}_keypoints.csv"
        bat_path = DATA_DIR / f"{vid_stem}_bat_boxes.csv"

        # Resume only on a readable keypoints CSV with the expected header.
        # Empty/truncated files from a crash or DataFrame([]).to_csv must be redone.
        if is_valid_keypoints_csv(kp_path):
            print(f"  SKIP {vid} — {kp_path.name} already exists")
            kp_paths.append(kp_path)
            if bat_path.exists():
                bat_paths.append(bat_path)
            continue

        try:
            kp_rows, bat_rows = _extract_video(vid, pose_model, bat_model, bat_class_id, device)
        except Exception as exc:
            print(f"  ERROR on {vid}: {exc} — skipping")
            failed.append(vid)
            continue

        # Header is always written (even for zero detections) so resume/merge
        # never hit pandas EmptyDataError on this file.
        write_keypoints_csv(kp_path, kp_rows)
        kp_paths.append(kp_path)
        if ENABLE_BAT_DETECTION:
            write_bat_csv(bat_path, bat_rows)
            bat_paths.append(bat_path)
        print(f"  Saved {len(kp_rows)} keypoint rows and {len(bat_rows)} bat rows for {vid}")

        # Release cached activation memory back to the allocator between videos.
        # Without this, CUDA/MPS caching allocators can hold onto peak-usage
        # memory for the life of the process even after Python's GC has
        # collected the tensors, so usage creeps upward over 183 videos.
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    # ── Do NOT write the combined CSV when running a slice ────────────────────
    #
    # BUG THIS FIXES: every batch runs in its own subprocess, and each one used to
    # write keypoints_raw.csv from *only its own slice*. Batch 2 clobbered batch 1,
    # batch 3 clobbered batch 2, and so on — so after a 10-batch run the "combined"
    # file contained just the final 3 videos, and the viz stage produced 3 skeleton
    # videos instead of 183. The per-video CSVs were always complete; only the
    # merged view was wrong.
    #
    # The per-video CSVs are the source of truth. Merge them explicitly via
    # merge_per_video_csvs() when a combined file is actually needed.
    is_slice = start is not None or end is not None
    if failed:
        print(f"\nPose extraction failed on {len(failed)} video(s): {failed}")

    if is_slice:
        print(f"\nSlice complete — wrote {len(kp_paths)} per-video CSV(s) to {DATA_DIR}")
        print("Combined keypoints_raw.csv NOT written (a slice must never overwrite the full merge).")
        if failed:
            raise RuntimeError(f"{len(failed)} video(s) failed pose extraction")
        return pd.DataFrame(), pd.DataFrame()

    merged = merge_per_video_csvs()
    if failed:
        raise RuntimeError(f"{len(failed)} video(s) failed pose extraction")
    return merged


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pose extraction (optionally a batch slice of the video list)")
    parser.add_argument("--start", type=int, default=None, help="Start index (inclusive) into the sorted video list")
    parser.add_argument("--end",   type=int, default=None, help="End index (exclusive) into the sorted video list")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip extraction; just merge existing per-video CSVs into keypoints_raw.csv")
    args = parser.parse_args()
    if args.merge_only:
        merge_per_video_csvs()
    else:
        run_pose_extraction(start=args.start, end=args.end)
