"""
Stage 4 — 2D Skeleton Overlay
Reads each video's per-video keypoints CSV, selects the batter, draws the
15-landmark skeleton on every frame, and saves annotated MP4s to DIAGNOSES_DIR
(src/output/videos/).

Landmark set (15): head, neck, L/R shoulder, elbow, wrist,
                   center_hip, L/R hip, knee, ankle.
All bone connections are drawn in red; each joint gets a unique color dot.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import re
import random
import cv2
import numpy as np
import pandas as pd
from config import DATA_DIR, DIAGNOSES_DIR, BATTING_VIDEOS_DIR, CONF_THRESHOLD, list_videos
from src.batter_selection import load_batter_keypoints

# Penn Action clips converted earlier are named NNNN_batting_video.mp4.
# The remaining named clips are the pro slow-motion reference videos.
NUMBERED_RE = re.compile(r"^\d{4}_batting_video$", re.IGNORECASE)

# ── Landmark colors (BGR) ────────────────────────────────────────────────────

LM_COLOR = {
    "head":           ( 80,  80,  80),
    "neck":           (  0, 255,   0),
    "left_shoulder":  (  0, 255, 255),
    "left_elbow":     (255,   0, 255),
    "left_wrist":     (255,   0, 200),
    "right_shoulder": (  0, 200,   0),
    "right_elbow":    (  0, 165, 255),
    "right_wrist":    (255, 255,   0),
    "center_hip":     (  0, 220, 180),
    "left_hip":       (200,   0, 255),
    "right_hip":      (  0, 200, 255),
    "left_knee":      (255, 200,   0),
    "right_knee":     (180,   0, 255),
    "left_ankle":     (  0,  80, 255),
    "right_ankle":    (255,  50, 180),
}
LM_RADIUS = 3
RED = (0, 0, 255)

CONNECTIONS = [
    ("head",          "neck"),
    ("neck",          "left_shoulder"),
    ("neck",          "right_shoulder"),
    ("neck",          "center_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow",    "left_wrist"),
    ("right_shoulder","right_elbow"),
    ("right_elbow",   "right_wrist"),
    ("center_hip",    "left_hip"),
    ("center_hip",    "right_hip"),
    ("left_hip",      "left_knee"),
    ("left_knee",     "left_ankle"),
    ("right_hip",     "right_knee"),
    ("right_knee",    "right_ankle"),
]

DIRECT = {
    "head":           "nose",
    "left_shoulder":  "left_shoulder",
    "left_elbow":     "left_elbow",
    "left_wrist":     "left_wrist",
    "right_shoulder": "right_shoulder",
    "right_elbow":    "right_elbow",
    "right_wrist":    "right_wrist",
    "left_hip":       "left_hip",
    "right_hip":      "right_hip",
    "left_knee":      "left_knee",
    "right_knee":     "right_knee",
    "left_ankle":     "left_ankle",
    "right_ankle":    "right_ankle",
}


def compute_landmarks(coco_joints: dict) -> dict:
    """
    Maps raw COCO keypoints → 15-landmark dict.
    neck = avg(shoulders), center_hip = avg(hips).
    Only includes landmarks above CONF_THRESHOLD.
    """
    lm = {}
    for name, coco in DIRECT.items():
        p = coco_joints.get(coco)
        if p and p[2] >= CONF_THRESHOLD:
            lm[name] = (p[0], p[1])

    def avg2(a, b):
        pa, pb = coco_joints.get(a), coco_joints.get(b)
        a_ok = pa and pa[2] >= CONF_THRESHOLD
        b_ok = pb and pb[2] >= CONF_THRESHOLD
        if a_ok and b_ok:
            return ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
        p = pa if a_ok else (pb if b_ok else None)
        return (p[0], p[1]) if p else None

    neck = avg2("left_shoulder", "right_shoulder")
    if neck:
        lm["neck"] = neck
    hip = avg2("left_hip", "right_hip")
    if hip:
        lm["center_hip"] = hip

    return lm


def draw_skeleton(frame, lm: dict):
    for (a, b) in CONNECTIONS:
        if a in lm and b in lm:
            cv2.line(frame,
                     (int(lm[a][0]), int(lm[a][1])),
                     (int(lm[b][0]), int(lm[b][1])),
                     RED, 2, cv2.LINE_AA)
    for name, (x, y) in lm.items():
        pt = (int(x), int(y))
        cv2.circle(frame, pt, LM_RADIUS,     LM_COLOR[name], -1, cv2.LINE_AA)
        cv2.circle(frame, pt, LM_RADIUS + 1, (0, 0, 0),       1, cv2.LINE_AA)


def draw_hud(frame, video_name, frame_idx, seg_id, total_frames, W):
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (W, 42), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)
    pct = int(frame_idx / max(total_frames - 1, 1) * 100)
    label = (f"{os.path.splitext(video_name)[0]}  |  "
             f"frame {frame_idx}/{total_frames}  |  seg {seg_id}  |  {pct}%")
    cv2.putText(frame, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def process_video(video_name: str, kp_df: pd.DataFrame):
    vkp = kp_df[kp_df["video"] == video_name]
    if len(vkp) == 0:
        print(f"  SKIP {video_name} — no keypoints found")
        return

    frame_joints: dict = {}
    for row in vkp.itertuples(index=False):
        frame_joints.setdefault(row.frame, {})[row.keypoint] = (row.x, row.y, row.confidence)
    frame_seg = vkp.groupby("frame")["segment_id"].first().to_dict()

    vid_path = str(BATTING_VIDEOS_DIR / video_name)
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video file: {vid_path}")

    writer = None
    try:
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  {video_name}  {W}x{H} @ {FPS:.1f}fps  ({total} frames)")

        out_path = str(DIAGNOSES_DIR / f"skeleton_2d_{os.path.splitext(video_name)[0]}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            coco = frame_joints.get(frame_idx, {})
            if coco:
                draw_skeleton(frame, compute_landmarks(coco))
            draw_hud(frame, video_name, frame_idx, frame_seg.get(frame_idx, "?"), total, W)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    print(f"    → {out_path}")


def _keypoints_for(video_name: str) -> pd.DataFrame:
    """
    Load one video's batter keypoints from its per-video CSV and assign segment_ids.

    Reads the per-video CSV rather than the combined keypoints_batter.csv on purpose:
    the full corpus is ~205 MB of CSV, which pandas expands several-fold in memory.
    Per-video loading keeps peak usage flat no matter how many clips are rendered.
    """
    return load_batter_keypoints(video_name)


def sample_videos(n_numbered: int = 10, n_slowmo: int = 3, seed: int = None) -> list:
    """
    Pick a reproducible random sample: `n_numbered` Penn Action clips
    (NNNN_batting_video.*) plus `n_slowmo` pro slow-motion clips.

    Only clips that actually have a per-video keypoints CSV are eligible, so a
    sample never contains a video that pose extraction hasn't processed.
    """
    rng = random.Random(seed)

    available = []
    for v in list_videos(BATTING_VIDEOS_DIR):
        if (DATA_DIR / f"{os.path.splitext(v)[0]}_keypoints.csv").exists():
            available.append(v)

    numbered = [v for v in available if NUMBERED_RE.match(os.path.splitext(v)[0])]
    slowmo   = [v for v in available if v not in numbered]

    picked  = rng.sample(numbered, min(n_numbered, len(numbered)))
    picked += rng.sample(slowmo,   min(n_slowmo,   len(slowmo)))

    print(f"Eligible: {len(numbered)} numbered, {len(slowmo)} slow-motion "
          f"(of {len(available)} with keypoints)")
    if len(numbered) < n_numbered or len(slowmo) < n_slowmo:
        print(f"  NOTE: requested {n_numbered} numbered + {n_slowmo} slow-motion; "
              f"sampling what is available.")
    return picked


def run_visualize_2d(videos: list = None, n_numbered: int = 10, n_slowmo: int = 3,
                     seed: int = None, all_videos: bool = False):
    """
    Render 2D skeleton overlays.

    Default: a random sample of 10 Penn Action clips + 3 slow-motion clips.
    `videos`     — explicit filenames, skips sampling.
    `all_videos` — render every clip that has keypoints (slow; ~187 videos).
    """
    DIAGNOSES_DIR.mkdir(parents=True, exist_ok=True)

    if videos:
        selected = list(videos)
    elif all_videos:
        selected = [v for v in list_videos(BATTING_VIDEOS_DIR)
                    if (DATA_DIR / f"{os.path.splitext(v)[0]}_keypoints.csv").exists()]
    else:
        selected = sample_videos(n_numbered, n_slowmo, seed)

    print(f"\nRendering {len(selected)} video(s):")
    for v in selected:
        print(f"  - {v}")
    print()

    ok, failed = 0, []
    for v in selected:
        try:
            kp = _keypoints_for(v)
            if kp.empty:
                print(f"  SKIP {v} — no keypoints CSV or empty")
                failed.append(v)
                continue
            process_video(v, kp)
            ok += 1
        except Exception as exc:
            print(f"  ERROR on {v}: {type(exc).__name__}: {exc}")
            failed.append(v)

    print(f"\nDone: {ok} rendered, {len(failed)} failed/skipped → {DIAGNOSES_DIR}")
    if failed:
        print("  failed/skipped: " + ", ".join(failed))
    return ok, failed


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Render 2D skeleton overlay videos")
    ap.add_argument("--videos", nargs="*", default=None, help="Explicit video filenames")
    ap.add_argument("--numbered", type=int, default=10, help="How many Penn Action clips to sample")
    ap.add_argument("--slowmo",   type=int, default=3,  help="How many slow-motion clips to sample")
    ap.add_argument("--seed",     type=int, default=None, help="Random seed for a reproducible sample")
    ap.add_argument("--all",      action="store_true", help="Render every video that has keypoints")
    a = ap.parse_args()
    run_visualize_2d(videos=a.videos, n_numbered=a.numbered,
                     n_slowmo=a.slowmo, seed=a.seed, all_videos=a.all)
