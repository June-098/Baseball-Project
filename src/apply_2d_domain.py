"""
Stage 5 — Apply the 2D Domain (Biomechanical Overlay)
=====================================================
Runs AFTER MotionBERT in the pipeline. This is the answer the agent team
(MEDA -> APA -> APE) arrived at for "how to effectively apply the 2D domain in
the video": instead of a plain 15-dot skeleton, we draw the *swing-relevant 2D
angles* directly on the footage and report them as live, color-coded numbers.

What it draws/computes per frame (all from the 2D keypoints in
each video's `*_keypoints.csv` after batter selection — no 3D, no bat
tracking required):

  1. Hip-shoulder separation  — angle between the shoulder line and the hip line.
     2D PROXY for rotational separation (true value needs 3D). Good ~25-45 deg.
  2. Spine tilt               — lean of the torso (center_hip -> neck) from vertical.
  3. Front-knee angle         — interior angle at the front knee (180 deg = straight).
     Front leg is chosen from handedness (righty -> left leg leads, lefty -> right).
  4. Wrist-path / attack angle — direction the hands are travelling, smoothed.
     This is the body-keypoint PROXY for MLB "attack angle" (ideal 5-20 deg).
     The headline value is taken at the estimated contact frame (peak hand speed).

IMPORTANT CAVEAT: every angle here is measured in the *image plane*, so it is
view-dependent — a true biomechanical reading needs the 3D pose (MotionBERT).
These are honest 2D proxies, which is exactly what Baseball Savant calls the
"approximable from body keypoints" tier.

Outputs (local layout under src/output/):
  videos/skeleton_2d_biomech_<video>.mp4  — annotated video, one per clip (DIAGNOSES_DIR)
  2d_metrics.json                         — per-segment metrics (OUTPUT_DIR = src/output)
"""
import sys
import os
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from config import DATA_DIR, BATTING_VIDEOS_DIR, OUTPUT_DIR, DIAGNOSES_DIR, CONF_THRESHOLD, list_videos
from src.visualize_2d import compute_landmarks, draw_skeleton
from src.batter_selection import load_batter_keypoints

# ── Ideal ranges (from 2026-06-24-baseball-savant-batting-parameters.md) ──────
IDEAL_ATTACK_ANGLE = (5.0, 20.0)     # MLB ideal attack angle band
GOOD_HIP_SHOULDER_SEP = 25.0         # elite hitters show 30-45 deg; >=25 is good

MIN_SEG_FRAMES = 20                  # below this, skip contact estimation (spurious tracks)
TRAIL_LEN = 10                       # frames of hand-path trail to draw
SMOOTH_WIN = 5                       # moving-average window for the hand path
MAX_VEL_FRAME_GAP = 2                # ignore velocity across holes larger than this (frames)

# Colors (BGR)
C_GREEN  = (0, 200,   0)
C_RED    = (0,   0, 255)
C_ORANGE = (0, 165, 255)
C_CYAN   = (255, 255, 0)
C_MAG    = (255, 0, 255)
C_YELLOW = (0, 255, 255)
C_WHITE  = (255, 255, 255)


# ── Geometry helpers ─────────────────────────────────────────────────────────
def _line_angle_deg(p, q):
    """Orientation of the line p->q in degrees, image coords (y down)."""
    return math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))


def _interior_angle(a, b, c):
    """Interior angle at vertex b formed by points a-b-c, in degrees [0,180]."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return None
    cosang = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def _handedness_from_filename(video_name):
    """Explicit filename tags only — generic 'left'/'right' substrings are too noisy."""
    n = video_name.lower()
    if "lefty" in n:
        return "lefty"
    if "righty" in n:
        return "righty"
    if "ichiro" in n:
        return "lefty"
    if "bautist" in n:
        return "righty"
    return None


def _handedness_from_pose(lm_by_frame):
    """
    Infer stance from which ankle travels farther (the stride / lead foot).
    Righty → left foot strides; lefty → right foot strides.
    Returns None when the signal is too weak to trust.
    """
    left_xs, right_xs = [], []
    for lm in lm_by_frame.values():
        if "left_ankle" in lm:
            left_xs.append(lm["left_ankle"][0])
        if "right_ankle" in lm:
            right_xs.append(lm["right_ankle"][0])
    if len(left_xs) < 5 or len(right_xs) < 5:
        return None
    left_travel = max(left_xs) - min(left_xs)
    right_travel = max(right_xs) - min(right_xs)
    if max(left_travel, right_travel) < 8:
        return None
    if left_travel > right_travel * 1.15:
        return "righty"
    if right_travel > left_travel * 1.15:
        return "lefty"
    return None


def _handedness(video_name, lm_by_frame=None):
    """
    Returns (handedness, source) where source is filename | pose | default.
    Filename tags win; otherwise stride-based pose; otherwise righty with a log.
    """
    tagged = _handedness_from_filename(video_name)
    if tagged:
        return tagged, "filename"
    if lm_by_frame:
        inferred = _handedness_from_pose(lm_by_frame)
        if inferred:
            return inferred, "pose"
    print(f"  NOTE: handedness unknown for {video_name}; defaulting to righty")
    return "righty", "default"


# ── Per-frame metrics from one frame's 15-landmark dict ──────────────────────
def frame_metrics(lm, handedness):
    """Returns dict of the static (single-frame) 2D metrics, None where missing."""
    m = {"hip_shoulder_sep": None, "spine_tilt": None, "front_knee_angle": None}

    # 1. Hip-shoulder separation (undirected line orientations, mapped to [0,90])
    if all(k in lm for k in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")):
        sh = _line_angle_deg(lm["left_shoulder"], lm["right_shoulder"])
        hp = _line_angle_deg(lm["left_hip"], lm["right_hip"])
        d = abs(sh - hp) % 180
        m["hip_shoulder_sep"] = round(min(d, 180 - d), 1)

    # 2. Spine tilt from vertical (image-up = (0,-1))
    if "neck" in lm and "center_hip" in lm:
        spine = (lm["neck"][0] - lm["center_hip"][0], lm["neck"][1] - lm["center_hip"][1])
        n = math.hypot(*spine)
        if n > 0:
            # angle to vertical-up vector (0,-1)
            cosang = (-spine[1]) / n
            cosang = max(-1.0, min(1.0, cosang))
            m["spine_tilt"] = round(math.degrees(math.acos(cosang)), 1)

    # 3. Front-knee interior angle
    side = "left" if handedness == "righty" else "right"
    hip, knee, ankle = f"{side}_hip", f"{side}_knee", f"{side}_ankle"
    if all(k in lm for k in (hip, knee, ankle)):
        ang = _interior_angle(lm[hip], lm[knee], lm[ankle])
        if ang is not None:
            m["front_knee_angle"] = round(ang, 1)

    return m


def _hands_point(coco):
    """Average of the two wrists (whichever passes confidence). None if neither."""
    pts = []
    for w in ("left_wrist", "right_wrist"):
        p = coco.get(w)
        if p and p[2] >= CONF_THRESHOLD:
            pts.append((p[0], p[1]))
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _moving_average(arr, win):
    """NaN-aware centered moving average over axis 0 of an (N,2) array."""
    out = arr.copy()
    half = win // 2
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        chunk = arr[lo:hi]
        valid = chunk[~np.isnan(chunk[:, 0])]
        if len(valid):
            out[i] = valid.mean(axis=0)
    return out


def wrist_path_angles(frames, hands_by_frame):
    """
    Build the per-frame attack-angle proxy (deg, +=upward) and hand speed.
    Returns dict frame -> {"attack_angle":..., "speed":...} and the contact frame.

    Attack angle is the path's inclination to horizontal in [-90, 90], independent
    of whether the hands travel left or right in the image (atan2(-dy, abs(dx))).
    Velocity is skipped across detection holes larger than MAX_VEL_FRAME_GAP so a
    gap cannot become a fake speed spike / contact frame.
    """
    T = len(frames)
    pos = np.full((T, 2), np.nan)
    for t, f in enumerate(frames):
        p = hands_by_frame.get(f)
        if p is not None:
            pos[t] = p

    pos_s = _moving_average(pos, SMOOTH_WIN)

    angles, speeds = {}, {}
    for t, f in enumerate(frames):
        lo, hi = max(0, t - 1), min(T - 1, t + 1)
        if lo == hi:
            continue
        if (int(frames[t]) - int(frames[lo]) > MAX_VEL_FRAME_GAP
                or int(frames[hi]) - int(frames[t]) > MAX_VEL_FRAME_GAP):
            continue
        if np.isnan(pos_s[lo, 0]) or np.isnan(pos_s[hi, 0]):
            continue
        dt = int(frames[hi] - frames[lo])
        if dt <= 0:
            continue
        dx = (pos_s[hi, 0] - pos_s[lo, 0]) / dt
        dy = (pos_s[hi, 1] - pos_s[lo, 1]) / dt
        # image y is down → negate so a rising bat is a positive attack angle.
        # abs(dx) folds left/right so a 15° upward path is ~15°, not ~165°.
        angles[f] = round(math.degrees(math.atan2(-dy, abs(dx))), 1)
        speeds[f] = math.hypot(dx, dy)

    contact = None
    if len(speeds) and T >= MIN_SEG_FRAMES:
        contact = max(speeds, key=speeds.get)

    per_frame = {f: {"attack_angle": angles.get(f), "speed": round(speeds.get(f, 0.0), 2)}
                 for f in frames if f in angles}
    return per_frame, contact, pos_s


# ── Drawing helpers ──────────────────────────────────────────────────────────
def _arc(frame, center, p1, p2, radius, color):
    c = (int(center[0]), int(center[1]))
    a1 = _line_angle_deg(center, p1)
    a2 = _line_angle_deg(center, p2)
    # draw the shorter sweep
    if abs(a2 - a1) > 180:
        if a2 > a1:
            a1 += 360
        else:
            a2 += 360
    cv2.ellipse(frame, c, (radius, radius), 0, min(a1, a2), max(a1, a2), color, 2, cv2.LINE_AA)


def _label(frame, text, org, color, scale=0.6, thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_biomech(frame, lm, metrics, attack, hand_trail, is_contact, handedness):
    # base skeleton (red bones + colored dots)
    draw_skeleton(frame, lm)

    # highlight shoulder & hip lines
    if "left_shoulder" in lm and "right_shoulder" in lm:
        cv2.line(frame, tuple(map(int, lm["left_shoulder"])), tuple(map(int, lm["right_shoulder"])),
                 C_CYAN, 2, cv2.LINE_AA)
    if "left_hip" in lm and "right_hip" in lm:
        cv2.line(frame, tuple(map(int, lm["left_hip"])), tuple(map(int, lm["right_hip"])),
                 C_MAG, 2, cv2.LINE_AA)

    # front-knee arc
    side = "left" if handedness == "righty" else "right"
    hip, knee, ankle = f"{side}_hip", f"{side}_knee", f"{side}_ankle"
    if metrics.get("front_knee_angle") is not None and all(k in lm for k in (hip, knee, ankle)):
        _arc(frame, lm[knee], lm[hip], lm[ankle], 26, C_YELLOW)
        _label(frame, f"{metrics['front_knee_angle']:.0f}",
                (int(lm[knee][0]) + 10, int(lm[knee][1])), C_YELLOW, 0.55, 1)

    # hand path trail + attack-angle arrow
    if len(hand_trail) >= 2:
        pts = np.array([p for p in hand_trail if p is not None], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(frame, [pts], False, C_YELLOW, 2, cv2.LINE_AA)
            tip = pts[-1]
            if attack is not None and attack.get("attack_angle") is not None:
                a = math.radians(attack["attack_angle"])
                end = (int(tip[0] + 45 * math.cos(a)), int(tip[1] - 45 * math.sin(a)))
                cv2.arrowedLine(frame, tuple(tip), end, C_YELLOW, 2, cv2.LINE_AA, tipLength=0.3)

    if is_contact:
        _label(frame, "CONTACT (est.)", (int(frame.shape[1] / 2) - 120, 70), C_RED, 0.9, 2)


def draw_hud(frame, video, frame_idx, seg_id, total, metrics, attack_now, contact_summary):
    W = frame.shape[1]
    panel_h = 150
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (340, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)

    _label(frame, f"{os.path.splitext(video)[0][:26]}  seg {seg_id}", (10, 22), C_WHITE, 0.5, 1)

    def col_attack(v):
        return C_GREEN if (v is not None and IDEAL_ATTACK_ANGLE[0] <= v <= IDEAL_ATTACK_ANGLE[1]) else C_ORANGE
    def col_sep(v):
        return C_GREEN if (v is not None and v >= GOOD_HIP_SHOULDER_SEP) else C_ORANGE

    sep = metrics.get("hip_shoulder_sep")
    tilt = metrics.get("spine_tilt")
    knee = metrics.get("front_knee_angle")
    aa = attack_now.get("attack_angle") if attack_now else None

    rows = [
        (f"Hip-Shoulder Sep: {sep if sep is not None else '--'} deg", col_sep(sep)),
        (f"Spine Tilt:       {tilt if tilt is not None else '--'} deg", C_WHITE),
        (f"Front-Knee Angle: {knee if knee is not None else '--'} deg", C_WHITE),
        (f"Attack Angle:     {aa if aa is not None else '--'} deg", col_attack(aa)),
    ]
    y = 48
    for text, color in rows:
        _label(frame, text, (10, y), color, 0.5, 1)
        y += 22

    if contact_summary and contact_summary.get("attack_angle") is not None:
        _label(frame, f"@contact AA: {contact_summary['attack_angle']} deg",
                (10, y), col_attack(contact_summary["attack_angle"]), 0.5, 1)
    _label(frame, "2D proxies (view-dependent)", (10, panel_h - 6), (180, 180, 180), 0.4, 1)


# ── Per-video processing ─────────────────────────────────────────────────────
def process_video(video_name, kp_df, metrics_out):
    vkp = kp_df[kp_df["video"] == video_name]
    if len(vkp) == 0:
        print(f"  SKIP {video_name} — no keypoints")
        return

    # frame -> {keypoint: (x,y,conf)} and frame -> segment_id
    coco_by_frame = {}
    for row in vkp.itertuples(index=False):
        coco_by_frame.setdefault(row.frame, {})[row.keypoint] = (row.x, row.y, row.confidence)
    seg_of_frame = vkp.groupby("frame")["segment_id"].first().to_dict()

    lm_by_frame, hands_by_frame = {}, {}
    for f, coco in coco_by_frame.items():
        lm_by_frame[f] = compute_landmarks(coco)
        hp = _hands_point(coco)
        if hp is not None:
            hands_by_frame[f] = hp

    handed, handed_src = _handedness(video_name, lm_by_frame)
    sm_by_frame = {f: frame_metrics(lm, handed) for f, lm in lm_by_frame.items()}

    # per-segment wrist-path angles + contact frame
    attack_by_frame, contact_frames, contact_summary_by_seg = {}, {}, {}
    for seg_id, sdf in vkp.groupby("segment_id"):
        frames = sorted(sdf["frame"].unique())
        per_frame, contact, _ = wrist_path_angles(frames, hands_by_frame)
        attack_by_frame.update(per_frame)
        if contact is not None:
            contact_frames[contact] = seg_id
            cs = {"frame": int(contact), **sm_by_frame.get(contact, {}),
                  "attack_angle": per_frame.get(contact, {}).get("attack_angle")}
            contact_summary_by_seg[seg_id] = cs

        seps = [sm_by_frame[f]["hip_shoulder_sep"] for f in frames if sm_by_frame[f]["hip_shoulder_sep"] is not None]
        knees = [sm_by_frame[f]["front_knee_angle"] for f in frames if sm_by_frame[f]["front_knee_angle"] is not None]
        metrics_out.append({
            "video": video_name,
            "segment_id": int(seg_id),
            "handedness": handed,
            "handedness_source": handed_src,
            "n_frames": len(frames),
            "contact_frame": int(contact) if contact is not None else None,
            "at_contact": contact_summary_by_seg.get(seg_id),
            "hip_shoulder_sep_range": [round(min(seps), 1), round(max(seps), 1)] if seps else None,
            "front_knee_angle_range": [round(min(knees), 1), round(max(knees), 1)] if knees else None,
            "per_frame": {
                str(f): {**sm_by_frame[f], "attack_angle": attack_by_frame.get(f, {}).get("attack_angle")}
                for f in frames
            },
        })

    vid_path = str(BATTING_VIDEOS_DIR / video_name)
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video file: {vid_path}")

    writer = None
    try:
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  {video_name}  {W}x{H} @ {FPS:.1f}fps  ({total} frames, {handed} via {handed_src})")

        out_path = str(DIAGNOSES_DIR / f"skeleton_2d_biomech_{os.path.splitext(video_name)[0]}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

        trail = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in lm_by_frame:
                lm = lm_by_frame[frame_idx]
                metrics = sm_by_frame[frame_idx]
                attack_now = attack_by_frame.get(frame_idx)
                hp = hands_by_frame.get(frame_idx)
                trail.append((int(hp[0]), int(hp[1])) if hp else None)
                trail = trail[-TRAIL_LEN:]
                seg_id = seg_of_frame.get(frame_idx, "?")
                cs = contact_summary_by_seg.get(seg_id)
                draw_biomech(frame, lm, metrics, attack_now,
                             [p for p in trail if p is not None], frame_idx in contact_frames, handed)
                draw_hud(frame, video_name, frame_idx, seg_id, total, metrics, attack_now, cs)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    print(f"    → {out_path}")


def run_apply_2d(video_filter=None):
    """video_filter: optional list of video filenames to process (default: all with keypoints)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSES_DIR.mkdir(parents=True, exist_ok=True)

    videos = [
        v for v in list_videos(BATTING_VIDEOS_DIR)
        if (DATA_DIR / f"{Path(v).stem}_keypoints.csv").exists()
    ]
    if video_filter:
        videos = [v for v in videos if v in video_filter]
    print(f"Applying 2D domain to {len(videos)} video(s) (per-video CSVs).")

    metrics_out = []
    failed = []
    for v in videos:
        try:
            kp = load_batter_keypoints(v)
            if kp.empty:
                print(f"  SKIP {v} — no keypoints CSV or empty")
                failed.append(v)
                continue
            process_video(v, kp, metrics_out)
        except Exception as exc:
            print(f"  ERROR on {v}: {type(exc).__name__}: {exc}")
            failed.append(v)

    out_json = OUTPUT_DIR / "2d_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nWrote {len(metrics_out)} segment metric records → {out_json}")
    if failed:
        print("  failed/skipped: " + ", ".join(failed))
    return metrics_out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Apply the 2D domain (biomech overlay)")
    ap.add_argument("--videos", nargs="*", help="Subset of video filenames to process")
    args = ap.parse_args()
    run_apply_2d(video_filter=args.videos)
