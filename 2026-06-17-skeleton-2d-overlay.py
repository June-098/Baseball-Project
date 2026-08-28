# ─────────────────────────────────────────────────────────────────────────────
# 2D Skeleton Overlay — batting video → annotated MP4
#
# 15 landmarks: head, neck, left/right shoulder, elbow, wrist,
#               center hip, left/right hip, knee, ankle
# All connection lines: red. Each landmark: distinct color dot.
# ─────────────────────────────────────────────────────────────────────────────

# %% ── 2D Skeleton Overlay ────────────────────────────────────────────────────

import cv2
import numpy as np
import pandas as pd
import os
from google.colab import drive
drive.mount('/content/drive')

DRIVE_PROJECT  = '/content/drive/MyDrive/Baseball Project'
VIDEO_FOLDER   = '/content/drive/MyDrive/Baseball Project/Batting videos'
CONF_THRESHOLD = 0.3
VIDEO_EXTS     = ('.mp4', '.mov', '.MOV', '.avi', '.AVI', '.MP4')

# ── 15 landmark colors (BGR) ──────────────────────────────────────────────────

LM_COLOR = {
    'head':           ( 80,  80,  80),   # dark gray
    'neck':           (  0, 255,   0),   # green
    'left_shoulder':  (  0, 255, 255),   # yellow
    'left_elbow':     (255,   0, 255),   # magenta
    'left_wrist':     (255,   0, 200),   # pink-magenta
    'right_shoulder': (  0, 200,   0),   # dark green
    'right_elbow':    (  0, 165, 255),   # orange
    'right_wrist':    (255, 255,   0),   # cyan
    'center_hip':     (  0, 220, 180),   # lime-green
    'left_hip':       (200,   0, 255),   # purple-pink
    'right_hip':      (  0, 200, 255),   # gold
    'left_knee':      (255, 200,   0),   # sky-blue
    'right_knee':     (180,   0, 255),   # violet
    'left_ankle':     (  0,  80, 255),   # orange-red
    'right_ankle':    (255,  50, 180),   # hot-pink
}
LM_RADIUS = 7   # dot radius

# ── Connections (all RED) ─────────────────────────────────────────────────────

RED = (0, 0, 255)

CONNECTIONS = [
    ('head',          'neck'),
    ('neck',          'left_shoulder'),
    ('neck',          'right_shoulder'),
    ('neck',          'center_hip'),          # spine
    ('left_shoulder', 'left_elbow'),
    ('left_elbow',    'left_wrist'),
    ('right_shoulder','right_elbow'),
    ('right_elbow',   'right_wrist'),
    ('center_hip',    'left_hip'),
    ('center_hip',    'right_hip'),
    ('left_hip',      'left_knee'),
    ('left_knee',     'left_ankle'),
    ('right_hip',     'right_knee'),
    ('right_knee',    'right_ankle'),
]

# ── COCO → 15-landmark mapping ────────────────────────────────────────────────

DIRECT = {
    'head':           'nose',
    'left_shoulder':  'left_shoulder',
    'left_elbow':     'left_elbow',
    'left_wrist':     'left_wrist',
    'right_shoulder': 'right_shoulder',
    'right_elbow':    'right_elbow',
    'right_wrist':    'right_wrist',
    'left_hip':       'left_hip',
    'right_hip':      'right_hip',
    'left_knee':      'left_knee',
    'right_knee':     'right_knee',
    'left_ankle':     'left_ankle',
    'right_ankle':    'right_ankle',
}


def compute_landmarks(coco_joints):
    """
    Convert raw COCO keypoints dict → 15-landmark dict.
    Computed: neck = avg(shoulders), center_hip = avg(hips).
    Returns: {name: (x, y)} for landmarks above CONF_THRESHOLD, else omitted.
    """
    lm = {}

    for name, coco in DIRECT.items():
        p = coco_joints.get(coco)
        if p and p[2] >= CONF_THRESHOLD:
            lm[name] = (p[0], p[1])

    def avg2(a, b):
        pa, pb = coco_joints.get(a), coco_joints.get(b)
        if pa and pb and pa[2] >= CONF_THRESHOLD and pb[2] >= CONF_THRESHOLD:
            return ((pa[0]+pb[0])/2, (pa[1]+pb[1])/2)
        p = pa if (pa and pa[2] >= CONF_THRESHOLD) else \
            pb if (pb and pb[2] >= CONF_THRESHOLD) else None
        return (p[0], p[1]) if p else None

    neck = avg2('left_shoulder', 'right_shoulder')
    if neck: lm['neck'] = neck

    hip  = avg2('left_hip', 'right_hip')
    if hip:  lm['center_hip'] = hip

    return lm


# ── Draw helpers ──────────────────────────────────────────────────────────────

def draw_skeleton(frame, lm):
    # 1. Lines (draw first so dots sit on top)
    for (a, b) in CONNECTIONS:
        if a in lm and b in lm:
            cv2.line(frame,
                     (int(lm[a][0]), int(lm[a][1])),
                     (int(lm[b][0]), int(lm[b][1])),
                     RED, 2, cv2.LINE_AA)
    # 2. Dots
    for name, (x, y) in lm.items():
        pt    = (int(x), int(y))
        color = LM_COLOR[name]
        cv2.circle(frame, pt, LM_RADIUS,     color,     -1, cv2.LINE_AA)
        cv2.circle(frame, pt, LM_RADIUS + 1, (0, 0, 0),  1, cv2.LINE_AA)

def draw_hud(frame, video_name, frame_idx, seg_id, total_frames, W):
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (W, 42), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)
    pct   = int(frame_idx / max(total_frames - 1, 1) * 100)
    label = (f"{os.path.splitext(video_name)[0]}  |  "
             f"frame {frame_idx}/{total_frames}  |  seg {seg_id}  |  {pct}%")
    cv2.putText(frame, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


# ── Per-video processing ──────────────────────────────────────────────────────

def process_video(video_name, kp_df):
    vkp = kp_df[kp_df['video'] == video_name]
    if len(vkp) == 0:
        print(f"  SKIP {video_name} — no keypoints found")
        return

    frame_joints, frame_seg = {}, {}
    for row in vkp.itertuples(index=False):
        frame_joints.setdefault(row.frame, {})[row.keypoint] = (row.x, row.y, row.confidence)
    frame_seg = vkp.groupby('frame')['segment_id'].first().to_dict()

    vid_path = os.path.join(VIDEO_FOLDER, video_name)
    cap      = cv2.VideoCapture(vid_path)
    W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {video_name}  {W}x{H} @ {FPS:.1f}fps  ({total} frames)")

    out_path = os.path.join(DRIVE_PROJECT,
                            f"skeleton_2d_{os.path.splitext(video_name)[0]}.mp4")
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        coco = frame_joints.get(frame_idx, {})
        if coco:
            draw_skeleton(frame, compute_landmarks(coco))
        draw_hud(frame, video_name, frame_idx, frame_seg.get(frame_idx, '?'), total, W)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"    → {out_path}")


# ── Run all videos ────────────────────────────────────────────────────────────

kp     = pd.read_csv(os.path.join(DRIVE_PROJECT, 'keypoints_batter.csv'))
videos = sorted([f for f in os.listdir(VIDEO_FOLDER) if f.endswith(VIDEO_EXTS)])
print(f"Found {len(videos)} video(s):\n  " + "\n  ".join(videos))

for v in videos:
    process_video(v, kp)

print("\nAll done.")
