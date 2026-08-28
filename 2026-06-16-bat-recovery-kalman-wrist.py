# ============================================================
# Bat Position Recovery: Wrist Proxy + Kalman Interpolation
# ============================================================
# Fills in bat position for every frame, even during fast swings
# where YOLO detection fails.
#
# Priority per frame:
#   1. Detected bat (bat_boxes_raw.csv) — most accurate
#   2. Kalman prediction (extrapolated from surrounding detections)
#   3. Wrist proxy (estimated from pose keypoints) — fallback
#
# Inputs:  keypoints_batter.csv  (pose + segment_id from Phase 1)
#          bat_boxes_raw.csv
# Output:  bat_boxes_filled.csv  (complete bat position every frame)
# ============================================================


# ── CELL A: Setup ───────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from google.colab.patches import cv2_imshow

DRIVE_PATH = "/content/drive/My Drive/Baseball Project"

kp_df  = pd.read_csv(f"{DRIVE_PATH}/keypoints_batter.csv")   # has segment_id
bat_df = pd.read_csv(f"{DRIVE_PATH}/bat_boxes_raw.csv")

# Pre-compute bat bbox center and dimensions
bat_df["bat_cx"] = (bat_df["bat_x1"] + bat_df["bat_x2"]) / 2
bat_df["bat_cy"] = (bat_df["bat_y1"] + bat_df["bat_y2"]) / 2
bat_df["bat_w"]  = bat_df["bat_x2"] - bat_df["bat_x1"]
bat_df["bat_h"]  = bat_df["bat_y2"] - bat_df["bat_y1"]

print("Loaded keypoints_batter:", kp_df.shape)
print("Loaded bat_boxes_raw:", bat_df.shape)
print("Segments:", kp_df.groupby(["video","segment_id"]).size().reset_index(name="rows"))


# ── CELL B: Kalman Filter class ─────────────────────────────
#
# Tracks bat center (cx, cy) using a constant-velocity model:
#   State: [cx, cy, vx, vy]  — position + velocity
#   We observe only [cx, cy] from the detected bat frames.
#
# When the bat is detected  → update() corrects the estimate.
# When detection is missing → predict() extrapolates along the arc.
#
# Tuning:
#   Q (process noise): how much the bat can deviate from constant velocity.
#      Velocity entries are large because a bat arc accelerates/decelerates.
#   R (measurement noise): how much we trust each YOLO detection (~20px).

class BatKalmanFilter:
    def __init__(self, cx, cy, vx=0.0, vy=0.0):
        self.x = np.array([cx, cy, vx, vy], dtype=float)

        # Uncertainty — start large so first detection dominates
        self.P = np.diag([200.0, 200.0, 500.0, 500.0])

        # State transition: position += velocity each frame
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)

        # We measure only position
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)

        # Process noise: trust velocity less (bat arc is curved, not perfectly linear)
        self.Q = np.diag([2.0, 2.0, 80.0, 80.0])

        # Measurement noise: ~20px uncertainty on YOLO center
        self.R = np.diag([20.0, 20.0])

    def predict(self):
        """Advance state by one frame without a measurement."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0], self.x[1]   # predicted cx, cy

    def update(self, cx, cy):
        """Correct state with a new detected measurement."""
        z   = np.array([cx, cy])
        y   = z - self.H @ self.x                     # innovation
        S   = self.H @ self.P @ self.H.T + self.R     # innovation covariance
        K   = self.P @ self.H.T @ np.linalg.inv(S)   # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[0], self.x[1]   # corrected cx, cy


# ── CELL C: Wrist-based bat proxy ──────────────────────────
#
# When neither detection nor Kalman is reliable, estimate the bat from
# the batter's wrist keypoints — which are available ~95% of frames.
#
# Geometry:
#   bat_knob ≈ midpoint of left_wrist + right_wrist (the grip center)
#   bat_direction ≈ forearm vector of the TOP hand, extended past the knob
#   bat_tip  = bat_knob + direction × bat_length_pixels
#
# Handedness:
#   Right-handed batter: right wrist = bottom (knob side)
#                        left  wrist = top (barrel side)
#                        direction = lw → rw flipped to knob→tip: rw - lw, reversed
#   Left-handed batter:  left  wrist = bottom (knob side)
#                        right wrist = top (barrel side)
#
# Bat length calibration (per segment, NOT per video):
#   shoulder_width_pixels varies with camera distance per clip.
#   Known values: human shoulder width ≈ 47.5 cm, bat ≈ 86 cm (34 inches)
#   → bat_length_px = shoulder_px * (86 / 47.5) ≈ shoulder_px * 1.81

BAT_TO_SHOULDER_RATIO = 86.0 / 47.5   # bat 86cm / shoulder 47.5cm

def detect_handedness(video_name):
    """Infer handedness from video filename."""
    vn = video_name.lower()
    if "lefty" in vn:
        return "left"
    if "righty" in vn:
        return "right"
    return "right"   # safe default

def get_kp(frame_df, name):
    """Extract (x, y) for a single keypoint from a frame's keypoint rows."""
    row = frame_df[frame_df["keypoint"] == name]
    if row.empty or row.iloc[0]["confidence"] < 0.3:
        return None
    return np.array([row.iloc[0]["x"], row.iloc[0]["y"]], dtype=float)

def estimate_bat_from_wrists(frame_df, handedness, bat_length_px):
    """
    Returns dict with knob, tip, center, and synthetic bbox — or None if
    insufficient keypoints.
    """
    lw = get_kp(frame_df, "left_wrist")
    rw = get_kp(frame_df, "right_wrist")
    le = get_kp(frame_df, "left_elbow")
    re = get_kp(frame_df, "right_elbow")

    if lw is None or rw is None:
        return None

    # Knob = bottom hand wrist
    #   Right-handed batter: LEFT hand is at the bottom → knob = left_wrist
    #   Left-handed batter:  RIGHT hand is at the bottom → knob = right_wrist
    knob = lw if handedness == "right" else rw

    # Bat direction: from knob (bottom hand) → toward top hand → tip
    #   Right-handed: left wrist (knob) → right wrist (top) → tip beyond
    #   Left-handed:  right wrist (knob) → left wrist (top) → tip beyond
    if handedness == "right":
        raw_dir = rw - lw   # knob(left) → top hand(right) → tip
    else:
        raw_dir = lw - rw   # knob(right) → top hand(left) → tip

    # Refine with top-hand elbow→wrist vector for better swing-phase orientation
    #   Right-handed: top hand = right → use right_elbow → right_wrist
    #   Left-handed:  top hand = left  → use left_elbow  → left_wrist
    if handedness == "right" and re is not None:
        elbow_dir = rw - re   # right elbow → right wrist (top hand)
        if np.linalg.norm(elbow_dir) > 5:
            raw_dir = 0.5 * raw_dir + 0.5 * elbow_dir
    elif handedness == "left" and le is not None:
        elbow_dir = lw - le   # left elbow → left wrist (top hand)
        if np.linalg.norm(elbow_dir) > 5:
            raw_dir = 0.5 * raw_dir + 0.5 * elbow_dir

    norm = np.linalg.norm(raw_dir)
    if norm < 5:
        return None   # wrists too close together to determine direction

    direction = raw_dir / norm
    tip = knob + direction * bat_length_px
    center = (knob + tip) / 2

    # Build a synthetic bbox around knob→tip line
    margin = 12
    x1 = float(min(knob[0], tip[0]) - margin)
    y1 = float(min(knob[1], tip[1]) - margin)
    x2 = float(max(knob[0], tip[0]) + margin)
    y2 = float(max(knob[1], tip[1]) + margin)

    return {
        "bat_x1": x1, "bat_y1": y1, "bat_x2": x2, "bat_y2": y2,
        "bat_cx": float(center[0]), "bat_cy": float(center[1]),
        "bat_w":  float(x2 - x1),  "bat_h":  float(y2 - y1),
        "bat_knob_x": float(knob[0]), "bat_knob_y": float(knob[1]),
        "bat_tip_x":  float(tip[0]),  "bat_tip_y":  float(tip[1]),
    }


# ── CELL D: Main recovery loop ──────────────────────────────
#
# For each video × segment:
#   1. Collect detected bat frames (source = "detected")
#   2. Run Kalman forward + backward through the segment
#      Frames in detection gaps → fill from Kalman (source = "kalman")
#   3. Any remaining missing frames → fill from wrist proxy (source = "wrist_proxy")
# Merge all three sources into bat_boxes_filled.csv

records = []

for (video, seg_id), seg_kp in kp_df.groupby(["video", "segment_id"]):
    handedness      = detect_handedness(video)
    frames_in_seg   = sorted(seg_kp["frame"].unique())
    f_min, f_max    = frames_in_seg[0], frames_in_seg[-1]

    # ── Calibrate bat length for THIS segment from its shoulder width ──
    ls = seg_kp[seg_kp["keypoint"] == "left_shoulder"][["frame","x","y"]]
    rs = seg_kp[seg_kp["keypoint"] == "right_shoulder"][["frame","x","y"]]
    sh_merge = ls.merge(rs, on="frame", suffixes=("_l","_r"))
    if not sh_merge.empty:
        sh_widths = np.sqrt((sh_merge["x_l"]-sh_merge["x_r"])**2 +
                            (sh_merge["y_l"]-sh_merge["y_r"])**2)
        shoulder_px  = sh_widths.median()   # median is robust to pose changes
        bat_length_px = shoulder_px * BAT_TO_SHOULDER_RATIO
    else:
        bat_length_px = 180   # fallback

    # ── Detected bat frames in this segment ──
    seg_bat = bat_df[
        (bat_df["video"] == video) &
        (bat_df["frame"] >= f_min) &
        (bat_df["frame"] <= f_max)
    ].set_index("frame")

    detected_set = set(seg_bat.index)

    # ── Kalman pass: forward through the full segment ──
    kf = None
    kalman_predictions = {}   # frame → (cx, cy, w, h)
    rolling_w, rolling_h = [], []

    for f in frames_in_seg:
        if f in detected_set:
            row   = seg_bat.loc[f]
            cx, cy = float(row["bat_cx"]), float(row["bat_cy"])
            w, h   = float(row["bat_w"]),  float(row["bat_h"])
            rolling_w.append(w)
            rolling_h.append(h)
            if kf is None:
                kf = BatKalmanFilter(cx, cy)
            else:
                kf.predict()
                kf.update(cx, cy)
        else:
            if kf is not None:
                pcx, pcy = kf.predict()
                # Use rolling median of detected bbox size during gaps
                pw = float(np.median(rolling_w)) if rolling_w else 80.0
                ph = float(np.median(rolling_h)) if rolling_h else 80.0
                kalman_predictions[f] = (pcx, pcy, pw, ph)

    # ── Wrist proxy: compute for all frames (used only where still missing) ──
    wrist_estimates = {}
    kp_by_frame = seg_kp.groupby("frame")

    for f, frame_df in kp_by_frame:
        if f not in detected_set and f not in kalman_predictions:
            est = estimate_bat_from_wrists(frame_df, handedness, bat_length_px)
            if est is not None:
                wrist_estimates[f] = est

    # ── Build output rows for every frame in this segment ──
    for f in frames_in_seg:
        base = {"video": video, "frame": f, "segment_id": int(seg_id),
                "bat_knob_x": None, "bat_knob_y": None,
                "bat_tip_x": None,  "bat_tip_y": None}

        if f in detected_set:
            row = seg_bat.loc[f]
            base.update({
                "bat_x1": float(row["bat_x1"]), "bat_y1": float(row["bat_y1"]),
                "bat_x2": float(row["bat_x2"]), "bat_y2": float(row["bat_y2"]),
                "bat_cx": float(row["bat_cx"]), "bat_cy": float(row["bat_cy"]),
                "bat_conf": float(row["bat_conf"]),
                "source": "detected",
            })
        elif f in kalman_predictions:
            cx, cy, w, h = kalman_predictions[f]
            base.update({
                "bat_x1": cx - w/2, "bat_y1": cy - h/2,
                "bat_x2": cx + w/2, "bat_y2": cy + h/2,
                "bat_cx": cx, "bat_cy": cy,
                "bat_conf": np.nan,
                "source": "kalman",
            })
        elif f in wrist_estimates:
            est = wrist_estimates[f]
            base.update({**est, "bat_conf": np.nan, "source": "wrist_proxy"})
        else:
            base.update({
                "bat_x1": np.nan, "bat_y1": np.nan,
                "bat_x2": np.nan, "bat_y2": np.nan,
                "bat_cx": np.nan, "bat_cy": np.nan,
                "bat_conf": np.nan,
                "source": "missing",
            })

        records.append(base)

filled_df = pd.DataFrame(records)
filled_df.to_csv(f"{DRIVE_PATH}/bat_boxes_filled.csv", index=False)
print(f"\nSaved bat_boxes_filled.csv — {len(filled_df)} rows")


# ── CELL E: Coverage summary ────────────────────────────────

print("\n── Coverage by segment ──")
summary = (
    filled_df.groupby(["video", "segment_id", "source"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)
for col in ["detected", "kalman", "wrist_proxy", "missing"]:
    if col not in summary.columns:
        summary[col] = 0
summary["total"] = summary[["detected","kalman","wrist_proxy","missing"]].sum(axis=1)
summary["coverage_pct"] = ((summary["total"] - summary["missing"]) / summary["total"] * 100).round(1)
print(summary.to_string(index=False))


# ── CELL F: Timeline visualization ──────────────────────────
#
# One row per segment showing frame-by-frame source color:
#   GREEN  = detected, BLUE = kalman, ORANGE = wrist_proxy, RED = missing

SOURCE_COLOR = {
    "detected":    "seagreen",
    "kalman":      "steelblue",
    "wrist_proxy": "darkorange",
    "missing":     "crimson",
}

for video, vid_df in filled_df.groupby("video"):
    segs = sorted(vid_df["segment_id"].unique())
    fig, axes = plt.subplots(len(segs), 1,
                             figsize=(16, max(2.5, len(segs) * 1.8)),
                             sharex=False)
    if len(segs) == 1:
        axes = [axes]

    fig.suptitle(f"{video} — bat recovery source per frame", fontsize=11, fontweight="bold")

    for ax, seg in zip(axes, segs):
        seg_df = vid_df[vid_df["segment_id"] == seg].sort_values("frame")
        frames  = seg_df["frame"].values
        sources = seg_df["source"].values
        colors  = [SOURCE_COLOR.get(s, "grey") for s in sources]

        ax.bar(frames, [1]*len(frames), color=colors, width=1.0)
        ax.set_ylabel(f"seg {seg}", fontsize=8)
        ax.set_yticks([])
        ax.set_xlim(frames[0]-1, frames[-1]+1)

        # Annotate % coverage
        n_missing = (seg_df["source"] == "missing").sum()
        n_total   = len(seg_df)
        pct       = (n_total - n_missing) / n_total * 100
        ax.set_title(f"Segment {seg} | frames {frames[0]}–{frames[-1]} | "
                     f"coverage {pct:.1f}%  "
                     f"({(seg_df['source']=='detected').sum()} detected / "
                     f"{(seg_df['source']=='kalman').sum()} kalman / "
                     f"{(seg_df['source']=='wrist_proxy').sum()} wrist_proxy / "
                     f"{n_missing} missing)",
                     fontsize=8, loc="left")

    # Legend
    handles = [mpatches.Patch(color=c, label=s) for s, c in SOURCE_COLOR.items()]
    fig.legend(handles=handles, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


# ── CELL G: Annotated video with source overlay ─────────────
#
# Re-renders the video with bat overlays colour-coded by source:
#   GREEN bbox  = detected by YOLO
#   BLUE bbox   = Kalman interpolated
#   ORANGE line = wrist proxy (draws knob→tip line, not just bbox)
#   RED banner  = truly missing

VIDEO_FILES = {
    "Chae_friend_Righty_Batting_V1.mov": f"{DRIVE_PATH}/Chae_friend_Righty_Batting_V1.mov",
    "Chae_friend_Righty_Batting_V2.MOV": f"{DRIVE_PATH}/Chae_friend_Righty_Batting_V2.MOV",
    "Chae_friend_Lefty_Batting_V1.mov":  f"{DRIVE_PATH}/Chae_friend_Lefty_Batting_V1.mov",
}

PREVIEW_EVERY_N = 12

def annotate_video_filled(video_name, video_path, filled_df):
    vid_filled = filled_df[filled_df["video"] == video_name].set_index("frame")

    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = f"{DRIVE_PATH}/bat_filled_{video_name.rsplit('.',1)[0]}.mp4"
    writer   = cv2.VideoWriter(out_path,
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (width, height))

    preview_frames = []

    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx not in vid_filled.index:
            writer.write(frame)
            continue

        row    = vid_filled.loc[frame_idx]
        source = row["source"]

        if source == "detected":
            color = (0, 200, 0)       # green
            x1,y1,x2,y2 = int(row.bat_x1),int(row.bat_y1),int(row.bat_x2),int(row.bat_y2)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            conf_str = f"conf={row.bat_conf:.2f}"

        elif source == "kalman":
            color = (200, 100, 0)     # blue
            x1,y1,x2,y2 = int(row.bat_x1),int(row.bat_y1),int(row.bat_x2),int(row.bat_y2)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            # Dashed effect (draw inner lighter rect)
            cv2.rectangle(frame, (x1+3,y1+3), (x2-3,y2-3), (255,180,80), 1)
            conf_str = "kalman"

        elif source == "wrist_proxy":
            color = (0, 140, 255)     # orange
            # Draw knob→tip line instead of bbox
            if not pd.isna(row.bat_knob_x):
                kx,ky = int(row.bat_knob_x), int(row.bat_knob_y)
                tx,ty = int(row.bat_tip_x),  int(row.bat_tip_y)
                cv2.line(frame, (kx,ky), (tx,ty), color, 3)
                cv2.circle(frame, (kx,ky), 5, (0,80,255), -1)   # knob dot
                cv2.circle(frame, (tx,ty), 5, (0,200,255), -1)  # tip dot
                cv2.putText(frame, "knob", (kx+4,ky-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,80,255), 1)
                cv2.putText(frame, "tip",  (tx+4,ty-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,255), 1)
            conf_str = "wrist proxy"
            x1 = int(row.bat_x1) if not pd.isna(row.bat_x1) else 0
            y1 = int(row.bat_y1) if not pd.isna(row.bat_y1) else 0

        else:   # missing
            cv2.rectangle(frame, (0,0), (width, 28), (0,0,150), -1)
            cv2.putText(frame, "MISSING — no bat data",
                        (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            conf_str = "missing"
            x1, y1 = 8, 35

        # Source label
        if source != "missing":
            label = f"seg={int(row.segment_id)}  {source}  {conf_str}"
            cv2.putText(frame, label, (x1, max(int(y1)-8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Frame counter
        cv2.putText(frame, f"f={frame_idx}", (6, height-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

        writer.write(frame)

        if frame_idx % PREVIEW_EVERY_N == 0:
            preview_frames.append((frame_idx, frame.copy()))

    cap.release()
    writer.release()
    print(f"  ✅ Saved → {out_path}")

    # Inline preview grid
    cols  = 4
    rows  = (len(preview_frames) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    axes = axes.flatten()
    for i, (fidx, frm) in enumerate(preview_frames):
        axes[i].imshow(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"f={fidx}", fontsize=7)
        axes[i].axis("off")
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{video_name} — bat recovery preview (every {PREVIEW_EVERY_N} frames)",
                 fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.show()


for vid_name, vid_path in VIDEO_FILES.items():
    print(f"\n{'='*60}\nProcessing: {vid_name}")
    annotate_video_filled(vid_name, vid_path, filled_df)
