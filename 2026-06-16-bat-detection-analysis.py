# ============================================================
# Bat Detection Analysis — Phase 2 QA
# Run in Google Colab (GPU not required for this notebook)
# Requires: bat_boxes_raw.csv, keypoints_batter.csv (from Phase 1)
# ============================================================

# ── CELL 1: Setup ──────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from google.colab.patches import cv2_imshow

DRIVE_PATH = "/content/drive/My Drive/Baseball Project"

bat_df  = pd.read_csv(f"{DRIVE_PATH}/bat_boxes_raw.csv")
kp_df   = pd.read_csv(f"{DRIVE_PATH}/keypoints_batter.csv")  # has segment_id from Phase 1

# Derive bbox center and rough height/width for later use
bat_df["bat_cx"]  = (bat_df["bat_x1"] + bat_df["bat_x2"]) / 2
bat_df["bat_cy"]  = (bat_df["bat_y1"] + bat_df["bat_y2"]) / 2
bat_df["bat_w"]   = bat_df["bat_x2"] - bat_df["bat_x1"]
bat_df["bat_h"]   = bat_df["bat_y2"] - bat_df["bat_y1"]

print("bat_boxes_raw shape:", bat_df.shape)
print("Videos:", bat_df["video"].unique())
print(bat_df.head(3))


# ── CELL 2: Confirm what (x1,y1) / (x2,y2) actually are ───
#
# IMPORTANT — read this before assuming x1,y1 = knob:
#
#   YOLO outputs an AXIS-ALIGNED bounding box (a horizontal rectangle
#   that encloses the bat). The four corners are always:
#       (x1, y1) = top-LEFT  of the box   ← NOT the knob
#       (x2, y2) = bottom-RIGHT of the box ← NOT the tip
#
#   In image coordinates, y=0 is the TOP of the frame, y increases DOWN.
#   So bat_y1 < bat_y2 always — and (x1,y1) is the upper-left corner of
#   the rectangle, not an endpoint of the bat itself.
#
#   When the bat is held diagonally (which it always is during a swing),
#   the actual knob and tip live somewhere INSIDE the bbox diagonals, not
#   at the corners. To get true knob/tip positions you need bat keypoint
#   detection (Phase 2.5).
#
# This cell overlays the bbox + labels its corners on sample frames so
# you can see this directly.
# ─────────────────────────────────────────────────────────────

VIDEO_FILES = {
    "Chae_friend_Righty_Batting_V1.mov":  f"{DRIVE_PATH}/Chae_friend_Righty_Batting_V1.mov",
    "Chae_friend_Righty_Batting_V2.MOV":  f"{DRIVE_PATH}/Chae_friend_Righty_Batting_V2.MOV",
    "Chae_friend_Lefty_Batting_V1.mov":   f"{DRIVE_PATH}/Chae_friend_Lefty_Batting_V1.mov",
}

def draw_bat_box_on_frame(video_path, video_name, frame_idx):
    """Extract a single frame from video and draw the bat bbox with labeled corners."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  Could not read frame {frame_idx} from {video_name}")
        return None

    row = bat_df[(bat_df["video"] == video_name) & (bat_df["frame"] == frame_idx)]
    if row.empty:
        print(f"  No bat detection in {video_name} frame {frame_idx}")
        return None

    r = row.iloc[0]
    x1, y1, x2, y2 = int(r.bat_x1), int(r.bat_y1), int(r.bat_x2), int(r.bat_y2)
    conf = r.bat_conf

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Label corners — so you can see these are just rectangle corners, not bat endpoints
    cv2.circle(frame, (x1, y1), 6, (255, 0, 0), -1)   # top-left: BLUE
    cv2.circle(frame, (x2, y2), 6, (0, 0, 255), -1)   # bottom-right: RED
    cv2.circle(frame, (x1, y2), 6, (0, 255, 255), -1) # bottom-left: YELLOW
    cv2.circle(frame, (x2, y1), 6, (255, 0, 255), -1) # top-right: MAGENTA
    cv2.circle(frame, (int(r.bat_cx), int(r.bat_cy)), 6, (0, 255, 0), -1) # center: GREEN

    cv2.putText(frame, f"(x1,y1) top-left",    (x1+5, y1+20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),   1)
    cv2.putText(frame, f"(x2,y2) bot-right",   (x2-150, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),   1)
    cv2.putText(frame, f"conf={conf:.2f}",      (x1, y1-10),    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),   1)
    cv2.putText(frame, f"frame {frame_idx}",    (10, 30),        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    return frame

# Show 3 sample frames per video (early / mid / late swing)
for vid_name, vid_path in VIDEO_FILES.items():
    frames_with_bat = bat_df[bat_df["video"] == vid_name]["frame"].values
    if len(frames_with_bat) == 0:
        print(f"No bat detections in {vid_name}")
        continue

    sample_frames = [
        frames_with_bat[len(frames_with_bat) // 6],   # early
        frames_with_bat[len(frames_with_bat) // 2],   # mid
        frames_with_bat[int(len(frames_with_bat) * 5/6)],  # late
    ]

    print(f"\n{'='*60}")
    print(f"Video: {vid_name}")
    print("Blue dot = (x1,y1) top-left corner")
    print("Red  dot = (x2,y2) bottom-right corner")
    print("Note: these are NOT the bat knob/tip — just bbox corners.")
    print("The bat diagonal runs roughly from top-left to bottom-right")
    print("(or bottom-left to top-right) depending on swing phase.")
    for fidx in sample_frames:
        frame = draw_bat_box_on_frame(vid_path, vid_name, fidx)
        if frame is not None:
            cv2_imshow(frame)


# ── CELL 3: Full per-frame bat tracking visualizer ─────────
#
# For each video this cell:
#   1. Reads every frame from the source video in Drive
#   2. Draws the bat bbox (GREEN) on frames where bat is detected
#      Draws a RED banner on frames where no bat was detected
#      Labels each frame with: frame number, segment_id, and bat_conf
#      Marks segment boundaries with a YELLOW divider line
#   3. Saves a fully annotated MP4 back to Drive for you to scrub through
#   4. Shows a quick preview grid of every Nth frame inline in Colab
# ─────────────────────────────────────────────────────────────

import os

# Merge bat detections with segment info from Phase 1
seg_lookup = kp_df[["video", "frame", "segment_id"]].drop_duplicates()
bat_seg    = bat_df.merge(seg_lookup, on=["video", "frame"], how="left")

# Segment start frames — used to draw divider lines
seg_starts = (
    kp_df.groupby(["video", "segment_id"])["frame"]
    .min()
    .reset_index()
    .rename(columns={"frame": "seg_start_frame"})
)

PREVIEW_EVERY_N = 15  # show 1 out of every N frames in the Colab preview grid

def annotate_video(video_name, video_path):
    """
    Read every frame, draw bat bbox + segment labels, write annotated MP4 to Drive.
    Also display a sampled preview grid inline.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ Could not open {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = f"{DRIVE_PATH}/bat_tracking_{video_name.rsplit('.', 1)[0]}.mp4"
    writer   = cv2.VideoWriter(out_path,
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (width, height))

    # Look up data for this video
    vid_bat  = bat_seg[bat_seg["video"] == video_name].set_index("frame")
    vid_segs = seg_starts[seg_starts["video"] == video_name]["seg_start_frame"].values

    preview_frames = []  # collect sampled frames for the Colab grid

    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        # ── Segment boundary divider ──
        if frame_idx in vid_segs and frame_idx > 0:
            cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 255), 6)
            cv2.putText(frame, f"── SEGMENT BOUNDARY ──",
                        (width // 2 - 160, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # ── Bat detection overlay ──
        if frame_idx in vid_bat.index:
            row     = vid_bat.loc[frame_idx]
            # If multiple detections on same frame (unlikely but safe), take first
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            x1 = int(row["bat_x1"]); y1 = int(row["bat_y1"])
            x2 = int(row["bat_x2"]); y2 = int(row["bat_y2"])
            conf   = row["bat_conf"]
            seg_id = row["segment_id"] if not pd.isna(row["segment_id"]) else "?"

            # Colour bbox by confidence: green (high) → yellow (mid) → red (low)
            if conf >= 0.7:
                color = (0, 220, 0)      # green
            elif conf >= 0.5:
                color = (0, 200, 200)    # yellow
            else:
                color = (0, 80, 255)     # orange-red

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Corner dots so you can judge whether corners align with bat ends
            cv2.circle(frame, (x1, y1), 5, (255, 80, 0),   -1)  # top-left  (BLUE)
            cv2.circle(frame, (x2, y2), 5, (0,  0,  255),  -1)  # bot-right (RED)

            seg_label = f"{int(seg_id)}" if seg_id != "?" else "?"
            label = f"seg={seg_label}  conf={conf:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        else:
            # No bat detected — red banner at top
            cv2.rectangle(frame, (0, 0), (width, 32), (0, 0, 180), -1)
            cv2.putText(frame, "NO BAT DETECTED",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Frame counter (bottom-left)
        cv2.putText(frame, f"frame {frame_idx}/{total-1}",
                    (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(frame)

        if frame_idx % PREVIEW_EVERY_N == 0:
            preview_frames.append((frame_idx, frame.copy()))

    cap.release()
    writer.release()
    print(f"  ✅ Saved annotated video → {out_path}")

    # ── Colab preview grid ──
    cols = 4
    rows = (len(preview_frames) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    for i, (fidx, frm) in enumerate(preview_frames):
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        axes[i].imshow(rgb)
        axes[i].set_title(f"frame {fidx}", fontsize=7)
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{video_name} — bat tracking preview (every {PREVIEW_EVERY_N} frames)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.show()


# Run for every video
for vid_name, vid_path in VIDEO_FILES.items():
    print(f"\n{'='*60}")
    print(f"Processing: {vid_name}")
    annotate_video(vid_name, vid_path)

print("\nAll annotated videos saved to Drive. Open them there to scrub frame-by-frame.")


# ── CELL 4: Plot confidence + detection rate across frame range per segment ──
#
# The "peak bat speed zone" is roughly the final 1/3 of each segment's
# frame range (mid-to-late swing through contact). We shade that zone
# to see whether detection rate/confidence drops right when we need it most.

def plot_segment_analysis(video_name, segment_id, bat_seg_df, kp_seg_df):
    """
    For one segment: plot per-frame bat_conf and mark frames with no detection.
    Shades the last third of the frame range as the peak bat speed zone.
    """
    frames_all   = sorted(kp_seg_df["frame"].unique())
    frames_bat   = bat_seg_df[
        (bat_seg_df["video"] == video_name) &
        (bat_seg_df["segment_id"] == segment_id)
    ].sort_values("frame")

    n = len(frames_all)
    f_min, f_max = frames_all[0], frames_all[-1]
    peak_start   = frames_all[int(n * 2/3)]   # last 1/3 = peak speed zone

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f"{video_name}  |  Segment {segment_id}", fontsize=11, fontweight="bold")

    # ── Top: confidence per detected frame ──
    ax1.scatter(frames_bat["frame"], frames_bat["bat_conf"],
                color="steelblue", s=20, zorder=3, label="bat_conf")
    ax1.axhline(0.5, color="orange", linestyle="--", linewidth=1, label="conf=0.5 threshold")
    ax1.axvspan(peak_start, f_max, alpha=0.12, color="red", label="peak speed zone")
    ax1.set_ylabel("bat_conf")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    # ── Bottom: detected vs. missing per frame ──
    detected_frames = set(frames_bat["frame"].values)
    detected = [1 if f in detected_frames else 0 for f in frames_all]
    colors   = ["steelblue" if d else "crimson" for d in detected]
    ax2.bar(frames_all, detected, color=colors, width=1.0, zorder=3)
    ax2.axvspan(peak_start, f_max, alpha=0.12, color="red")
    ax2.set_ylabel("Detected (1) / Missing (0)")
    ax2.set_xlabel("Frame")
    ax2.set_yticks([0, 1])

    # Legend for bar colors
    ax2.legend(handles=[
        mpatches.Patch(color="steelblue", label="bat detected"),
        mpatches.Patch(color="crimson",   label="bat missing"),
        mpatches.Patch(color="red",       alpha=0.2, label="peak speed zone"),
    ], fontsize=8, loc="lower left")

    plt.tight_layout()
    plt.show()

    # Text summary
    n_detected = len(frames_bat)
    n_total    = len(frames_all)
    n_peak_all = sum(1 for f in frames_all if f >= peak_start)
    n_peak_det = sum(1 for f in frames_bat["frame"] if f >= peak_start)
    print(f"  Overall:    {n_detected}/{n_total} frames detected  ({n_detected/n_total*100:.1f}%)")
    print(f"  Peak zone:  {n_peak_det}/{n_peak_all} frames detected  ({n_peak_det/n_peak_all*100:.1f}%)")
    if n_peak_det / max(n_peak_all, 1) < 0.7:
        print("  ⚠️  Detection drops in the peak speed zone — consider fine-tuning.")
    else:
        print("  ✅ Detection holds up through the peak speed zone.")
    print()

# Run for every video × segment
for (vid, seg), grp_kp in kp_df.groupby(["video", "segment_id"]):
    print(f"\n{'─'*60}")
    print(f"Segment {seg}  |  Frames {grp_kp['frame'].min()}–{grp_kp['frame'].max()}")
    plot_segment_analysis(vid, seg, bat_seg, grp_kp)


# ── CELL 5: Overall confidence distribution by video ────────
fig, axes = plt.subplots(1, len(bat_df["video"].unique()), figsize=(14, 4), sharey=True)
for ax, (vid, grp) in zip(axes, bat_df.groupby("video")):
    ax.hist(grp["bat_conf"], bins=20, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(grp["bat_conf"].mean(), color="red", linestyle="--",
               label=f"mean={grp['bat_conf'].mean():.2f}")
    ax.axvline(0.5, color="orange", linestyle=":", label="0.5 threshold")
    ax.set_title(vid.split("_Batting")[0].replace("Chae_friend_", ""), fontsize=9)
    ax.set_xlabel("bat_conf")
    ax.legend(fontsize=7)
axes[0].set_ylabel("Frame count")
fig.suptitle("Bat Detection Confidence Distribution by Video", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.show()
