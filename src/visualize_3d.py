"""
Stage 3c — 3D Skeleton Visualization
Renders an animated MP4 per segment using matplotlib 3D, saved to the
Batting Diagnoses folder as skeleton_3d_<video>_seg<id>.mp4.

VIEW FIX (2026-06-24): the camera now faces the batter from the FRONT, matching
how the source video is shot. The earlier render used the wrong axis as "up"
(it mapped dim1→height), which produced a sideways/squished skeleton.

Verified MotionBERT axis convention for this project's output:
  dim 0 = lateral (image left-right)
  dim 1 = depth   (toward / away from camera)
  dim 2 = vertical, with UP = -dim 2
So: matplotlib X ← dim0, matplotlib Y ← dim1 (depth), matplotlib Z(up) ← -dim2,
viewed near-front (elev=8, azim=-85).
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from config import DIAGNOSES_DIR, OUTPUT_DIR, H36M_JOINTS, H36M_BONES, BONE_COLOR_MAP

# Front-on camera, slight angle so it still reads as 3D.
VIEW_ELEV, VIEW_AZIM = 8, -85


def render_skeleton_video(video_name, seg_id, pose_3d, frames, out_path, fps=30):
    T = len(frames)
    xs_all = pose_3d[:, :, 0]          # lateral
    ys_all = pose_3d[:, :, 1]          # depth
    zs_all = -pose_3d[:, :, 2]         # height (up = -dim2)

    m = 0.15
    xlim = (xs_all.min() - m, xs_all.max() + m)
    ylim = (ys_all.min() - m, ys_all.max() + m)
    zlim = (zs_all.min() - m, zs_all.max() + m)

    fig = plt.figure(figsize=(7, 9), facecolor="#0A0A0A")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0A0A0A")

    def draw_frame(t):
        ax.cla()
        ax.set_facecolor("#0A0A0A")
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        # true proportions so the body isn't distorted
        ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
        ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        ax.grid(False)

        joints = pose_3d[t]
        xs, ys, zs = joints[:, 0], joints[:, 1], -joints[:, 2]

        for bone_idx, (i, j) in enumerate(H36M_BONES):
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]],
                    color=BONE_COLOR_MAP[bone_idx], linewidth=3, alpha=0.95)
        ax.scatter(xs, ys, zs, c="white", s=16, zorder=5, depthshade=False)
        for w in ("LWrist", "RWrist"):
            k = H36M_JOINTS.index(w)
            ax.scatter(xs[k], ys[k], zs[k], c="#FFFF00", s=50, zorder=6, depthshade=False)

        # subtle ground plane at the feet
        floor = zs_all.min()
        for gx in np.linspace(xlim[0], xlim[1], 5):
            ax.plot([gx, gx], list(ylim), [floor, floor], color="#1F1F1F", linewidth=0.5)
        for gy in np.linspace(ylim[0], ylim[1], 5):
            ax.plot(list(xlim), [gy, gy], [floor, floor], color="#1F1F1F", linewidth=0.5)

        pct = int(t / max(T - 1, 1) * 100)
        ax.set_title(f"{os.path.splitext(video_name)[0]}  |  seg {seg_id}  |  "
                     f"frame {frames[t]}  ({pct}%)", color="white", fontsize=10, pad=6)

    anim = animation.FuncAnimation(fig, draw_frame, frames=T, interval=int(1000 / fps))
    writer = animation.FFMpegWriter(fps=fps, bitrate=2500,
                                    extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    anim.save(out_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def run_visualize_3d(results_3d: dict):
    """Render from in-memory inference output (dict of (video,seg)->{pose_3d,frames,T})."""
    DIAGNOSES_DIR.mkdir(parents=True, exist_ok=True)
    for (video, seg_id), data in results_3d.items():
        vname = os.path.splitext(video)[0]
        out_mp4 = str(DIAGNOSES_DIR / f"skeleton_3d_{vname}_seg{seg_id}.mp4")
        print(f"Rendering {vname} seg={seg_id} ({data['T']} frames) ...")
        render_skeleton_video(video, seg_id, data["pose_3d"], data["frames"], out_mp4)
    print(f"\nAll 3D skeleton videos saved to {DIAGNOSES_DIR}.")


def run_visualize_3d_from_json(json_path=None, videos=None):
    """Re-render straight from keypoints_3d.json (no MotionBERT / GPU needed)."""
    json_path = Path(json_path) if json_path else (OUTPUT_DIR / "keypoints_3d.json")
    DIAGNOSES_DIR.mkdir(parents=True, exist_ok=True)
    data = json.load(open(json_path))
    from collections import defaultdict
    segs = defaultdict(list)
    for r in data:
        segs[(r["video"], r["segment_id"])].append(r)
    for (video, seg_id) in sorted(segs):
        if videos and video not in videos:
            continue
        recs = sorted(segs[(video, seg_id)], key=lambda r: r["frame"])
        frames = [r["frame"] for r in recs]
        pose = np.array([[[r["joints"][n]["x"], r["joints"][n]["y"], r["joints"][n]["z"]]
                          for n in H36M_JOINTS] for r in recs], dtype=np.float32)
        vname = os.path.splitext(video)[0]
        out_mp4 = str(DIAGNOSES_DIR / f"skeleton_3d_{vname}_seg{seg_id}.mp4")
        print(f"Rendering {vname} seg={seg_id} ({len(frames)} frames) ...")
        render_skeleton_video(video, seg_id, pose, frames, out_mp4)
    print(f"\nDone. 3D skeleton videos in {DIAGNOSES_DIR}.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", help="subset of video filenames")
    args = ap.parse_args()
    run_visualize_3d_from_json(videos=args.videos)
