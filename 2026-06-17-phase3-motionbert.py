# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — 2D → 3D Body Lifting with MotionBERT
#
# Input:   keypoints_batter.csv  (from Phase 1 batter_selector.py)
# Output:  keypoints_3d.json     (x, y, z per joint per frame per segment)
#          3D skeleton MP4       (animated visualization per segment)
#
# Run in Google Colab with GPU (T4).
# Each # %% block is one Colab cell.
# ─────────────────────────────────────────────────────────────────────────────


# %% ── CELL A  Setup: Mount Drive, Clone MotionBERT, Install Deps ─────────────

from google.colab import drive
drive.mount('/content/drive')

import os, sys

FOLDER_ID     = '10Lh0gzVyJUGAp4uoMy6YBKajhZXuIpAZ'   # Batting Diagnoses
import subprocess as _sp
_r = _sp.run(['find', '/content/drive', '-type', 'd', '-name', 'Batting Diagnoses', '-maxdepth', '7'],
              capture_output=True, text=True, timeout=30)
_found = [p for p in _r.stdout.strip().split('\n') if p]
DRIVE_PROJECT = _found[0] if _found else '/content/drive/MyDrive/Batting Diagnoses'
print(f"Drive folder: {DRIVE_PROJECT}")

os.chdir('/content')
if not os.path.exists('MotionBERT'):
    os.system('git clone https://github.com/Walter0807/MotionBERT.git')

os.chdir('/content/MotionBERT')
os.system('pip install -q einops timm')

sys.path.insert(0, '/content/MotionBERT')
print("Setup complete. MotionBERT repo ready.")


# %% ── CELL B  Load MotionBERT Pose3D Model ────────────────────────────────────
#
# Checkpoint: MB_ft_h36m  (DSTformer fine-tuned on Human3.6M)
#
# HOW TO GET THE CHECKPOINT:
#   1. Go to https://github.com/Walter0807/MotionBERT
#   2. Find "Model Zoo" or "Checkpoint" section in the README
#   3. Download the pose3d / MB_ft_h36m checkpoint
#   4. Place it at: /content/MotionBERT/checkpoint/pose3d/best_epoch.bin
#   OR: update GDRIVE_ID below from the link in the README and gdown will fetch it.

import torch
from functools import partial
import torch.nn as nn
from lib.model.DSTformer import DSTformer

CKPT_PATH = '/content/MotionBERT/checkpoint/pose3d/best_epoch.bin'
os.makedirs('/content/MotionBERT/checkpoint/pose3d', exist_ok=True)

if not os.path.exists(CKPT_PATH):
    # Replace with the Google Drive file ID from the MotionBERT README
    GDRIVE_ID = "REPLACE_WITH_ID_FROM_README"
    os.system(f'gdown {GDRIVE_ID} -O {CKPT_PATH}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

model_pos = DSTformer(
    dim_in=3,          # x, y, confidence  (COCO 17 joints)
    dim_out=3,         # x, y, z           (H36M 17 joints output)
    dim_feat=256,
    dim_rep=512,
    depth=5,
    num_heads=8,
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    maxlen=243,
    num_joints=17,
)

checkpoint = torch.load(CKPT_PATH, map_location='cpu')
# Handle different checkpoint key conventions
state = (checkpoint.get('model')
         or checkpoint.get('model_pos')
         or checkpoint)
model_pos.load_state_dict(state, strict=True)
model_pos.eval().to(device)
print("MotionBERT loaded.")


# %% ── CELL C  Load & Preprocess keypoints_batter.csv → Per-Segment Tensors ──

import numpy as np
import pandas as pd

# ── Joint order constants ────────────────────────────────────────────────────

# COCO 17: input to MotionBERT (matches our CSV keypoint names)
COCO_JOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]

# H36M 17: output from MotionBERT
H36M_JOINTS = [
    'Hip', 'RHip', 'RKnee', 'RAnkle',
    'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Neck', 'Head',
    'LShoulder', 'LElbow', 'LWrist',
    'RShoulder', 'RElbow', 'RWrist',
]

# Bone connections for H36M skeleton (index pairs)
H36M_BONES = [
    # Spine chain
    (0, 7), (7, 8), (8, 9), (9, 10),
    # Left leg
    (0, 4), (4, 5), (5, 6),
    # Right leg
    (0, 1), (1, 2), (2, 3),
    # Left arm
    (8, 11), (11, 12), (12, 13),
    # Right arm
    (8, 14), (14, 15), (15, 16),
]

BONE_COLORS = {
    'spine':     '#FFFFFF',
    'left_leg':  '#00FFFF',
    'right_leg': '#FFA500',
    'left_arm':  '#4FC3F7',
    'right_arm': '#EF5350',
}
BONE_COLOR_MAP = [
    '#FFFFFF', '#FFFFFF', '#FFFFFF', '#FFFFFF',   # spine x4
    '#00FFFF', '#00FFFF', '#00FFFF',               # left leg
    '#FFA500', '#FFA500', '#FFA500',               # right leg
    '#4FC3F7', '#4FC3F7', '#4FC3F7',               # left arm
    '#EF5350', '#EF5350', '#EF5350',               # right arm
]

# ── Load data ────────────────────────────────────────────────────────────────

KP_PATH = os.path.join(DRIVE_PROJECT, 'keypoints_batter.csv')
if not os.path.exists(KP_PATH):
    # Generate from keypoints_raw.csv if batter CSV is not yet saved
    print("keypoints_batter.csv not found — generating from keypoints_raw.csv ...")
    RAW_PATH = os.path.join(DRIVE_PROJECT, 'keypoints_raw.csv')
    raw = pd.read_csv(RAW_PATH)

    # Inline batter selector (same logic as batter_selector.py)
    def select_batter(df):
        df = df.copy()
        df['bbox_area'] = (df['bbox_x2'] - df['bbox_x1']) * (df['bbox_y2'] - df['bbox_y1'])
        out = []
        for video, vdf in df.groupby('video', sort=False):
            pft = vdf.groupby(['frame', 'track_id'])['bbox_area'].first().reset_index()
            bpf = (pft.sort_values('bbox_area', ascending=False)
                      .groupby('frame').first().reset_index()[['frame', 'track_id']]
                      .sort_values('frame').reset_index(drop=True))
            bpf['segment_id'] = (bpf['track_id'] != bpf['track_id'].shift()).cumsum() - 1
            out.append(vdf.merge(bpf, on=['frame', 'track_id'], how='inner'))
        return pd.concat(out, ignore_index=True).drop(columns=['bbox_area'])

    kp = select_batter(raw)
    kp.to_csv(KP_PATH, index=False)
    print(f"Saved {len(kp)} rows → {KP_PATH}")
else:
    kp = pd.read_csv(KP_PATH)
    print(f"Loaded keypoints_batter.csv: {len(kp)} rows")

print(f"Segments: {kp.groupby(['video','segment_id']).ngroups}")


# ── Preprocessing functions ──────────────────────────────────────────────────

def get_video_dims(video_name):
    """
    Read actual pixel dimensions from the source video file.
    MotionBERT normalization is image-width-based, so we need W and H.
    """
    import cv2
    vpath = os.path.join(DRIVE_PROJECT, video_name)
    cap = cv2.VideoCapture(vpath)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if W == 0 or H == 0:
        raise RuntimeError(f"Could not read dimensions for {video_name}. "
                           f"Check the file is at {vpath}.")
    return W, H


def build_segment_tensor(seg_df, W, H, window=243):
    """
    Convert one segment's DataFrame rows into a padded (window, 17, 3) numpy array.
    The 3 channels are [x_norm, y_norm, confidence].

    Normalization: MotionBERT image-width-based normalization.
      x_norm = x / W * 2 - 1          maps [0, W] → [-1, 1]
      y_norm = y / W * 2 - H/W        maps [0, H] → [-H/W, H/W], preserving aspect ratio
    This matches exactly how MotionBERT was trained (normalize_screen_coordinates).

    Short segments (< window) are edge-padded symmetrically.
    Long segments are handled with a sliding window in Cell D.
    """
    frames = sorted(seg_df['frame'].unique())
    T = len(frames)
    frame_idx = {f: t for t, f in enumerate(frames)}

    seq = np.zeros((T, 17, 3), dtype=np.float32)   # (T, J, [x, y, conf])

    for row in seg_df.itertuples(index=False):
        t = frame_idx[row.frame]
        j = COCO_JOINTS.index(row.keypoint)
        seq[t, j] = [row.x, row.y, row.confidence]

    # MotionBERT image-width normalization (matches training-time preprocessing)
    seq_norm = seq.copy()
    seq_norm[:, :, 0] = seq[:, :, 0] / W * 2 - 1          # x: [0,W] → [-1,1]
    seq_norm[:, :, 1] = seq[:, :, 1] / W * 2 - H / W      # y: [0,H] → [-H/W, H/W]
    # confidence stays in [0, 1]

    return seq_norm, T, frames


def sliding_window_inference(seq_norm, T, model, device, window=243):
    """
    Run MotionBERT on a sequence of any length.

    For sequences shorter than `window`: pad + single forward pass.
    For sequences longer than `window`:  overlapping windows, average overlaps.

    Returns: np.ndarray of shape (T, 17, 3)  — 3D joints in H36M order.
    """
    if T <= window:
        pad_l = (window - T) // 2
        pad_r  = window - T - pad_l
        padded = np.pad(seq_norm, ((pad_l, pad_r), (0, 0), (0, 0)), mode='edge')
        x = torch.tensor(padded[None], dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(x).cpu().numpy()[0]           # (243, 17, 3)
        return out[pad_l: pad_l + T]                  # (T, 17, 3)
    else:
        # Accumulate with overlap-add
        out_sum   = np.zeros((T, 17, 3), dtype=np.float64)
        out_count = np.zeros((T, 1, 1),  dtype=np.float64)
        stride = window // 2
        for start in range(0, T, stride):
            end = min(start + window, T)
            chunk = seq_norm[start:end]
            if len(chunk) < window:
                pad_r = window - len(chunk)
                chunk = np.pad(chunk, ((0, pad_r), (0, 0), (0, 0)), mode='edge')
            x = torch.tensor(chunk[None], dtype=torch.float32).to(device)
            with torch.no_grad():
                out_chunk = model(x).cpu().numpy()[0]    # (243, 17, 3)
            valid = min(window, T - start)
            out_sum  [start:start + valid] += out_chunk[:valid]
            out_count[start:start + valid] += 1
        return (out_sum / out_count).astype(np.float32)


# Read video dimensions once per video (needed for correct normalization)
print("Reading video dimensions...")
video_dims = {}
for video_name in kp['video'].unique():
    W, H = get_video_dims(video_name)
    video_dims[video_name] = (W, H)
    print(f"  {video_name}: {W}x{H}")

# Build per-segment tensors (store in a dict, don't run model yet)
segments = {}
for (video, seg_id), seg_df in kp.groupby(['video', 'segment_id']):
    W, H = video_dims[video]
    seq_norm, T, frames = build_segment_tensor(seg_df, W, H)
    segments[(video, seg_id)] = {'seq_norm': seq_norm, 'T': T, 'frames': frames}
    print(f"  {video}  seg={seg_id}  T={T} frames")

print(f"\nTotal segments ready: {len(segments)}")


# %% ── CELL D  Run MotionBERT Inference → 3D Poses ───────────────────────────

results_3d = {}  # (video, seg_id) → {'frames': [...], 'pose_3d': (T,17,3)}

for (video, seg_id), seg in segments.items():
    pose_3d = sliding_window_inference(
        seg['seq_norm'], seg['T'], model_pos, device
    )
    results_3d[(video, seg_id)] = {
        'frames':   seg['frames'],
        'pose_3d':  pose_3d,         # (T, 17, 3) — H36M joints
    }
    print(f"✓ {video}  seg={seg_id}  output shape: {pose_3d.shape}")

print("\nInference complete.")


# %% ── CELL E  Save keypoints_3d.json ────────────────────────────────────────

import json

records = []
for (video, seg_id), data in results_3d.items():
    for t, frame in enumerate(data['frames']):
        joints = data['pose_3d'][t]        # (17, 3)
        frame_record = {
            'video':      video,
            'segment_id': int(seg_id),
            'frame':      int(frame),
            'joints': {
                name: {
                    'x': float(joints[j, 0]),
                    'y': float(joints[j, 1]),
                    'z': float(joints[j, 2]),
                }
                for j, name in enumerate(H36M_JOINTS)
            }
        }
        records.append(frame_record)

OUT_JSON = os.path.join(DRIVE_PROJECT, 'keypoints_3d.json')
with open(OUT_JSON, 'w') as f:
    json.dump(records, f, indent=2)

print(f"Saved {len(records)} frame records → {OUT_JSON}")
print(f"Example entry:\n{json.dumps(records[0], indent=2)}")


# %% ── CELL F  3D Skeleton Visualization → Animated MP4 ──────────────────────

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

def render_skeleton_video(video_name, seg_id, pose_3d, frames, out_path, fps=30):
    """
    Render a 3D animated skeleton video for one segment.

    pose_3d : (T, 17, 3) array in H36M joint order.

    AXIS MAPPING (key fix):
      MotionBERT outputs H36M camera-space coordinates:
        dim 0 → X  (left-right)
        dim 1 → Y  (UP in 3D — head positive, feet negative)
        dim 2 → Z  (depth, towards/away from camera)
      Matplotlib 3D has Z as the vertical "up" axis.
      So we remap:
        matplotlib X  ←  H36M X  (left-right, unchanged)
        matplotlib Y  ←  H36M Z  (depth)
        matplotlib Z  ←  H36M Y  (height → mpl "up" axis)
      If the skeleton appears upside-down, negate the last line: zs = -pose_3d[:,1]
    """
    T = len(frames)

    # Remap for correct orientation
    xs_all = pose_3d[:, :, 0]          # (T, 17) left-right
    ys_all = pose_3d[:, :, 2]          # (T, 17) depth
    zs_all = pose_3d[:, :, 1]          # (T, 17) height → mpl Z (up)

    margin = 0.2
    xlim = (xs_all.min() - margin, xs_all.max() + margin)
    ylim = (ys_all.min() - margin, ys_all.max() + margin)
    zlim = (zs_all.min() - margin, zs_all.max() + margin)

    fig = plt.figure(figsize=(10, 9), facecolor='#0A0A0A')
    ax  = fig.add_subplot(111, projection='3d', facecolor='#0A0A0A')

    def draw_frame(t):
        ax.cla()
        ax.set_facecolor('#0A0A0A')
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        ax.set_xlabel('X (left-right)', color='#555', fontsize=7)
        ax.set_ylabel('Z (depth)',      color='#555', fontsize=7)
        ax.set_zlabel('Y (height)',     color='#555', fontsize=7)
        ax.tick_params(colors='#333', labelsize=6)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        ax.grid(True, color='#1A1A1A', linewidth=0.4)
        ax.view_init(elev=15, azim=-70)

        joints = pose_3d[t]    # (17, 3)
        xs = joints[:, 0]
        ys = joints[:, 2]      # depth
        zs = joints[:, 1]      # height

        # Draw bones
        for bone_idx, (i, j) in enumerate(H36M_BONES):
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]],
                    color=BONE_COLOR_MAP[bone_idx], linewidth=2.5, alpha=0.95)

        # Draw joints
        ax.scatter(xs, ys, zs, c='white', s=18, zorder=5, depthshade=False)

        # Highlight wrists (yellow)
        wi_l = H36M_JOINTS.index('LWrist')
        wi_r = H36M_JOINTS.index('RWrist')
        ax.scatter(xs[wi_l], ys[wi_l], zs[wi_l], c='#FFFF00', s=55, zorder=6, depthshade=False)
        ax.scatter(xs[wi_r], ys[wi_r], zs[wi_r], c='#FFFF00', s=55, zorder=6, depthshade=False)

        # Ground plane (at ankle level)
        floor_z = zs_all.min()
        gx = np.linspace(xlim[0], xlim[1], 6)
        gy = np.linspace(ylim[0], ylim[1], 6)
        for gxi in gx:
            ax.plot([gxi, gxi], [ylim[0], ylim[1]], [floor_z, floor_z],
                    color='#1F1F1F', linewidth=0.5)
        for gyi in gy:
            ax.plot([xlim[0], xlim[1]], [gyi, gyi], [floor_z, floor_z],
                    color='#1F1F1F', linewidth=0.5)

        progress = int(t / max(T - 1, 1) * 100)
        ax.set_title(
            f'{os.path.splitext(video_name)[0]}  |  seg {seg_id}  '
            f'|  frame {frames[t]}  ({progress}%)',
            color='white', fontsize=10, pad=10,
        )

    anim = animation.FuncAnimation(fig, draw_frame, frames=T, interval=int(1000/fps))
    writer = animation.FFMpegWriter(fps=fps, bitrate=2500,
                                    extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    anim.save(out_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# Render one MP4 per (video, segment)
for (video, seg_id), data in results_3d.items():
    vname   = os.path.splitext(video)[0]
    out_mp4 = os.path.join(DRIVE_PROJECT, f'skeleton_3d_{vname}_seg{seg_id}.mp4')
    print(f"Rendering {vname} seg={seg_id}  ({data['T']} frames) ...")
    render_skeleton_video(
        video_name=video,
        seg_id=seg_id,
        pose_3d=data['pose_3d'],
        frames=data['frames'],
        out_path=out_mp4,
    )

print("\nAll videos saved to Drive/Baseball Project.")

# ── Inline preview: mid-swing frame per segment ──────────────────────────────
n_segs = len(results_3d)
fig, axes = plt.subplots(1, min(n_segs, 4), figsize=(5 * min(n_segs, 4), 5),
                          subplot_kw={'projection': '3d'},
                          facecolor='#0A0A0A')
if n_segs == 1:
    axes = [axes]

for ax, ((video, seg_id), data) in zip(axes, list(results_3d.items())[:4]):
    ax.set_facecolor('#0A0A0A')
    mid_t  = data['T'] // 2
    joints = data['pose_3d'][mid_t]

    # Same axis remap as render_skeleton_video
    xs = joints[:, 0]
    ys = joints[:, 2]   # depth
    zs = joints[:, 1]   # height → mpl Z (up)

    for bone_idx, (i, j) in enumerate(H36M_BONES):
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]],
                color=BONE_COLOR_MAP[bone_idx], linewidth=2)
    ax.scatter(xs, ys, zs, c='white', s=15, depthshade=False)
    ax.set_xlabel('X', color='#555', fontsize=7)
    ax.set_ylabel('Z(depth)', color='#555', fontsize=7)
    ax.set_zlabel('Y(height)', color='#555', fontsize=7)
    ax.set_title(f'seg {seg_id}  (mid-swing)', color='white', fontsize=9)
    ax.tick_params(colors='#333', labelsize=6)
    ax.view_init(elev=15, azim=-70)

fig.patch.set_facecolor('#0A0A0A')
plt.tight_layout()
plt.show()
print("Preview complete.")
