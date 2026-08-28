# ═════════════════════════════════════════════════════════════════════════════
# COLAB — 3D POSE ESTIMATION (MotionBERT 2D → 3D lifting)
# Clean, re-runnable version of notebook cells 14–18.
# Use a GPU runtime (Runtime → Change runtime type → T4 GPU).
#
# Input : <DRIVE>/Batting Key Point/keypoints_batter.csv   (or <DRIVE>/keypoints_batter.csv)
# Output: <DRIVE>/keypoints_3d.json                         (x,y,z per joint/frame/segment)
#         <DRIVE>/skeleton_3d_<video>_seg<id>.mp4           (animated 3D skeleton per swing)
#
# "Lifting" = inferring depth (z) from a single-camera 2D skeleton. MotionBERT
# (DSTformer, H36M-finetuned, ~37mm error) is the model.
# ═════════════════════════════════════════════════════════════════════════════

# %% ── CELL 1 · Setup ─────────────────────────────────────────────────────────
import os, sys, json, math
from google.colab import drive
drive.mount('/content/drive')

DRIVE = '/content/drive/My Drive/Baseball Project'
DIAG = f'{DRIVE}/Batting Diagnoses'          # CONVENTION: all output videos go here
VIDEO_FOLDER = f'{DRIVE}/Batting Videos'
os.makedirs(DIAG, exist_ok=True)
KP_CSV = f'{DRIVE}/Batting Key Point/keypoints_batter.csv'      # fallback: f'{DRIVE}/keypoints_batter.csv'
CKPT = '/content/drive/My Drive/models/MotionBERT/best_epoch.bin'
WINDOW = 243                                                    # MotionBERT temporal window

os.chdir('/content')
if not os.path.exists('MotionBERT'):
    os.system('git clone https://github.com/Walter0807/MotionBERT.git')
os.system('pip install -q einops timm')
sys.path.insert(0, '/content/MotionBERT')
print('Setup complete.')

# %% ── CELL 2 · Load DSTformer + checkpoint ──────────────────────────────────
import torch, torch.nn as nn
from functools import partial
from lib.model.DSTformer import DSTformer

assert os.path.exists(CKPT), (
    f'Checkpoint not found at {CKPT}. Download MB_ft_h36m best_epoch.bin from the '
    'MotionBERT README and place it there (gdown <id> -O {CKPT}).')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

model_pos = DSTformer(dim_in=3, dim_out=3, dim_feat=512, dim_rep=512, depth=5,
                      num_heads=8, mlp_ratio=2,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6), maxlen=243, num_joints=17)
ckpt = torch.load(CKPT, map_location='cpu')
state = ckpt.get('model') or ckpt.get('model_pos') or ckpt
state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
model_pos.load_state_dict(state, strict=True)
model_pos.eval().to(device)
print('MotionBERT loaded.')

# %% ── CELL 3 · Joint constants ──────────────────────────────────────────────
import numpy as np, pandas as pd, cv2

COCO_JOINTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
               'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
               'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
               'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
H36M_JOINTS = ['Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
               'Spine', 'Thorax', 'Neck', 'Head', 'LShoulder', 'LElbow', 'LWrist',
               'RShoulder', 'RElbow', 'RWrist']
H36M_BONES = [(0, 7), (7, 8), (8, 9), (9, 10), (0, 4), (4, 5), (5, 6), (0, 1), (1, 2),
              (2, 3), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]
BONE_COLOR = ['#FFFFFF']*4 + ['#00FFFF']*3 + ['#FFA500']*3 + ['#4FC3F7']*3 + ['#EF5350']*3

# %% ── CELL 4 · Preprocess CSV → per-segment normalized tensors ───────────────
kp = pd.read_csv(KP_CSV)
print(f'Loaded {len(kp)} rows, {kp.groupby(["video","segment_id"]).ngroups} segments')

def video_dims(name):
    cap = cv2.VideoCapture(f'{VIDEO_FOLDER}/{name}')
    W, H = int(cap.get(3)), int(cap.get(4)); cap.release()
    if not W or not H:
        raise RuntimeError(f'Cannot read dims for {name}')
    return W, H

def segment_tensor(seg_df, W, H):
    """(T,17,3) = [x_norm, y_norm, conf]. MotionBERT image-width normalization."""
    frames = sorted(seg_df['frame'].unique()); fi = {f: t for t, f in enumerate(frames)}
    seq = np.zeros((len(frames), 17, 3), np.float32)
    for r in seg_df.itertuples(index=False):
        seq[fi[r.frame], COCO_JOINTS.index(r.keypoint)] = [r.x, r.y, r.confidence]
    seq[:, :, 0] = seq[:, :, 0] / W * 2 - 1          # x → [-1,1]
    seq[:, :, 1] = seq[:, :, 1] / W * 2 - H / W      # y → preserves aspect ratio
    return seq, frames

def infer(seq, model, device, window=WINDOW):
    T = len(seq)
    if T <= window:
        pl = (window - T) // 2; pr = window - T - pl
        x = torch.tensor(np.pad(seq, ((pl, pr), (0, 0), (0, 0)), 'edge')[None]).float().to(device)
        with torch.no_grad():
            return model(x).cpu().numpy()[0][pl:pl + T]
    out = np.zeros((T, 17, 3)); cnt = np.zeros((T, 1, 1)); stride = window // 2
    for s in range(0, T, stride):
        ch = seq[s:s + window]
        if len(ch) < window:
            ch = np.pad(ch, ((0, window - len(ch)), (0, 0), (0, 0)), 'edge')
        x = torch.tensor(ch[None]).float().to(device)
        with torch.no_grad():
            oc = model(x).cpu().numpy()[0]
        v = min(window, T - s); out[s:s + v] += oc[:v]; cnt[s:s + v] += 1
    return (out / cnt).astype(np.float32)

dims = {v: video_dims(v) for v in kp['video'].unique()}
results = {}
for (video, seg), sdf in kp.groupby(['video', 'segment_id']):
    W, H = dims[video]
    seq, frames = segment_tensor(sdf, W, H)
    results[(video, seg)] = {'frames': frames, 'pose_3d': infer(seq, model_pos, device)}
    print(f'  {video} seg{seg}: {len(frames)} frames → {results[(video, seg)]["pose_3d"].shape}')
print('Inference complete.')

# %% ── CELL 5 · Save keypoints_3d.json ────────────────────────────────────────
records = []
for (video, seg), data in results.items():
    for t, frame in enumerate(data['frames']):
        j = data['pose_3d'][t]
        records.append({'video': video, 'segment_id': int(seg), 'frame': int(frame),
                        'joints': {n: {'x': float(j[i, 0]), 'y': float(j[i, 1]), 'z': float(j[i, 2])}
                                   for i, n in enumerate(H36M_JOINTS)}})
json.dump(records, open(f'{DRIVE}/keypoints_3d.json', 'w'), indent=2)
print(f'Saved {len(records)} records → {DRIVE}/keypoints_3d.json')

# %% ── CELL 6 · Render animated 3D skeleton MP4 per segment ───────────────────
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
# Vertical axis = z (index 2), up = -z. mpl Z(up) ← -z, mpl Y(depth) ← y, mpl X ← x.

def render(video, seg, pose, frames, out, fps=30):
    xs_a, ys_a, zs_a = pose[:, :, 0], pose[:, :, 1], -pose[:, :, 2]
    m = 0.2
    xl = (xs_a.min()-m, xs_a.max()+m); yl = (ys_a.min()-m, ys_a.max()+m); zl = (zs_a.min()-m, zs_a.max()+m)
    fig = plt.figure(figsize=(9, 8), facecolor='#0A0A0A')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0A0A0A')
    def frame_draw(t):
        ax.cla(); ax.set_facecolor('#0A0A0A')
        ax.set_xlim(xl); ax.set_ylim(yl); ax.set_zlim(zl)
        # FRONT-on camera (matches how the video is shot); true proportions, no distortion.
        ax.view_init(elev=8, azim=-85)
        ax.set_box_aspect((xl[1]-xl[0], yl[1]-yl[0], zl[1]-zl[0]))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.grid(False)
        j = pose[t]; xs, ys, zs = j[:, 0], j[:, 1], -j[:, 2]
        for bi, (a, b) in enumerate(H36M_BONES):
            ax.plot([xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]], color=BONE_COLOR[bi], lw=2.5)
        ax.scatter(xs, ys, zs, c='white', s=16, depthshade=False)
        for w in ('LWrist', 'RWrist'):
            i = H36M_JOINTS.index(w); ax.scatter(xs[i], ys[i], zs[i], c='#FFFF00', s=50, depthshade=False)
        ax.set_title(f'{os.path.splitext(video)[0]} seg{seg} f{frames[t]}', color='white', fontsize=9)
    anim = animation.FuncAnimation(fig, frame_draw, frames=len(frames), interval=1000/fps)
    anim.save(out, writer=animation.FFMpegWriter(fps=fps, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']), dpi=110)
    plt.close(fig); print(f'  → {out}')

for (video, seg), data in results.items():
    render(video, seg, data['pose_3d'], data['frames'],
           f'{DIAG}/skeleton_3d_{os.path.splitext(video)[0]}_seg{seg}.mp4')
print(f'All 3D skeleton videos saved to {DIAG}.')
