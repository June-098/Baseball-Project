# ═════════════════════════════════════════════════════════════════════════════
# COLAB — BAT TRACKING with a TrackNet-style temporal model
# Use a GPU runtime. This is the path MEMORY.md flagged for the deferred bat
# tracking: a purpose-built temporal model instead of frame-by-frame YOLO.
#
# WHY TrackNet (and the honest catch): plain YOLO collapses to ~17% detection in
# the contact zone because an 80 mph bat is a motion blur on a 30fps phone.
# TrackNet takes THREE consecutive frames at once, so motion is a feature, not a
# problem. The catch: it is NOT zero-shot — it must be TRAINED on labeled bat
# positions, and the labels MUST include blurred contact-zone frames (that is the
# whole point). Plan for ~200-500 labeled frames across your clips.
#
# We track TWO points — the bat KNOB and the bat TIP — so we get orientation, not
# just a dot. The sweet spot ≈ knob + 0.7*(tip-knob) is what attack angle needs.
#
# Pipeline (cells below):
#   1 Setup     2 Model     3 Labels (YOLO pseudo + manual)     4 Dataset+aug
#   5 Train     6 Inference → bat_track.csv + annotated MP4
# Output: <DRIVE>/models/TrackNet/tracknet_bat.pt , <DRIVE>/bat_track.csv ,
#         <DRIVE>/bat_tracknet_<video>.mp4
# ═════════════════════════════════════════════════════════════════════════════

# %% ── CELL 1 · Setup & config ────────────────────────────────────────────────
import os, math, json, random, glob
import numpy as np, pandas as pd, cv2, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from google.colab import drive
drive.mount('/content/drive')

DRIVE = '/content/drive/My Drive/Baseball Project'
DIAG = f'{DRIVE}/Batting Diagnoses'                    # CONVENTION: output videos go here
VIDEO_FOLDER = f'{DRIVE}/Batting Videos'
os.makedirs(DIAG, exist_ok=True)
LABELS_CSV = f'{DRIVE}/bat_labels.csv'                 # columns: video,frame,knob_x,knob_y,tip_x,tip_y
YOLO_BOXES = f'{DRIVE}/bat_boxes_raw.csv'              # for pseudo-labels (optional bootstrap)
# Videos to leave OUT of training (the slow-mo pro clips bloat the cache + add easy frames).
# Bootstrap skips these, and training ignores them — so you don't have to edit the CSV.
EXCLUDE_VIDEOS = ['Ichiro Slowmo.mp4', 'Jimmy Rollins Slowmo.mp4', 'Jose Bautistia Slowmo.mp4']
CKPT_DIR = f'{DRIVE}/models/TrackNet'; os.makedirs(CKPT_DIR, exist_ok=True)
CKPT = f'{CKPT_DIR}/tracknet_bat.pt'

IN_H, IN_W = 288, 512          # network input size (divisible by 8)
SIGMA = 4.0                    # Gaussian radius (output-res px)
N_POINTS = 2                   # knob, tip
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)


# %% ── CELL 2 · TrackNet-style temporal U-Net (3 frames → 2 heatmaps) ─────────
def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

class TrackNetBat(nn.Module):
    """VGG-style encoder + symmetric decoder with skips. Input 9ch (3 RGB frames),
    output N_POINTS sigmoid heatmaps for the MIDDLE frame."""
    def __init__(self, n_pts=N_POINTS):
        super().__init__()
        self.e1 = nn.Sequential(cbr(9, 64), cbr(64, 64))
        self.e2 = nn.Sequential(cbr(64, 128), cbr(128, 128))
        self.e3 = nn.Sequential(cbr(128, 256), cbr(256, 256), cbr(256, 256))
        self.bott = nn.Sequential(cbr(256, 512), cbr(512, 512))
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.d3 = nn.Sequential(cbr(512 + 256, 256), cbr(256, 256))
        self.d2 = nn.Sequential(cbr(256 + 128, 128), cbr(128, 128))
        self.d1 = nn.Sequential(cbr(128 + 64, 64), cbr(64, 64))
        self.out = nn.Conv2d(64, n_pts, 1)

    def forward(self, x):
        e1 = self.e1(x)                 # H
        e2 = self.e2(self.pool(e1))     # H/2
        e3 = self.e3(self.pool(e2))     # H/4
        b = self.bott(self.pool(e3))    # H/8
        d = self.d3(torch.cat([self.up(b), e3], 1))   # H/4
        d = self.d2(torch.cat([self.up(d), e2], 1))   # H/2
        d = self.d1(torch.cat([self.up(d), e1], 1))   # H
        return torch.sigmoid(self.out(d))


# %% ── CELL 3 · Labels ────────────────────────────────────────────────────────
# You need a labels CSV: video,frame,knob_x,knob_y,tip_x,tip_y  (pixel coords in
# the ORIGINAL frame). Two ways to build it:
#
# (A) BOOTSTRAP from YOLO boxes — free, but only covers the EASY (slow-bat) frames.
#     Run this to seed bat_labels.csv, THEN you MUST add contact-zone frames by hand.
def bootstrap_from_yolo(yolo_csv=YOLO_BOXES, out=LABELS_CSV, conf_min=0.5):
    df = pd.read_csv(yolo_csv)
    rows = []
    for r in df.itertuples(index=False):
        if r.video in EXCLUDE_VIDEOS:
            continue
        if getattr(r, 'bat_conf', 1.0) < conf_min:
            continue
        # bat ≈ the long diagonal of its box; knob/tip = the two diagonal ends
        x1, y1, x2, y2 = r.bat_x1, r.bat_y1, r.bat_x2, r.bat_y2
        if (x2 - x1) >= (y2 - y1):      # wide box → roughly horizontal bat
            knob, tip = (x1, y2), (x2, y1)
        else:                            # tall box → roughly vertical bat
            knob, tip = (x1, y1), (x2, y2)
        rows.append({'video': r.video, 'frame': r.frame,
                     'knob_x': knob[0], 'knob_y': knob[1], 'tip_x': tip[0], 'tip_y': tip[1]})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'Bootstrapped {len(rows)} pseudo-labels → {out}. '
          'NOTE: knob/tip from a box are approximate — and contact frames are missing. Add them by hand.')

# (B) MANUAL (the important frames). Recommended: CVAT or Roboflow "keypoint"
#     project, two points per frame (knob, tip), export to the CSV schema above —
#     CVAT is already in the project stack (MEMORY.md). Prioritize the ~6-10 frames
#     around each contact (where YOLO fails). For a quick in-Colab clicker:
def click_annotate(video, frames, out=LABELS_CSV):
    """Minimal in-Colab annotator: shows each frame, you click KNOB then TIP."""
    from IPython.display import display, Javascript
    from google.colab.output import eval_js
    import base64
    cap = cv2.VideoCapture(f'{VIDEO_FOLDER}/{video}')
    existing = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    rows = []
    for fr in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, img = cap.read()
        if not ok:
            continue
        _, buf = cv2.imencode('.jpg', img); b64 = base64.b64encode(buf).decode()
        js = Javascript('''
          async function click2(src){
            const img=new Image(); img.src=src; await img.decode();
            const c=document.createElement('canvas'); c.width=img.width; c.height=img.height;
            document.body.appendChild(c); c.getContext('2d').drawImage(img,0,0);
            const pts=[]; const lab=document.createElement('div');
            lab.textContent='Click KNOB then TIP'; document.body.appendChild(lab);
            return await new Promise(res=>{c.onclick=e=>{const r=c.getBoundingClientRect();
              pts.push([(e.clientX-r.left)*img.width/r.width,(e.clientY-r.top)*img.height/r.height]);
              if(pts.length==2){c.remove();lab.remove();res(pts);}};});
          }''')
        display(js)
        pts = eval_js(f'click2("data:image/jpeg;base64,{b64}")')
        rows.append({'video': video, 'frame': int(fr), 'knob_x': pts[0][0], 'knob_y': pts[0][1],
                     'tip_x': pts[1][0], 'tip_y': pts[1][1]})
    cap.release()
    pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).to_csv(out, index=False)
    print(f'Saved {len(rows)} manual labels → {out}')

# Example:
# bootstrap_from_yolo()
# click_annotate('Chae_friend_Righty_Batting_V1.mov', [60,62,64,66,68,70])   # contact zone


# %% ── CELL 4 · Dataset (3-frame stacks → Gaussian heatmaps) + augmentation ───
def gaussian(cx, cy, h, w, sigma=SIGMA):
    """Gaussian blob, computed only in a small window around (cx,cy) (fast)."""
    hm = np.zeros((h, w), np.float32)
    if cx is None or np.isnan(cx):
        return hm
    cx, cy = float(cx), float(cy); r = int(4 * sigma)
    x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
    y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return hm
    ys, xs = np.mgrid[y0:y1, x0:x1]
    hm[y0:y1, x0:x1] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return hm

def motion_blur(img, k):
    if k < 3:
        return img
    ang = random.uniform(0, math.pi); kern = np.zeros((k, k), np.float32)
    cx = (k - 1) / 2
    for i in range(k):
        x = int(round(cx + (i - cx) * math.cos(ang))); y = int(round(cx + (i - cx) * math.sin(ang)))
        if 0 <= x < k and 0 <= y < k:
            kern[y, x] = 1
    s = kern.sum(); kern = kern / s if s else kern
    return cv2.filter2D(img, -1, kern)

# ── Frame cache: decode every NEEDED frame ONCE into RAM (the speed fix) ──────
# The old dataset re-seeked H.264 video on the Drive mount 3x per sample per epoch
# (random seek + decode + network I/O) → GPU starved. We now read each video
# sequentially once, keep only the resized frames we need, and train from RAM.
import shutil
FRAME_CACHE = {}     # (video, frame_idx) -> uint8 (IN_H, IN_W, 3) BGR, already resized
VID_DIMS = {}        # video -> (W0, H0) original size, for scaling the labels

def _load_labels(csv=LABELS_CSV):
    """Read labels, dropping any video in EXCLUDE_VIDEOS (e.g. the slow-mo pro clips)."""
    df = pd.read_csv(csv)
    if EXCLUDE_VIDEOS:
        df = df[~df['video'].isin(EXCLUDE_VIDEOS)].reset_index(drop=True)
    return df

def build_frame_cache(labels_csv, local_dir='/content/_vids'):
    lab = _load_labels(labels_csv)
    os.makedirs(local_dir, exist_ok=True)
    FRAME_CACHE.clear(); VID_DIMS.clear()
    for video, vdf in lab.groupby('video'):
        need = {f for fr in vdf['frame'].astype(int) for f in (fr - 1, fr, fr + 1) if f >= 0}
        local = f'{local_dir}/{video}'
        if not os.path.exists(local):                     # copy off the slow Drive mount once
            shutil.copy(f'{VIDEO_FOLDER}/{video}', local)
        cap = cv2.VideoCapture(local)
        VID_DIMS[video] = (int(cap.get(3)), int(cap.get(4)))
        maxf = max(need); idx = 0
        while idx <= maxf:                                # ONE sequential pass, no seeking
            ok, img = cap.read()
            if not ok:
                break
            if idx in need:
                FRAME_CACHE[(video, idx)] = cv2.resize(img, (IN_W, IN_H))
            idx += 1
        cap.release()
        print(f'  cached {sum(1 for k in FRAME_CACHE if k[0]==video)} frames from {video}')
    gb = len(FRAME_CACHE) * IN_W * IN_H * 3 / 1e9
    print(f'Frame cache ready: {len(FRAME_CACHE)} frames (~{gb:.2f} GB in RAM)')

class BatDataset(Dataset):
    """Item: 3 consecutive frames (t-1,t,t+1) → heatmaps(knob,tip) for frame t.
    Reads from FRAME_CACHE (RAM) — no video I/O. Augmentation still varies per epoch."""
    def __init__(self, items, train=True):
        self.items = items; self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r = self.items[i]
        f1 = FRAME_CACHE.get((r.video, r.frame))
        f0 = FRAME_CACHE.get((r.video, r.frame - 1), f1)
        f2 = FRAME_CACHE.get((r.video, r.frame + 1), f1)
        W0, H0 = VID_DIMS[r.video]
        sx, sy = IN_W / W0, IN_H / H0
        frames = [f0, f1, f2]                              # already resized to IN_W x IN_H
        kx, ky, tx, ty = r.knob_x * sx, r.knob_y * sy, r.tip_x * sx, r.tip_y * sy
        if self.train:
            if random.random() < 0.6:                     # motion blur (the key aug)
                k = random.choice([5, 9, 13, 17]); frames = [motion_blur(f, k) for f in frames]
            if random.random() < 0.5:                     # horizontal flip
                frames = [f[:, ::-1] for f in frames]; kx = IN_W - kx; tx = IN_W - tx
            if random.random() < 0.5:                     # brightness
                g = random.uniform(0.7, 1.3); frames = [np.clip(f * g, 0, 255).astype(np.uint8) for f in frames]
        x = np.concatenate([f[:, :, ::-1].transpose(2, 0, 1) for f in frames], 0).astype(np.float32) / 255.0
        y = np.stack([gaussian(kx, ky, IN_H, IN_W), gaussian(tx, ty, IN_H, IN_W)], 0)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y)


# %% ── CELL 5 · Train ─────────────────────────────────────────────────────────
def wbce(pred, target, w_pos=20.0):
    """Weighted BCE — heatmaps are mostly zeros, so up-weight the positive blob."""
    pred = pred.clamp(1e-6, 1 - 1e-6)
    return -(w_pos * target * pred.log() + (1 - target) * (1 - pred).log()).mean()

def train(epochs=40, bs=8, lr=1e-3, val_frac=0.15, workers=2):
    import time
    if not FRAME_CACHE:
        print('Building frame cache (one-time)...'); build_frame_cache(LABELS_CSV)
    lab = _load_labels().sort_values(['video', 'frame']).reset_index(drop=True)
    print(f'Training on {len(lab)} labels across {lab.video.nunique()} videos: {sorted(lab.video.unique())}')
    items = list(lab.itertuples(index=False)); random.Random(0).shuffle(items)
    n_val = max(1, int(len(items) * val_frac))
    va = BatDataset(items[:n_val], train=False)
    tr = BatDataset(items[n_val:], train=True)
    dl = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=workers,
                    pin_memory=True, persistent_workers=workers > 0)
    vdl = DataLoader(va, batch_size=bs, num_workers=workers, pin_memory=True,
                     persistent_workers=workers > 0)
    net = TrackNetBat().to(device); opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best = 1e9
    for ep in range(epochs):
        t0 = time.time(); net.train(); tl = 0
        for x, y in dl:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(); loss = wbce(net(x), y); loss.backward(); opt.step(); tl += loss.item()
        net.eval(); vl = 0
        with torch.no_grad():
            for x, y in vdl:
                vl += wbce(net(x.to(device)), y.to(device)).item()
        vl /= max(len(vdl), 1); sched.step()
        print(f'epoch {ep+1}/{epochs}  train {tl/len(dl):.3f}  val {vl:.3f}  ({time.time()-t0:.0f}s)')
        if vl < best:
            best = vl; torch.save(net.state_dict(), CKPT); print(f'  saved → {CKPT}')
    print('Training done. Best val', round(best, 3))

# train()   # uncomment after you have enough labels (incl. contact-zone frames)
# If you hit RAM limits from the cache, train on fewer labels (e.g. drop the slow-mo
# pro clips) or set workers=0. If you hit GPU OOM, lower bs (e.g. bs=4).


# %% ── CELL 6 · Inference → bat_track.csv + annotated MP4 ─────────────────────
SWEET = 0.7   # sweet spot fraction from knob → tip

def peak(hm, thr=0.3):
    i = int(hm.argmax()); y, x = divmod(i, hm.shape[1]); v = float(hm[y, x])
    return (x, y, v) if v >= thr else (None, None, v)

def track_video(video, conf_thr=0.3):
    net = TrackNetBat().to(device); net.load_state_dict(torch.load(CKPT, map_location=device)); net.eval()
    cap = cv2.VideoCapture(f'{VIDEO_FOLDER}/{video}')
    W0, H0 = int(cap.get(3)), int(cap.get(4)); FPS = cap.get(5) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    sx, sy = W0 / IN_W, H0 / IN_H
    out = cv2.VideoWriter(f'{DIAG}/bat_tracknet_{os.path.splitext(video)[0]}.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W0, H0))
    rows, trail = [], []
    small = [cv2.resize(f, (IN_W, IN_H))[:, :, ::-1].transpose(2, 0, 1) for f in frames]
    for t in range(len(frames)):
        a, b, c = small[max(t-1, 0)], small[t], small[min(t+1, len(frames)-1)]
        x = torch.from_numpy(np.concatenate([a, b, c], 0)[None].astype(np.float32) / 255.0).to(device)
        with torch.no_grad():
            hm = net(x)[0].cpu().numpy()
        (kx, ky, kc), (tx, ty, tc) = peak(hm[0], conf_thr), peak(hm[1], conf_thr)
        rec = {'video': video, 'frame': t, 'knob_x': None, 'knob_y': None, 'tip_x': None,
               'tip_y': None, 'sweet_x': None, 'sweet_y': None, 'knob_conf': round(kc, 3), 'tip_conf': round(tc, 3)}
        frame = frames[t]
        if kx is not None and tx is not None:
            KX, KY, TX, TY = kx*sx, ky*sy, tx*sx, ty*sy
            SXp, SYp = KX + SWEET*(TX-KX), KY + SWEET*(TY-KY)
            rec.update({'knob_x': KX, 'knob_y': KY, 'tip_x': TX, 'tip_y': TY, 'sweet_x': SXp, 'sweet_y': SYp})
            cv2.line(frame, (int(KX), int(KY)), (int(TX), int(TY)), (0, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(frame, (int(KX), int(KY)), 5, (255, 0, 0), -1)      # knob blue
            cv2.circle(frame, (int(TX), int(TY)), 5, (0, 0, 255), -1)      # tip red
            cv2.circle(frame, (int(SXp), int(SYp)), 4, (0, 255, 0), -1)    # sweet green
            trail.append((int(SXp), int(SYp)))
        trail = trail[-15:]
        if len(trail) >= 2:
            cv2.polylines(frame, [np.array(trail, np.int32)], False, (0, 255, 0), 2, cv2.LINE_AA)
        rows.append(rec); out.write(frame)
    out.release()
    df = pd.DataFrame(rows)
    cov = df['sweet_x'].notna().mean() * 100
    print(f'{video}: tracked {cov:.0f}% of frames → annotated MP4 in Batting Diagnoses')
    return df   # CSV is written by track_all() so multiple videos accumulate

def track_all(videos=None):
    """Track every clip (minus EXCLUDE_VIDEOS) and write ONE combined bat_track.csv.
    Pass a list to track specific clips, e.g. track_all(['Chae_friend_Righty_Batting_V1.mov'])."""
    if videos is None:
        videos = sorted(v for v in os.listdir(VIDEO_FOLDER)
                        if v.lower().endswith(('.mp4', '.mov', '.avi')) and v not in EXCLUDE_VIDEOS)
    full = pd.concat([track_video(v) for v in videos], ignore_index=True)
    full.to_csv(f'{DRIVE}/bat_track.csv', index=False)
    print(f'\nCombined {len(full)} rows from {len(videos)} clip(s) → {DRIVE}/bat_track.csv')
    return full

# track_all(['Chae_friend_Righty_Batting_V1.mov'])   # smoke-test one clip first
# track_all()                                          # then run all your clips
