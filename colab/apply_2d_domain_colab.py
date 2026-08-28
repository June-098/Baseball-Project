# ─────────────────────────────────────────────────────────────────────────────
# COLAB CELL — Apply the 2D Domain (Biomechanical Overlay)
# Paste this AFTER the MotionBERT cells in YOLO_Baseball.ipynb.
#
# It draws the four swing-relevant 2D angles on each batting video and writes a
# per-segment metrics file. 2D only — no 3D, no bat tracking required.
#   Hip-shoulder separation · Spine tilt · Front-knee angle · Attack-angle proxy
#
# Outputs → /content/drive/My Drive/Baseball Project/
#   skeleton_2d_biomech_<video>.mp4   (one per clip)
#   2d_metrics.json                    (per-segment: per-frame + summary)
#
# NOTE: every angle is a 2D, view-dependent PROXY (the "approximable from body
# keypoints" tier on Baseball Savant). Exact values need 3D / bat tracking.
# ─────────────────────────────────────────────────────────────────────────────
import os, json, math
import cv2, numpy as np, pandas as pd
from google.colab import drive
drive.mount('/content/drive')

DRIVE_PROJECT = '/content/drive/My Drive/Baseball Project'
DIAG          = f'{DRIVE_PROJECT}/Batting Diagnoses'     # CONVENTION: output videos go here
VIDEO_FOLDER  = f'{DRIVE_PROJECT}/Batting Videos'
KP_CSV        = f'{DRIVE_PROJECT}/Batting Key Point/keypoints_batter.csv'   # or {DRIVE_PROJECT}/keypoints_batter.csv
os.makedirs(DIAG, exist_ok=True)
CONF          = 0.3
VIDEO_EXTS    = ('.mp4', '.mov', '.MOV', '.avi', '.AVI', '.MP4')

IDEAL_AA = (5.0, 20.0)        # MLB ideal attack-angle band
GOOD_SEP = 25.0               # good hip-shoulder separation
MIN_SEG_FRAMES, TRAIL_LEN, SMOOTH_WIN = 20, 10, 5

C_GREEN, C_RED, C_ORANGE = (0, 200, 0), (0, 0, 255), (0, 165, 255)
C_CYAN, C_MAG, C_YELLOW, C_WHITE = (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)
RED = (0, 0, 255)

# ── COCO → 15-landmark mapping (neck = avg shoulders, center_hip = avg hips) ──
DIRECT = {'head': 'nose', 'left_shoulder': 'left_shoulder', 'left_elbow': 'left_elbow',
          'left_wrist': 'left_wrist', 'right_shoulder': 'right_shoulder', 'right_elbow': 'right_elbow',
          'right_wrist': 'right_wrist', 'left_hip': 'left_hip', 'right_hip': 'right_hip',
          'left_knee': 'left_knee', 'right_knee': 'right_knee', 'left_ankle': 'left_ankle',
          'right_ankle': 'right_ankle'}
LM_COLOR = {'head': (80, 80, 80), 'neck': (0, 255, 0), 'left_shoulder': (0, 255, 255),
            'left_elbow': (255, 0, 255), 'left_wrist': (255, 0, 200), 'right_shoulder': (0, 200, 0),
            'right_elbow': (0, 165, 255), 'right_wrist': (255, 255, 0), 'center_hip': (0, 220, 180),
            'left_hip': (200, 0, 255), 'right_hip': (0, 200, 255), 'left_knee': (255, 200, 0),
            'right_knee': (180, 0, 255), 'left_ankle': (0, 80, 255), 'right_ankle': (255, 50, 180)}
CONNECTIONS = [('head', 'neck'), ('neck', 'left_shoulder'), ('neck', 'right_shoulder'),
               ('neck', 'center_hip'), ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
               ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
               ('center_hip', 'left_hip'), ('center_hip', 'right_hip'), ('left_hip', 'left_knee'),
               ('left_knee', 'left_ankle'), ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')]


def compute_landmarks(coco):
    lm = {}
    for name, c in DIRECT.items():
        p = coco.get(c)
        if p and p[2] >= CONF:
            lm[name] = (p[0], p[1])
    def avg2(a, b):
        pa, pb = coco.get(a), coco.get(b)
        ao, bo = (pa and pa[2] >= CONF), (pb and pb[2] >= CONF)
        if ao and bo: return ((pa[0]+pb[0])/2, (pa[1]+pb[1])/2)
        p = pa if ao else (pb if bo else None)
        return (p[0], p[1]) if p else None
    n = avg2('left_shoulder', 'right_shoulder');  h = avg2('left_hip', 'right_hip')
    if n: lm['neck'] = n
    if h: lm['center_hip'] = h
    return lm


def line_ang(p, q): return math.degrees(math.atan2(q[1]-p[1], q[0]-p[0]))


def interior(a, b, c):
    v1, v2 = (a[0]-b[0], a[1]-b[1]), (c[0]-b[0], c[1]-b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 == 0 or n2 == 0: return None
    return math.degrees(math.acos(max(-1, min(1, (v1[0]*v2[0]+v1[1]*v2[1])/(n1*n2)))))


def handed(v):
    n = v.lower()
    if 'lefty' in n or 'ichiro' in n: return 'lefty'
    return 'righty'


def fmetrics(lm, hd):
    m = {'hip_shoulder_sep': None, 'spine_tilt': None, 'front_knee_angle': None}
    if all(k in lm for k in ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip')):
        d = abs(line_ang(lm['left_shoulder'], lm['right_shoulder']) - line_ang(lm['left_hip'], lm['right_hip'])) % 180
        m['hip_shoulder_sep'] = round(min(d, 180-d), 1)
    if 'neck' in lm and 'center_hip' in lm:
        sp = (lm['neck'][0]-lm['center_hip'][0], lm['neck'][1]-lm['center_hip'][1]); n = math.hypot(*sp)
        if n: m['spine_tilt'] = round(math.degrees(math.acos(max(-1, min(1, -sp[1]/n)))), 1)
    side = 'left' if hd == 'righty' else 'right'
    if all(f'{side}_{j}' in lm for j in ('hip', 'knee', 'ankle')):
        a = interior(lm[f'{side}_hip'], lm[f'{side}_knee'], lm[f'{side}_ankle'])
        if a is not None: m['front_knee_angle'] = round(a, 1)
    return m


def hands_pt(coco):
    pts = [(coco[w][0], coco[w][1]) for w in ('left_wrist', 'right_wrist') if coco.get(w) and coco[w][2] >= CONF]
    return (sum(p[0] for p in pts)/len(pts), sum(p[1] for p in pts)/len(pts)) if pts else None


def mov_avg(arr, win):
    out = arr.copy(); half = win//2
    for i in range(len(arr)):
        ch = arr[max(0, i-half):min(len(arr), i+half+1)]; v = ch[~np.isnan(ch[:, 0])]
        if len(v): out[i] = v.mean(axis=0)
    return out


def wrist_path(frames, hands):
    T = len(frames); pos = np.full((T, 2), np.nan)
    for t, f in enumerate(frames):
        if hands.get(f) is not None: pos[t] = hands[f]
    ps = mov_avg(pos, SMOOTH_WIN); ang, spd = {}, {}
    for t, f in enumerate(frames):
        lo, hi = max(0, t-1), min(T-1, t+1)
        if hi == lo or np.isnan(ps[lo, 0]) or np.isnan(ps[hi, 0]): continue
        dx, dy = ps[hi, 0]-ps[lo, 0], ps[hi, 1]-ps[lo, 1]
        ang[f] = round(math.degrees(math.atan2(-dy, dx)), 1); spd[f] = math.hypot(dx, dy)
    contact = max(spd, key=spd.get) if (spd and T >= MIN_SEG_FRAMES) else None
    return {f: {'attack_angle': ang.get(f), 'speed': round(spd.get(f, 0), 2)} for f in frames if f in ang}, contact


def draw_skel(frame, lm):
    for a, b in CONNECTIONS:
        if a in lm and b in lm:
            cv2.line(frame, tuple(map(int, lm[a])), tuple(map(int, lm[b])), RED, 2, cv2.LINE_AA)
    for name, (x, y) in lm.items():
        cv2.circle(frame, (int(x), int(y)), 3, LM_COLOR[name], -1, cv2.LINE_AA)
        cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 0), 1, cv2.LINE_AA)


def arc(frame, c, p1, p2, r, col):
    a1, a2 = line_ang(c, p1), line_ang(c, p2)
    if abs(a2-a1) > 180: a1, a2 = (a1+360, a2) if a2 > a1 else (a1, a2+360)
    cv2.ellipse(frame, (int(c[0]), int(c[1])), (r, r), 0, min(a1, a2), max(a1, a2), col, 2, cv2.LINE_AA)


def label(frame, t, org, col, s=0.5, th=1):
    cv2.putText(frame, t, org, cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), th+2, cv2.LINE_AA)
    cv2.putText(frame, t, org, cv2.FONT_HERSHEY_SIMPLEX, s, col, th, cv2.LINE_AA)


def process(video, kp, out):
    vkp = kp[kp['video'] == video]
    if not len(vkp): return
    hd = handed(video)
    coco_f = {}
    for r in vkp.itertuples(index=False):
        coco_f.setdefault(r.frame, {})[r.keypoint] = (r.x, r.y, r.confidence)
    seg_of = vkp.groupby('frame')['segment_id'].first().to_dict()
    lm_f = {f: compute_landmarks(c) for f, c in coco_f.items()}
    sm_f = {f: fmetrics(lm_f[f], hd) for f in coco_f}
    hands = {f: hands_pt(c) for f, c in coco_f.items() if hands_pt(c)}

    aa_f, contact_f, cs_seg = {}, {}, {}
    for seg, sdf in vkp.groupby('segment_id'):
        frames = sorted(sdf['frame'].unique())
        pf, contact = wrist_path(frames, hands); aa_f.update(pf)
        if contact is not None:
            contact_f[contact] = seg
            cs_seg[seg] = {'frame': int(contact), **sm_f.get(contact, {}),
                           'attack_angle': pf.get(contact, {}).get('attack_angle')}
        seps = [sm_f[f]['hip_shoulder_sep'] for f in frames if sm_f[f]['hip_shoulder_sep'] is not None]
        knees = [sm_f[f]['front_knee_angle'] for f in frames if sm_f[f]['front_knee_angle'] is not None]
        out.append({'video': video, 'segment_id': int(seg), 'handedness': hd, 'n_frames': len(frames),
                    'contact_frame': int(contact) if contact is not None else None, 'at_contact': cs_seg.get(seg),
                    'hip_shoulder_sep_range': [round(min(seps), 1), round(max(seps), 1)] if seps else None,
                    'front_knee_angle_range': [round(min(knees), 1), round(max(knees), 1)] if knees else None,
                    'per_frame': {str(f): {**sm_f[f], 'attack_angle': aa_f.get(f, {}).get('attack_angle')} for f in frames}})

    cap = cv2.VideoCapture(f'{VIDEO_FOLDER}/{video}')
    W, H = int(cap.get(3)), int(cap.get(4)); FPS = cap.get(5) or 30.0; total = int(cap.get(7))
    print(f'  {video}  {W}x{H}@{FPS:.0f}  {total}f  ({hd})')
    wr = cv2.VideoWriter(f'{DIAG}/skeleton_2d_biomech_{os.path.splitext(video)[0]}.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    trail, i = [], 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if i in lm_f:
            lm, m, an = lm_f[i], sm_f[i], aa_f.get(i); hp = hands.get(i)
            trail.append((int(hp[0]), int(hp[1])) if hp else None); trail = trail[-TRAIL_LEN:]
            draw_skel(frame, lm)
            if 'left_shoulder' in lm and 'right_shoulder' in lm:
                cv2.line(frame, tuple(map(int, lm['left_shoulder'])), tuple(map(int, lm['right_shoulder'])), C_CYAN, 2, cv2.LINE_AA)
            if 'left_hip' in lm and 'right_hip' in lm:
                cv2.line(frame, tuple(map(int, lm['left_hip'])), tuple(map(int, lm['right_hip'])), C_MAG, 2, cv2.LINE_AA)
            side = 'left' if hd == 'righty' else 'right'
            if m.get('front_knee_angle') is not None and all(f'{side}_{j}' in lm for j in ('hip', 'knee', 'ankle')):
                arc(frame, lm[f'{side}_knee'], lm[f'{side}_hip'], lm[f'{side}_ankle'], 26, C_YELLOW)
                label(frame, f"{m['front_knee_angle']:.0f}", (int(lm[f'{side}_knee'][0])+10, int(lm[f'{side}_knee'][1])), C_YELLOW)
            tp = [p for p in trail if p]
            if len(tp) >= 2:
                cv2.polylines(frame, [np.array(tp, np.int32)], False, C_YELLOW, 2, cv2.LINE_AA)
                if an and an.get('attack_angle') is not None:
                    a = math.radians(an['attack_angle']); t = tp[-1]
                    cv2.arrowedLine(frame, tuple(t), (int(t[0]+45*math.cos(a)), int(t[1]-45*math.sin(a))), C_YELLOW, 2, cv2.LINE_AA, tipLength=0.3)
            if i in contact_f: label(frame, 'CONTACT (est.)', (W//2-120, 70), C_RED, 0.9, 2)
            # HUD
            bar = frame.copy(); cv2.rectangle(bar, (0, 0), (340, 150), (0, 0, 0), -1); cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)
            sep, tilt, knee = m['hip_shoulder_sep'], m['spine_tilt'], m['front_knee_angle']
            aav = an['attack_angle'] if an else None
            ca = C_GREEN if (aav is not None and IDEAL_AA[0] <= aav <= IDEAL_AA[1]) else C_ORANGE
            csep = C_GREEN if (sep is not None and sep >= GOOD_SEP) else C_ORANGE
            label(frame, f'{os.path.splitext(video)[0][:26]}  seg {seg_of.get(i,"?")}', (10, 22), C_WHITE)
            for k, (txt, col) in enumerate([(f'Hip-Shoulder Sep: {sep} deg', csep), (f'Spine Tilt:       {tilt} deg', C_WHITE),
                                            (f'Front-Knee Angle: {knee} deg', C_WHITE), (f'Attack Angle:     {aav} deg', ca)]):
                label(frame, txt, (10, 48+22*k), col)
            label(frame, '2D proxies (view-dependent)', (10, 144), (180, 180, 180), 0.4)
        wr.write(frame); i += 1
    cap.release(); wr.release()


kp = pd.read_csv(KP_CSV)
videos = sorted([v for v in kp['video'].unique()])
print(f'Applying 2D domain to {len(videos)} video(s).')
metrics = []
for v in videos:
    process(v, kp, metrics)
with open(f'{DRIVE_PROJECT}/2d_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'\nWrote {len(metrics)} segment records → {DRIVE_PROJECT}/2d_metrics.json')
