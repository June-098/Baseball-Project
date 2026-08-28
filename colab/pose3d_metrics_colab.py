# ═════════════════════════════════════════════════════════════════════════════
# COLAB — 3D BIOMECHANICAL METRICS (view-independent)
# Run AFTER pose3d_estimation (needs keypoints_3d.json). CPU is fine.
#
# These are the TRUE 3D versions of the swing angles the 2D overlay could only
# approximate — they do NOT depend on camera viewpoint.
#
# MotionBERT axis convention (verified on this project): vertical = z, UP = -z;
# horizontal (transverse) plane = (x, y). All metrics are angles → scale-free.
#
# Output: <DRIVE>/metrics_3d.json
# ═════════════════════════════════════════════════════════════════════════════
import json, math
import numpy as np
from collections import defaultdict
from google.colab import drive
drive.mount('/content/drive')

DRIVE = '/content/drive/My Drive/Baseball Project'
IN_JSON = f'{DRIVE}/keypoints_3d.json'
OUT_JSON = f'{DRIVE}/metrics_3d.json'
VERT, UP = 2, -1.0          # z vertical, up = -z
HORIZ = (0, 1)              # x,y transverse plane
MIN_SEG_FRAMES = 20

def jp(r, n):
    j = r['joints'][n]; return np.array([j['x'], j['y'], j['z']])
def ang(a, b):
    a = a/(np.linalg.norm(a)+1e-9); b = b/(np.linalg.norm(b)+1e-9)
    return math.degrees(math.acos(max(-1, min(1, float(np.dot(a, b))))))
def undirected(a, b):
    t = ang(a, b); return min(t, 180-t)
def interior(a, b, c):
    return ang(a-b, c-b)
def handed(v):
    return 'lefty' if ('lefty' in v.lower() or 'ichiro' in v.lower()) else 'righty'

def seg_metrics(records, hd):
    rs = sorted(records, key=lambda r: r['frame']); frames = [r['frame'] for r in rs]
    lead = 'L' if hd == 'righty' else 'R'
    hands = np.array([(jp(r, 'LWrist')+jp(r, 'RWrist'))/2 for r in rs])
    vel = np.gradient(hands, axis=0); spd = np.linalg.norm(vel, axis=1)
    ci = int(np.argmax(spd)) if len(rs) >= MIN_SEG_FRAMES else len(rs)//2
    sep = [undirected(jp(r, 'RHip')-jp(r, 'LHip'), jp(r, 'RShoulder')-jp(r, 'LShoulder')) for r in rs]
    ext = [interior(jp(r, f'{lead}Shoulder'), jp(r, f'{lead}Elbow'), jp(r, f'{lead}Wrist')) for r in rs]
    attack = [math.degrees(math.atan2(UP*vel[i][VERT], math.hypot(vel[i][HORIZ[0]], vel[i][HORIZ[1]]))) for i in range(len(rs))]
    sv = jp(rs[ci], 'RShoulder')-jp(rs[ci], 'LShoulder')
    tilt = math.degrees(math.atan2(UP*sv[VERT], math.hypot(sv[HORIZ[0]], sv[HORIZ[1]])))
    return {'handedness': hd, 'n_frames': len(rs), 'contact_frame': int(frames[ci]),
            'hip_shoulder_sep': {'max': round(max(sep), 1), 'at_contact': round(sep[ci], 1)},
            'lead_elbow_ext': {'max': round(max(ext), 1), 'at_contact': round(ext[ci], 1)},
            'shoulder_tilt_at_contact': round(tilt, 1),
            'attack_angle_3d_at_contact': round(attack[ci], 1),
            '_note': 'attack_angle_3d depends on the estimated contact frame; reliable after Phase 4 event detection.'}

data = json.load(open(IN_JSON))
segs = defaultdict(list)
for r in data:
    segs[(r['video'], r['segment_id'])].append(r)
out = []
for (video, seg) in sorted(segs):
    if len(segs[(video, seg)]) < MIN_SEG_FRAMES:
        continue
    m = {'video': video, 'segment_id': int(seg), **seg_metrics(segs[(video, seg)], handed(video))}
    out.append(m)
    print(f"  {video[:26]:26s} seg{seg}: sep_max={m['hip_shoulder_sep']['max']:.0f} "
          f"ext_max={m['lead_elbow_ext']['max']:.0f} attack@c={m['attack_angle_3d_at_contact']:.0f}")
json.dump(out, open(OUT_JSON, 'w'), indent=2)
print(f'\nWrote {len(out)} segment records → {OUT_JSON}')
print('Reference: hip-shoulder sep elite ~30-45 deg at launch; lead-arm extension >140 deg near full.')
