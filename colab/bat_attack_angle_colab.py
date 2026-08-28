# ═════════════════════════════════════════════════════════════════════════════
# COLAB — ATTACK ANGLE from the tracked bat (EXACT, not the wrist proxy)
# Run AFTER tracknet_bat_tracking (needs bat_track.csv with sweet-spot path).
#
# Attack angle (MLB Glossary): the vertical angle at which the bat's SWEET SPOT is
# travelling at the point of impact. 0° = flat, + = upward, − = downward.
# Ideal band: 5°–20°  (a 62-point wOBA gap vs. outside the band).
#
# Now that we track the bat sweet spot directly, this is the real metric — the 2D
# overlay's wrist-path version was only a proxy for exactly this.
#
# Contact frame = peak sweet-spot speed (swap in Phase-4 event detection when ready).
# Bonus: swing-path tilt (plane angle over the 40 ms before contact) + attack
# direction (pull/oppo), both also from the bat path.
#
# Output: <DRIVE>/bat_attack_angle.json
# ═════════════════════════════════════════════════════════════════════════════
import json, math
import numpy as np, pandas as pd
from google.colab import drive
drive.mount('/content/drive')

DRIVE = '/content/drive/My Drive/Baseball Project'
BAT_CSV = f'{DRIVE}/bat_track.csv'
OUT = f'{DRIVE}/bat_attack_angle.json'
SMOOTH_WIN = 5
IDEAL = (5.0, 20.0)

def smooth(a, w):
    out = a.copy(); h = w // 2
    for i in range(len(a)):
        ch = a[max(0, i-h):min(len(a), i+h+1)]; v = ch[~np.isnan(ch[:, 0])]
        if len(v):
            out[i] = v.mean(0)
    return out

def analyze(video, df, fps=30.0):
    df = df.sort_values('frame')
    frames = df['frame'].to_numpy()
    pos = df[['sweet_x', 'sweet_y']].to_numpy(dtype=float)   # image px; y is DOWN
    ps = smooth(pos, SMOOTH_WIN)
    vel = np.gradient(ps, axis=0)
    spd = np.linalg.norm(vel, axis=1)
    if np.all(np.isnan(spd)):
        return None
    ci = int(np.nanargmax(spd))
    # attack angle: + = upward → negate dy (image y is down)
    aa = math.degrees(math.atan2(-vel[ci, 1], vel[ci, 0]))
    # attack direction: horizontal sign of travel (pull/oppo is handedness-relative)
    attack_dir = 'pull-ish' if vel[ci, 0] > 0 else 'oppo-ish'
    # swing-path tilt: angle of the bat-path chord over ~40 ms before contact
    n40 = max(1, int(round(0.040 * fps)))
    j = max(0, ci - n40)
    dx, dy = ps[ci, 0] - ps[j, 0], ps[ci, 1] - ps[j, 1]
    tilt = math.degrees(math.atan2(-dy, abs(dx) + 1e-9)) if not (np.isnan(dx) or np.isnan(dy)) else None
    in_band = IDEAL[0] <= aa <= IDEAL[1]
    return {'video': video, 'contact_frame': int(frames[ci]),
            'attack_angle_deg': round(aa, 1), 'ideal_band': list(IDEAL), 'in_ideal_band': bool(in_band),
            'swing_path_tilt_deg': None if tilt is None else round(tilt, 1),
            'attack_direction': attack_dir,
            'coverage_pct': round(float(np.mean(~np.isnan(pos[:, 0]))) * 100, 1),
            '_note': 'contact = peak sweet-spot speed; replace with Phase-4 event detection for precision.'}

bat = pd.read_csv(BAT_CSV)
out = []
for video, df in bat.groupby('video'):
    r = analyze(video, df)
    if r:
        out.append(r)
        flag = '✅ ideal' if r['in_ideal_band'] else '✗ outside'
        print(f"{video[:30]:30s} attack={r['attack_angle_deg']:+5.1f}° {flag}  "
              f"tilt={r['swing_path_tilt_deg']}  cover={r['coverage_pct']}%")
json.dump(out, open(OUT, 'w'), indent=2)
print(f'\nWrote {len(out)} records → {OUT}')
print('Reminder: low coverage near contact = the bat tracker needs more contact-zone training labels.')
