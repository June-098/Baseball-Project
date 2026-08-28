"""
Stage 6 — 3D Biomechanical Metrics (view-independent)
=====================================================
Reads keypoints_3d.json (MotionBERT output, H36M joints) and computes the swing
metrics that the 2D overlay could only *approximate*. Because these come from the
lifted 3D pose, they are NOT view-dependent — this removes the "2D proxy" caveat.

MotionBERT axis convention (verified empirically on this project's output):
  vertical axis = z (index 2), with UP = -z
  horizontal (transverse) plane = (x, y)
All metrics here are ANGLES, so MotionBERT's root-relative scaling is irrelevant.

Metrics per segment:
  hip_shoulder_sep   — angle between the hip line and shoulder line projected on the
                       horizontal plane (true axial separation). max over swing +
                       value at contact. Elite ~30-45 deg at launch.
  lead_elbow_ext     — interior angle shoulder-elbow-wrist of the lead arm
                       (180 = straight). max over swing + at contact.
  shoulder_tilt      — tilt of the shoulder line vs horizontal, at contact.
  attack_angle_3d    — vertical angle of the hands' velocity at contact (+ = up).
                       STILL contact-dependent → reliable only once Phase 4 event
                       detection lands. Reported but flagged.

Contact frame = peak hand speed (lightweight stand-in for event detection).

Output: metrics_3d.json in OUTPUT_DIR (Drive project root).
"""
import sys
import json
import math
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from config import OUTPUT_DIR

# axis convention
VERT = 2          # z is vertical
UP_SIGN = -1.0    # up = -z
HORIZ = (0, 1)    # x, y form the transverse plane

MIN_SEG_FRAMES = 20


def _jp(rec, name):
    j = rec["joints"][name]
    return np.array([j["x"], j["y"], j["z"]], dtype=np.float64)


def _angle(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))


def _undirected(a, b):
    """Angle between two undirected lines, [0,90]."""
    t = _angle(a, b)
    return min(t, 180.0 - t)


def _interior(a, b, c):
    """Interior angle at vertex b (a-b-c), [0,180]."""
    return _angle(a - b, c - b)


def _horiz(v):
    return np.array([v[HORIZ[0]], v[HORIZ[1]]])


def segment_metrics(records, handedness):
    rs = sorted(records, key=lambda r: r["frame"])
    frames = [r["frame"] for r in rs]
    lead = "L" if handedness == "righty" else "R"  # front arm leads the swing

    # contact = peak hand (mid-wrist) speed
    hands = np.array([(_jp(r, "LWrist") + _jp(r, "RWrist")) / 2 for r in rs])
    vel = np.gradient(hands, axis=0)
    spd = np.linalg.norm(vel, axis=1)
    ci = int(np.argmax(spd)) if len(rs) >= MIN_SEG_FRAMES else len(rs) // 2

    sep, ext, attack = [], [], []
    for i, r in enumerate(rs):
        sep.append(_undirected(_jp(r, "RHip") - _jp(r, "LHip"),
                               _jp(r, "RShoulder") - _jp(r, "LShoulder")))
        ext.append(_interior(_jp(r, f"{lead}Shoulder"), _jp(r, f"{lead}Elbow"), _jp(r, f"{lead}Wrist")))
        v = vel[i]
        attack.append(math.degrees(math.atan2(UP_SIGN * v[VERT], math.hypot(v[HORIZ[0]], v[HORIZ[1]]))))

    sv = _jp(rs[ci], "RShoulder") - _jp(rs[ci], "LShoulder")
    sh_tilt = math.degrees(math.atan2(UP_SIGN * sv[VERT], math.hypot(sv[HORIZ[0]], sv[HORIZ[1]])))

    return {
        "handedness": handedness,
        "n_frames": len(rs),
        "contact_frame": int(frames[ci]),
        "hip_shoulder_sep": {"max": round(max(sep), 1), "at_contact": round(sep[ci], 1)},
        "lead_elbow_ext": {"max": round(max(ext), 1), "at_contact": round(ext[ci], 1)},
        "shoulder_tilt_at_contact": round(sh_tilt, 1),
        "attack_angle_3d_at_contact": round(attack[ci], 1),
        "_note": "attack_angle_3d depends on the estimated contact frame; reliable after Phase 4 event detection.",
        "per_frame": {
            str(f): {"hip_shoulder_sep": round(sep[i], 1),
                     "lead_elbow_ext": round(ext[i], 1),
                     "attack_angle_3d": round(attack[i], 1)}
            for i, f in enumerate(frames)
        },
    }


def _handedness(video_name):
    return "lefty" if ("lefty" in video_name.lower() or "ichiro" in video_name.lower()) else "righty"


def run_compute_3d_metrics(in_json=None, out_json=None):
    in_json = Path(in_json) if in_json else (OUTPUT_DIR / "keypoints_3d.json")
    out_json = Path(out_json) if out_json else (OUTPUT_DIR / "metrics_3d.json")

    data = json.load(open(in_json))
    segs = defaultdict(list)
    for r in data:
        segs[(r["video"], r["segment_id"])].append(r)

    out = []
    for (video, seg_id) in sorted(segs):
        recs = segs[(video, seg_id)]
        if len(recs) < MIN_SEG_FRAMES:
            continue
        m = segment_metrics(recs, _handedness(video))
        m = {"video": video, "segment_id": int(seg_id), **m}
        out.append(m)
        c = m["hip_shoulder_sep"]
        print(f"  {video[:26]:26s} seg{seg_id}: sep_max={c['max']:.0f} "
              f"ext_max={m['lead_elbow_ext']['max']:.0f} attack@c={m['attack_angle_3d_at_contact']:.0f}")

    json.dump(out, open(out_json, "w"), indent=2)
    print(f"\nWrote {len(out)} segment records → {out_json}")
    return out


if __name__ == "__main__":
    run_compute_3d_metrics()
