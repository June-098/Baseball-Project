"""
Stage 3b — Save 3D Keypoints
Serializes results_3d dict to data/keypoints_3d.json.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from config import DATA_DIR, H36M_JOINTS


def save_keypoints_3d(results_3d: dict) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for (video, seg_id), data in results_3d.items():
        for t, frame in enumerate(data["frames"]):
            joints = data["pose_3d"][t]   # (17, 3)
            records.append({
                "video": video,
                "segment_id": int(seg_id),
                "frame": int(frame),
                "joints": {
                    name: {"x": float(joints[j, 0]),
                           "y": float(joints[j, 1]),
                           "z": float(joints[j, 2])}
                    for j, name in enumerate(H36M_JOINTS)
                },
            })

    out_path = DATA_DIR / "keypoints_3d.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} frame records → {out_path}")
    return out_path
