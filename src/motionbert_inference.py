"""
Stage 3 — MotionBERT Inference
Reads keypoints_batter.csv, normalizes 2D poses per-segment,
runs DSTformer to produce (T, 17, 3) H36M joint coordinates.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import torch
from config import DATA_DIR, BATTING_VIDEOS_DIR, COCO_JOINTS


def get_video_dims(video_name: str):
    vpath = str(BATTING_VIDEOS_DIR / video_name)
    cap = cv2.VideoCapture(vpath)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if W == 0 or H == 0:
        raise RuntimeError(
            f"Could not read dimensions for {video_name}. "
            f"Check the file exists at {vpath}."
        )
    return W, H


def build_segment_tensor(seg_df: pd.DataFrame, W: int, H: int, window: int = 243):
    """
    Converts one segment's rows into a (T, 17, 3) array [x_norm, y_norm, conf].
    MotionBERT image-width normalization:
      x_norm = x/W * 2 - 1
      y_norm = y/W * 2 - H/W   (preserves aspect ratio)
    """
    frames = sorted(seg_df["frame"].unique())
    T = len(frames)
    frame_idx = {f: t for t, f in enumerate(frames)}

    seq = np.zeros((T, 17, 3), dtype=np.float32)
    for row in seg_df.itertuples(index=False):
        t = frame_idx[row.frame]
        j = COCO_JOINTS.index(row.keypoint)
        seq[t, j] = [row.x, row.y, row.confidence]

    seq_norm = seq.copy()
    seq_norm[:, :, 0] = seq[:, :, 0] / W * 2 - 1
    seq_norm[:, :, 1] = seq[:, :, 1] / W * 2 - H / W

    return seq_norm, T, frames


def sliding_window_inference(seq_norm, T: int, model, device, window: int = 243):
    """
    Runs MotionBERT on sequences of any length.
    Short sequences (< window): pad + single forward pass.
    Long sequences: overlapping windows averaged at overlaps.
    Returns (T, 17, 3) array of H36M 3D joints.
    """
    if T <= window:
        pad_l = (window - T) // 2
        pad_r = window - T - pad_l
        padded = np.pad(seq_norm, ((pad_l, pad_r), (0, 0), (0, 0)), mode="edge")
        x = torch.tensor(padded[None], dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(x).cpu().numpy()[0]
        return out[pad_l: pad_l + T]
    else:
        out_sum = np.zeros((T, 17, 3), dtype=np.float64)
        out_count = np.zeros((T, 1, 1), dtype=np.float64)
        stride = window // 2
        for start in range(0, T, stride):
            chunk = seq_norm[start: start + window]
            if len(chunk) < window:
                chunk = np.pad(chunk, ((0, window - len(chunk)), (0, 0), (0, 0)), mode="edge")
            x = torch.tensor(chunk[None], dtype=torch.float32).to(device)
            with torch.no_grad():
                out_chunk = model(x).cpu().numpy()[0]
            valid = min(window, T - start)
            out_sum[start: start + valid] += out_chunk[:valid]
            out_count[start: start + valid] += 1
        return (out_sum / out_count).astype(np.float32)


def run_motionbert_inference(model_pos, device) -> dict:
    kp_path = DATA_DIR / "keypoints_batter.csv"
    kp = pd.read_csv(kp_path)
    print(f"Loaded keypoints_batter.csv: {len(kp)} rows, "
          f"{kp.groupby(['video','segment_id']).ngroups} segments")

    print("\nReading video dimensions...")
    video_dims = {}
    for video_name in kp["video"].unique():
        W, H = get_video_dims(video_name)
        video_dims[video_name] = (W, H)
        print(f"  {video_name}: {W}x{H}")

    print("\nBuilding segment tensors...")
    segments = {}
    for (video, seg_id), seg_df in kp.groupby(["video", "segment_id"]):
        W, H = video_dims[video]
        seq_norm, T, frames = build_segment_tensor(seg_df, W, H)
        segments[(video, seg_id)] = {"seq_norm": seq_norm, "T": T, "frames": frames}
        print(f"  {video}  seg={seg_id}  T={T} frames")

    print(f"\nRunning MotionBERT inference on {len(segments)} segments...")
    results_3d = {}
    for (video, seg_id), seg in segments.items():
        pose_3d = sliding_window_inference(seg["seq_norm"], seg["T"], model_pos, device)
        results_3d[(video, seg_id)] = {
            "frames": seg["frames"],
            "pose_3d": pose_3d,
            "T": seg["T"],
        }
        print(f"  ✓ {video}  seg={seg_id}  output: {pose_3d.shape}")

    print("\nInference complete.")
    return results_3d


if __name__ == "__main__":
    from src.motionbert_setup import setup_motionbert
    from src.motionbert_model import load_motionbert_model
    setup_motionbert()
    model_pos, device = load_motionbert_model()
    run_motionbert_inference(model_pos, device)
