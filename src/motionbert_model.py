"""
MotionBERT Model Loader
Loads the DSTformer checkpoint and returns (model, device).
Requires setup_motionbert() to have been called first.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from functools import partial
from config import MOTIONBERT_DIR, MOTIONBERT_CKPT

if str(MOTIONBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MOTIONBERT_DIR))

from lib.model.DSTformer import DSTformer


def load_motionbert_model():
    if not MOTIONBERT_CKPT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {MOTIONBERT_CKPT}\n"
            "Download best_epoch.bin from the MotionBERT README "
            "(MB_ft_h36m) and place it at that path."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model_pos = DSTformer(
        dim_in=3,
        dim_out=3,
        dim_feat=512,
        dim_rep=512,
        depth=5,
        num_heads=8,
        mlp_ratio=2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        maxlen=243,
        num_joints=17,
    )

    checkpoint = torch.load(str(MOTIONBERT_CKPT), map_location="cpu")
    state = (
        checkpoint.get("model")
        or checkpoint.get("model_pos")
        or checkpoint
    )
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

    model_pos.load_state_dict(state, strict=True)
    model_pos.eval().to(device)
    print("MotionBERT loaded.")
    return model_pos, device


if __name__ == "__main__":
    from src.motionbert_setup import setup_motionbert
    setup_motionbert()
    load_motionbert_model()
