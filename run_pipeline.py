"""
Baseball CV Pipeline — main entry point.

Run the full pipeline:
    python run_pipeline.py

Run a single stage:
    python run_pipeline.py --stage pose
    python run_pipeline.py --stage batter
    python run_pipeline.py --stage motionbert
    python run_pipeline.py --stage viz3d
    python run_pipeline.py --stage viz2d

Stages in order:
    pose       → data/keypoints_raw.csv
    batter     → data/keypoints_batter.csv
    motionbert → data/keypoints_3d.json
    viz3d      → data/skeleton_3d_*.mp4
    viz2d      → data/skeleton_2d_*.mp4
    apply2d    → Drive/skeleton_2d_biomech_*.mp4 + 2d_metrics.json
    metrics3d  → Drive/metrics_3d.json (view-independent 3D biomech metrics)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.batter_selection import run_batter_selection
from src.visualize_2d import run_visualize_2d
from src.apply_2d_domain import run_apply_2d
from src.compute_3d_metrics import run_compute_3d_metrics

# pose_extraction is imported lazily inside its stage: it pulls in ultralytics/torch,
# which the torch-free stages (batter, viz2d, apply2d) must not require.


def run_motionbert_stages(stage=None):
    from src.motionbert_setup import setup_motionbert
    from src.motionbert_model import load_motionbert_model
    from src.motionbert_inference import run_motionbert_inference
    from src.save_3d import save_keypoints_3d
    from src.visualize_3d import run_visualize_3d

    setup_motionbert()
    model_pos, device = load_motionbert_model()
    results_3d = run_motionbert_inference(model_pos, device)
    save_keypoints_3d(results_3d)

    if stage in ("viz3d", None):
        run_visualize_3d(results_3d)

    return results_3d


def main():
    parser = argparse.ArgumentParser(description="Baseball CV pipeline")
    parser.add_argument(
        "--stage",
        choices=["pose", "batter", "motionbert", "viz3d", "viz2d", "apply2d", "metrics3d"],
        help="Run a single stage. Omit to run the full pipeline.",
    )
    args = parser.parse_args()

    if args.stage == "pose":
        from src.pose_extraction import run_pose_extraction
        run_pose_extraction()
    elif args.stage == "batter":
        run_batter_selection()
    elif args.stage == "motionbert":
        run_motionbert_stages(stage="motionbert")
    elif args.stage == "viz3d":
        run_motionbert_stages(stage="viz3d")
    elif args.stage == "viz2d":
        run_visualize_2d()
    elif args.stage == "apply2d":
        run_apply_2d()
    elif args.stage == "metrics3d":
        run_compute_3d_metrics()
    else:
        print("=== Running full pipeline ===\n")
        from src.pose_extraction import run_pose_extraction
        run_pose_extraction()
        run_batter_selection()
        run_motionbert_stages()
        run_visualize_2d()
        run_apply_2d()
        run_compute_3d_metrics()
        print("\n=== Pipeline complete. Outputs in data/ ===")


if __name__ == "__main__":
    main()
