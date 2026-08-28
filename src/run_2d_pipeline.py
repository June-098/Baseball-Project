"""
Full local 2D pipeline in one command:
    Stage 1 — Pose extraction (batched, subprocess per batch)
    Stage 2 — Batter selection + 2D skeleton overlay (per-video CSVs)

Usage (from the project root, e.g. Personal Baseball Project/):
    python src/run_2d_pipeline.py                  # default batch size (20)
    python src/run_2d_pipeline.py --batch-size 10   # smaller batches, more restarts

Why Stage 1 is still subprocess-batched but Stage 2 runs in-process here:
Stage 1 loads the YOLO pose model and decodes every frame of every video —
that's the memory-heavy part, so each batch gets its own subprocess (see
pose_extraction.py's stream=True fix and run_pose_batches.py).
Stage 2 only touches pandas/OpenCV (no torch/ultralytics), reads one
video at a time, and writes output as they go.

Output:
  src/output/data/*_keypoints.csv     — one per video (source of truth)
  src/output/videos/skeleton_2d_*.mp4 — 2D skeleton overlay
  Combined keypoints_raw.csv / keypoints_batter.csv are NOT written here.
  Use `python src/pose_extraction.py --merge-only` then
  `python src/batter_selection.py` if you need the combined files.
  `python src/apply_2d_domain.py` reads per-video CSVs directly.

Resumable: if Stage 1 is interrupted partway through, rerun this same
command — pose_extraction.py skips any video whose per-video CSV already
exists and is valid in src/output/data/, so completed videos aren't reprocessed.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.run_pose_batches import run_pose_batches


def main():
    parser = argparse.ArgumentParser(description="Full local 2D pipeline: pose (batched) -> batter -> skeleton overlay")
    parser.add_argument("--batch-size", type=int, default=20,
                         help="Videos per pose-extraction subprocess batch (default: 20)")
    parser.add_argument("--numbered", type=int, default=10,
                         help="How many Penn Action clips to render (default: 10)")
    parser.add_argument("--slowmo", type=int, default=3,
                         help="How many slow-motion clips to render (default: 3)")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed, for a reproducible sample")
    parser.add_argument("--all", action="store_true",
                         help="Render every video that has keypoints (slow)")
    parser.add_argument("--skip-pose", action="store_true",
                         help="Skip Stage 1 and render from existing per-video CSVs")
    args = parser.parse_args()

    if args.skip_pose:
        print("=== Stage 1/2 skipped (--skip-pose): using existing per-video CSVs ===")
    else:
        print("=== Stage 1/2: Pose extraction (batched) ===\n")
        ok = run_pose_batches(args.batch_size)
        if not ok:
            print("\nOne or more Stage 1 batches failed. Rerun this command to retry — "
                  "already-processed videos are skipped automatically. "
                  "Stopping before the render stage so bad keypoint data doesn't propagate.")
            sys.exit(1)

    print("\n=== Stage 2/2: Batter selection + 2D skeleton overlay (per video) ===")
    from src.visualize_2d import run_visualize_2d
    rendered, failed = run_visualize_2d(
        n_numbered=args.numbered,
        n_slowmo=args.slowmo,
        seed=args.seed,
        all_videos=args.all,
    )

    print("\n=== Pipeline complete ===")
    print(f"Keypoint CSVs  -> src/output/data/   (one per video)")
    print(f"Skeleton MP4s  -> src/output/videos/ ({rendered} rendered)")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
