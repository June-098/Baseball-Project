"""
Run pose extraction in batches, each in its own subprocess.

Why a subprocess per batch instead of just looping inside one process:
looping still leaves one long-lived Python interpreter holding the YOLO
models, PyTorch/CUDA-or-MPS caching allocators, and any memory fragmentation
that builds up over the run. torch.cuda.empty_cache()/torch.mps.empty_cache()
(already called between videos in pose_extraction.py) helps, but a caching
allocator can still fragment or hold onto memory for the life of the
process. Actually exiting the process after each batch is the only way to
guarantee the OS reclaims everything before the next batch starts.

This script only lists the video folder (via config.py, which has no heavy
imports) to compute batch boundaries — it does not import ultralytics/torch
itself, so it stays lightweight regardless of batch size or count.

Usage (from the project root, e.g. Personal Baseball Project/):
    python src/run_pose_batches.py                  # default batch size (20)
    python src/run_pose_batches.py --batch-size 10   # smaller batches, more restarts

Resumable: pose_extraction.py already skips any video whose per-video CSV
exists and is valid in src/output/data/, so if this script is interrupted
(or one batch's subprocess crashes), just rerun the same command — completed
videos won't be reprocessed.
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import BATTING_VIDEOS_DIR, list_videos


def run_pose_batches(batch_size: int = 20) -> bool:
    """
    Run pose extraction in subprocess batches.

    Returns True if every batch exited 0. Uses list_videos() so batch indices
    match pose_extraction.py's slice into the same sorted list.
    """
    total = len(list_videos(BATTING_VIDEOS_DIR))
    if total == 0:
        raise RuntimeError(f"No video files found in {BATTING_VIDEOS_DIR}")

    num_batches = (total + batch_size - 1) // batch_size
    print(f"{total} video(s) found in {BATTING_VIDEOS_DIR.name}/ — "
          f"running {num_batches} batch(es) of up to {batch_size}\n")

    pose_script = Path(__file__).resolve().parent / "pose_extraction.py"
    failed_batches = []

    for i in range(num_batches):
        start, end = i * batch_size, min((i + 1) * batch_size, total)
        print(f"\n=== Batch {i + 1}/{num_batches}  (videos {start}–{end - 1}) ===")
        result = subprocess.run(
            [sys.executable, str(pose_script), "--start", str(start), "--end", str(end)],
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            print(f"  BATCH {i + 1} exited with code {result.returncode} — continuing to next batch")
            failed_batches.append(i + 1)

    print("\n=== All batches complete ===")
    if failed_batches:
        print(f"Batches with a non-zero exit code (check output above): {failed_batches}")
        print("Rerun this same command to retry — completed videos are skipped automatically.")
        return False

    print("All batches finished cleanly.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run pose extraction in batched subprocesses")
    parser.add_argument("--batch-size", type=int, default=20,
                         help="Number of videos per subprocess batch (default: 20)")
    args = parser.parse_args()
    ok = run_pose_batches(args.batch_size)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
