"""
MotionBERT Setup
Clones the MotionBERT repo into the project root if not already present,
installs required extras, and adds it to sys.path.
"""
import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MOTIONBERT_DIR


def setup_motionbert():
    if not MOTIONBERT_DIR.exists():
        print("Cloning MotionBERT...")
        subprocess.run(
            ["git", "clone", "https://github.com/Walter0807/MotionBERT.git",
             str(MOTIONBERT_DIR)],
            check=True,
        )
    else:
        print(f"MotionBERT already present at {MOTIONBERT_DIR}")

    subprocess.run(["pip", "install", "-q", "einops", "timm"], check=True)

    if str(MOTIONBERT_DIR) not in sys.path:
        sys.path.insert(0, str(MOTIONBERT_DIR))

    print("MotionBERT setup complete.")


if __name__ == "__main__":
    setup_motionbert()
