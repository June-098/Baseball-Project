from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# Local run (no Google Drive mount): everything lives under this project folder.
BATTING_VIDEOS_DIR = PROJECT_ROOT / "Batting Videos"
# All pipeline outputs land under src/output, split into data (CSV/JSON) and videos (MP4).
OUTPUT_DIR = PROJECT_ROOT / "src" / "output"
DATA_DIR = OUTPUT_DIR / "data"
# CONVENTION: every stage that writes a VIDEO saves it here.
DIAGNOSES_DIR = OUTPUT_DIR / "videos"

# ── Video transcription (coaching-reference research) ────────────────────────
# Source videos you have downloaded yourself go here; transcripts are written as
# markdown alongside the other research notes.
GRADUM_DIR = PROJECT_ROOT / "Gradum Gswing"
RESOURCES_DIR = PROJECT_ROOT / "Baseball Resources"
TRANSCRIPTS_DIR = RESOURCES_DIR / "transcripts"
# Trimmed, embedding-ready output from src/transcript_trimming.py
TRANSCRIPTS_CLEAN_DIR = RESOURCES_DIR / "transcripts_clean"
# Canonical RAG corpus: edited .md notes. All rag_*.py scripts read and write here.
RAG_RESOURCES_DIR = RESOURCES_DIR / "RAG Resources"
RAG_INDEX_DIR = RAG_RESOURCES_DIR / "rag_index"

MOTIONBERT_DIR = PROJECT_ROOT / "MotionBERT"
# Prefer a local checkpoint copy; fall back to the one already on Drive (170 MB).
_LOCAL_CKPT = PROJECT_ROOT / "models" / "MotionBERT" / "best_epoch.bin"
_DRIVE_CKPT = Path("G:/My Drive/models/MotionBERT/best_epoch.bin")
MOTIONBERT_CKPT = _LOCAL_CKPT if _LOCAL_CKPT.exists() else _DRIVE_CKPT

# Video formats accepted anywhere in the pipeline.
#
# LOWERCASE ONLY — always compare against filename.lower(). A mixed-case tuple is a
# trap: "clip.MOV".lower() is "clip.mov", which can never match a ".MOV" entry, so
# uppercase entries silently do nothing while looking like they handle the case.
VIDEO_EXTS = (
    ".mp4", ".m4v", ".mov", ".qt",                            # MPEG-4 / QuickTime
    ".avi", ".wmv", ".asf",                                   # Windows
    ".mkv", ".webm",                                          # Matroska / WebM
    ".mpg", ".mpeg", ".mpe", ".m2v",                          # MPEG-1/2
    ".ts", ".mts", ".m2ts",                                   # transport streams
    ".flv", ".f4v",                                           # Flash
    ".3gp", ".3g2",                                           # mobile
    ".ogv", ".rm", ".rmvb", ".vob", ".divx", ".mxf",          # misc
)

CONF_THRESHOLD = 0.3


def is_video(name) -> bool:
    """True if `name` looks like a video file, regardless of extension case."""
    return str(name).lower().endswith(VIDEO_EXTS)


def list_videos(directory=None) -> list:
    """
    Sorted list of video filenames in `directory` (defaults to BATTING_VIDEOS_DIR).

    Every stage must use this so they all agree on which files count as videos and
    in what order — batch slicing in pose_extraction.py depends on a stable order.
    """
    d = Path(directory) if directory is not None else BATTING_VIDEOS_DIR
    return sorted(
        (f.name for f in Path(d).iterdir() if f.is_file() and is_video(f.name)),
        key=str.lower,
    )

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

COCO_JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

H36M_JOINTS = [
    "Hip", "RHip", "RKnee", "RAnkle",
    "LHip", "LKnee", "LAnkle",
    "Spine", "Thorax", "Neck", "Head",
    "LShoulder", "LElbow", "LWrist",
    "RShoulder", "RElbow", "RWrist",
]

H36M_BONES = [
    (0, 7), (7, 8), (8, 9), (9, 10),   # spine chain
    (0, 4), (4, 5), (5, 6),             # left leg
    (0, 1), (1, 2), (2, 3),             # right leg
    (8, 11), (11, 12), (12, 13),        # left arm
    (8, 14), (14, 15), (15, 16),        # right arm
]

BONE_COLOR_MAP = [
    "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF",
    "#00FFFF", "#00FFFF", "#00FFFF",
    "#FFA500", "#FFA500", "#FFA500",
    "#4FC3F7", "#4FC3F7", "#4FC3F7",
    "#EF5350", "#EF5350", "#EF5350",
]
