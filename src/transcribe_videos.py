"""
Local video transcription — turns coaching videos into timestamped markdown notes.

Runs entirely on your machine (no upload, no API). Built for extracting coaching
concepts from instructional footage you have downloaded yourself, so the material
can be turned into structured reference notes for the feedback engine.

    Input :  Gradum Gswing/*.mp4|mov|mkv|...        (whatever you drop in)
    Output:  Baseball Resources/transcripts/<name>.md

Why faster-whisper rather than openai-whisper: it wraps CTranslate2, which is
roughly 4x faster on CPU at the same accuracy and uses far less memory. There is
no MPS/Metal backend, so on Apple Silicon it runs on CPU with int8 quantization —
still comfortably faster than realtime on an M-series chip.

── Setup (once) ─────────────────────────────────────────────────────────────
    pip install faster-whisper
    # ffmpeg must be on PATH:  brew install ffmpeg

── Usage ────────────────────────────────────────────────────────────────────
    python src/transcribe_videos.py                      # all new videos, small model
    python src/transcribe_videos.py --model medium       # more accurate, slower
    python src/transcribe_videos.py --language en        # skip auto-detect
    python src/transcribe_videos.py --videos "clip.mp4"  # just one
    python src/transcribe_videos.py --force              # re-transcribe everything

Model sizes (accuracy vs. speed on CPU):
    tiny   ~75 MB   fastest, noticeably error-prone
    base   ~145 MB  usable for rough notes
    small  ~480 MB  DEFAULT — good balance for coaching speech
    medium ~1.5 GB  clearly better on jargon and names
    large-v3 ~3 GB  best, several times slower

Note on source material: this is for building your own research notes from
content you have legitimate access to. Transcripts of instructional videos are
the creator's work — keep the output for personal study and cite the source in
anything derived from it, rather than redistributing the text.
"""
import sys
import subprocess
import tempfile
import argparse
import shutil
import time
from pathlib import Path
from datetime import timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import GRADUM_DIR, TRANSCRIPTS_DIR, is_video, list_videos


# ── helpers ──────────────────────────────────────────────────────────────────

def _check_prereqs() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH.\n"
            "  macOS:  brew install ffmpeg"
        )
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "faster-whisper is not installed.\n"
            "  pip install faster-whisper"
        )


def _pick_device() -> tuple[str, str]:
    """
    CTranslate2 supports CUDA and CPU only — there is no Metal/MPS backend, so
    Apple Silicon runs on CPU. int8 quantization is the right default there:
    roughly 2x faster than float32 with negligible accuracy loss for speech.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"


def _extract_audio(video: Path, wav_out: Path) -> None:
    """
    Whisper wants 16 kHz mono PCM. Doing the conversion once with ffmpeg is far
    faster and more reliable than letting the library decode arbitrary containers.
    """
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", str(video),
         "-vn",                    # drop video
         "-ac", "1",               # mono
         "-ar", "16000",           # 16 kHz
         "-acodec", "pcm_s16le",
         str(wav_out)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _ts(seconds: float) -> str:
    """Seconds -> H:MM:SS (or M:SS for short clips)."""
    td = timedelta(seconds=int(seconds))
    total = int(td.total_seconds())
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _write_markdown(video: Path, segments: list, info, out_path: Path,
                    model_name: str, elapsed: float) -> None:
    lines = [
        f"# {video.stem}",
        "",
        f"- **Source file:** `{video.name}`",
        f"- **Detected language:** {info.language} "
        f"(confidence {info.language_probability:.2f})",
        f"- **Duration:** {_ts(info.duration)}",
        f"- **Model:** faster-whisper `{model_name}`",
        f"- **Transcribed:** {time.strftime('%Y-%m-%d %H:%M')} "
        f"(took {elapsed/60:.1f} min)",
        "",
        "> Auto-generated transcript for personal research notes. "
        "Speech-to-text makes mistakes on names and jargon — verify anything "
        "you rely on against the source video.",
        "",
        "---",
        "",
        "## Transcript",
        "",
    ]
    for seg in segments:
        text = seg.text.strip()
        if text:
            lines.append(f"**[{_ts(seg.start)}]** {text}")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ── main ─────────────────────────────────────────────────────────────────────

def transcribe_one(video: Path, model, model_name: str, out_dir: Path,
                   language: str = None) -> Path:
    out_path = out_dir / f"{video.stem}.md"
    print(f"\n  {video.name}")

    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        print("    extracting audio ...", flush=True)
        _extract_audio(video, wav)

        print("    transcribing ...", flush=True)
        segments, info = model.transcribe(
            str(wav),
            language=language,          # None => auto-detect
            beam_size=5,
            vad_filter=True,            # drop silence; big speedup on edited video
            vad_parameters={"min_silence_duration_ms": 500},
        )
        # segments is a lazy generator — consume it so timing is accurate
        segments = list(segments)

    elapsed = time.time() - started
    _write_markdown(video, segments, info, out_path, model_name, elapsed)

    speed = info.duration / elapsed if elapsed else 0
    print(f"    -> {out_path.name}  ({len(segments)} segments, "
          f"{elapsed/60:.1f} min, {speed:.1f}x realtime)")
    return out_path


def run(input_dir: Path = None, output_dir: Path = None, model_name: str = "small",
        language: str = None, videos: list = None, force: bool = False) -> tuple[int, list]:
    _check_prereqs()
    from faster_whisper import WhisperModel

    in_dir  = Path(input_dir)  if input_dir  else GRADUM_DIR
    out_dir = Path(output_dir) if output_dir else TRANSCRIPTS_DIR

    if not in_dir.exists():
        raise RuntimeError(
            f"Input folder not found: {in_dir}\n"
            f"Create it and put your downloaded videos inside."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Which videos?
    if videos:
        targets = [in_dir / v for v in videos]
        missing = [t for t in targets if not t.exists()]
        if missing:
            raise RuntimeError("Not found: " + ", ".join(m.name for m in missing))
    else:
        targets = [in_dir / v for v in list_videos(in_dir)]

    if not targets:
        print(f"No videos found in {in_dir}")
        print(f"  (recognized extensions include .mp4 .mov .mkv .webm .m4v .avi ...)")
        return 0, []

    # Resume: skip anything already transcribed
    pending = targets if force else [
        t for t in targets if not (out_dir / f"{t.stem}.md").exists()
    ]
    skipped = len(targets) - len(pending)

    print(f"Input : {in_dir}")
    print(f"Output: {out_dir}")
    print(f"Found {len(targets)} video(s); {skipped} already transcribed, "
          f"{len(pending)} to process.")
    if not pending:
        print("Nothing to do. Use --force to re-transcribe.")
        return 0, []

    device, compute = _pick_device()
    print(f"\nLoading model '{model_name}' on {device} ({compute}) ...")
    print("  (first run downloads the model — this happens once)")
    model = WhisperModel(model_name, device=device, compute_type=compute)

    done, failed = 0, []
    for v in pending:
        try:
            transcribe_one(v, model, model_name, out_dir, language)
            done += 1
        except subprocess.CalledProcessError as exc:
            err = (exc.stderr or b"").decode(errors="replace").strip().splitlines()
            print(f"    ERROR (ffmpeg): {err[-1] if err else 'audio extraction failed'}")
            failed.append(v.name)
        except Exception as exc:
            print(f"    ERROR: {type(exc).__name__}: {exc}")
            failed.append(v.name)

    print(f"\nDone: {done} transcribed, {len(failed)} failed -> {out_dir}")
    if failed:
        print("  failed: " + ", ".join(failed))
    return done, failed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Transcribe local videos to timestamped markdown (faster-whisper)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input",  default=None, help=f"Video folder (default: {GRADUM_DIR.name}/)")
    ap.add_argument("--output", default=None, help="Transcript folder (default: Baseball Resources/transcripts/)")
    ap.add_argument("--model",  default="small",
                    choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                    help="Whisper model size (default: small)")
    ap.add_argument("--language", default=None,
                    help="Force language code e.g. 'en', 'ko'. Omit to auto-detect.")
    ap.add_argument("--videos", nargs="*", default=None, help="Specific filenames only")
    ap.add_argument("--force", action="store_true", help="Re-transcribe even if output exists")
    a = ap.parse_args()

    try:
        _, failed = run(a.input, a.output, a.model, a.language, a.videos, a.force)
        sys.exit(1 if failed else 0)
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        sys.exit(2)
