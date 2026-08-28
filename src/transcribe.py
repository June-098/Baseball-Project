#!/usr/bin/env python3

import argparse
from pathlib import Path

from faster_whisper import WhisperModel


def transcribe(
    input_path: Path,
    output_path: Path,
    model_name: str,
    language: str | None,
) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"File not found: {input_path}")

    model = WhisperModel(model_name, device="auto", compute_type="default")

    segments, info = model.transcribe(
        str(input_path),
        language=language,
        vad_filter=True,
    )

    with output_path.open("w", encoding="utf-8") as output_file:
        for segment in segments:
            output_file.write(segment.text.strip() + "\n")

    detected = info.language or "unknown"
    print(f"Detected language: {detected}")
    print(f"Transcript saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio or video file into text."
    )
    parser.add_argument("input", type=Path, help="Input audio or video file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output text file; defaults to the input filename with a .txt extension",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="small",
        help="Whisper model: tiny, base, small, medium, or large-v3",
    )
    parser.add_argument(
        "-l",
        "--language",
        help="Optional language code, such as en, es, fr, ko, or zh",
    )
    args = parser.parse_args()

    output_path = args.output or args.input.with_suffix(".txt")
    transcribe(args.input, output_path, args.model, args.language)


if __name__ == "__main__":
    main()