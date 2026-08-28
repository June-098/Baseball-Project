"""
Transcript trimming — turn raw Whisper transcripts into embedding-ready baseball content.

Reads the markdown produced by src/transcribe_videos.py and removes everything that
isn't instructional baseball content, then emits chunks ready for a vector store.

    Input :  Baseball Resources/transcripts/*.md
    Output:  Baseball Resources/transcripts_clean/<name>.md          human-readable
             Baseball Resources/transcripts_clean/chunks.jsonl        for embedding
             Baseball Resources/transcripts_clean/_removal_report.md  audit trail

── What it does ─────────────────────────────────────────────────────────────
1. Strips the "**[0:00]**" timestamp prefixes.
2. Rejoins Whisper's mid-sentence line breaks. Whisper cuts on audio pauses, not
   grammar, so a single sentence is routinely split across two segments
   ("...all the way down to Little" / "League not knowing how to..."). Embedding
   those fragments separately produces garbage retrieval, so sentences are
   reassembled before anything else happens.
3. Drops greetings, self-introductions, episode announcements, sign-offs,
   subscribe/promo lines, and the music-bed lyrics that open and close episodes.
4. Keeps everything that looks like baseball instruction.
5. Chunks the survivors for embedding, preserving a timestamp per chunk so the
   chatbot can cite the exact moment in the source video.

── Design note: it errs toward keeping ──────────────────────────────────────
A false removal silently destroys knowledge the chatbot will never recover. A
false keep is merely noise that retrieval can rank down. So a line is only cut
when a high-precision rule fires AND the line contains no baseball vocabulary.
Every removal is written to _removal_report.md with its reason — read it once
before trusting the output.

── Usage ────────────────────────────────────────────────────────────────────
    python src/transcript_trimming.py                 # trim everything
    python src/transcript_trimming.py --dry-run       # preview, write nothing
    python src/transcript_trimming.py --report-only   # just the audit report
    python src/transcript_trimming.py --chunk-words 180 --overlap 40
"""
import re
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import TRANSCRIPTS_DIR, TRANSCRIPTS_CLEAN_DIR


# ── Domain vocabulary ────────────────────────────────────────────────────────
# The single most important safeguard: if a line contains any of these, it is
# instructional content and is never removed, whatever else matches.
BASEBALL_TERMS = {
    # swing mechanics
    "swing", "swings", "swinging", "bat", "bats", "barrel", "knob", "handle",
    "hands", "hand", "wrist", "wrists", "elbow", "elbows", "shoulder", "shoulders",
    "hip", "hips", "pelvis", "torso", "trunk", "spine", "core", "knee", "knees",
    "leg", "legs", "foot", "feet", "ankle", "stride", "load", "loading", "gather",
    "separation", "separate", "rotation", "rotate", "rotational", "turn", "turning",
    "extension", "extend", "follow", "through", "finish", "posture", "balance",
    "weight", "shift", "transfer", "coil", "torque", "leverage", "connection",
    # contact / path
    "contact", "launch", "angle", "attack", "path", "plane", "bat path", "swing path",
    "level", "uppercut", "downward", "upward", "trajectory", "direction", "timing",
    "sweet", "spot", "point", "impact", "square", "squared",
    # pitching / pitch types
    "pitch", "pitches", "pitcher", "pitching", "fastball", "curveball", "curve",
    "slider", "changeup", "change-up", "breaking", "offspeed", "off-speed",
    "velocity", "spin", "mound", "release", "delivery", "windup", "arm slot",
    "strike", "strikes", "ball", "balls", "zone", "strike zone", "count",
    # positions / field
    "hitter", "hitters", "hitting", "batter", "batters", "batting", "plate",
    "inside", "outside", "high", "low", "away", "middle", "gap", "field",
    "infield", "outfield", "opposite", "pull", "line drive", "grounder",
    "fly ball", "home run", "base", "bases", "dugout", "cage", "tee", "drill",
    "drills", "practice", "deck", "at-bat", "at bat", "atbat", "inning",
    # player / level
    "player", "players", "professional", "pro", "college", "collegiate",
    "high school", "youth", "little league", "major league", "mlb", "coach",
    "coaching", "hitter's", "athlete",
    # analytical
    "exit", "velo", "mechanics", "mechanic", "kinetic", "sequence", "sequencing",
    "degrees", "degree", "measurement", "data", "average", "percentage",
    # plate discipline / field locations — short coaching cues live here
    "chase", "chasing", "center", "backspin", "topspin", "whiff", "swing-and-miss",
    "approach", "adjust", "adjustment", "trigger", "cue", "rhythm", "tempo",
}

# Multi-word terms checked separately (the set above is token-matched)
BASEBALL_PHRASES = [
    "strike zone", "bat path", "swing path", "line drive", "fly ball",
    "home run", "at bat", "little league", "major league", "high school",
    "exit velocity", "launch angle", "attack angle", "contact point",
    "opposite field", "breaking ball", "off speed", "back side", "front side",
    "top hand", "bottom hand", "lead arm", "back elbow", "hip load",
]


# ── Boilerplate patterns ─────────────────────────────────────────────────────
# Deliberately high-precision. Each is anchored to phrasing that only appears in
# framing material, never mid-lesson.
GREETING_PATTERNS = [
    r"^\s*(hi|hey|hello|what'?s up|good morning|good afternoon|welcome)\b",
    r"\bwelcome (back )?to\b",
    r"\bthanks for (watching|joining|tuning)\b",
    r"\bthank you for (watching|joining|tuning)\b",
    # Bare pleasantries that don't open with a greeting word. "How you doing?"
    # slipped through the anchored pattern above and survived into a chunk.
    r"^\s*how (are|you|is|'?s)\b.{0,25}\??\s*$",
    r"^\s*how'?s (everybody|everyone|it going|things)\b",
    r"^\s*what'?s (going on|happening|good)\b",
    r"\bare back\b.{0,25}\b(20\d\d|this (year|season|week))\b",   # "…are back 2020"
]

SELF_INTRO_PATTERNS = [
    r"\bthis is [A-Z][a-z]+.{0,40}\bwith\b",          # "this is Nathan ... with Gradum"
    r"\bmy name is\b",
    r"\bi'?m [A-Z][a-z]+ (and|with|from)\b",
    r"\bwith (gradum|gratum|gswing|g-?swing)\b",
]

EPISODE_META_PATTERNS = [
    r"\bthis is (our |the )?(first |second |third )?\b.{0,20}\b(episode|vlog|video)\b",
    r"\bepisode (number )?\w+ of\b",
    r"\bteaching tuesday\b.{0,30}\b(episode|vlog)\b",
    r"^\s*this is episode\b",
]

SIGNOFF_PATTERNS = [
    r"\bstay tuned\b",
    r"\bsee you (next|later|guys)\b",
    r"\bnext week'?s?\b.{0,30}\b(teaching|episode|video)\b",
    r"\buntil next time\b",
    r"\bthat'?s (it|all) for\b",
    r"\bwe'?ll see you\b",
    r"\bcatch you (next|later)\b",
]

PROMO_PATTERNS = [
    r"\b(subscribe|like and subscribe|hit the bell|smash that)\b",
    r"\bcheck out (our|the) (website|link|channel)\b",
    r"\blink in (the )?(bio|description)\b",
    r"\bfollow us on\b",
    r"\bdm us\b",
    r"\bsign up (for|at)\b",
]

# Lyric / music-bed detection. Episode intros and outros run over a music bed and
# Whisper transcribes the vocals, producing song lyrics that are (a) not baseball
# content and (b) somebody else's copyrighted work — they should not end up in a
# database or get surfaced by the chatbot.
LYRIC_HINT_PATTERNS = [
    r"\b(yeah|uh|ooh|oh|na na|la la|woah|whoa)\b.*\b(yeah|uh|ooh|oh|na na|la la|woah|whoa)\b",
    r"\bi (won'?t|can'?t) stop\b",
    r"\bknock(ing)? (him|it|them|you) (out|down)\b",
    r"\b(bitch|nigga|shit|fuck|damn|hoe)\b",           # profanity: instruction never has it
    r"\blike i'?m at the\b",
    r"\bwith the hits\b",
]

MUSIC_MARKERS = [
    r"^\s*\[?\s*(music|applause|laughter|intro|outro|instrumental)\s*\]?\s*$",
    r"^\s*\(\s*(music|applause|singing)\s*\)\s*$",
    r"^\s*♪",
]

# Short-utterance handling.
#
# Coaching audio is full of one- and two-word acknowledgements ("Nice.", "Beautiful.",
# "One more.") that carry no retrievable information. But short sentences are ALSO
# where the densest instruction lives — "Don't chase it." is a complete lesson on
# plate discipline in three words. Length alone is therefore not a safe signal, so a
# short line is only dropped when it reads as pure acknowledgement AND contains no
# instructional verb.
PRAISE_FILLER = {
    "beautiful", "nice", "perfect", "good", "great", "awesome", "excellent",
    "yes", "yeah", "yep", "okay", "ok", "right", "sure", "exactly", "correct",
    "alright", "boom", "again", "more", "one", "very", "there", "here", "go",
    "you", "we", "it", "that", "this", "next", "thing", "key", "please",
    "questions", "any", "less", "time", "show", "me", "ahead", "change",
}

INSTRUCTION_VERBS = {
    "don't", "dont", "do", "not", "no", "never", "always", "keep", "stay", "get",
    "drive", "watch", "look", "see", "move", "turn", "load", "stride", "extend",
    "rotate", "hit", "swing", "take", "let", "use", "feel", "think", "start",
    "stop", "chase", "pull", "push", "stick", "hold", "throw", "catch", "square",
    "finish", "follow", "control", "relax", "tighten", "shorten", "lengthen",
}

SEG_RE = re.compile(r"^\*\*\[(\d+):(\d{2})(?::(\d{2}))?\]\*\*\s*(.*)$")


def is_pure_filler(text: str) -> bool:
    """True only for short acknowledgements with no instructional content."""
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens or len(tokens) > 4:
        return False
    if set(tokens) & INSTRUCTION_VERBS:
        return False          # imperative coaching — keep
    return all(t in PRAISE_FILLER for t in tokens)


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_seconds(m: re.Match) -> int:
    a, b, c = m.group(1), m.group(2), m.group(3)
    return int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)


def _fmt_ts(sec: int) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def has_baseball_content(text: str) -> bool:
    low = text.lower()
    if any(p in low for p in BASEBALL_PHRASES):
        return True
    tokens = set(re.findall(r"[a-z']+", low))
    return bool(tokens & BASEBALL_TERMS)


def _matches(text: str, patterns) -> str | None:
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return p
    return None


def classify(sentence: str, pos_ratio: float, ts: int, duration: int) -> tuple[bool, str]:
    """
    Decide whether to keep `sentence`.

    pos_ratio  0.0 = start of video, 1.0 = end
    Returns (keep, reason_if_removed)
    """
    text = sentence.strip()

    if len(text) < 3:
        return False, "empty/too short"

    # Explicit music markers are always noise.
    if _matches(text, MUSIC_MARKERS):
        return False, "music/sound marker"

    baseball = has_baseball_content(text)

    # Intro/outro zone: first 12% or last 12% of the video, or the literal first
    # and last 25 seconds. Music beds and framing live here.
    in_edge_zone = (
        pos_ratio <= 0.12 or pos_ratio >= 0.88
        or ts <= 25 or (duration and ts >= duration - 25)
    )

    # Lyrics: only ever removed in the edge zone and only with no baseball content.
    if not baseball and in_edge_zone and _matches(text, LYRIC_HINT_PATTERNS):
        return False, "music/lyrics"

    # Profanity outside the edge zone with no baseball content is still not
    # instruction — almost always a stray lyric or banter.
    if not baseball and _matches(text, LYRIC_HINT_PATTERNS[-3:]):
        return False, "non-instructional (lyric/banter)"

    # Boilerplate categories. Baseball vocabulary always wins — "welcome back, today
    # we're covering hip load" must survive.
    if not baseball:
        for name, pats in (
            ("greeting", GREETING_PATTERNS),
            ("self-introduction", SELF_INTRO_PATTERNS),
            ("episode announcement", EPISODE_META_PATTERNS),
            ("sign-off", SIGNOFF_PATTERNS),
            ("promo/CTA", PROMO_PATTERNS),
        ):
            if _matches(text, pats):
                return False, name

    # Pure acknowledgement ("Sure.", "Nice.", "One more.") — but never at the cost of
    # short imperatives like "Don't chase it.", which is_pure_filler() protects.
    if not baseball and is_pure_filler(text):
        return False, "filler"

    # Bare proper nouns / mis-transcribed names left over as their own sentence.
    tokens = re.findall(r"[a-z']+", text.lower())
    if not baseball and len(tokens) <= 2 and not (set(tokens) & INSTRUCTION_VERBS):
        return False, "name/fragment"

    return True, ""


# ── parsing ──────────────────────────────────────────────────────────────────

def parse_transcript(path: Path) -> tuple[dict, list]:
    """
    Returns (metadata, segments) where segments is [(seconds, text), ...].
    """
    raw = path.read_text(encoding="utf-8")

    meta = {"title": path.stem, "source_file": "", "duration_s": 0, "language": ""}
    m = re.search(r"\*\*Source file:\*\*\s*`([^`]+)`", raw)
    if m:
        meta["source_file"] = m.group(1)
    m = re.search(r"\*\*Detected language:\*\*\s*(\w+)", raw)
    if m:
        meta["language"] = m.group(1)
    m = re.search(r"\*\*Duration:\*\*\s*([\d:]+)", raw)
    if m:
        parts = [int(x) for x in m.group(1).split(":")]
        meta["duration_s"] = (parts[0] * 3600 + parts[1] * 60 + parts[2]
                              if len(parts) == 3 else parts[0] * 60 + parts[1])

    body = raw.split("## Transcript", 1)[-1]
    segments = []
    for line in body.splitlines():
        m = SEG_RE.match(line.strip())
        if m:
            text = m.group(4).strip()
            if text:
                segments.append((_to_seconds(m), text))
    return meta, segments


def rejoin_sentences(segments: list) -> list:
    """
    Whisper splits on audio pauses, not grammar. Concatenate every segment, then
    re-split on sentence boundaries, carrying the timestamp of the segment where
    each sentence began so chunks stay citable back to the video.
    """
    if not segments:
        return []

    parts, offsets, cursor = [], [], 0
    for ts, text in segments:
        offsets.append((cursor, ts))
        parts.append(text)
        cursor += len(text) + 1

    joined = " ".join(parts)

    def ts_at(char_idx: int) -> int:
        best = offsets[0][1]
        for start, ts in offsets:
            if start <= char_idx:
                best = ts
            else:
                break
        return best

    sentences, pos = [], 0
    for m in re.finditer(r"[^.!?]+[.!?]+|\S[^.!?]*$", joined):
        s = m.group().strip()
        if s:
            sentences.append((ts_at(m.start()), s))
        pos = m.end()
    return sentences


def chunk_sentences(kept: list, chunk_words: int, overlap: int) -> list:
    """Group kept sentences into ~chunk_words chunks with sentence-level overlap."""
    chunks, cur, cur_words, cur_ts = [], [], 0, None
    for ts, sent in kept:
        if cur_ts is None:
            cur_ts = ts
        w = len(sent.split())
        if cur_words + w > chunk_words and cur:
            chunks.append((cur_ts, " ".join(cur)))
            # overlap: carry trailing sentences forward
            back, wc = [], 0
            for s in reversed(cur):
                if wc >= overlap:
                    break
                back.insert(0, s)
                wc += len(s.split())
            cur, cur_words, cur_ts = back, wc, ts
        cur.append(sent)
        cur_words += w
    if cur:
        chunks.append((cur_ts, " ".join(cur)))
    return chunks


# ── main ─────────────────────────────────────────────────────────────────────

def process_file(path: Path, chunk_words: int, overlap: int):
    meta, segments = parse_transcript(path)
    sentences = rejoin_sentences(segments)
    duration = meta["duration_s"]
    last_ts = sentences[-1][0] if sentences else 0
    span = duration or last_ts or 1

    kept, removed = [], []
    for ts, sent in sentences:
        keep, reason = classify(sent, ts / span if span else 0, ts, duration)
        (kept if keep else removed).append((ts, sent, reason))

    kept_pairs = [(ts, s) for ts, s, _ in kept]
    chunks = chunk_sentences(kept_pairs, chunk_words, overlap)
    return meta, sentences, kept, removed, chunks


def run(in_dir: Path = None, out_dir: Path = None, chunk_words: int = 220,
        overlap: int = 45, dry_run: bool = False, report_only: bool = False):
    in_dir = Path(in_dir) if in_dir else TRANSCRIPTS_DIR
    out_dir = Path(out_dir) if out_dir else TRANSCRIPTS_CLEAN_DIR

    if not in_dir.exists():
        raise RuntimeError(f"Transcript folder not found: {in_dir}")

    files = sorted(in_dir.glob("*.md"))
    if not files:
        raise RuntimeError(f"No .md transcripts in {in_dir}")

    if not (dry_run or report_only):
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input : {in_dir}")
    print(f"Output: {out_dir}{'  (DRY RUN — nothing written)' if dry_run else ''}")
    print(f"Found {len(files)} transcript(s)\n")

    all_chunks, report, totals = [], [], {"sent": 0, "kept": 0, "removed": 0}
    reason_counts = {}

    for path in files:
        meta, sentences, kept, removed, chunks = process_file(path, chunk_words, overlap)

        totals["sent"] += len(sentences)
        totals["kept"] += len(kept)
        totals["removed"] += len(removed)
        for _, _, r in removed:
            reason_counts[r] = reason_counts.get(r, 0) + 1

        pct = 100 * len(kept) / len(sentences) if sentences else 0
        print(f"  {path.stem[:52]:52s} {len(sentences):4d} sent -> "
              f"{len(kept):4d} kept ({pct:5.1f}%), {len(chunks):3d} chunks")

        # cleaned markdown
        if not (dry_run or report_only):
            lines = [
                f"# {meta['title']}", "",
                f"- **Source video:** `{meta['source_file']}`",
                f"- **Duration:** {_fmt_ts(meta['duration_s'])}",
                f"- **Sentences kept:** {len(kept)} of {len(sentences)}",
                "", "---", "",
            ]
            for ts, sent, _ in kept:
                lines.append(f"`[{_fmt_ts(ts)}]` {sent}")
                lines.append("")
            (out_dir / path.name).write_text("\n".join(lines), encoding="utf-8")

        for i, (ts, text) in enumerate(chunks):
            all_chunks.append({
                "id": f"{path.stem}::{i:03d}",
                "text": text,
                "source_title": meta["title"],
                "source_file": meta["source_file"],
                "timestamp_s": ts,
                "timestamp": _fmt_ts(ts),
                "chunk_index": i,
                "word_count": len(text.split()),
            })

        report.append((path.stem, sentences, kept, removed))

    # chunks.jsonl
    if not (dry_run or report_only):
        with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # audit report
    if not dry_run:
        rl = ["# Removal report", "",
              "Every sentence the trimmer dropped, with the rule that fired. "
              "Skim this before trusting the cleaned output — if instruction was "
              "cut, loosen the matching rule in `src/transcript_trimming.py`.", "",
              f"- Sentences in: **{totals['sent']}**",
              f"- Kept: **{totals['kept']}** "
              f"({100*totals['kept']/max(totals['sent'],1):.1f}%)",
              f"- Removed: **{totals['removed']}**", "",
              "## Removals by reason", "", "| Reason | Count |", "|---|---|"]
        for r, c in sorted(reason_counts.items(), key=lambda x: -x[1]):
            rl.append(f"| {r} | {c} |")
        rl += ["", "---", ""]
        for stem, sentences, kept, removed in report:
            rl.append(f"## {stem}")
            rl.append("")
            if not removed:
                rl.append("_Nothing removed._")
                rl.append("")
                continue
            rl.append("| Time | Reason | Removed text |")
            rl.append("|---|---|---|")
            for ts, sent, reason in removed:
                safe = sent.replace("|", "\\|")
                if len(safe) > 110:
                    safe = safe[:107] + "..."
                rl.append(f"| {_fmt_ts(ts)} | {reason} | {safe} |")
            rl.append("")
        if not report_only or True:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "_removal_report.md").write_text("\n".join(rl), encoding="utf-8")

    print(f"\n{'─'*70}")
    print(f"Sentences: {totals['sent']}  kept: {totals['kept']} "
          f"({100*totals['kept']/max(totals['sent'],1):.1f}%)  "
          f"removed: {totals['removed']}")
    print(f"Chunks for embedding: {len(all_chunks)}")
    print("\nRemovals by reason:")
    for r, c in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"   {c:4d}  {r}")
    if not dry_run:
        print(f"\nWrote -> {out_dir}")
        print( "  <name>.md            cleaned transcripts")
        if not report_only:
            print("  chunks.jsonl         embedding-ready chunks")
        print("  _removal_report.md   AUDIT THIS FIRST")
    return all_chunks, totals


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Trim Whisper transcripts to baseball content")
    ap.add_argument("--input",  default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--chunk-words", type=int, default=220,
                    help="Target words per chunk (default 220, ~300 tokens)")
    ap.add_argument("--overlap", type=int, default=45,
                    help="Overlap words between chunks (default 45)")
    ap.add_argument("--dry-run", action="store_true", help="Preview only, write nothing")
    ap.add_argument("--report-only", action="store_true", help="Write only the audit report")
    a = ap.parse_args()

    try:
        run(a.input, a.output, a.chunk_words, a.overlap, a.dry_run, a.report_only)
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        sys.exit(2)
