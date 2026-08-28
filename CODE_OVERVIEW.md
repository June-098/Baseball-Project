# Code Overview — what runs, and what we built

*Updated 27 August 2026. Companion to `RUNBOOK.md`, which covers run order.*

Two independent systems share this repo. They have no code in common except `config.py`.

| System | Input | Output | Status |
|---|---|---|---|
| **A. Computer Vision** | batting videos | skeleton MP4s, keypoint CSVs | 2D working, 3D not working |
| **B. RAG Q&A** | coaching transcripts | cited text answers | retrieval working, abstention blocked |

---

# Part 1 — Which files are valid to run

Every file below compiles cleanly. "Runnable" means it has a `__main__` block and is meant
to be invoked directly. "Library" means other files import it and running it does nothing useful.

## A. Baseball analysis (computer vision)

| File | Run it? | What it does |
|---|---|---|
| `src/run_2d_pipeline.py` | ✅ **primary entry point** | Runs the whole 2D pipeline: pose → batter selection → skeleton overlay |
| `src/run_pose_batches.py` | ✅ | Pose extraction only, batched across subprocesses |
| `src/pose_extraction.py` | ✅ | Single stage. Supports `--start/--end` slicing and `--merge-only` |
| `src/batter_selection.py` | ✅ | Single stage. Usually called by the others |
| `src/visualize_2d.py` | ✅ | Skeleton rendering with clip sampling |
| `src/apply_2d_domain.py` | ✅ | Biomechanics angle overlay (separation, spine tilt, knee, hand path) |
| `run_pipeline.py` *(root)* | ⚠️ legacy | Older orchestrator including 3D stages. Superseded by `run_2d_pipeline.py` |

**Not currently working — 3D lifting chain.** These compile and are wired correctly, but the
stage itself is broken and blocks everything downstream of it:
`src/motionbert_setup.py`, `src/motionbert_model.py`, `src/motionbert_inference.py`,
`src/save_3d.py` (library), `src/visualize_3d.py`, `src/compute_3d_metrics.py`.

## B. RAG question answering

| File                         | Run it?                   | What it does                                                               |
| ---------------------------- | ------------------------- | -------------------------------------------------------------------------- |
| `src/transcribe_videos.py`   | ✅                         | Video → timestamped markdown transcripts                                   |
| `src/transcript_trimming.py` | ✅                         | Transcripts → cleaned text + `chunks.jsonl`                                |
| `src/rag_ingest.py`          | ✅ **and library**         | Builds the index. Also exports the `Retriever` class the other two import  |
| `src/rag_eval.py`            | ✅                         | Measures retrieval quality. Also `--inspect` for single-question debugging |
| `src/rag_answer.py`          | ✅ **product entry point** | Question → cited answer, with abstention                                   |

## Library only — never run directly

`config.py` — paths and constants. Every script imports it.
`src/save_3d.py` — helper for the 3D chain.

## Dead weight worth cleaning up

**`src/transcribe.py`** duplicates `transcribe_videos.py`. It is a 63-line minimal version
against the 264-line one that has device auto-selection, resume support, batch processing,
and error isolation. Two files doing one job will eventually get edited in the wrong place.
Delete it or rename it clearly.

**Root-level dated scripts** (`2026-06-16-bat-detection-analysis.py`,
`2026-06-16-bat-recovery-kalman-wrist.py`, `2026-06-17-phase3-motionbert.py`,
`2026-06-17-skeleton-2d-overlay.py`) are notebook exports superseded by `src/`. None have a
`__main__` block. Keep for reference, but they are not part of the pipeline.

---

# Part 2 — What we built, by module

## `run_2d_pipeline.py` — the CV orchestrator

**Problem.** The original work lived in a Colab notebook that exhausted RAM partway through
and had to be babysat cell by cell.

**What we built.** A single command that chains three stages and manages memory as a
first-class concern.

The memory work is the substance here. Ultralytics buffers results for an entire video before
returning, which meant every frame stayed resident at once. Switching to streaming inference
made only one frame resident at a time. On top of that, each batch of videos runs in its own
subprocess, because PyTorch's caching allocator does not reliably return memory to the OS even
after garbage collection — exiting the process is the only guarantee. Device selection is
automatic across CUDA, Apple Silicon MPS, and CPU.

**The bug worth remembering.** After the first full run, only 3 of 183 videos had rendered.
Nothing errored. The cause was that each batch subprocess wrote the combined `keypoints_raw.csv`
from *only its own slice*, so batch 10 silently overwrote batches 1 through 9. The per-video
CSVs had been correct the entire time; only the merged view was wrong. The fix was to make
per-video files the source of truth and require an explicit merge step. A slice can no longer
write the combined file at all.

We also found a dead code path: `VIDEO_EXTS` contained `.MOV` and `.MP4`, but the comparison
lowercased filenames first, so those uppercase entries could never match. Replaced with 26
lowercase extensions behind shared `is_video()` and `list_videos()` helpers.

## `transcript_trimming.py` — text cleaning

**Problem.** Raw Whisper output is roughly 9% greetings, sign-offs, promotional lines, and
music-bed lyrics. That noise competes with real instruction during retrieval.

**What we built.** A classifier that keeps 91% of content (874 of 959 sentences) and logs
every removal with the rule that fired.

Two decisions shaped it. First, **sentence reassembly**: Whisper splits on audio pauses rather
than grammar, so a single sentence routinely spans two segments. Embedding those fragments
separately produces meaningless vectors, so sentences are rejoined before anything else runs.
Second, **bias toward keeping**: a wrong removal destroys knowledge permanently, while a wrong
keep is just noise that ranks low. A line is only cut when a high-precision rule fires *and* it
contains no baseball vocabulary.

**A bug we caught in testing.** The first version dropped any sentence under four words as
filler. That removed "Don't chase it." — a complete lesson on plate discipline. Short sentences
are where the densest coaching lives. The rule now only drops short lines that read as pure
acknowledgement and contain no instructional verb.

## `rag_ingest.py` — indexing and retrieval

**Problem.** Turn 83 text chunks into something searchable by meaning, not just keywords.

**What we built.** Local ChromaDB storing 384-dimension embeddings from `bge-small-en-v1.5`,
plus a BM25 keyword index, fused at query time.

Three techniques, each added for a measured reason:

**Contextual prefixes.** Mid-video chunks rarely name their own topic; the coach just says "it".
Each chunk embeds with its video title wrapped around it so topic carries into the vector.

**Hybrid retrieval.** Vector search handles paraphrase well and exact strings poorly. Named
drills and player names are exact-string problems, so BM25 runs alongside and the two rankings
fuse through Reciprocal Rank Fusion, which combines ranks rather than incomparable scores.

**Cross-encoder reranking.** A bi-encoder embeds question and passage separately, measuring
topical similarity. A cross-encoder reads them together and scores actual relevance, which is
sharper for both ordering and abstention.

**Two defects the evaluation caught.** Equal-weight rank fusion measured *worse* than vector
alone (recall@5 0.85 versus 0.95). On a small corpus where every chunk says "ball" and "swing",
BM25's ranking is close to noise, and equal weighting let it outvote good semantic results.
Fixed with vector-dominant weights and a minimum BM25 score floor. Separately, "How do I hit a
curveball?" returned a high-pitch video, because the curve ball transcript says "breaking ball"
throughout and "curveball" once — and a five-word title prefix on a 220-word chunk was only
about 2% of the token mass. Fixed by wrapping the title at both ends.

## `rag_eval.py` — the measurement harness

**Problem.** Without a number that moves, you cannot tell whether a change helped.

**What we built.** A harness that runs a labeled question set and reports recall@1/3/5, mean
reciprocal rank, coverage separation, and a threshold sweep. It never ships; it is an
instrument.

The question set has two halves. In-coverage questions must retrieve a known source.
Out-of-coverage questions must be refused. Both are required, because the second half is what
sets the abstention threshold.

**Coverage separation** is the concept that made the abstention problem legible. It measures
the gap between the lowest-scoring in-coverage question and the highest-scoring out-of-coverage
one. If that gap is negative, the two groups overlap and *no threshold can separate them* —
which is exactly what the data showed. The harness now refuses to suggest a threshold when the
distributions overlap, rather than emitting a number that looks authoritative and is not.

`--inspect <ID>` dumps the actual chunks retrieved for one question, which distinguishes the two
causes of a miss: retrieval ranked the right chunk too low, or the corpus genuinely does not
answer it. Those need opposite fixes, and no metric can tell them apart.

**Results.** recall@1 0.75 → 0.80, recall@5 0.85 → 0.90, MRR 0.775 → 0.829. Coverage gap
remains negative, which is the open issue.

## `rag_answer.py` — generation

**Problem.** Retrieval returns five raw transcript chunks. Users need prose.

**What we built.** A pipeline that scores, abstains, retrieves, generates, and then validates.

**Abstention runs before the LLM call.** Vector search always returns its top five, whether or
not anything relevant exists — there is no "no results" state. Without a floor, an
out-of-coverage question yields a confident answer assembled from unrelated material. Checking
the score first is cheaper and impossible to talk around.

**Grounding is hybrid.** Standard baseball terms may be defined from general knowledge and are
marked as such. Every *coaching* claim must come from the transcripts with a video and timestamp
citation. That separation keeps the product honest about what it actually knows.

**Validation runs in code.** A prompt is a request; validation is a guarantee. The most important
check asks whether any retrieved video title actually appears in the answer. If not, the model
answered from its own training knowledge rather than the transcripts — the exact failure the
product exists to prevent, and one that is invisible because a hallucinated answer reads exactly
like a grounded one. Failures retry once, then abstain.

---

# Part 3 — Housekeeping notes

**`chunks.jsonl` and the cleaned `.md` files are out of sync.** `chunks.jsonl` covers 20 videos;
the `transcripts_clean/` folder currently holds 12 hand-edited `.md` files. This does not break
anything — `rag_ingest.py` reads `chunks.jsonl` only, and the `.md` files are for human reading.
Worth knowing so the difference is not mistaken for data loss.

**The cleaned `.md` files are hand-curated.** They have been renamed and given section headers
that the trimmer never produced. **Re-running `transcript_trimming.py` would overwrite that
work.** Copy the folder aside first if you ever need to regenerate.

**All 21 in-coverage golden questions still match the current corpus.** Verified against
`chunks.jsonl` on 27 August. One thing to tidy: IC-21 ("How do I hit inside pitches?") carries
a note copied from IC-20 that says "timing/rhythm concept", which does not describe it.
