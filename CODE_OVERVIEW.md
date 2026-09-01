# Code Overview — what runs, and what we built

*Updated 31 August 2026. Companion to `RUNBOOK.md`, which covers run order.*

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
| `src/rag_chunk.py`           | library (also `__main__` stats) | Parses Markdown into parent/child chunks. Ingest calls this.          |
| `src/rag_ingest.py`          | ✅ **run this first**     | Builds the index. Exports `Retriever` — eval and answer import it          |
| `src/rag_eval.py`            | ✅ after ingest           | Measures retrieval. `--inspect` for one golden question                    |
| `src/rag_answer.py`          | ✅ **product entry point** | Question → prose answer. Imports `Retriever`; does not rebuild the index  |
| `src/transcribe_videos.py`   | ✅                         | Video → timestamped markdown transcripts                                   |
| `src/transcript_trimming.py` | ✅                         | Older Whisper-clean path. Live ingest no longer reads its `chunks.jsonl`   |

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

**Run this before `rag_eval.py` or `rag_answer.py`.** Those two files do not build the
index. They import `Retriever` from here and read whatever is already on disk under
`Baseball Resources/RAG Resources/rag_index/`. If you change chunking or the embedding
model, rebuild here first or you are measuring yesterday's index.

```
python src/rag_ingest.py --rebuild     # 1. write chunks + Chroma + BM25
python src/rag_eval.py                 # 2. recall / MRR on golden questions
python src/rag_answer.py "..."         # 3. prose answers from the same Retriever
```

**Problem.** Coaching notes must be searchable by meaning ("load my hips") and by exact
names ("Ferris Wheel"), and the LLM needs enough surrounding explanation to write a real
answer. Equal-sized character slices solved search but cut sentences in half. Fixed-width
RCTS (300 characters, 80 overlap) peaked at recall@5 **0.955** on 22 in-coverage questions
with **416** slices — then lost context at generation time.

**What we built.** A two-file indexing path: `rag_ingest.py` owns the disk index and
search; `rag_chunk.py` owns Markdown → parent/child records. The rest of this section
is the call chain, then how Chroma and BM25 actually store and fuse those records.

### How `rag_chunk.py` starts when you run ingest

You never run `rag_chunk.py` first. It is a library. `python src/rag_ingest.py` (or
`--rebuild`) hits the `__main__` block, which calls `build()`. `build()` calls
`load_chunks()`. If you did not pass `--chunks`, `load_chunks()` lists every top-level
`.md` in `Baseball Resources/RAG Resources/` (skipping files whose names start with
`_` and anything inside `rag_index/`) and, for each file, calls `chunks_from_markdown(path)`.

That helper is the only handshake:

```
chunks_from_markdown(path)
    from rag_chunk import children_from_markdown
    return children_from_markdown(path)
```

The import happens **inside the function**, so ingest can start even if transformers is
slow to load. From there `children_from_markdown` does the real work, in this order:

1. Read the note. Strip YAML front matter (kept as tags) and decorative `---` rules.
2. `parse_blocks` walks the file as Markdown: headings, paragraphs, lists, tables, quotes.
3. `blocks_to_leaves` turns heading stacks into **leaves** (one heading path + its body).
4. `coalesce_leaves` merges consecutive short siblings so a five-word heading is not its
   own retrieval unit.
5. `pack_leaf` packs each leaf into **children** of roughly 120–220 embedding-model
   tokens. Definitions can stay shorter. Oversize paragraphs split on sentence boundaries.
6. For every child, `choose_parent_text` picks a **parent** window of roughly 300–600
   tokens — usually the smallest ancestor section that still fits, otherwise neighboring
   leaves. That parent is what the LLM later reads.
7. `build_embedding_text` prefixes the child with `Document` / `Section` / `Content type`
   so the mean-pooled vector contains the topic, not just a paragraph that says "it".

Each child is one dict. The fields that matter downstream:

| Field | Role |
|---|---|
| `id` | Stable key, e.g. `Attack Angle::001`. Same id in Chroma and BM25. |
| `text` | Display Markdown (what you would show a reader). |
| `embedding_text` | Hierarchy header + body. This is what gets **embedded** and what BM25 tokenizes. |
| `parent_id` / `parent_text` | Generation window. Not embedded as its own vector. |
| `source_title`, `section_path`, `content_type` | Metadata stored on the Chroma row. |

`build()` then writes every child as one JSON line to `rag_index/chunks.jsonl`. That file
is the human-readable copy of the index. Eval and parent expansion read it back by id.

Running `python src/rag_chunk.py` by itself only prints per-note stats. It does **not**
write Chroma or BM25. Ingest is what persists.

### How local Chroma stores the embeddings

Chroma here is a **folder on disk**, not a server. `build()` does:

```
client = chromadb.PersistentClient(path=.../rag_index/chroma)
collection = client.create_collection("baseball_transcripts",
                                      metadata={"hnsw:space": "cosine"})
```

`PersistentClient` means every `add` is written under `rag_index/chroma/` (SQLite plus
HNSW graph files). Restarting Python does not rebuild anything — `Retriever` later calls
`get_collection` on the same path.

The embedding model is `BAAI/bge-small-en-v1.5` (384 dimensions, ~130 MB, downloaded
once by sentence-transformers). Ingest embeds `embedding_text` for every child, with
`normalize_embeddings=True`, then `collection.add(...)` stores four aligned lists:

- **ids** — the child ids
- **embeddings** — the 384-d vectors (the searchable thing)
- **documents** — the clean `text` (returned to you at query time)
- **metadatas** — source title, section path, parent_id, content type, etc.

HNSW is an approximate nearest-neighbor graph over those vectors. Cosine **distance** is
what Chroma stores; `Retriever.vector_search` converts it to **similarity** with
`1 - distance` so 1.0 is identical and 0.0 is orthogonal.

A question is **not** compared as raw text. `vector_search` embeds:

```
Represent this sentence for searching relevant passages: <question>
```

That prefix is required by bge models on the **query** side only. Documents were already
embedded at ingest without it. The result is the top-k children whose vectors point the
same way as the question.

`--rebuild` deletes the `chroma/` folder first, then recreates the collection. Without
`--rebuild`, ingest still deletes and recreates the named collection so you do not get
duplicate ids.

Chroma does **not** know about BM25. It is only the vector store + document store.

### What the BM25 index is, and how it works with Chroma

BM25 (Okapi BM25, via `rank_bm25.BM25Okapi`) is a **keyword** ranker. It does not use
vectors. It tokenizes each `embedding_text` into lowercase words (`tokenize()` keeps
`a-z`, `0-9`, and apostrophes) and, at query time, scores how well the query's words
match each document: rare words (high IDF) count more than "swing" or "ball", and longer
documents are down-weighted.

Ingest pickle-dumps two things into `rag_index/bm25.pkl`:

```
{"bm25": BM25Okapi([token lists in child order]), "ids": [same child ids]}
```

The id list is the join key. BM25 returns an array of scores aligned to that list;
`keyword_search` sorts, keeps the top-k, and maps positions back to ids. It does **not**
store the passage text. When a BM25-only hit is not already in the vector results,
`search()` loads the text from Chroma with `col.get(ids=[cid])`.

So the two indexes are complementary views of the **same children**:

| | Chroma | BM25 |
|---|---|---|
| Good at | paraphrase ("load my hips" → gather) | exact names ("Ferris Wheel", "Money Gap") |
| Input at ingest | 384-d vector of `embedding_text` | word counts of the same `embedding_text` |
| Query | embed the question, HNSW cosine | tokenize the question, BM25 score |
| Returns | id + text + metadata + similarity | id + keyword score |

**Hybrid search** (`Retriever.search`) pulls a pool from each (about `max(4k, 20)`),
drops BM25 hits below score **2.0** (BM25 always returns *something*, even when nothing
matched), then fuses **ranks** with weighted Reciprocal Rank Fusion:

```
score(id) += vector_weight / (60 + rank + 1)     # weight 1.0
score(id) += bm25_weight  / (60 + rank + 1)     # weight 0.35
```

Fusion is on rank, not on raw cosine vs raw BM25, because those magnitudes are not
comparable. Vector gets the larger weight because equal-weight fusion used to *lose* to
vector-only on this corpus (recall@5 0.85 vs 0.95) — every note says "ball" and "swing",
so BM25's ranking is close to noise unless a distinctive term actually fired.

The hierarchy header on `embedding_text` is why prefixes still matter: mid-note chunks
rarely name the document, so "curveball" used to lose to a high-pitch video when the note
said "breaking ball" everywhere and "curveball" once. Putting `Document` / `Section` /
`Content type` in the string that both Chroma and BM25 see puts the title in the vector
*and* in the keyword index.

Optional `--rerank` then runs a cross-encoder on the fused pool. `rag_answer.py` does
**not** turn that on. Abstention still reads `best_vector_score` (top cosine).

When `expand_parent=True` (answer path), child hits are replaced by `parent_text` and
deduped by `parent_id`, so five children from one section become one parent for Gemma.
Eval keeps `expand_parent=False` so recall measures whether the right *note* was found.

**What evaluation says now (hybrid, children, 22 in-coverage).** Parent-child recall@1
**0.773**, recall@5 **0.909**, MRR **0.841**. Misses: IC-15 (rollover) and IC-20 (athletic
stance). MRR fell versus RCTS because rank-1 collapsed: mean-pooling a section-sized child
dilutes a needle sentence that used to own a 50-token slice. That is the small-to-big
tradeoff, not a broken parser. Generation is the part that improved — the model sees the
parent, not a mid-sentence fragment.

**Smoke-test retrieval without calling the LLM:**

```
python src/rag_ingest.py --query "curveball" --full
```

That prints the **parent** text the answer step would feed Gemma. It is not an answer.

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

**Results (29 August, parent-child index).** Hybrid child retrieval: recall@1 0.773,
recall@5 0.909, MRR 0.841 on 22 in-coverage questions. Coverage gap remains negative
(cosine still cannot separate IC from OOC). Misses: IC-15, IC-20. Rerank is still the
better abstention *signal*; it is not wired into `answer_question`.

## `rag_answer.py` — generation

**Problem.** Retrieval returns five raw transcript chunks. Users need prose.

**What we built.** A pipeline that scores, abstains, retrieves, generates, and then validates.
Generation is local **Gemma 4 E4B** via Ollama (`gemma4:e4b` at `http://127.0.0.1:11434`).
Anthropic helpers remain in the file, commented out. Retrieval is unchanged: hybrid
Chroma + BM25, then `expand_parent=True` so Gemma sees parent text.

**Abstention runs before the LLM call.** Vector search always returns its top five, whether or
not anything relevant exists — there is no "no results" state. Without a floor, an
out-of-coverage question yields a confident answer assembled from unrelated material. Checking
the score first is cheaper and impossible to talk around.

**Grounding is hybrid.** Standard baseball terms may be defined from general knowledge and
are marked "(general definition)". Coaching claims must come from the retrieved excerpts.
The answer body is prose only — no `[Video | section @ 0:00]` in the paragraphs. `print_answer`
appends **Sources** and **Retrieved from** from the hit list.

**Validation runs in code.** Banned medical/injury language is rejected. Leftover bracket
citations are stripped, then rejected if they remain. Failures retry once, then abstain.
Abstention still uses cosine vs `ABSTAIN_THRESHOLD` (0.42) *before* the LLM call.

**Golden set (generation check, not retrieval metrics).** `--goldenset` is not a separate
script. Use `--golden` (or the alias `--goldenset`). This is *not* `rag_eval.py`. Eval
only scores whether search found the right note; `--golden` asks Gemma every question and
checks "answered vs abstained".

```
python src/rag_answer.py --golden
```

Terminal: in-coverage should print `OK` (answered), out-of-coverage `OK` (abstained).
JSON with the full answers: `Baseball Resources/RAG Resources/rag_index/answer_report.json`.

#### `_ollama_chat` function:
messages
   │
   ▼
payload ───────────── Python dict
   │
   │ json.dumps()
   ▼
JSON string
   │
   │ encode("utf-8")
   ▼
bytes
   │
   │ HTTP POST
   ▼
Ollama :11434/api/chat
   │
   │ model inference
   ▼
response bytes
   │
   │ decode("utf-8")
   ▼
JSON string
   │
   │ json.loads()
   ▼
data ──────────────── Python dict
   │
   ▼
data["message"]["content"]
   │
   ▼
text
   │
   ▼
return


---

# Part 3 — Housekeeping notes

**Ingest reads `Baseball Resources/RAG Resources/*.md` and writes `rag_index/`.** It does
not read `transcripts_clean/chunks.jsonl`. `chunks.jsonl` in the index folder is an *output*
of ingest (one JSON object per child). Re-running `transcript_trimming.py` does not refresh
the live RAG index.

**The `.md` notes in RAG Resources are hand-curated** (section headers, cleaned coaching
prose). Do not overwrite them from a trimmer run without a backup.

**Golden set: 22 in-coverage + 10 out-of-coverage.** `rag_eval.py` only checks whether
search found the expected *note title*. `note` in the golden file is not an LLM prompt.
`ideal_answer` is a future grading key, not required to run `rag_answer.py`.
