# Runbook — what each script does and the order to run them

Two **independent** pipelines live in this repo. They share `config.py` and nothing else.

| Pipeline | Input | Output | Purpose |
|---|---|---|---|
| **A. Computer Vision** | batting videos | skeleton MP4s, keypoint CSVs | measure body mechanics |
| **B. RAG / Q&A** | coaching videos | cited text answers | answer hitting questions |

You can run either without touching the other.

---

## Shared

### `config.py` — *library, never run directly*

Single source of truth for paths and constants. Every script imports from it, so changing
a folder location here changes it everywhere.

| Provides | Used by |
|---|---|
| `BATTING_VIDEOS_DIR`, `DATA_DIR`, `DIAGNOSES_DIR` | CV pipeline |
| `TRANSCRIPTS_DIR`, `TRANSCRIPTS_CLEAN_DIR`, `RESOURCES_DIR` | RAG pipeline |
| `VIDEO_EXTS`, `is_video()`, `list_videos()` | both — one definition of "is this a video" |
| `KEYPOINT_NAMES`, `H36M_JOINTS`, `CONF_THRESHOLD` | CV pipeline |

---

# Pipeline B — RAG / Q&A  *(current focus)*

## Step 1 · `src/transcribe_videos.py`

**Video files → markdown transcripts**

| Function | Does |
|---|---|
| `_check_prereqs()` | Fails fast if ffmpeg or faster-whisper is missing |
| `_pick_device()` | CUDA → CPU+int8. CTranslate2 has no Metal backend, so Apple Silicon uses CPU |
| `_extract_audio()` | ffmpeg → 16 kHz mono WAV (what Whisper expects) |
| `transcribe_one()` | One video → timestamped markdown |
| `run()` | Loops the folder, skips already-done files, isolates failures |

**Why it exists:** you can't search video. This turns speech into text.

```bash
python src/transcribe_videos.py                 # all new videos in Gradum Gswing/
python src/transcribe_videos.py --model medium  # more accurate, slower
```

**Rerun when:** you add videos. Completed ones are skipped automatically.

---

## Step 2 · `src/transcript_trimming.py`

**Raw transcripts → clean, embedding-ready chunks**

| Function | Does | Why it matters |
|---|---|---|
| `parse_transcript()` | Pulls metadata + `**[0:00]** text` segments | strips timestamp markup |
| `rejoin_sentences()` | Reassembles sentences Whisper split mid-clause | **critical** — Whisper cuts on pauses, not grammar. Embedding fragments produces garbage vectors |
| `has_baseball_content()` | Domain vocabulary check | the safety net: if a line has baseball terms it is never removed |
| `is_pure_filler()` | "Nice." yes, "Don't chase it." no | length alone is not a filler signal |
| `classify()` | Keep/remove + reason | biased toward keeping |
| `chunk_sentences()` | Groups into ~220-word chunks with overlap | retrieval unit |

**Why it exists:** raw transcripts are ~9% greetings, sign-offs, promos, and music-bed
lyrics. That noise competes with real instruction during retrieval.

```bash
python src/transcript_trimming.py            # writes cleaned md + chunks.jsonl
python src/transcript_trimming.py --dry-run  # preview, writes nothing
```

**Always read** `transcripts_clean/_removal_report.md` afterward — it lists every cut
sentence and the rule that fired. Nothing is deleted silently.

**Rerun when:** you add transcripts, or change chunk size.

---

## Step 3 · `src/rag_ingest.py`

**chunks.jsonl → searchable index** (also holds the retriever used by steps 4 and 5)

| Function                        | Does                                                                                                                 |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `contextual_text()`             | Prepends `[Video: Title]` before embedding. Mid-video chunks often never name their topic — the coach just says "it" |
| `build()`                       | Embeds everything → ChromaDB, plus a BM25 keyword index                                                              |
| `Retriever.vector_search()`     | Semantic search — good at paraphrase                                                                                 |
| `Retriever.keyword_search()`    | BM25 — good at exact strings ("Ferris Wheel", "Mookie Betts")                                                        |
| `Retriever.search()`            | Fuses both with Reciprocal Rank Fusion                                                                               |
| `Retriever.best_vector_score()` | Top similarity — the number abstention reads                                                                         |

**Why it exists:** this is the vector database. ChromaDB is *embedded* — vectors live in
`Baseball Resources/rag_index/chroma/` on your disk, not on someone's server.

```bash
python src/rag_ingest.py                              # build the index
python src/rag_ingest.py --rebuild                    # wipe + rebuild
python src/rag_ingest.py --query "hit a curveball"    # smoke test (raw chunks, not prose)
```

**Rerun when:** chunks change, or you swap the embedding model.

---

## Step 4 · `src/rag_eval.py`

**Measure retrieval before building on it**

| Function | Does |
|---|---|
| `load_golden()` | Reads `golden_questions.jsonl` (20 in-coverage + 10 out-of-coverage) |
| `hit_rank()` | Rank of the first correct source, or None |
| `evaluate()` | recall@1/3/5 + MRR |
| `coverage_separation()` | Gap between in-coverage and out-of-coverage similarity — **this gap IS the abstention threshold** |
| `sweep_thresholds()` | Accuracy at each candidate threshold |

**Why it exists:** without a number that moves, you cannot tell whether a change helped.
It also produces the threshold Step 5 needs. **Never ships** — it's a measuring instrument.

```bash
python src/rag_eval.py --compare-hybrid --sweep
```

**Rerun after every change** to chunking, embedding model, or retrieval settings.

---

## Step 5 · `src/rag_answer.py`

**Question → cited prose answer**

| Function | Does |
|---|---|
| `answer_question()` | Orchestrates: score → abstain? → retrieve → generate → validate |
| `build_context()` | Formats chunks with titles + timestamps for the prompt |
| `validate_answer()` | **Enforces policy in code** (see below) |
| `abstain_response()` | The refusal, with corpus scope so the user learns what *is* covered |
| `run_golden()` | End-to-end check across all 30 questions |

**Why it exists:** Steps 1–4 return raw chunks. This turns them into readable answers.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python src/rag_answer.py "what is launch angle and how do I improve it?"
python src/rag_answer.py --interactive
python src/rag_answer.py --golden
```

⚠️ Set `ABSTAIN_THRESHOLD` from your own `rag_eval.py --sweep` first. The committed
value (0.42) is a placeholder.

---

## Run order — Pipeline B

```bash
cd "Personal Baseball Project"
source .venv/bin/activate
pip install chromadb sentence-transformers rank-bm25 anthropic

python src/transcribe_videos.py        # 1. videos  -> transcripts      (skip if done)
python src/transcript_trimming.py      # 2. clean   -> chunks.jsonl
python src/rag_ingest.py               # 3. embed   -> ChromaDB + BM25
python src/rag_eval.py --compare-hybrid --sweep   # 4. MEASURE. read the report.
#    -> edit ABSTAIN_THRESHOLD in src/rag_answer.py using the sweep result
python src/rag_answer.py --golden      # 5. end-to-end check
python src/rag_answer.py --interactive # 5. use it
```

**Each step consumes the previous step's output.** Skipping ahead fails with a clear error
(no chunks, no index). Step 4 is the one people skip and shouldn't — it's the only place
you learn whether retrieval actually works, and it produces the threshold Step 5 requires.

---

# Pipeline A — Computer Vision

| Script | Role |
|---|---|
| `src/pose_extraction.py` | YOLO pose + ByteTrack → per-video keypoint CSVs. `--start/--end` for batching |
| `src/batter_selection.py` | Largest-bbox track per frame = the batter; assigns `segment_id` per swing |
| `src/visualize_2d.py` | Draws the 15-point skeleton → MP4. Samples 10 Penn Action + 3 slow-motion by default |
| `src/run_2d_pipeline.py` | **Runs all three.** The one to use |
| `src/run_pose_batches.py` | Pose stage only, batched |
| `src/apply_2d_domain.py` | Biomech angle overlay (hip-shoulder separation, spine tilt, knee, hand-path) |
| `src/motionbert_*.py`, `save_3d.py`, `visualize_3d.py` | 2D→3D lifting. **Currently not working** |
| `src/compute_3d_metrics.py` | View-independent 3D angles. Blocked on the above |
| `run_pipeline.py` (root) | Older orchestrator for all stages incl. 3D |

```bash
python src/run_2d_pipeline.py --seed 42               # full 2D pipeline
python src/run_2d_pipeline.py --skip-pose --seed 42   # re-render only
```

**Known state (Aug 2026):** 2D overlay works. Pose extraction is partial. Batter
selection/segmentation and 3D lifting are not working. See the roadmap — Q1 is a recovery
quarter.

---

# `validate_answer()` — why it exists

**A prompt is a request. Validation is a guarantee.**

The system prompt tells the model to cite every coaching claim. It usually complies.
"Usually" is the problem: LLM instruction-following degrades on edge cases, long contexts,
and unusual questions — and when it fails, **the failure is invisible**. An ungrounded
answer reads exactly like a grounded one. Same confident tone, same fluent prose, no
citation.

So `validate_answer()` re-checks the finished text in code:

```python
def validate_answer(answer, hits):
    if len(answer.strip()) < 40:                     # 1. generation failed
        return False, "answer too short"

    cited = any(title_stem(h) in answer.lower() for h in hits)
    if not cited:                                    # 2. THE important one
        return False, "no citation matching any retrieved source"

    for pat in BANNED_PATTERNS:                      # 3. safety / legal
        if re.search(pat, answer, re.I):
            return False, f"banned claim: ..."

    return True, ""
```

**Check 2 is the one that matters.** If no retrieved video title appears in the answer, the
model answered from its own training knowledge instead of your transcripts. That is exactly
the failure the product is built to prevent — the answer may even be *correct*, but it isn't
*grounded*, and you can't tell which without checking.

**Check 3** blocks injury, medical, diagnostic, and guarantee language. Your users include
minors, and no ground truth in this corpus supports injury claims. Encoding it here means
nobody has to remember the rule.

**On failure:** retry once with the rejection reason attached, then abstain. Unvalidated
text is never shown.

**Its limit, stated honestly:** it verifies a citation *exists*, not that it's *accurate*.
It's a floor, not a ceiling. Catching wrong-but-present citations needs an LLM judge — worth
adding later, unnecessary now.

**Why this is the same idea as the roadmap's MEDA/APA/APE rule:** the retrieval layer
decides what is true; the language model only explains it. `validate_answer()` is what makes
that boundary real instead of aspirational — enforced in code rather than requested in a
prompt.
