"""
Chunk size / overlap sweep — find the settings that retrieve best, by measuring.

Runs a grid of (chunk_words, overlap) combinations. For each one it rebuilds
chunks, rebuilds the index, runs the golden-set evaluation, and records the
metrics. Then it prints a ranked table and tells you which setting won.

    python src/rag_chunk_sweep.py                                  # default grid
    python src/rag_chunk_sweep.py --sizes 120 220 400 --overlaps 0 45
    python src/rag_chunk_sweep.py --apply                          # write the winner

── Why a sweep instead of a rule of thumb ───────────────────────────────────
Chunk size trades two failure modes against each other:

  TOO SMALL   The answer gets split across a boundary. Retrieval finds half of
              it, and the LLM sees an incomplete passage.

  TOO LARGE   One relevant sentence gets averaged together with 300 words of
              other material. Embeddings mean-pool, so the signal you care
              about gets diluted. You already saw this exact effect with title
              dilution: five title words inside a 220-word chunk were roughly
              2% of the token mass and barely moved the vector.

Overlap exists to soften the first failure. Repeating a few sentences at each
boundary means a concept that straddles two chunks appears whole in at least
one. It costs storage and adds near-duplicate results, so more is not better.

The right values depend on how the source material is written: monologue
coaching video chunks differently than dense technical prose. There is no
universal number, which is why this measures instead of asserting.

── What it does NOT touch ───────────────────────────────────────────────────
Your hand-edited .md files in transcripts_clean/ are never rewritten. The sweep
generates chunk data in a temp file and only writes chunks.jsonl if you pass
--apply. The ChromaDB index IS rebuilt on each iteration and is restored to the
winning (or original) configuration at the end.
"""
import sys
import json
import shutil
import tempfile
import argparse
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TRANSCRIPTS_DIR, TRANSCRIPTS_CLEAN_DIR, RESOURCES_DIR

REPORT_PATH = RESOURCES_DIR / "rag_index" / "chunk_sweep_report.md"

DEFAULT_SIZES    = [120, 180, 220, 300, 400]
DEFAULT_OVERLAPS = [0, 25, 45, 80]


def build_chunks(chunk_words: int, overlap: int, out_path: Path) -> int:
    """
    Regenerate chunk data at the given settings, writing ONLY to out_path.

    Calls transcript_trimming's own functions rather than the script, so the
    cleaned .md files (which are hand-curated) are never rewritten.
    """
    from transcript_trimming import process_file, _fmt_ts

    rows = []
    for path in sorted(TRANSCRIPTS_DIR.glob("*.md")):
        meta, _sentences, _kept, _removed, chunks = process_file(path, chunk_words, overlap)
        for i, (ts, text) in enumerate(chunks):
            rows.append({
                "id": f"{path.stem}::{i:03d}",
                "text": text,
                "source_title": meta["title"],
                "source_file": meta.get("source_file", ""),
                "timestamp_s": ts,
                "timestamp": _fmt_ts(ts),
                "chunk_index": i,
                "word_count": len(text.split()),
            })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def evaluate_config(chunks_path: Path, k: int, rerank: bool) -> dict:
    """Rebuild the index from chunks_path, then score it on the golden set."""
    import rag_ingest
    from rag_eval import load_golden, evaluate, coverage_separation

    rag_ingest.build(chunks_path, rebuild=True)

    retriever = rag_ingest.Retriever()
    ic, ooc = load_golden()

    res = evaluate(retriever, ic, k, use_hybrid=True, rerank=rerank)
    sep = coverage_separation(retriever, ic, ooc,
                              signal="rerank" if rerank else "vector")

    return {
        "recall@1": res["recall@1"],
        "recall@3": res["recall@3"],
        f"recall@{k}": res[f"recall@{k}"],
        "mrr": res["mrr"],
        "gap": sep.get("gap"),
        "misses": [m["id"] for m in res["misses"]],
    }


def run(sizes, overlaps, k=5, rerank=False, apply_best=False):
    chunks_live = TRANSCRIPTS_CLEAN_DIR / "chunks.jsonl"
    backup = None
    if chunks_live.exists():
        backup = chunks_live.with_suffix(".jsonl.sweep-backup")
        shutil.copy2(chunks_live, backup)
        print(f"Backed up current chunks.jsonl -> {backup.name}")

    combos = [(s, o) for s in sizes for o in overlaps if o < s]
    print(f"\nTesting {len(combos)} configurations "
          f"(overlap must be < chunk size)\n")
    print(f"  {'size':>5} {'ovlp':>5} {'chunks':>7} {'r@1':>6} {'r@3':>6} "
          f"{'r@'+str(k):>6} {'mrr':>6} {'gap':>8}")
    print("  " + "-" * 60)

    results = []
    tmpdir = Path(tempfile.mkdtemp(prefix="chunk_sweep_"))
    try:
        for size, ovl in combos:
            tmp_chunks = tmpdir / f"chunks_{size}_{ovl}.jsonl"
            try:
                n = build_chunks(size, ovl, tmp_chunks)
                m = evaluate_config(tmp_chunks, k, rerank)
                gap = m["gap"]
                gap_s = f"{gap:+.3f}" if gap is not None else "   n/a"
                print(f"  {size:>5} {ovl:>5} {n:>7} "
                      f"{m['recall@1']:>6.3f} {m['recall@3']:>6.3f} "
                      f"{m[f'recall@{k}']:>6.3f} {m['mrr']:>6.3f} {gap_s:>8}")
                results.append({"chunk_words": size, "overlap": ovl,
                                "n_chunks": n, **m})
            except Exception as exc:
                print(f"  {size:>5} {ovl:>5}   FAILED: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=1)

        if not results:
            print("\nNo configuration completed. Nothing changed.")
            return []

        # Rank by recall@k first, then MRR as the tie-breaker. Recall decides
        # whether the right passage is reachable at all; MRR only reorders it.
        ranked = sorted(results,
                        key=lambda r: (r[f"recall@{k}"], r["mrr"]), reverse=True)
        best = ranked[0]

        print(f"\n{'='*62}")
        print(f"BEST: chunk_words={best['chunk_words']}  overlap={best['overlap']}")
        print(f"      recall@{k}={best[f'recall@{k}']:.3f}  mrr={best['mrr']:.3f}  "
              f"chunks={best['n_chunks']}")
        if best["misses"]:
            print(f"      still missing: {', '.join(best['misses'])}")

        spread = ranked[0][f"recall@{k}"] - ranked[-1][f"recall@{k}"]
        if spread < 0.06:
            print(f"\n  NOTE: only {spread:.3f} recall separates best from worst.")
            print("  On a corpus this small that difference is within noise.")
            print("  Prefer the simpler setting and revisit after adding videos.")

        write_report(ranked, k, rerank)

        if apply_best:
            src = tmpdir / f"chunks_{best['chunk_words']}_{best['overlap']}.jsonl"
            shutil.copy2(src, chunks_live)
            import rag_ingest
            rag_ingest.build(chunks_live, rebuild=True)
            print(f"\nApplied. chunks.jsonl regenerated at "
                  f"chunk_words={best['chunk_words']}, overlap={best['overlap']}")
            print("Set the same values as defaults in transcript_trimming.py "
                  "so future runs match.")
        elif backup:
            shutil.copy2(backup, chunks_live)
            import rag_ingest
            rag_ingest.build(chunks_live, rebuild=True)
            print("\nRestored your original chunks.jsonl and index. "
                  "Re-run with --apply to keep the winner.")

        return ranked
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def write_report(ranked, k, rerank):
    lines = [
        "# Chunk size / overlap sweep", "",
        f"- Configurations tested: **{len(ranked)}**",
        f"- Ranking signal: recall@{k}, then MRR",
        f"- Reranking: {'on' if rerank else 'off'}", "",
        "Ranked best first.", "",
        f"| chunk_words | overlap | chunks | recall@1 | recall@3 | recall@{k} | MRR | gap |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in ranked:
        gap = f"{r['gap']:+.3f}" if r.get("gap") is not None else "n/a"
        lines.append(
            f"| {r['chunk_words']} | {r['overlap']} | {r['n_chunks']} | "
            f"{r['recall@1']:.3f} | {r['recall@3']:.3f} | {r[f'recall@{k}']:.3f} | "
            f"{r['mrr']:.3f} | {gap} |")

    best = ranked[0]
    lines += ["", "## Winner", "",
              f"**chunk_words={best['chunk_words']}, overlap={best['overlap']}** "
              f"— recall@{k} {best[f'recall@{k}']:.3f}, MRR {best['mrr']:.3f}", ""]
    if best["misses"]:
        lines.append(f"Still missing: {', '.join(best['misses'])}")
        lines.append("")
    lines += [
        "## Reading this", "",
        "Small differences are not real. On a corpus of this size, a recall gap "
        "under about 0.05 is one or two questions changing rank, which is noise. "
        "Prefer the simpler configuration when results are close.", "",
        "If the same setting wins on both recall and MRR, that is a genuine signal. "
        "If recall is flat and only MRR moves, ordering improved but reachability "
        "did not, which matters less when the LLM sees all top-k chunks anyway.", "",
        "Re-run this after the corpus grows. The optimum shifts with corpus size "
        "and with how many chunks compete for each query.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport -> {REPORT_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sweep chunk size and overlap")
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--overlaps", type=int, nargs="+", default=DEFAULT_OVERLAPS)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--rerank", action="store_true",
                    help="Evaluate with cross-encoder reranking (slower)")
    ap.add_argument("--apply", action="store_true",
                    help="Keep the winning configuration instead of restoring")
    a = ap.parse_args()
    try:
        run(a.sizes, a.overlaps, a.k, a.rerank, a.apply)
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        sys.exit(2)
