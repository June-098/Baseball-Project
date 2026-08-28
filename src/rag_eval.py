"""
RAG evaluation harness — measure retrieval before building the chatbot on top of it.

Runs the golden question set through the retriever and reports numbers that move
when you change something. Without this you are guessing whether a change helped.

    Input :  Baseball Resources/golden_questions.jsonl
    Output:  Baseball Resources/RAG Resources/rag_index/eval_report.md

── Usage ────────────────────────────────────────────────────────────────────
    python src/rag_eval.py                     # full report
    python src/rag_eval.py --compare-hybrid    # hybrid vs vector-only
    python src/rag_eval.py --sweep             # abstention threshold sweep
    python src/rag_eval.py --verbose           # show every failure

── What it measures ─────────────────────────────────────────────────────────

recall@k   Did an expected source appear in the top k? The headline number.
           If retrieval misses, no amount of prompt work saves the answer.

MRR        Mean Reciprocal Rank. recall@5 treats rank 1 and rank 5 as equal;
           MRR does not. Rising MRR at flat recall means better ordering.

Separation The gap between top similarity on in-coverage vs out-of-coverage
           questions. This gap IS the abstention threshold — if the two
           distributions overlap, no threshold can separate them and you need
           better retrieval before abstention can work at all.
"""
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import RESOURCES_DIR, RAG_INDEX_DIR

GOLDEN_PATH = RESOURCES_DIR / "golden_questions.jsonl"
REPORT_PATH = RAG_INDEX_DIR / "eval_report.md"


def load_golden(path: Path = None) -> tuple[list, list]:
    path = path or GOLDEN_PATH
    if not path.exists():
        raise RuntimeError(f"No golden set at {path}")
    rows = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "_comment" in r:
            continue
        rows.append(r)
    ic  = [r for r in rows if r.get("type") == "in_coverage"]
    ooc = [r for r in rows if r.get("type") == "out_of_coverage"]
    if not ic:
        raise RuntimeError("Golden set has no in_coverage questions")
    return ic, ooc


def hit_rank(results: list, expect_sources: list) -> int | None:
    """1-based rank of the first result whose source matches. None = miss."""
    for i, r in enumerate(results, 1):
        title = r["meta"]["source_title"]
        if any(exp.lower() in title.lower() for exp in expect_sources):
            return i
    return None


def evaluate(retriever, questions: list, k: int, use_hybrid: bool,
             rerank: bool = False) -> dict:
    ranks, rows = [], []
    for q in questions:
        res = retriever.search(q["q"], k=k, use_hybrid=use_hybrid, rerank=rerank)
        rank = hit_rank(res, q.get("expect_sources", []))
        ranks.append(rank)
        rows.append({
            "id": q["id"], "q": q["q"], "rank": rank,
            "top_score": retriever.best_vector_score(q["q"]),
            "top_source": res[0]["meta"]["source_title"] if res else "",
            "note": q.get("note", ""),
        })

    n = len(questions)
    found = [r for r in ranks if r is not None]
    return {
        "n": n,
        "recall@1": sum(1 for r in ranks if r == 1) / n,
        "recall@3": sum(1 for r in found if r <= 3) / n,
        f"recall@{k}": len(found) / n,
        "mrr": sum(1.0 / r for r in found) / n if n else 0.0,
        "misses": [r for r in rows if r["rank"] is None],
        "rows": rows,
    }


def coverage_separation(retriever, ic: list, ooc: list, signal: str = "vector") -> dict:
    """
    signal: "vector" = raw cosine, "rerank" = cross-encoder score.

    Cosine from a bi-encoder measures TOPICAL similarity, so every baseball
    question lands in a narrow 0.6-0.8 band whether or not the corpus answers it.
    That compression is why the in/out distributions overlap. The cross-encoder
    reads query and passage together and scores relevance, which spreads them out.
    """
    score_fn = (retriever.best_rerank_score if signal == "rerank"
                else retriever.best_vector_score)
    ic_scores  = [score_fn(q["q"]) for q in ic]
    ooc_scores = [score_fn(q["q"]) for q in ooc] if ooc else []
    out = {
        "signal": signal,
        "ic_min": min(ic_scores), "ic_mean": sum(ic_scores)/len(ic_scores),
        "ic_scores": ic_scores, "ooc_scores": ooc_scores,
        "ic_rows":  [{"id": q["id"], "q": q["q"], "score": s} for q, s in zip(ic, ic_scores)],
        "ooc_rows": [{"id": q["id"], "q": q["q"], "score": s} for q, s in zip(ooc, ooc_scores)],
    }
    if ooc_scores:
        out["ooc_max"]  = max(ooc_scores)
        out["ooc_mean"] = sum(ooc_scores)/len(ooc_scores)
        out["gap"]      = out["ic_min"] - out["ooc_max"]
        # A threshold is only meaningful if the two distributions don't overlap.
        out["suggested_threshold"] = (
            round((out["ic_min"] + out["ooc_max"]) / 2, 3) if out["gap"] > 0 else None
        )
    return out


def print_separation_detail(sep: dict):
    """Show which questions sit at the boundary — the negative-gap culprits."""
    print("\n  Lowest-scoring IN-COVERAGE (these set the floor):")
    for r in sorted(sep["ic_rows"], key=lambda x: x["score"])[:5]:
        print(f"    {r['score']:+.3f}  {r['id']}  {r['q'][:56]}")
    if sep.get("ooc_rows"):
        print("\n  Highest-scoring OUT-OF-COVERAGE (these break separation):")
        for r in sorted(sep["ooc_rows"], key=lambda x: -x["score"])[:5]:
            print(f"    {r['score']:+.3f}  {r['id']}  {r['q'][:56]}")


def sweep_thresholds(sep: dict) -> list:
    """
    For each candidate threshold, how many questions are handled correctly?

    A threshold is a dial between two error types:
      too LOW  -> answers everything, including questions with no coverage
      too HIGH -> abstains on questions the corpus genuinely answers

    Accuracy = (in-coverage answered + out-of-coverage abstained) / total, so the
    peak is the best available compromise. If the peak is well under 100%, the
    signal itself cannot separate the two classes and no threshold will fix it.

    The candidate range adapts to the signal: cosine lives in 0..1, cross-encoder
    scores are logits roughly in -12..+12.
    """
    ic, ooc = sep["ic_scores"], sep["ooc_scores"]
    all_scores = ic + ooc
    lo, hi = min(all_scores), max(all_scores)
    span = hi - lo or 1.0
    candidates = [round(lo + span * i / 24, 3) for i in range(25)]

    rows = []
    for t in candidates:
        answered  = sum(1 for s in ic if s >= t)              # correct: answer
        abstained = sum(1 for s in ooc if s < t)              # correct: abstain
        total = len(ic) + len(ooc)
        rows.append({
            "threshold": t,
            "ic_answered": answered, "ic_total": len(ic),
            "ooc_abstained": abstained, "ooc_total": len(ooc),
            "accuracy": (answered + abstained) / total if total else 0,
        })
    return rows


def inspect_question(qid: str, k: int = 5, rerank: bool = True):
    """
    Dump the chunks retrieved for one golden question, so you can read them and
    decide the thing no metric can decide for you: does this corpus actually
    answer the question?

    A "miss" has two very different causes, and they need opposite fixes:
      - retrieval failed  -> the right chunk exists but ranked too low  -> tune retrieval
      - corpus gap        -> nothing here answers it                    -> fix the LABEL
                                                                           or add content
    """
    from rag_ingest import Retriever

    ic, ooc = load_golden()
    q = next((x for x in ic + ooc if x["id"].lower() == qid.lower()), None)
    if q is None:
        raise RuntimeError(f"No golden question with id {qid!r}")

    r = Retriever()
    hits = r.search(q["q"], k=k, rerank=rerank)

    print(f"\n{'='*72}")
    print(f"{q['id']}  [{q['type']}]")
    print(f"Q: {q['q']}")
    if q.get("expect_sources"):
        print(f"Expected: {', '.join(q['expect_sources'])}")
    if q.get("note"):
        print(f"Note: {q['note']}")
    print(f"\nvector score : {r.best_vector_score(q['q']):+.3f}")
    try:
        print(f"rerank score : {r.best_rerank_score(q['q']):+.3f}   (>0 relevant, <0 not)")
    except Exception:
        pass
    print("=" * 72)

    for i, h in enumerate(hits, 1):
        title = h["meta"]["source_title"]
        match = ""
        if q.get("expect_sources"):
            match = ("  <-- EXPECTED SOURCE"
                     if any(e.lower() in title.lower() for e in q["expect_sources"])
                     else "")
        rs = f" rerank={h['rerank_score']:+.2f}" if "rerank_score" in h else ""
        print(f"\n{i}. {title} @ {h['meta']['timestamp']}{rs}{match}")
        print(f"   {h['text'][:600]}")
        if len(h["text"]) > 600:
            print("   [...]")

    print(f"\n{'='*72}")
    print("Read the chunks above and decide:")
    print("  - If they DO answer the question -> retrieval is fine; widen")
    print("    expect_sources in golden_questions.jsonl to include what came back.")
    print("  - If they DON'T -> the corpus has a gap. Move this question to")
    print("    out_of_coverage, or add videos that cover the topic.")


def run(k: int = 5, compare_hybrid: bool = False, sweep: bool = False,
        verbose: bool = False, rerank: bool = False, signal: str = "vector"):
    from rag_ingest import Retriever

    ic, ooc = load_golden()
    print(f"Golden set: {len(ic)} in-coverage, {len(ooc)} out-of-coverage\n")

    print("Loading retriever ...")
    r = Retriever()

    label = "hybrid + rerank" if rerank else "hybrid (vector + BM25)"
    print("\n" + "=" * 72)
    print(f"RETRIEVAL — {label}")
    print("=" * 72)
    hybrid = evaluate(r, ic, k, use_hybrid=True, rerank=rerank)
    for key in ["recall@1", "recall@3", f"recall@{k}", "mrr"]:
        print(f"  {key:12s} {hybrid[key]:.3f}")

    vector = None
    if compare_hybrid:
        print("\n" + "=" * 72)
        print("RETRIEVAL — vector only")
        print("=" * 72)
        vector = evaluate(r, ic, k, use_hybrid=False)
        for key in ["recall@1", "recall@3", f"recall@{k}", "mrr"]:
            delta = hybrid[key] - vector[key]
            arrow = "+" if delta > 0 else ""
            print(f"  {key:12s} {vector[key]:.3f}   ({label} {arrow}{delta:.3f})")
        if hybrid[f"recall@{k}"] < vector[f"recall@{k}"]:
            print("\n  WARNING: hybrid is WORSE than vector alone.")
            print("  Lower BM25_WEIGHT or raise BM25_MIN_SCORE in rag_ingest.py,")
            print("  or just run vector-only — on a small corpus BM25 is often noise.")

    if hybrid["misses"]:
        print(f"\n  MISSES ({len(hybrid['misses'])}):")
        for m in hybrid["misses"]:
            print(f"    {m['id']}  {m['q'][:58]}")
            print(f"        got: {m['top_source'][:56]} (sim {m['top_score']:.3f})")
            if m["note"]:
                print(f"        note: {m['note'][:70]}")

    if verbose:
        print("\n  ALL RESULTS:")
        for row in hybrid["rows"]:
            mark = "OK  " if row["rank"] else "MISS"
            rk = f"@{row['rank']}" if row["rank"] else " - "
            print(f"    {mark} {rk}  sim={row['top_score']:.3f}  {row['id']}  {row['q'][:46]}")

    print("\n" + "=" * 72)
    print(f"COVERAGE SEPARATION — signal: {signal}")
    print("=" * 72)
    sep = coverage_separation(r, ic, ooc, signal=signal)
    print(f"  in-coverage  min {sep['ic_min']:+.3f}   mean {sep['ic_mean']:+.3f}")
    if ooc:
        print(f"  out-of-cov   max {sep['ooc_max']:+.3f}   mean {sep['ooc_mean']:+.3f}")
        print(f"  gap          {sep['gap']:+.3f}")
        if sep["suggested_threshold"]:
            print(f"\n  SUGGESTED THRESHOLD: {sep['suggested_threshold']}")
            print("  (clean separation — a threshold here abstains correctly)")
        else:
            print("\n  WARNING: distributions OVERLAP. No threshold separates them.")
            if signal == "vector":
                print("  Try:  python src/rag_eval.py --signal rerank --sweep")
                print("  Cosine measures topic, not relevance — every baseball question")
                print("  scores high. The cross-encoder scores relevance and separates better.")
        print_separation_detail(sep)

    sweep_rows = []
    if sweep and ooc:
        print("\n" + "=" * 72)
        print("THRESHOLD SWEEP")
        print("=" * 72)
        print(f"  {'thresh':>7}  {'answered':>12}  {'abstained':>12}  {'accuracy':>9}")
        sweep_rows = sweep_thresholds(sep)
        best = max(sweep_rows, key=lambda x: x["accuracy"])
        for row in sweep_rows:
            star = " <-- best" if row is best else ""
            print(f"  {row['threshold']:>7.2f}  "
                  f"{row['ic_answered']:>4}/{row['ic_total']:<7}  "
                  f"{row['ooc_abstained']:>4}/{row['ooc_total']:<7}  "
                  f"{row['accuracy']:>8.1%}{star}")

    # report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RAG evaluation report", "",
        f"- In-coverage questions: **{len(ic)}**",
        f"- Out-of-coverage questions: **{len(ooc)}**", "",
        "## Retrieval (hybrid)", "", "| Metric | Value |", "|---|---|",
    ]
    for key in ["recall@1", "recall@3", f"recall@{k}", "mrr"]:
        lines.append(f"| {key} | {hybrid[key]:.3f} |")
    if vector:
        lines += ["", "## Hybrid vs vector-only", "",
                  "| Metric | Vector | Hybrid | Delta |", "|---|---|---|---|"]
        for key in ["recall@1", "recall@3", f"recall@{k}", "mrr"]:
            d = hybrid[key] - vector[key]
            lines.append(f"| {key} | {vector[key]:.3f} | {hybrid[key]:.3f} | {d:+.3f} |")
    if hybrid["misses"]:
        lines += ["", "## Misses", "", "| ID | Question | Top result | Sim |", "|---|---|---|---|"]
        for m in hybrid["misses"]:
            lines.append(f"| {m['id']} | {m['q']} | {m['top_source'][:44]} | {m['top_score']:.3f} |")
    lines += ["", f"## Coverage separation (signal: {sep['signal']})", "",
              f"- in-coverage min: **{sep['ic_min']:+.3f}**"]
    if ooc:
        lines += [f"- out-of-coverage max: **{sep['ooc_max']:+.3f}**",
                  f"- gap: **{sep['gap']:+.3f}**",
                  f"- suggested threshold: **{sep.get('suggested_threshold') or 'NONE — distributions overlap'}**"]

        # Per-question scores. Without these the negative gap is a number you
        # can't act on; with them you can see exactly which questions collide,
        # and whether the fault is retrieval or a mislabeled golden question.
        lines += ["", "### In-coverage, lowest first", "",
                  "| Score | ID | Question |", "|---|---|---|"]
        for r in sorted(sep["ic_rows"], key=lambda x: x["score"]):
            lines.append(f"| {r['score']:+.3f} | {r['id']} | {r['q']} |")

        lines += ["", "### Out-of-coverage, highest first", "",
                  "| Score | ID | Question |", "|---|---|---|"]
        for r in sorted(sep["ooc_rows"], key=lambda x: -x["score"]):
            lines.append(f"| {r['score']:+.3f} | {r['id']} | {r['q']} |")

        # Name the specific collisions.
        overlap_ic  = [r for r in sep["ic_rows"]  if r["score"] < sep["ooc_max"]]
        overlap_ooc = [r for r in sep["ooc_rows"] if r["score"] > sep["ic_min"]]
        if overlap_ic or overlap_ooc:
            lines += ["", "### Overlap — the questions blocking a clean threshold", ""]
            if overlap_ic:
                lines.append("**In-coverage scoring below the OOC maximum** — either "
                             "retrieval is failing them, or the corpus genuinely doesn't "
                             "answer them and they are mislabeled:")
                lines.append("")
                for r in sorted(overlap_ic, key=lambda x: x["score"]):
                    lines.append(f"- `{r['score']:+.3f}` **{r['id']}** — {r['q']}")
                lines.append("")
            if overlap_ooc:
                lines.append("**Out-of-coverage scoring above the IC minimum** — these are "
                             "topically close enough that the corpus looks relevant:")
                lines.append("")
                for r in sorted(overlap_ooc, key=lambda x: -x["score"]):
                    lines.append(f"- `{r['score']:+.3f}` **{r['id']}** — {r['q']}")
                lines.append("")
            lines.append("Inspect any of these with: "
                         "`python src/rag_eval.py --inspect <ID>`")
    if sweep_rows:
        lines += ["", "## Threshold sweep", "",
                  "| Threshold | Answered (IC) | Abstained (OOC) | Accuracy |", "|---|---|---|---|"]
        for row in sweep_rows:
            lines.append(f"| {row['threshold']:.2f} | {row['ic_answered']}/{row['ic_total']} "
                         f"| {row['ooc_abstained']}/{row['ooc_total']} | {row['accuracy']:.1%} |")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport -> {REPORT_PATH}")
    return hybrid, sep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate RAG retrieval")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--compare-hybrid", action="store_true", help="Also run vector-only")
    ap.add_argument("--sweep", action="store_true", help="Sweep abstention thresholds")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--rerank", action="store_true",
                    help="Cross-encoder rerank the retrieval results")
    ap.add_argument("--signal", choices=["vector", "rerank"], default="vector",
                    help="Score used for coverage separation / threshold (default vector)")
    ap.add_argument("--inspect", metavar="ID", default=None,
                    help="Dump retrieved chunks for one golden question (e.g. IC-15)")
    a = ap.parse_args()
    try:
        if a.inspect:
            inspect_question(a.inspect, a.k, rerank=True)
        else:
            run(a.k, a.compare_hybrid, a.sweep, a.verbose, a.rerank, a.signal)
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        sys.exit(2)
