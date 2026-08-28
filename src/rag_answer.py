"""
RAG answer generation — retrieved chunks -> professional, cited prose.

This is the "AG" in RAG. `rag_ingest.py` finds relevant chunks; this turns them
into an answer a coach or athlete would actually want to read.

    python src/rag_answer.py "what is launch angle and how do I improve it?"
    python src/rag_answer.py --interactive
    python src/rag_answer.py --golden          # run the whole golden set

── The pipeline for one question ────────────────────────────────────────────
    1. score = top cosine similarity for the question
    2. score < ABSTAIN_THRESHOLD  ->  refuse, no LLM call at all
    3. otherwise retrieve top-k chunks (hybrid)
    4. LLM writes prose under the grounding policy below
    5. validate_answer() checks citations + banned claims IN CODE
    6. failed validation -> one retry -> abstain

── Grounding policy: HYBRID ─────────────────────────────────────────────────
Standard baseball terminology may be defined from general knowledge, because
"launch angle is the vertical angle the ball leaves the bat" is a dictionary
fact, not a claim these coaches own. But every *coaching* claim — what to do,
what is correct, what to feel — must come from the transcripts and carry a
[Video @ timestamp] citation.

Definitions get a "(general definition)" marker so a reader can always tell
which sentences are grounded in the source videos and which are background.
That separation is the whole point: it keeps the product honest about what it
actually knows, and it keeps the coaches' instruction attributed to them.

── Setup ────────────────────────────────────────────────────────────────────
    pip install anthropic
    export ANTHROPIC_API_KEY="sk-ant-..."
"""
import os
import re
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAG_INDEX_DIR

# ── Tunables ─────────────────────────────────────────────────────────────────
#
# ABSTAIN_SIGNAL and ABSTAIN_THRESHOLD MUST come from the same rag_eval.py run.
# The two signals live on completely different scales:
#
#     "vector"  cosine similarity, range 0.0 .. 1.0    -> thresholds like 0.55
#     "rerank"  cross-encoder logits, range ~-12 .. +12 -> thresholds like -6.35
#
# Mixing them fails SILENTLY and in the worst direction. A rerank threshold of
# -6.35 checked against a cosine score of 0.65 always passes, so the system
# answers everything and never abstains. Nothing errors; you just lose the
# guardrail. _check_signal_scale() below catches that mismatch at startup.
#
# Set these from your own sweep:
#     python src/rag_eval.py --sweep                          -> use "vector"
#     python src/rag_eval.py --signal rerank --sweep           -> use "rerank"
# Take the best-accuracy threshold, then bias slightly toward abstaining
# (subtract ~0.03 for vector, ~0.5 for rerank).
ABSTAIN_SIGNAL = "vector"          # "vector" | "rerank"
ABSTAIN_THRESHOLD = 0.42


def _check_signal_scale(signal: str, threshold: float) -> None:
    """Fail loudly on an obvious signal/threshold mismatch rather than silently."""
    if signal == "vector" and not (0.0 <= threshold <= 1.0):
        raise RuntimeError(
            f"ABSTAIN_SIGNAL='vector' expects a cosine threshold in 0.0-1.0, got "
            f"{threshold}.\nThat looks like a cross-encoder value. Either set "
            f"ABSTAIN_SIGNAL='rerank', or re-run:\n"
            f"    python src/rag_eval.py --sweep"
        )
    if signal == "rerank" and 0.0 <= threshold <= 1.0:
        print(f"  WARNING: ABSTAIN_SIGNAL='rerank' with threshold {threshold}, which "
              f"is in the cosine range.\n  Cross-encoder thresholds are usually "
              f"negative. Verify against: python src/rag_eval.py --signal rerank --sweep")

TOP_K = 5
MODEL = "claude-sonnet-5"
MAX_TOKENS = 900

CORPUS_SCOPE = ("hitting mechanics — swing plane, launch and attack angle, hip load "
                "and the gather, hand separation, lead-arm connection, contact point, "
                "and hitting specific pitch locations")

SYSTEM_PROMPT = f"""You answer baseball hitting questions for athletes and coaches, \
using transcript excerpts from Gradum Gswing instructional videos.

GROUNDING RULES — these are strict:

1. COACHING CLAIMS must come from the excerpts. Anything about what to do, what is
   correct, what to feel, or how to fix something requires a citation in the form
   [Video Title @ 0:00]. Never invent coaching advice.

2. DEFINITIONS of standard baseball terms (launch angle, exit velocity, attack
   angle) may use general knowledge. Mark these "(general definition)". Keep them
   to one sentence.

3. If the excerpts do not answer the question, say so plainly. Do not fill gaps.

4. NEVER give medical, injury, or diagnostic advice. No claims about injury risk.

5. Coaching cues are these coaches' methods, not universal law. Write "Gradum
   teaches..." or "these coaches emphasize...", not "you must...".

STYLE:
- Write in clean prose. Do NOT stitch transcript fragments together verbatim —
  paraphrase the idea in your own words and cite where it came from.
- 2-4 short paragraphs. Lead with the direct answer.
- Plain language. Define jargon on first use.
- No greeting, no sign-off.

FORMAT:
- Prose answer with inline [Video Title @ timestamp] citations.
- Then a "Sources:" line listing each video and timestamp used, once each.
"""

USER_TEMPLATE = """Transcript excerpts:

{context}

---

Question: {question}

Answer using the rules given. Cite every coaching claim."""

BANNED_PATTERNS = [
    r"\binjur(y|ies|ed)\b", r"\bdiagnos(e|is|ed)\b", r"\bsee a doctor\b",
    r"\bmedical\b", r"\bphysical therap", r"\bguarantee[ds]?\b",
    r"\bwill definitely\b", r"\bcures?\b", r"\bpain\b",
]


# ── core ─────────────────────────────────────────────────────────────────────

def build_context(hits: list) -> str:
    return "\n\n".join(
        f"[{h['meta']['source_title']} @ {h['meta']['timestamp']}]\n{h['text']}"
        for h in hits
    )


def validate_answer(answer: str, hits: list) -> tuple[bool, str]:
    """
    Enforce the policy in code, not just in the prompt. A prompt is a request;
    this is the check that actually holds.
    """
    if not answer or len(answer.strip()) < 40:
        return False, "answer too short"

    # At least one citation matching a retrieved source.
    cited = False
    for h in hits:
        # strip the trailing "[youtube_id]" from titles before matching
        stem = re.sub(r"\s*\[[A-Za-z0-9_\-]+\]\s*$", "", h["meta"]["source_title"]).strip()
        if stem and stem[:22].lower() in answer.lower():
            cited = True
            break
    if not cited:
        return False, "no citation matching any retrieved source"

    for pat in BANNED_PATTERNS:
        m = re.search(pat, answer, re.IGNORECASE)
        if m:
            return False, f"banned claim: {m.group()!r}"

    return True, ""


def abstain_response(question: str, score: float) -> dict:
    return {
        "question": question,
        "answer": (
            "I don't have material covering that. These transcripts cover "
            f"{CORPUS_SCOPE}."
        ),
        "sources": [],
        "abstained": True,
        "score": score,
        "reason": "below_threshold",
    }


def answer_question(question: str, retriever=None, k: int = TOP_K,
                    threshold: float = ABSTAIN_THRESHOLD, verbose: bool = False,
                    signal: str = ABSTAIN_SIGNAL) -> dict:
    from rag_ingest import Retriever
    retriever = retriever or Retriever()

    _check_signal_scale(signal, threshold)

    # Same function rag_eval.py uses for the sweep, so the threshold you read out
    # of the eval report is directly comparable to the score checked here.
    score = (retriever.best_rerank_score(question) if signal == "rerank"
             else retriever.best_vector_score(question))
    if verbose:
        print(f"  {signal} score {score:+.3f} (threshold {threshold})")

    # Abstain BEFORE spending an API call — cheaper, and nothing to talk around.
    if score < threshold:
        return abstain_response(question, score)

    hits = retriever.search(question, k=k)
    context = build_context(hits)

    try:
        import anthropic
    except ImportError:
        raise RuntimeError("pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY in your environment")

    client = anthropic.Anthropic()

    last_reason = ""
    for attempt in (1, 2):
        msg = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": USER_TEMPLATE.format(context=context, question=question)
                + ("\n\nYour previous answer was rejected: " + last_reason +
                   ". Fix it." if attempt == 2 else "")
            }],
        )
        text = "".join(b.text for b in msg.content if b.type == "text").strip()

        ok, reason = validate_answer(text, hits)
        if ok:
            return {
                "question": question,
                "answer": text,
                "sources": [{
                    "title": h["meta"]["source_title"],
                    "timestamp": h["meta"]["timestamp"],
                    "chunk_id": h["id"],
                } for h in hits],
                "abstained": False,
                "score": score,
                "attempts": attempt,
            }
        last_reason = reason
        if verbose:
            print(f"  attempt {attempt} rejected: {reason}")

    # Two failures -> abstain rather than ship something unvalidated.
    out = abstain_response(question, score)
    out["reason"] = f"validation_failed: {last_reason}"
    out["answer"] = ("I found related material but couldn't produce a properly "
                     "sourced answer. Try rephrasing the question.")
    return out


def print_answer(res: dict):
    print()
    if res["abstained"]:
        print(f"[ABSTAINED — similarity {res['score']:.3f}]  {res.get('reason','')}")
        print(res["answer"])
        return
    print(res["answer"])
    if res["sources"]:
        seen, lines = set(), []
        for s in res["sources"]:
            key = (s["title"], s["timestamp"])
            if key not in seen:
                seen.add(key)
                lines.append(f"  - {s['title']} @ {s['timestamp']}")
        print("\nRetrieved from:")
        print("\n".join(lines))
    print(f"\n[similarity {res['score']:.3f}, {res.get('attempts',1)} attempt(s)]")


def run_golden(threshold: float, k: int, signal: str = ABSTAIN_SIGNAL):
    """Run the golden set end to end: in-coverage should answer, OOC should abstain."""
    from rag_eval import load_golden
    from rag_ingest import Retriever

    ic, ooc = load_golden()
    r = Retriever()
    results, correct = [], 0

    print(f"\n{'='*72}\nIN-COVERAGE ({len(ic)}) — should answer\n{'='*72}")
    for q in ic:
        res = answer_question(q["q"], r, k, threshold, signal=signal)
        ok = not res["abstained"]
        correct += ok
        print(f"  {'OK  ' if ok else 'MISS'} [{res['score']:.3f}] {q['id']}  {q['q'][:52]}")
        results.append({**res, "id": q["id"], "expected": "answer"})

    print(f"\n{'='*72}\nOUT-OF-COVERAGE ({len(ooc)}) — should abstain\n{'='*72}")
    for q in ooc:
        res = answer_question(q["q"], r, k, threshold, signal=signal)
        ok = res["abstained"]
        correct += ok
        print(f"  {'OK  ' if ok else 'LEAK'} [{res['score']:.3f}] {q['id']}  {q['q'][:52]}")
        results.append({**res, "id": q["id"], "expected": "abstain"})

    total = len(ic) + len(ooc)
    print(f"\nCorrect behaviour: {correct}/{total} ({correct/total:.1%})")

    out = RAG_INDEX_DIR / "answer_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Answer baseball questions from the transcripts")
    ap.add_argument("question", nargs="*", help="Question to answer")
    ap.add_argument("-k", type=int, default=TOP_K)
    ap.add_argument("--threshold", type=float, default=ABSTAIN_THRESHOLD)
    ap.add_argument("--signal", choices=["vector", "rerank"], default=ABSTAIN_SIGNAL,
                    help="Score used for the abstention check. MUST match the "
                         "rag_eval.py run the threshold came from.")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--golden", action="store_true", help="Run the full golden set")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    try:
        if a.golden:
            run_golden(a.threshold, a.k, a.signal)
        elif a.interactive:
            from rag_ingest import Retriever
            r = Retriever()
            print("Ask a baseball hitting question (Ctrl-C to quit)\n")
            while True:
                try:
                    q = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if q:
                    print_answer(answer_question(q, r, a.k, a.threshold, a.verbose, a.signal))
                    print()
        elif a.question:
            print_answer(answer_question(" ".join(a.question), None, a.k,
                                         a.threshold, a.verbose, a.signal))
        else:
            ap.print_help()
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        sys.exit(2)
