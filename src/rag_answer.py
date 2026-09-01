"""
RAG answer generation — retrieved chunks -> professional, cited prose.

This is the "AG" in RAG. `rag_ingest.py` finds relevant chunks; this turns them
into an answer a coach or athlete would actually want to read.

    python src/rag_answer.py "what is launch angle and how do I improve it?"
    python src/rag_answer.py --interactive
    python src/rag_answer.py --golden          # run the whole golden set

── The pipeline for one question ────────────────────────────────────────────
    1. score = top cosine similarity for the question
    2. retrieve top-k unless this is a swing-report or off-topic question
    3. local Gemma writes the answer (report first, then notes, then generic)
    4. validate_answer() checks banned claims and strips drills from refusals
    5. failed validation -> one retry -> abstain only on safety failure

── Grounding policy: NOTES FIRST, THEN GENERIC COACHING ─────────────────────
Prefer the retrieved batting notes for hitting how-tos. If they include a
drill AND the athlete asked how to hit/practice/fix, reproduce Setup and
Execution as numbered lists. If the question is off-topic or about a
SwingLens report, do not attach a drill. Medical/injury claims are banned.

── Setup ────────────────────────────────────────────────────────────────────
    ollama serve                  # if it is not already running
    ollama list                   # confirm gemma4:e4b is installed
    Optional .env overrides: OLLAMA_MODEL, OLLAMA_HOST
"""
import os
import re
import sys
import json
import argparse
import urllib.error
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAG_INDEX_DIR

load_dotenv(PROJECT_ROOT / ".env")

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
# 0.0 = never refuse for low cosine. Out-of-coverage still gets a generic answer.
# Raise this again if you want the old abstain-before-Gemma gate.
ABSTAIN_THRESHOLD = 0.0
TOP_K = 5
# MODEL = "claude-sonnet-5"   # previous Anthropic generator
MODEL = "gemma4:e4b"
MAX_TOKENS = 2200


# Anthropic cloud generation — kept for a one-file revert. Not called.
# def _anthropic_api_key() -> str:
#     return os.environ.get("ANTHROPIC_API_KEY", "").strip()
#
# def _anthropic_workspace_id() -> str:
#     return os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
#
# def _anthropic_client():
#     import anthropic
#     key = _anthropic_api_key()
#     if not key:
#         raise RuntimeError(
#             "ANTHROPIC_API_KEY is missing. Copy .env.example to .env and add your key."
#         )
#     workspace = _anthropic_workspace_id()
#     headers = {"anthropic-workspace-id": workspace} if workspace else None
#     return anthropic.Anthropic(api_key=key, default_headers=headers)


def _ollama_host() -> str:
    # when _ollama_host() is called, it returns the value of the OLLAMA_HOST environment variable, or the default value of "http://127.0.0.1:11434" if the environment variable is not set.
    return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def _ollama_model() -> str:
    # when _ollama_model() is called, it returns the value of the OLLAMA_MODEL environment variable, or the default value of "gemma4:e4b" if the environment variable is not set.
    return os.environ.get("OLLAMA_MODEL", MODEL).strip() or MODEL


def _ollama_chat(messages: list, max_tokens: int = MAX_TOKENS) -> str:
    """One non-streaming chat turn against the local Ollama daemon."""
    payload = {
        "model": _ollama_model(),
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.2,
        },
    }
    req = urllib.request.Request(
        f"{_ollama_host()}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404:
            raise RuntimeError(
                f"Ollama does not have model {_ollama_model()!r}. "
                "Run `ollama list` and `ollama pull gemma4:e4b`."
            ) from exc
        raise RuntimeError(f"Ollama HTTP {exc.code}: {body[:400]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {_ollama_host()}. Start it with: ollama serve"
        ) from exc

    text = ((data.get("message") or {}).get("content") or "").strip()
    if not text:
        raise RuntimeError(f"Ollama returned an empty reply from {_ollama_model()}")
    return text


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


CORPUS_SCOPE = (
    "batting mechanics only: athletic stance and the power position, hip load "
    "and the gather, trigger and timing, the kinetic chain (legs to torso to "
    "hands), hip-shoulder separation, hand separation, lead-arm connection, "
    "top-hand barrel control, bat path and swing plane, attack angle, launch "
    "angle, contact point, barrel speed with control, direction through the "
    "ball and follow-through (delayed rollover / money gap / direction), hitting pitch "
    "locations (inside, outside, high, breaking balls), and contact quality "
    "(square contact, line drives, fly balls, ground balls, pop-ups, and "
    "extra-base hits including home runs)"
) # CORPUS_SCOPE is a string that describes the scope of the corpus that the model is trained on.

SYSTEM_PROMPT = f"""Imagine yourself as a professional baseball hitting coach.
You answer baseball hitting questions, and questions about this athlete's
SwingLens swing report when one is attached.

GROUNDING RULES:

1. SWING REPORT. If a SwingLens report is in the user message, treat it as
   ground truth for THIS clip. Questions about the prototype score,
   checkpoints, lead-knee, torso lean, shoulder-hip gap, head travel, swing
   path, or why a number appeared ARE in scope. Explain the number from the
   report (and the scoring recipe in the report). Do not use batting-note
   drills on those questions.

2. PREFER THE EXCERPTS when they cover a hitting question that is not about
   the swing report. Paraphrase; do not dump the notes verbatim.

3. DRILLS only when they asked how to hit a pitch, how to practice, how to
   fix a mechanical issue, or what a named drill is. Then:
   - If the excerpts name a drill, use that drill. Copy its Setup and
     Execution as numbered lists. If the notes give a video URL and timestamp,
     include them as: For a reference, watch this video (URL) at timestamp X:XX
     if you are confused with the writings. If the excerpts contain an http
     or https URL, you must paste that exact URL (do not drop it).
   - Do not invent YouTube links. Only use URLs that appear in the excerpts.
   - If the excerpts have no drill, write a practical tee or front-toss drill
     in the same Setup / Execution numbered format.
   Use this shape:

   ### {{Drill name}}
   One sentence on what the hitter should feel.

   ### Setup
   1. ...
   2. ...

   ### Execution
   1. ...
   2. ...

   NEVER attach a drill to a refusal, a score explanation, or an off-topic
   reply. If you are refusing, stop after 1-2 sentences.

4. REFUSALS. If the question is not hitting mechanics and not about this
   report, reply in 1-2 sentences that you only cover hitting and this swing
   report. Then STOP. No drill, no practice plan, no extra coaching.

5. If the excerpts do not answer a hitting question, still give a complete
   hitting answer from general coaching — unless the question is off-topic,
   in which case use the refusal rule. Do not say "the notes do not contain..."
   as the whole reply.

6. NEVER give medical, injury, or diagnostic advice. No claims about injury risk.

7. Do not put attribution in the answer. No "(general coaching)", no
   "(general definition)", no "Gradum Gswing", no "these notes teach...", no
   "Source:", no source names, no [Video | Section @ 0:00], and no Sources list.

STYLE:
- Hitting how-to: 1-2 short paragraphs, then the drill block. Add a short
  Common mistakes list when the notes have one.
- Report questions: 1-3 short paragraphs that use the report numbers. No drill.
- Refusals: 1-2 sentences only.
- Plain language. Define jargon on first use.
- No greeting, no sign-off.
- Follow-ups: stay consistent with earlier coach replies unless the athlete
  changes topic. Do not repeat the full drill unless they ask.
"""

USER_TEMPLATE = """{history}{analysis}Retrieved batting notes (ignore them if they are empty or off-topic, and ignore them entirely for swing-report or off-topic questions):

{context}

---

Question: {question}

Answer using the rules given. Include a full Setup/Execution drill ONLY when
they asked how to hit, how to practice, how to fix a swing, or what a named
drill is. If they asked about this report, or the question is off-topic, do
not include a drill. If the notes contain a video URL and a drill is required,
paste that exact URL and timestamp into the drill block. Do not include
(general coaching), Gradum, citations, source names, or a source list."""

BANNED_PATTERNS = [
    r"\binjur(y|ies|ed)\b", r"\bdiagnos(e|is|ed)\b", r"\bsee a doctor\b",
    r"\bmedical\b", r"\bphysical therap", r"\bguarantee[ds]?\b",
    r"\bwill definitely\b", r"\bcures?\b", r"\bpain\b",
]
# Bracket cites the model used to paste into prose, e.g. [Attack Angle | ... @ 0:00]
_INLINE_CITE_RE = re.compile(r"\s*\[[^\[\]]*(?:@|\|)[^\[\]]*\]")
_SOURCES_BLOCK_RE = re.compile(
    r"\n+\s*(?:Sources|Retrieved from)\s*:\s*\n.*\Z", re.IGNORECASE | re.DOTALL
)
_ATTRIBUTION_RE = re.compile(
    r"\s*\((?:general (?:coaching|definition)|Gradum Gswing)\)"
    r"|\s*Source:\s*Gradum Gswing\.?",
    re.IGNORECASE,
)
_REPORT_RE = re.compile(
    r"\b(prototype score|checkpoint score|lead[ -]?knee|torso lean|"
    r"shoulder[ –-]?hip|head travel|swing path(?: shape)?|"
    r"(?:this|my) (?:swing )?report|"
    r"why (?:did|do|was|is) (?:i|my|the).{0,80}(?:score|grade)|"
    r"how (?:is|was|do you|did you|does).{0,50}"
    r"(?:score|computed|calculated|measured))\b",
    re.IGNORECASE,
)
_OFF_TOPIC_RE = re.compile(
    r"\b(throw a slider|pitching velocity|field a ground|shortstop|"
    r"slide into|steal a base|frame pitches|as a catcher|"
    r"sacrifice bunt|arm care|elbow injury|eat before a game|"
    r"how much does a good baseball bat cost|nutrition)\b",
    re.IGNORECASE,
)
_REFUSAL_RE = re.compile(
    r"\b(outside (?:of )?the scope|out of (?:the )?scope|"
    r"i cannot (?:provide|answer)|i don'?t (?:cover|have material)|"
    r"only cover(?:s)? (?:baseball )?hitting)\b",
    re.IGNORECASE,
)
_DRILL_TITLE_RE = re.compile(
    r"^(?:#{1,3}\s+)?[\w][\w '’.-]*\bdrill\b|"
    r"^#{1,3}\s+(?:Setup|Execution)\b",
    re.IGNORECASE | re.MULTILINE,
)
_NAMED_DRILL_INLINE_RE = re.compile(
    r"(?i)\b(?:drop[ -]?bat|phone[ -]?booth|ferris[ -]?wheel)\s+drill\b",
)


# ── core ─────────────────────────────────────────────────────────────────────

def build_context(hits: list) -> str:
    # build_context() is a function that builds the context for the question.
    # packs retrieved parents as labeled excerpts so Gemma can use them, without putting those labels in the user-facing answer.
    return "\n\n".join(
        f"[{h['meta']['source_title']} | {h['meta']['timestamp']}]\n{h['text']}"
        for h in hits
    )


def strip_source_markup(text: str) -> str:
    """Remove cites, source lists, and leftover attribution tags."""
    text = _SOURCES_BLOCK_RE.sub("", text)
    text = _INLINE_CITE_RE.sub("", text)
    text = _ATTRIBUTION_RE.sub("", text)
    return re.sub(r"[ \t]+\n", "\n", text).strip()


def is_report_question(question: str) -> bool:
    return bool(_REPORT_RE.search(question or ""))


def is_off_topic(question: str) -> bool:
    """Pitching, fielding, medical, commerce — not hitting and not this report."""
    if is_report_question(question):
        return False
    return bool(_OFF_TOPIC_RE.search(question or ""))


def looks_like_drill(answer: str) -> bool:
    return bool(_DRILL_TITLE_RE.search(answer or ""))


def format_analysis(analysis: str | None) -> str:
    text = (analysis or "").strip()
    if not text:
        return ""
    return (
        "This athlete's current SwingLens report (2D on-device analysis of "
        "THEIR clip). Treat it as ground truth for questions about this swing:\n\n"
        f"{text}\n\n---\n\n"
    )


def skip_note_retrieval(question: str) -> bool:
    return is_report_question(question) or is_off_topic(question)


def validate_answer(answer: str, hits: list, question: str = "") -> tuple[bool, str]:
    """
    Enforce the policy in code, not just in the prompt. A prompt is a request;
    this is the check that actually holds.
    """
    if not answer or len(answer.strip()) < 20:
        return False, "answer too short"

    if _INLINE_CITE_RE.search(answer) or _SOURCES_BLOCK_RE.search("\n" + answer):
        return False, "answer still contains source citations; write prose only"

    for pat in BANNED_PATTERNS:
        m = re.search(pat, answer, re.IGNORECASE)
        if m:
            return False, f"banned claim: {m.group()!r}"

    drill_forbidden = (
        is_report_question(question)
        or is_off_topic(question)
        or bool(_REFUSAL_RE.search(answer))
    )
    if drill_forbidden and (
        looks_like_drill(answer) or _NAMED_DRILL_INLINE_RE.search(answer)
    ):
        return False, "do not include a drill on a refusal, off-topic, or report question"

    return True, ""


def format_history(history: list | None) -> str:
    """Prior turns for follow-up questions. Empty string on the first ask."""
    if not history:
        return ""
    parts = []
    for turn in history[-8:]:
        role = "Athlete" if turn.get("role") == "user" else "Coach"
        content = (turn.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}")
    if not parts:
        return ""
    return "Earlier in this conversation:\n\n" + "\n\n".join(parts) + "\n\n---\n\n"


def search_query(question: str, history: list | None) -> str:
    """Blend the latest follow-up with the previous athlete question for retrieval."""
    if not history:
        return question
    prior = [t.get("content", "").strip() for t in history if t.get("role") == "user"]
    prior = [p for p in prior if p]
    if not prior:
        return question
    return f"{prior[-1]} {question}".strip()


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
                    signal: str = ABSTAIN_SIGNAL, history: list | None = None,
                    analysis: str | None = None) -> dict:
    from rag_ingest import Retriever

    _check_signal_scale(signal, threshold)

    if skip_note_retrieval(question):
        hits = []
        score = 1.0
        if is_report_question(question) and (analysis or "").strip():
            context = (
                "(Answer from the SwingLens report attached above. "
                "Do not use batting-note drills.)"
            )
        elif is_report_question(question):
            context = (
                "(No swing report is attached. Tell them to analyze a clip "
                "first. One or two sentences. No drill.)"
            )
        else:
            context = (
                "(This question is outside hitting and this swing report. "
                "Refuse in 1-2 sentences. No drill.)"
            )
        if verbose:
            print(f"  skip notes ({'report' if is_report_question(question) else 'off-topic'})")
    else:
        retriever = retriever or Retriever()
        query = search_query(question, history)
        # Same function rag_eval.py uses for the sweep, so the threshold you read out
        # of the eval report is directly comparable to the score checked here.
        score = (retriever.best_rerank_score(query) if signal == "rerank"
                 else retriever.best_vector_score(query))
        if verbose:
            print(f"  {signal} score {score:+.3f} (threshold {threshold})")

        # Low cosine used to refuse here. OOC questions now still get a Gemma answer.
        if score < threshold:
            return abstain_response(question, score)

        hits = retriever.search(query, k=k, expand_parent=True)
        context = build_context(hits) if hits else "(none retrieved)"

    # Anthropic cloud generation (previous path):
    # try:
    #     import anthropic
    # except ImportError:
    #     raise RuntimeError("pip install anthropic")
    # client = _anthropic_client()
    # msg = client.messages.create(
    #     model="claude-sonnet-5",
    #     max_tokens=MAX_TOKENS,
    #     system=SYSTEM_PROMPT,
    #     messages=[{"role": "user", "content": USER_TEMPLATE.format(...)}],
    # )
    # text = strip_source_markup(
    #     "".join(b.text for b in msg.content if b.type == "text").strip()
    # )

    last_reason = ""
    for attempt in (1, 2):
        user_content = USER_TEMPLATE.format(
            history=format_history(history),
            analysis=format_analysis(analysis),
            context=context,
            question=question,
        )
        if attempt == 2:
            user_content += (
                "\n\nYour previous answer was rejected: " + last_reason + ". Fix it."
            )
        text = strip_source_markup(_ollama_chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]))

        ok, reason = validate_answer(text, hits, question)
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
    out["answer"] = ("I could not produce a safe answer for that question. "
                     "Try rephrasing, and avoid injury or medical topics.")
    return out


def print_answer(res: dict):
    print()
    if res["abstained"]:
        print(f"[ABSTAINED — similarity {res['score']:.3f}]  {res.get('reason','')}")
        print(res["answer"])
        return
    print(res["answer"])
    print(f"\n[similarity {res['score']:.3f}, {res.get('attempts',1)} attempt(s)]")


def run_golden(threshold: float, k: int, signal: str = ABSTAIN_SIGNAL):
    """Run the golden set: every question should get an answer (generic is OK)."""
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

    print(f"\n{'='*72}\nOUT-OF-COVERAGE ({len(ooc)}) — generic answer is OK\n{'='*72}")
    for q in ooc:
        res = answer_question(q["q"], r, k, threshold, signal=signal)
        ok = not res["abstained"]
        correct += ok
        print(f"  {'OK  ' if ok else 'MISS'} [{res['score']:.3f}] {q['id']}  {q['q'][:52]}")
        results.append({**res, "id": q["id"], "expected": "answer"})

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
    ap.add_argument("--golden", "--goldenset", action="store_true",
                    help="Run the full golden set (also --goldenset)")
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
