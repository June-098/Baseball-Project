# RAG Build Plan — Baseball Q\&A Assistant

**14 August 2026 · Corpus: 17 videos → 874 sentences → 83 chunks**

Build order is deliberate: **eval → retrieval → abstention → generation.** Most projects invert it and spend weeks tuning prompts around retrieval that was broken from the start. You cannot tell whether a prompt change helped if you have no number that moves.

---

## Phase 0 — Setup (30 min)

cd "Personal Baseball Project"

source .venv/bin/activate

pip install chromadb sentence-transformers rank-bm25

- [x] Install deps  
- [x] `python src/rag_ingest.py` — builds ChromaDB \+ BM25 (\~1 min, downloads a 130 MB model once)  
- [x] Smoke test: `python src/rag_ingest.py --query "how do I hit a curveball"`  
- [x] Confirm the curve ball video ranks \#1. If not, stop and debug ingestion before continuing.

---

## Phase 1 — Eval harness (already built; 2 hrs of your input)

The harness exists. What it needs is *your* judgment about what a correct answer looks like.

- [x] Run the baseline: `python src/rag_eval.py --compare-hybrid --sweep`  
- [ ] Read `Baseball Resources/rag_index/eval_report.md`  
- [ ] Expand `golden_questions.jsonl` from 20 → 30 in-coverage questions  
- [ ] Expand out-of-coverage from 10 → 15  
- [ ] Record the baseline numbers below — every later change is measured against these

| Metric       | Baseline     | Target      |
| :----------- | :----------- | :---------- |
| recall@1     | 0.8\_\_\_    | \> 0.60     |
| recall@5     | 0.9\_\_\_    | **\> 0.85** |
| MRR          | 0.829\_\_\_  | \> 0.70     |
| Coverage gap | -9.123\_\_\_ | **\> 0.10** |

**How to write a good golden question.** Phrase it the way an athlete would, not the way the coach does. `"Why do I keep popping up?"` is a better test than `"What is launch angle?"` because the first exposes the vocabulary gap and the second cannot fail.

{"id": "IC-21", "type": "in\_coverage",

 "q": "My hands keep drifting away from my body. How do I fix that?",

 "expect\_sources": \["Connection Lead Arm", "How to Separate your Hands"\],

 "note": "athlete symptom phrasing, no matching video title"}

**Reading the report.** `recall@5` is the headline. `MRR` rising while recall stays flat means ordering improved — still a win. The **coverage gap** (in-coverage minimum minus out-of-coverage maximum) decides whether abstention is even possible: if it's negative, the two distributions overlap and no threshold can separate them. Fix retrieval first.

---

## Phase 2 — Retrieval tuning (3–4 hrs)

Change one variable at a time and re-run the harness after each. That's the whole method.

- [ ] **Hybrid vs vector-only.** `--compare-hybrid`. Expect hybrid to win on named entities (IC-08 Ferris Wheel, IC-09 Mookie Betts, IC-13 Money Gap). If it doesn't, BM25 is misconfigured.  
        
- [ ] **Contextual prefixes on/off.** Already on. Comment out the prefix in `contextual_text()`, rebuild, re-run. Expect a measurable drop — that's the value of the feature, and worth confirming rather than assuming.  
        
- [ ] **Chunk size.** Regenerate at 120 and 400 words, re-ingest, compare: `bash python src/transcript_trimming.py --chunk-words 120 --overlap 25 python src/rag_ingest.py --rebuild && python src/rag_eval.py`   
        
- [ ] **Embedding model.** Swap `EMBED_MODEL` in `rag_ingest.py`: `BAAI/bge-small-en-v1.5` (current) → `BAAI/bge-base-en-v1.5` (larger, slower) → `all-MiniLM-L6-v2` (baseline). Watch **IC-04** specifically — attack angle vs launch angle is where weak models collapse two distinct concepts.  
        
- [ ] **Query expansion**, only if the symptom-phrased questions (IC-14, IC-15) still miss:  
        
      \`\`\`python  
        
      SYNONYMS \= {  
        
          "popping up":   "under the ball, swing plane, launch angle",  
        
          "rolling over": "topping the ball, hands, connection",  
        
          "chase":        "plate discipline, outside pitch",  
        
      }  
        
      def expand(q):  
        
          extra \= \[v for k, v in SYNONYMS.items() if k in q.lower()\]  
        
          return q \+ (" " \+ " ".join(extra) if extra else "")  
        
      \`\`\`  
        
      This is a \*\*pre-retrieval\*\* step. It is not prompt engineering and not a guardrail.

**Stop when recall@5 \> 0.85.** Chasing 0.95 on 83 chunks is overfitting to a corpus you're about to grow.

---

## Phase 3 — Abstention (1–2 hrs)

- [ ] Run `python src/rag_eval.py --sweep`  
- [ ] Read the suggested threshold and the sweep table  
- [ ] Pick the threshold with the best accuracy, then **subtract \~0.03** — bias toward abstaining. A wrong answer costs trust; an abstention costs nothing.  
- [ ] Hard-code it and test all 10 out-of-coverage questions return the refusal  
- [ ] Add 5 adversarial near-misses: questions using heavy baseball vocabulary about uncovered topics (OOC-10 "how much does a bat cost" is the template — the corpus says "bat" constantly, so word overlap alone must not trigger an answer)

ABSTAIN\_THRESHOLD \= 0.42   \# set from your own sweep, not copied

def answer(question, retriever):

    score \= retriever.best\_vector\_score(question)

    if score \< ABSTAIN\_THRESHOLD:

        return {"answer": "I don't have material covering that. These transcripts "

                          "cover hitting mechanics — swing plane, launch and attack "

                          "angle, hip load, hand separation, and pitch location.",

                "sources": \[\], "abstained": True, "score": score}

    hits \= retriever.search(question, k=5)

    return generate(question, hits)

Note it abstains **before** calling the LLM — cheaper and impossible to talk around.

---

## Phase 4 — Generation (2–3 hrs)

- [ ] Write the system prompt (below)  
- [ ] Wire Claude API first — do not debug retrieval and local inference simultaneously  
- [ ] Enforce citations in code, not just in the prompt  
- [ ] Run all 30 golden questions and read every answer by hand once  
- [ ] Only then swap in a local model

SYSTEM \= """You answer baseball hitting questions using ONLY the transcript

excerpts provided. These are from Gradum Gswing instructional videos.

Rules:

\- Use only the excerpts. Never add outside baseball knowledge.

\- Cite every claim as \[Video Title @ timestamp\].

\- If the excerpts don't answer the question, say so. Do not fill gaps.

\- Never give medical or injury advice.

\- Coaching cues are opinions of these coaches — attribute them, don't state

  them as universal fact.

"""

def generate(question, hits):

    ctx \= "\\n\\n".join(

        f"\[{h\['meta'\]\['source\_title'\]} @ {h\['meta'\]\['timestamp'\]}\]\\n{h\['text'\]}"

        for h in hits

    )

    \# ... call the model with SYSTEM \+ ctx \+ question ...

**Validate the output, don't trust it:**

def validate(answer, hits):

    titles \= {h\["meta"\]\["source\_title"\] for h in hits}

    if not any(t.split(" \[")\[0\]\[:20\] in answer for t in titles):

        return None          \# no citation \-\> reject, retry once, then abstain

    banned \= \["injury", "diagnose", "you should see a doctor", "guaranteed"\]

    if any(b in answer.lower() for b in banned):

        return None

    return answer

This mirrors the MEDA/APA/APE rule already in the roadmap: **the retrieval layer decides what is true, the language model only explains it.** Same boundary, different subsystem.

---

## Phase 5 — Scale (after 1–4 are green)

- [ ] Transcribe 50+ videos  
- [ ] Re-ingest, re-run the harness  
- [ ] Expect recall to *drop* — more chunks means more competition. That's normal and it's why the baseline numbers matter  
- [ ] Re-sweep the abstention threshold (coverage widened, so the gap moves)  
- [ ] Add new golden questions for newly covered topics

---

## Definition of done

| Gate | Target |
| :---- | :---- |
| recall@5 on golden set | \> 0.85 |
| Out-of-coverage abstention | 10/10 |
| Answers carrying a citation | 100% |
| Banned-claim violations | 0 |
| Adversarial near-misses handled | 5/5 |

---

## Timebox

| Phase | Effort |
| :---- | :---- |
| 0 Setup | 0.5 hr |
| 1 Eval harness | 2 hr (mostly writing questions) |
| 2 Retrieval tuning | 3–4 hr |
| 3 Abstention | 1–2 hr |
| 4 Generation | 2–3 hr |
| **Total to working prototype** | **\~10 hr** |

One focused weekend. The corpus is small enough that every experiment runs in seconds — which is exactly why tuning now, before 50+ videos, is the cheap moment to do it.  
