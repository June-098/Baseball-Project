"""
RAG ingestion — parent-child chunks -> local ChromaDB + BM25 index.

This file is the retrieval layer. Run it *before* rag_eval.py or rag_answer.py.
Those scripts import Retriever from here; they do not rebuild the index.

    python src/rag_ingest.py --rebuild
    python src/rag_eval.py
    python src/rag_answer.py "what is launch angle?"

    Input :  Baseball Resources/RAG Resources/*.md
    Output:  Baseball Resources/RAG Resources/rag_index/chroma/
             Baseball Resources/RAG Resources/rag_index/bm25.pkl
             Baseball Resources/RAG Resources/rag_index/chunks.jsonl

Full narrative (why prefixes, hybrid, rerank, parent-child): CODE_OVERVIEW.md
under "`rag_ingest.py` — indexing and retrieval".

── Setup ────────────────────────────────────────────────────────────────────
    pip install chromadb sentence-transformers rank-bm25 transformers

── Usage ────────────────────────────────────────────────────────────────────
    python src/rag_ingest.py                      # build the index
    python src/rag_ingest.py --rebuild            # wipe and rebuild
    python src/rag_ingest.py --query "curveball"  # smoke-test (parents, not an answer)

── What actually gets stored ────────────────────────────────────────────────

1. PARENT-CHILD. rag_chunk.py cuts Markdown structurally. Children (~120–220
   embedding-model tokens) are embedded and searched. rag_answer expands a hit
   to parent_text (~300–600 tokens) for the LLM. Eval ranks children.

2. EMBEDDING STRING. Not the raw paragraph. Hierarchy header plus body:

       Document: Contact Point
       Section: Step-by-Step Explanation > Inside pitch
       Content type: hitting mechanics instruction

       The hitter turns more tightly...

   That replaced the old [Source: Title] … [Topic: Title] wrap. Same job: mid-note
   chunks say "it"; the header puts the topic in the mean-pooled vector.

3. HYBRID SEARCH. Vector (bge-small-en-v1.5, 384-d, cosine) + BM25, fused with
   weighted RRF (vector 1.0, BM25 0.35, BM25 floor 2.0). Equal-weight fusion
   used to lose to vector-only on this corpus. Optional cross-encoder: --rerank.
"""
import sys
import json
import pickle
import shutil
import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import RAG_RESOURCES_DIR, RAG_INDEX_DIR

RAG_DIR    = RAG_INDEX_DIR
CHROMA_DIR = RAG_DIR / "chroma"
BM25_PATH  = RAG_DIR / "bm25.pkl"
CHUNKS_PATH = RAG_DIR / "chunks.jsonl"
COLLECTION = "baseball_transcripts"

# bge-small-en-v1.5: 384-dim, ~130MB, consistently strong on short-passage
# retrieval and cheap enough to re-embed the whole corpus in seconds.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Cross-encoder for reranking. ~80MB, reads query+document together rather than
# embedding them separately, so it scores relevance far more sharply than cosine.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Hybrid fusion weights. Vector dominates because on this corpus BM25 is close to
# noise for conceptual queries — see the comment in Retriever.search().
VEC_WEIGHT  = 1.0
BM25_WEIGHT = 0.35
# BM25 hits scoring below this are dropped rather than fused. BM25 always returns
# its top-k even when nothing genuinely matched; this filters that out.
BM25_MIN_SCORE = 2.0


def _check_deps():
    missing = []
    for mod, pkg in [("chromadb", "chromadb"),
                     ("sentence_transformers", "sentence-transformers"),
                     ("rank_bm25", "rank-bm25"),
                     ("transformers", "transformers")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise RuntimeError("Missing packages:\n  pip install " + " ".join(missing))


def list_rag_markdown(directory: Path = None) -> list:
    """Top-level .md notes in RAG Resources. Skips _underscored drafts and the index folder."""
    d = Path(directory) if directory is not None else RAG_RESOURCES_DIR
    if not d.exists():
        raise RuntimeError(f"RAG Resources folder not found: {d}")
    files = sorted(
        p for p in d.glob("*.md")
        if p.is_file() and not p.name.startswith("_")
    )
    if not files:
        raise RuntimeError(f"No .md files in {d}")
    return files


def chunks_from_markdown(path: Path, **_ignored) -> list:
    """Parent-child children for one note. Extra kwargs ignored (old RCTS API)."""
    from rag_chunk import children_from_markdown
    return children_from_markdown(path)


def load_jsonl(path: Path) -> list:
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    if not rows:
        raise RuntimeError(f"{path} is empty")
    return rows


def write_chunks_jsonl(chunks: list, path: Path = None) -> Path:
    path = path or CHUNKS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return path


def load_chunks(path: Path = None) -> list:
    """
    Default: chunk every .md note in RAG Resources.
    Pass a .jsonl path (or --chunks) to load a prebuilt file instead.
    """
    if path is not None:
        if not path.exists():
            raise RuntimeError(f"No chunks at {path}")
        return load_jsonl(path)

    files = list_rag_markdown()
    chunks = []
    for p in files:
        rows = chunks_from_markdown(p)
        n_parents = len({c.get("parent_id") for c in rows})
        print(f"  {p.stem[:52]:52s} {len(rows):3d} children / {n_parents:3d} parents")
        chunks.extend(rows)
    if not chunks:
        raise RuntimeError(f"No embeddable text in {RAG_RESOURCES_DIR}")
    return chunks


def clean_title(source_title: str) -> str:
    """Strip the trailing '[youtube_id]' so titles read naturally."""
    return re.sub(r"\s*\[[A-Za-z0-9_\-]+\]\s*$", "", source_title).strip()


def contextual_text(chunk: dict) -> str:
    """What actually gets embedded — hierarchy prefix plus the child Markdown body."""
    if chunk.get("embedding_text"):
        return chunk["embedding_text"]
    title = clean_title(chunk["source_title"])
    return f"[Source: {title}] {chunk['text']} [Topic: {title}]"


def tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9']+", text.lower())


def build(chunks_path: Path = None, rebuild: bool = False):
    _check_deps()
    import chromadb
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi

    if chunks_path is None:
        print(f"Source: {RAG_RESOURCES_DIR}")
    chunks = load_chunks(chunks_path)
    write_chunks_jsonl(chunks)
    print(f"Loaded {len(chunks)} chunks  -> {CHUNKS_PATH}")

    if rebuild and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print("Wiped existing Chroma store")
    RAG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [contextual_text(c) for c in chunks]
    print("Embedding ...")
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    ).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    # cosine matches the normalized embeddings above
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    col.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],          # clean text for display
        metadatas=[{
            "source_title": c["source_title"],
            "source_file":  c.get("source_file", ""),
            "timestamp":    c.get("timestamp", ""),
            "timestamp_s":  int(c.get("timestamp_s", 0) or 0),
            "chunk_index":  int(c.get("chunk_index", 0) or 0),
            "parent_id":    c.get("parent_id", ""),
            "content_type": c.get("content_type", ""),
            "section_path": c.get("section_path", ""),
        } for c in chunks],
    )
    print(f"Chroma collection '{COLLECTION}' -> {col.count()} docs")

    # BM25 over the same contextual text so titles are keyword-searchable too
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": [c["id"] for c in chunks]}, f)
    print(f"BM25 index -> {BM25_PATH}")
    print(f"\nIndex ready at {RAG_DIR}")


# ── retrieval ────────────────────────────────────────────────────────────────

class Retriever:
    """Hybrid retriever. Load once, query many times."""

    def __init__(self):
        _check_deps()
        import chromadb
        from sentence_transformers import SentenceTransformer

        if not CHROMA_DIR.exists():
            raise RuntimeError("No index. Run: python src/rag_ingest.py")

        self.model = SentenceTransformer(EMBED_MODEL)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.col = self.client.get_collection(COLLECTION)
        self._ce = None          # cross-encoder, loaded lazily on first rerank
        self._by_id = {}
        if CHUNKS_PATH.exists():
            self._by_id = {c["id"]: c for c in load_jsonl(CHUNKS_PATH)}

        with open(BM25_PATH, "rb") as f:
            blob = pickle.load(f)
        self.bm25 = blob["bm25"]
        self.bm25_ids = blob["ids"]

    def vector_search(self, query: str, k: int = 10) -> list:
        # bge models expect this prefix on queries (not on documents)
        q = "Represent this sentence for searching relevant passages: " + query
        emb = self.model.encode([q], normalize_embeddings=True).tolist()
        res = self.col.query(query_embeddings=emb, n_results=k)
        out = []
        for i, cid in enumerate(res["ids"][0]):
            out.append({
                "id": cid,
                "text": res["documents"][0][i],
                "meta": res["metadatas"][0][i],
                # Chroma returns cosine DISTANCE; similarity = 1 - distance
                "score": 1.0 - res["distances"][0][i],
            })
        return out

    def keyword_search(self, query: str, k: int = 10) -> list:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [{"id": self.bm25_ids[i], "score": float(scores[i])} for i in ranked]

    def search(self, query: str, k: int = 5, rrf_k: int = 60,
               use_hybrid: bool = True, vec_weight: float = VEC_WEIGHT,
               bm25_weight: float = BM25_WEIGHT,
               bm25_floor: float = BM25_MIN_SCORE, rerank: bool = False,
               expand_parent: bool = False) -> list:
        """
        Weighted Reciprocal Rank Fusion.

        WHY WEIGHTED: the first version fused vector and BM25 rankings with equal
        weight, and measured WORSE than vector alone (recall@5 0.85 vs 0.95).
        On an 83-chunk corpus where every chunk says "ball", "swing" and "hit",
        BM25's ranking for a conceptual query is close to noise — and equal-weight
        RRF let that noise outvote a good semantic ranking.

        Two corrections:
          - vector gets the larger weight (it is the better signal here)
          - BM25 hits below `bm25_floor` are dropped entirely rather than fused,
            so keyword search only contributes when it actually matched something
            distinctive ("Ferris Wheel", "Mookie Betts", "Money Gap")

        Fusion is still on RANK, so cosine and BM25 magnitudes never need to be
        made comparable.
        """
        pool = max(k * 4, 20)
        vec = self.vector_search(query, k=pool)
        if not use_hybrid:
            results = self._rerank(query, vec, k=pool) if rerank else vec
            if expand_parent:
                return self._expand_to_parents(results, k)
            return results[:k]

        kw = [h for h in self.keyword_search(query, k=pool)
              if h["score"] >= bm25_floor]

        fused = {}
        for rank, hit in enumerate(vec):
            fused.setdefault(hit["id"], 0.0)
            fused[hit["id"]] += vec_weight / (rrf_k + rank + 1)
        for rank, hit in enumerate(kw):
            fused.setdefault(hit["id"], 0.0)
            fused[hit["id"]] += bm25_weight / (rrf_k + rank + 1)

        vec_by_id = {h["id"]: h for h in vec}
        order = sorted(fused.items(), key=lambda x: -x[1])
        candidates = order[:pool]

        results = []
        for cid, rrf in candidates:
            hit = vec_by_id.get(cid)
            if hit is None:
                got = self.col.get(ids=[cid])
                hit = {"id": cid, "text": got["documents"][0],
                       "meta": got["metadatas"][0], "score": None}
            results.append({**hit, "rrf": rrf})

        if rerank:
            results = self._rerank(query, results, k=pool)
        if expand_parent:
            return self._expand_to_parents(results, k)
        return results[:k]

    def _expand_to_parents(self, hits: list, k: int) -> list:
        """Child vectors retrieve; the LLM receives the parent (deduped)."""
        out = []
        seen = set()
        for h in hits:
            rec = self._by_id.get(h["id"], {})
            pid = rec.get("parent_id") or h["id"]
            if pid in seen:
                continue
            seen.add(pid)
            parent = rec.get("parent_text") or h["text"]
            meta = dict(h.get("meta") or {})
            if rec.get("section_path"):
                meta["timestamp"] = rec["section_path"]
            if rec.get("parent_id"):
                meta["parent_id"] = rec["parent_id"]
            out.append({
                **h,
                "text": parent,
                "child_text": rec.get("text", h["text"]),
                "meta": meta,
            })
            if len(out) >= k:
                break
        return out

    def _rerank(self, query: str, candidates: list, k: int) -> list:
        """
        Cross-encoder reranking.

        A bi-encoder embeds query and document separately, so it measures topical
        similarity — which is why every baseball question scores 0.6-0.8 against
        this corpus and the in/out-of-coverage distributions overlap. A
        cross-encoder reads query and document TOGETHER and scores actual
        relevance, which separates far more sharply.

        Adds ~100ms per query and ~80MB of model. Worth it: it improves ordering
        AND produces the signal abstention needs.
        """
        if not candidates:
            return []
        ce = self._cross_encoder()
        pairs = [(query, c["text"]) for c in candidates]
        scores = ce.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda c: -c["rerank_score"])[:k]

    def _cross_encoder(self):
        if self._ce is None:
            from sentence_transformers import CrossEncoder
            self._ce = CrossEncoder(RERANK_MODEL)
        return self._ce

    def best_rerank_score(self, query: str, pool: int = 10) -> float:
        """
        Sharper abstention signal than raw cosine. Cross-encoder scores are
        logits, typically negative for irrelevant pairs and positive for relevant
        ones, so the natural threshold sits near 0 rather than somewhere in a
        compressed 0.6-0.8 band.
        """
        vec = self.vector_search(query, k=pool)
        if not vec:
            return -10.0
        ce = self._cross_encoder()
        return float(max(ce.predict([(query, h["text"]) for h in vec])))

    def best_vector_score(self, query: str) -> float:
        """Top cosine similarity — the signal the abstention threshold reads."""
        hits = self.vector_search(query, k=1)
        return hits[0]["score"] if hits else 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build / query the RAG index")
    ap.add_argument("--chunks", default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--query", default=None, help="Smoke-test a query instead of building")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--no-hybrid", action="store_true", help="Vector only")
    ap.add_argument("--rerank", action="store_true", help="Cross-encoder rerank")
    ap.add_argument("--full", action="store_true",
                    help="Print each chunk in full instead of a preview")
    ap.add_argument("--preview-chars", type=int, default=400,
                    help="Preview length when not using --full (default 400)")
    a = ap.parse_args()

    try:
        if a.query:
            r = Retriever()
            hits = r.search(a.query, k=a.k, use_hybrid=not a.no_hybrid, rerank=a.rerank,
                            expand_parent=True)
            print(f"\nQuery: {a.query!r}")
            print(f"Top vector similarity: {r.best_vector_score(a.query):.3f}")
            if a.rerank:
                print(f"Top rerank score:      {r.best_rerank_score(a.query):+.3f}"
                      "   (>0 relevant, <0 not)")
            print("\nNOTE: this is RAW RETRIEVAL — the chunks fed to the LLM, not an\n"
                  "answer. Prose answers come from the generation step (rag_answer.py).\n")
            for i, h in enumerate(hits, 1):
                s = f"{h['score']:.3f}" if h["score"] is not None else "  -  "
                print(f"{i}. [{s}] {h['meta']['source_title'][:48]} @{h['meta']['timestamp']}"
                      f"  ({len(h['text'].split())} words)")
                shown = h["text"]
                if a.full:
                    body = shown
                else:
                    body = shown
                    if len(body) > a.preview_chars:
                        body = body[:a.preview_chars].rsplit(" ", 1)[0] + " […]"
                print(f"   {body}\n")
                child = h.get("child_text")
                if child and child != h["text"] and a.full:
                    print(f"   --- child hit ({len(child.split())} words) ---\n   {child}\n")
        else:
            build(Path(a.chunks) if a.chunks else None, a.rebuild)
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        sys.exit(2)
