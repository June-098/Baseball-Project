"""
RAG ingestion — chunks.jsonl -> local ChromaDB + BM25 index.

Builds the retrieval layer for the baseball Q&A assistant. Everything runs on your
machine: ChromaDB persists to a folder, embeddings come from a local
sentence-transformers model, and no data leaves the laptop.

    Input :  Baseball Resources/RAG Resources/*.md
    Output:  Baseball Resources/RAG Resources/rag_index/chroma/     ChromaDB store
             Baseball Resources/RAG Resources/rag_index/bm25.pkl    keyword index
             Baseball Resources/RAG Resources/rag_index/chunks.jsonl

── Setup ────────────────────────────────────────────────────────────────────
    pip install chromadb sentence-transformers rank-bm25

── Usage ────────────────────────────────────────────────────────────────────
    python src/rag_ingest.py                      # build the index
    python src/rag_ingest.py --rebuild            # wipe and rebuild
    python src/rag_ingest.py --query "curveball"  # smoke-test retrieval

── Two design decisions worth knowing ───────────────────────────────────────

1. CONTEXTUAL PREFIXES. Each chunk is embedded as

       [Source: How to hit a curve ball] So when the ball breaks down and away...

   Mid-video chunks often never name their own topic — the coach just says "it".
   Without the prefix, a question about curveballs can miss the very video that
   answers it. Prepending the title carries topic into the vector. The prefix is
   embedded but NOT shown to the user; `text` stays clean for display.

2. HYBRID RETRIEVAL. Vector search is strong on paraphrase ("load my hips" ->
   "the gather") and weak on exact strings. Named drills and player names are
   exact-string problems, so BM25 runs alongside and the two rankings are fused
   with Reciprocal Rank Fusion. Neither alone covers this corpus.
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

CHUNK_WORDS = 220
CHUNK_OVERLAP = 45

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
                     ("rank_bm25", "rank-bm25")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise RuntimeError("Missing packages:\n  pip install " + " ".join(missing))


_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")
_SOURCE_VIDEO_RE = re.compile(r"\*\*Source video:\*\*\s*`([^`]+)`")
_META_LINE_RE = re.compile(
    r"^\s*-\s+\*\*(Source video|Duration|Sentences kept):", re.IGNORECASE
)


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


def _section_blocks(md: str) -> list:
    """Split markdown into (heading, body) pairs. Lead text before the first heading is kept."""
    sections, heading, body_lines = [], "", []
    for line in md.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            body = "\n".join(body_lines).strip()
            if body:
                sections.append((heading, body))
            heading = m.group(2).strip()
            body_lines = []
        else:
            body_lines.append(line)
    body = "\n".join(body_lines).strip()
    if body:
        sections.append((heading, body))
    return sections


def _paragraph_blocks(body: str) -> list:
    """Keep tables intact; split the rest on blank lines. Drop front-matter meta lines."""
    blocks, current, in_table = [], [], False
    for line in body.splitlines():
        if _META_LINE_RE.match(line) or line.strip() == "---":
            continue
        if line.strip().startswith("|"):
            if not in_table and current:
                blocks.append("\n".join(current).strip())
                current = []
            in_table = True
            current.append(line)
            continue
        if in_table:
            blocks.append("\n".join(current).strip())
            current, in_table = [], False
        if not line.strip():
            if current:
                blocks.append("\n".join(current).strip())
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [b for b in blocks if b]


def _pack_blocks(blocks: list, chunk_words: int, overlap: int) -> list:
    chunks, cur, cur_words = [], [], 0
    for block in blocks:
        w = len(block.split())
        if cur and cur_words + w > chunk_words:
            chunks.append("\n\n".join(cur))
            back, wc = [], 0
            for b in reversed(cur):
                if wc >= overlap:
                    break
                back.insert(0, b)
                wc += len(b.split())
            cur, cur_words = back, wc
        cur.append(block)
        cur_words += w
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def chunks_from_markdown(path: Path, chunk_words: int = CHUNK_WORDS,
                         overlap: int = CHUNK_OVERLAP) -> list:
    raw = path.read_text(encoding="utf-8")
    m = _SOURCE_VIDEO_RE.search(raw)
    source_file = m.group(1) if m else path.name
    title = path.stem
    rows = []
    idx = 0
    for heading, body in _section_blocks(raw):
        packed = _pack_blocks(_paragraph_blocks(body), chunk_words, overlap)
        for text in packed:
            if heading:
                text = f"{heading}\n\n{text}"
            rows.append({
                "id": f"{title}::{idx:03d}",
                "text": text,
                "source_title": title,
                "source_file": source_file,
                "timestamp_s": 0,
                "timestamp": heading or "note",
                "chunk_index": idx,
                "word_count": len(text.split()),
            })
            idx += 1
    return rows


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
        print(f"  {p.stem[:52]:52s} {len(rows):3d} chunk(s)")
        chunks.extend(rows)
    if not chunks:
        raise RuntimeError(f"No embeddable text in {RAG_RESOURCES_DIR}")
    return chunks


def clean_title(source_title: str) -> str:
    """Strip the trailing '[youtube_id]' so titles read naturally."""
    return re.sub(r"\s*\[[A-Za-z0-9_\-]+\]\s*$", "", source_title).strip()


def contextual_text(chunk: dict) -> str:
    """
    What actually gets embedded — the title wrapped around the chunk body.

    TITLE DILUTION: embeddings mean-pool over tokens, so a 5-word prefix on a
    220-word chunk contributes ~2% of the signal. That was measurably too weak —
    "How do I hit a curveball?" retrieved a high-pitch video over the curve ball
    video, because the curve ball transcript says "breaking ball" throughout and
    "curveball" only once, while the title (which does say it) barely registered.

    Repeating the title at both ends roughly doubles its weight for a few tokens
    of cost. Cheap, and it targets exactly the failure we measured.
    """
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
            "timestamp_s":  c.get("timestamp_s", 0),
            "chunk_index":  c.get("chunk_index", 0),
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
               bm25_floor: float = BM25_MIN_SCORE, rerank: bool = False) -> list:
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
            out = vec[:k]
            return self._rerank(query, vec, k) if rerank else out

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
        candidates = order[: pool if rerank else k]

        results = []
        for cid, rrf in candidates:
            hit = vec_by_id.get(cid)
            if hit is None:
                got = self.col.get(ids=[cid])
                hit = {"id": cid, "text": got["documents"][0],
                       "meta": got["metadatas"][0], "score": None}
            results.append({**hit, "rrf": rrf})

        return self._rerank(query, results, k) if rerank else results[:k]

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
            hits = r.search(a.query, k=a.k, use_hybrid=not a.no_hybrid, rerank=a.rerank)
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
                if a.full:
                    body = h["text"]
                else:
                    # Cut on a word boundary so previews don't end mid-word.
                    body = h["text"]
                    if len(body) > a.preview_chars:
                        body = body[:a.preview_chars].rsplit(" ", 1)[0] + " […]"
                print(f"   {body}\n")
        else:
            build(Path(a.chunks) if a.chunks else None, a.rebuild)
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        sys.exit(2)
