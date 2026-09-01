"""
Parent-child chunking for the coaching notes.

Children are the retrieval unit (embedded). Parents are the generation unit
(passed to the LLM). Notes are parsed as Markdown, not sliced to a fixed
character width.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

CHILD_SOFT_MIN = 120
CHILD_SOFT_MAX = 220
CHILD_ATOMIC_MIN = 40
CHILD_ATOMIC_MAX = 100
CHILD_HARD_MAX = 240
PARENT_MIN = 300
PARENT_MAX = 600
OVERLAP_MIN = 30
OVERLAP_MAX = 60

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_HR_RE = re.compile(r"^-{3,}$|^\*{3,}$|^_{3,}$")
_UL_RE = re.compile(r"^(\s*)([-*+])\s+")
_OL_RE = re.compile(r"^(\s*)(\d+[.)])\s+")
_YAML_KEY_RE = re.compile(r"^[\w-]+\s*:")
_YOUTUBE_ID_RE = re.compile(r"\[[A-Za-z0-9_-]{8,}\]")
_SOURCE_VIDEO_RE = re.compile(r"\*\*Source video:\*\*\s*`([^`]+)`")
_META_LIST_RE = re.compile(
    r"^\s*[-*]\s+\*\*(Source video|Duration|Sentences kept):\*\*", re.I
)
_NUM_ONLY_RE = re.compile(r"^\d+[.)]$")
_LEADING_NUM_RE = re.compile(r"^\d+[.)]\s+")
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])["\u201d\']?\s+(?=[A-Z\u201c"\'(])')
_PARTIAL_END_RE = re.compile(r"""[.!?]["'\u201d\u2019)]?$|:$""")
_LEAD_PUNCT_RE = re.compile(r"^[\s.:,;!?)\]]+")
_DEF_HEADING_RE = re.compile(
    r"(?i)\b(what is|what does|explained|definition|means|versus|vs\.?)\b"
)

_tokenizer = None


def set_tokenizer(tok) -> None:
    global _tokenizer
    _tokenizer = tok


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        try:
            _tokenizer = AutoTokenizer.from_pretrained(
                EMBED_MODEL, local_files_only=True
            )
        except Exception:
            _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
        _tokenizer.model_max_length = 100_000
    return _tokenizer


def count_tokens(text: str) -> int:
    if not text or not text.strip():
        return 0
    return len(_get_tokenizer().encode(text, add_special_tokens=False))


def clean_heading_title(title: str) -> str:
    return _LEADING_NUM_RE.sub("", title).strip()


@dataclass
class Block:
    kind: str  # para, list, table, quote
    text: str


@dataclass
class Leaf:
    path: tuple[str, ...]
    heading_lines: tuple[str, ...]
    blocks: list[Block] = field(default_factory=list)

    @property
    def heading_line(self) -> str:
        return self.heading_lines[-1] if self.heading_lines else ""


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """YAML between opening --- ... --- at the start of the file; otherwise {}."""
    m = re.match(r"\A\s*---\n(.*)\n---\s*\n?", raw, re.DOTALL)
    if not m:
        return {}, raw
    inner = m.group(1)
    if re.match(r"\s*#+\s", inner) or not _YAML_KEY_RE.search(inner):
        return {}, raw
    meta = _parse_simple_yaml(inner)
    return meta, raw[m.end():]


def _parse_simple_yaml(text: str) -> dict:
    meta: dict = {}
    key = None
    for line in text.splitlines():
        if re.match(r"^\s*-\s+", line) and key is not None:
            item = re.sub(r"^\s*-\s+", "", line).strip()
            cur = meta.get(key)
            if not isinstance(cur, list):
                meta[key] = [] if cur in (None, "") else [cur]
            meta[key].append(item)
            continue
        km = re.match(r"^([\w-]+)\s*:\s*(.*)$", line)
        if km:
            key = km.group(1)
            val = km.group(2).strip()
            meta[key] = val if val else []
    return meta


def strip_horizontal_rules(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if _HR_RE.match(line.strip()):
            continue
        lines.append(line)
    return "\n".join(lines)


def _is_list_line(line: str) -> bool:
    return bool(_UL_RE.match(line) or _OL_RE.match(line))


def _is_list_continuation(line: str) -> bool:
    if not line.strip():
        return False
    if line.startswith(" ") or line.startswith("\t"):
        return True
    return False


def parse_blocks(body: str) -> list:
    """Linear scan: heading markers plus coherent body blocks."""
    lines = body.splitlines()
    n = len(lines)
    i = 0
    out: list = []
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        hm = _HEADING_RE.match(line)
        if hm:
            out.append(("heading", len(hm.group(1)), hm.group(2).strip(), line.strip()))
            i += 1
            continue
        if stripped.startswith("|"):
            acc = []
            while i < n and lines[i].strip().startswith("|"):
                acc.append(lines[i])
                i += 1
            out.append(Block("table", "\n".join(acc).strip()))
            continue
        if stripped.startswith(">"):
            acc = []
            while i < n and lines[i].strip().startswith(">"):
                acc.append(lines[i])
                i += 1
            out.append(Block("quote", "\n".join(acc).strip()))
            continue
        if _is_list_line(line):
            acc = [line]
            i += 1
            while i < n:
                nxt = lines[i]
                if _is_list_line(nxt) or _is_list_continuation(nxt):
                    acc.append(nxt)
                    i += 1
                    continue
                if not nxt.strip() and i + 1 < n and _is_list_line(lines[i + 1]):
                    acc.append(nxt)
                    i += 1
                    continue
                break
            out.append(Block("list", "\n".join(acc).rstrip()))
            continue
        acc = [line]
        i += 1
        while i < n:
            nxt = lines[i]
            st = nxt.strip()
            if not st:
                break
            if (_HEADING_RE.match(nxt) or _HR_RE.match(st) or st.startswith("|")
                    or st.startswith(">") or _is_list_line(nxt)):
                break
            acc.append(nxt)
            i += 1
        out.append(Block("para", "\n".join(acc).strip()))
    return out


def _keep_h1(title: str) -> bool:
    return not _YOUTUBE_ID_RE.search(title)


def blocks_to_leaves(items: list) -> list[Leaf]:
    leaves: list[Leaf] = []
    stack: list[tuple[int, str, str]] = []
    buf: list[Block] = []

    def flush():
        nonlocal buf
        if not buf:
            stack_ok = [(lv, t, ln) for lv, t, ln in stack if lv > 1 or _keep_h1(t)]
            if not stack_ok:
                return
            # heading-only: no leaf; the next leaf inherits the stack
            return
        usable = [(lv, t, ln) for lv, t, ln in stack if lv > 1 or _keep_h1(t)]
        leaves.append(Leaf(
            path=tuple(t for _, t, _ in usable),
            heading_lines=tuple(ln for _, _, ln in usable),
            blocks=buf,
        ))
        buf = []

    for item in items:
        if isinstance(item, tuple) and item[0] == "heading":
            _, level, title, line = item
            flush()
            stack = [s for s in stack if s[0] < level]
            stack.append((level, title, line))
            continue
        buf.append(item)
    flush()
    return [lf for lf in leaves if any(b.text.strip() for b in lf.blocks)]


def _is_meta_list(block: Block) -> bool:
    if block.kind != "list":
        return False
    items = [ln for ln in block.text.splitlines() if ln.strip()]
    return bool(items) and all(_META_LIST_RE.match(ln) for ln in items)


def _is_partial_sentence(text: str) -> bool:
    t = text.strip()
    if not t or _NUM_ONLY_RE.match(t):
        return True
    if _HEADING_RE.match(t.split("\n", 1)[0]):
        return False
    if t.endswith("|"):
        return False
    return _PARTIAL_END_RE.search(t) is None


def merge_content_blocks(blocks: list[Block]) -> list[Block]:
    """Glue fragments that must not stand alone."""
    cleaned = [b for b in blocks if not _is_meta_list(b) and b.text.strip()]
    out: list[Block] = []
    for b in cleaned:
        if not out:
            out.append(b)
            continue
        prev = out[-1]
        if _NUM_ONLY_RE.match(prev.text.strip()):
            out[-1] = Block(b.kind, prev.text.strip() + " " + b.text.lstrip())
            continue
        if prev.kind == "para" and prev.text.rstrip().endswith(":") and b.kind == "list":
            out[-1] = Block("list", prev.text.rstrip() + "\n\n" + b.text)
            continue
        if _is_partial_sentence(prev.text) and b.kind == "para":
            out[-1] = Block("para", prev.text.rstrip() + " " + b.text.lstrip())
            continue
        if _is_partial_sentence(prev.text) and b.kind in ("list", "quote", "table"):
            out[-1] = Block(b.kind, prev.text.rstrip() + "\n\n" + b.text)
            continue
        out.append(b)
    return out


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if out and _LEAD_PUNCT_RE.match(p):
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out


def _repair_lead(text: str) -> str:
    return _LEAD_PUNCT_RE.sub("", text).strip()


def is_definition(path: tuple[str, ...], text: str) -> bool:
    heading = path[-1] if path else ""
    if _DEF_HEADING_RE.search(heading):
        return True
    first = text.strip().split("\n", 1)[0]
    return bool(re.match(
        r"(?i)^.{0,100}?\b(is|are|means|refers to)\b", first
    ))


def is_atomic(block: Block, path: tuple[str, ...]) -> bool:
    n = count_tokens(block.text)
    if block.kind in ("table", "quote"):
        return True
    if n < CHILD_ATOMIC_MIN or n > CHILD_ATOMIC_MAX:
        return False
    return is_definition(path, block.text)


def overlap_prefix(text: str) -> str:
    sents = split_sentences(text)
    if not sents:
        return ""
    acc: list[str] = []
    for s in reversed(sents):
        trial = " ".join([s] + acc)
        t = count_tokens(trial)
        if t > OVERLAP_MAX and acc:
            break
        acc.insert(0, s)
        if t >= OVERLAP_MIN:
            if count_tokens(" ".join(acc)) > OVERLAP_MAX and len(acc) > 1:
                acc = acc[1:]
            break
    return " ".join(acc).strip()


def _split_oversize_para(text: str, budget: int) -> list[str]:
    sents = split_sentences(text)
    if not sents:
        return [_repair_lead(text)] if text.strip() else []
    chunks: list[str] = []
    buf: list[str] = []
    for s in sents:
        s = _repair_lead(s)
        if not s:
            continue
        trial = " ".join(buf + [s]).strip()
        if buf and count_tokens(trial) > budget:
            chunks.append(" ".join(buf).strip())
            buf = [s]
        else:
            buf.append(s)
    if buf:
        leftover = " ".join(buf).strip()
        if chunks and count_tokens(leftover) < CHILD_ATOMIC_MIN:
            chunks[-1] = (chunks[-1] + " " + leftover).strip()
        else:
            chunks.append(leftover)
    return [c for c in (_repair_lead(c) for c in chunks) if c]


def pack_leaf(leaf: Leaf) -> list[str]:
    """Body strings for one leaf (heading attached later)."""
    blocks = merge_content_blocks(leaf.blocks)
    if not blocks:
        return []
    htok = count_tokens(leaf.heading_line) if leaf.heading_line else 0
    body_max = max(CHILD_SOFT_MAX - htok, 80)
    body_min = max(min(CHILD_SOFT_MIN - htok, body_max - 20), 40)
    hard = max(CHILD_HARD_MAX - htok, body_max)

    units: list[tuple[str, str]] = []  # (kind, text)
    for b in blocks:
        n = count_tokens(b.text)
        if b.kind == "para" and n > hard:
            for piece in _split_oversize_para(b.text, hard):
                units.append(("para", piece))
        else:
            units.append((b.kind, b.text))

    chunks: list[str] = []
    buf: list[str] = []
    buf_tok = 0

    def emit(with_overlap: bool) -> None:
        nonlocal buf, buf_tok
        body = _repair_lead("\n\n".join(buf).strip())
        if not body:
            buf, buf_tok = [], 0
            return
        chunks.append(body)
        if with_overlap:
            ov = overlap_prefix(body)
            buf = [ov] if ov else []
            buf_tok = count_tokens(ov) if ov else 0
        else:
            buf, buf_tok = [], 0

    for kind, text in units:
        text = _repair_lead(text)
        if not text:
            continue
        ut = count_tokens(text)
        atomic = is_atomic(Block(kind, text), leaf.path)
        if buf and buf_tok + ut > body_max and buf_tok >= body_min:
            emit(with_overlap=True)
        elif buf and atomic and buf_tok >= body_min:
            emit(with_overlap=False)
        buf.append(text)
        buf_tok = count_tokens("\n\n".join(buf))

    if buf:
        leftover = _repair_lead("\n\n".join(buf).strip())
        lt = count_tokens(leftover)
        if chunks and lt < CHILD_ATOMIC_MIN:
            chunks[-1] = (chunks[-1] + "\n\n" + leftover).strip()
        else:
            chunks.append(leftover)
    return chunks


def join_leaves(leaves: list[Leaf]) -> str:
    parts: list[str] = []
    prev: tuple[str, ...] = ()
    for leaf in leaves:
        for i, title in enumerate(leaf.path):
            if i >= len(prev) or prev[i] != title:
                if i < len(leaf.heading_lines):
                    parts.append(leaf.heading_lines[i])
        body = "\n\n".join(b.text for b in leaf.blocks if b.text.strip())
        if body:
            parts.append(body)
        prev = leaf.path
    return "\n\n".join(p for p in parts if p.strip())


def merge_two_leaves(a: Leaf, b: Leaf) -> Leaf:
    n = 0
    while n < len(a.path) and n < len(b.path) and a.path[n] == b.path[n]:
        n += 1
    blocks: list[Block] = []
    for line in a.heading_lines[n:]:
        blocks.append(Block("para", line))
    blocks.extend(a.blocks)
    for line in b.heading_lines[n:]:
        blocks.append(Block("para", line))
    blocks.extend(b.blocks)
    return Leaf(
        path=a.path[:n],
        heading_lines=a.heading_lines[:n],
        blocks=blocks,
    )


def _leaf_tokens(leaf: Leaf) -> int:
    return count_tokens(join_leaves([leaf]))


def _leaf_is_atomic_def(leaf: Leaf) -> bool:
    body = "\n\n".join(b.text for b in leaf.blocks)
    n = count_tokens(body)
    return is_definition(leaf.path, body) and CHILD_ATOMIC_MIN <= n <= CHILD_ATOMIC_MAX


def coalesce_leaves(leaves: list[Leaf]) -> list[Leaf]:
    """Merge consecutive short siblings under the same parent heading up to ~220 tokens."""
    out: list[Leaf] = []
    for leaf in leaves:
        if not out:
            out.append(leaf)
            continue
        prev = out[-1]
        same_parent = (
            len(prev.path) >= 2 and len(leaf.path) >= 2
            and prev.path[:-1] == leaf.path[:-1]
        )
        if not same_parent or _leaf_is_atomic_def(prev) or _leaf_is_atomic_def(leaf):
            out.append(leaf)
            continue
        if _leaf_tokens(prev) >= CHILD_SOFT_MIN:
            out.append(leaf)
            continue
        merged = merge_two_leaves(prev, leaf)
        if _leaf_tokens(merged) > CHILD_SOFT_MAX:
            out.append(leaf)
            continue
        out[-1] = merged
    return out


def _with_heading(body: str, heading_line: str) -> str:
    body = _repair_lead(body)
    if heading_line:
        return f"{heading_line}\n\n{body}".strip()
    return body


def grow_neighbor_leaves(leaves: list[Leaf], idx: int) -> str:
    """Widen to adjacent sections until the parent range. Stays inside this note."""
    lo = hi = idx

    def md() -> str:
        return join_leaves(leaves[lo:hi + 1])

    while count_tokens(md()) < PARENT_MIN:
        grew = False
        if hi + 1 < len(leaves):
            hi += 1
            grew = True
            if count_tokens(md()) > PARENT_MAX:
                hi -= 1
                break
        if count_tokens(md()) >= PARENT_MIN:
            break
        if lo - 1 >= 0:
            lo -= 1
            grew = True
            if count_tokens(md()) > PARENT_MAX:
                lo += 1
                break
        if not grew:
            break
    return md()


def grow_child_window(child_texts: list[str], idx: int, heading_line: str) -> str:
    lo = hi = idx

    def md() -> str:
        return _with_heading("\n\n".join(child_texts[lo:hi + 1]), heading_line)

    while count_tokens(md()) < PARENT_MIN:
        grew = False
        if hi + 1 < len(child_texts):
            hi += 1
            grew = True
            if count_tokens(md()) > PARENT_MAX:
                hi -= 1
                break
        if count_tokens(md()) >= PARENT_MIN:
            break
        if lo - 1 >= 0:
            lo -= 1
            grew = True
            if count_tokens(md()) > PARENT_MAX:
                lo += 1
                break
        if not grew:
            break
    return md()


def choose_parent_text(leaves: list[Leaf], leaf_idx: int,
                       leaf_child_texts: list[str], child_i: int) -> str:
    leaf = leaves[leaf_idx]
    ancestors: list[str] = [join_leaves(leaves)]
    for depth in range(1, len(leaf.path) + 1):
        prefix = leaf.path[:depth]
        group = [L for L in leaves if L.path[:depth] == prefix]
        ancestors.append(join_leaves(group))

    chosen = None
    for md in ancestors:
        if count_tokens(md) <= PARENT_MAX:
            chosen = md
            break
    if chosen is None:
        if len(leaf_child_texts) > 1:
            return grow_child_window(leaf_child_texts, child_i, leaf.heading_line)
        return grow_neighbor_leaves(leaves, leaf_idx)

    if count_tokens(chosen) < PARENT_MIN:
        grown = grow_neighbor_leaves(leaves, leaf_idx)
        if PARENT_MIN <= count_tokens(grown) <= PARENT_MAX:
            return grown
        if count_tokens(grown) > count_tokens(chosen) and count_tokens(grown) <= PARENT_MAX:
            return grown
    return chosen


def strip_leading_headings(md: str) -> str:
    lines = md.splitlines()
    i = 0
    while i < len(lines) and (
        not lines[i].strip() or _HEADING_RE.match(lines[i])
    ):
        i += 1
    return "\n".join(lines[i:]).strip()


def infer_content_type(path: tuple[str, ...], tags: list) -> str:
    tags_l = [str(t).lower() for t in (tags or [])]
    blob = " ".join(path).lower()
    if any("drill" in t for t in tags_l) or "drill" in blob:
        return "hitting drill"
    if _DEF_HEADING_RE.search(blob):
        return "definition"
    return "hitting mechanics instruction"


def build_embedding_text(doc_title: str, path: tuple[str, ...],
                         content_type: str, display_md: str) -> str:
    section = " > ".join(clean_heading_title(p) for p in path) or doc_title
    body = strip_leading_headings(display_md) or display_md.strip()
    return (
        f"Document: {doc_title}\n"
        f"Section: {section}\n"
        f"Content type: {content_type}\n\n"
        f"{body}"
    )


def children_from_markdown(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    front, body = parse_front_matter(raw)
    body = strip_horizontal_rules(body)
    m = _SOURCE_VIDEO_RE.search(raw)
    source_file = m.group(1) if m else path.name
    title = path.stem
    tags = front.get("tags") if isinstance(front.get("tags"), list) else []

    items = parse_blocks(body)
    leaves = coalesce_leaves(blocks_to_leaves(items))
    if not leaves:
        return []

    packed: list[list[str]] = [pack_leaf(lf) for lf in leaves]
    children: list[dict] = []
    idx = 0
    pending_parent: dict[tuple, str] = {}

    for li, leaf in enumerate(leaves):
        bodies = packed[li]
        for ci, body in enumerate(bodies):
            display = _with_heading(body, leaf.heading_line)
            display = _repair_lead(display)
            ctype = infer_content_type(leaf.path, tags)
            parent_text = choose_parent_text(leaves, li, bodies, ci)
            parent_key = (title, parent_text)
            if parent_key not in pending_parent:
                pending_parent[parent_key] = f"{title}::p{len(pending_parent):03d}"
            children.append({
                "id": f"{title}::{idx:03d}",
                "text": display,
                "embedding_text": build_embedding_text(
                    title, leaf.path, ctype, display
                ),
                "parent_id": pending_parent[parent_key],
                "parent_text": parent_text,
                "source_title": title,
                "source_file": source_file,
                "timestamp_s": 0,
                "timestamp": " > ".join(
                    clean_heading_title(p) for p in leaf.path
                ) or "note",
                "section_path": " > ".join(
                    clean_heading_title(p) for p in leaf.path
                ),
                "content_type": ctype,
                "chunk_index": idx,
                "word_count": len(display.split()),
                "token_count": count_tokens(display),
                "parent_tokens": count_tokens(parent_text),
                "role": "child",
                "tags": tags,
            })
            idx += 1
    return children


def leading_punct_violations(chunks: list[dict]) -> list[str]:
    bad = []
    for c in chunks:
        body = strip_leading_headings(c["text"])
        if body and _LEAD_PUNCT_RE.match(body):
            bad.append(c["id"])
    return bad


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import RAG_RESOURCES_DIR

    files = sorted(
        p for p in RAG_RESOURCES_DIR.glob("*.md")
        if p.is_file() and not p.name.startswith("_")
    )
    all_chunks = []
    print(f"{'note':52s} {'n':>4} {'tok med':>8} {'parent med':>10}")
    for p in files:
        rows = children_from_markdown(p)
        all_chunks.extend(rows)
        toks = sorted(r["token_count"] for r in rows) or [0]
        pars = sorted(r["parent_tokens"] for r in rows) or [0]
        med = toks[len(toks) // 2]
        pmed = pars[len(pars) // 2]
        print(f"{p.stem[:52]:52s} {len(rows):4d} {med:8d} {pmed:10d}")

    bad = leading_punct_violations(all_chunks)
    hrs = [c["id"] for c in all_chunks if re.search(r"^---+$", c["text"], re.M)]
    yaml_leak = [c["id"] for c in all_chunks if re.search(r"^tags:\s*$", c["text"], re.M)]
    print(f"\n{len(all_chunks)} children")
    print(f"leading punct: {len(bad)}  {bad[:8]}")
    print(f"decorative ---: {len(hrs)}")
    print(f"yaml leak: {len(yaml_leak)}")
    sample = next(
        (c for c in all_chunks
         if c["source_title"] == "Contact Point"
         and "Inside pitch" in c["text"]),
        None,
    )
    if sample:
        print("\n--- Contact Point / Inside pitch embedding_text ---")
        print(sample["embedding_text"][:700])
        print("\n--- parent (first 500) ---")
        print(sample["parent_text"][:500])
