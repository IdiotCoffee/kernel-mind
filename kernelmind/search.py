import os
import re
import math
from typing import List, Tuple

import chromadb
from rank_bm25 import BM25Okapi

from kernelmind.embeddings.local_backend import LocalEmbeddingBackend
from kernelmind.utils.rewriter import QueryRewriter
from kernelmind.synthesis import synthesize_answer

# Reranker imports (transformers)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
except Exception:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    pipeline = None

# ----------------------------------
# Init
# ----------------------------------
_EMBEDDER = LocalEmbeddingBackend()
_REWRITER = QueryRewriter()

# Reranker will be created lazily when first used
_RERANKER = None
_RERANKER_MODEL = "BAAI/bge-reranker-base"

CANDIDATE_MULTIPLIER = 12

TYPE_BOOST = {
    "function": 0.20,
    "method":   0.18,
    "class":    0.10,
    "import":   0.02,
    "file":     0.00,
    None:       0.00,
}

BLOCKED_FOLDERS = [
    "tests/", "test/",
    "docs/", "docs_src/",
    "examples/", "example/",
    "tutorial/", "tutorials/",
    "benchmarks/", "scripts/",
    "migrations/",
]

token_pattern = re.compile(r"\w+")

CALL_PATTERN = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\("
)

def tokenize(text):
    return token_pattern.findall((text or "").lower())

def should_allow(path: str, query: str):
    p = (path or "").lower()
    if "test" in query.lower() or "docs" in query.lower():
        return True
    for bad in BLOCKED_FOLDERS:
        if bad in p:
            return False
    return True

def pretty(indoc, inmeta, indists):
    for i in range(len(indoc)):
        meta = inmeta[i]
        dist = indists[i] if i < len(indists) else None

        print("\n=== Result", i + 1, "===")
        print("Path   :", meta.get("path"))
        print("Name   :", meta.get("name"))
        print("Qualified:", meta.get("qualified_name"))
        print("Type   :", meta.get("type"))
        print("Repo   :", meta.get("repo"))
        print("Score  :", dist)
        print("\nCode:\n")
        print(indoc[i])

def extract_called_symbols(text: str):
    found = set()
    if not text:
        return found
    for match in CALL_PATTERN.finditer(text):
        sym = match.group(1)
        tok = sym.split(".")[-1]
        if tok and tok not in ("if", "for", "while", "return", "class", "with", "async"):
            found.add(tok)
    return found

def _meta_matches_symbol(meta: dict, sym: str):
    if not meta:
        return False
    name = meta.get("name") or ""
    qual = meta.get("qualified_name") or ""
    if name == sym:
        return True
    if qual == sym:
        return True
    if qual.split(".") and qual.split(".")[-1] == sym:
        return True
    return False

def expand_call_chain(initial_chunks, repo_name, collection, depth=2, per_symbol=6):
    seen = set()
    expanded = []

    def key_of(meta):
        return (meta.get("path"), meta.get("qualified_name") or meta.get("name"), meta.get("type"))

    for doc, meta, dist in initial_chunks:
        if not meta:
            continue
        expanded.append((doc, meta, dist))
        seen.add(key_of(meta))

    frontier = initial_chunks

    for level in range(depth):
        if not frontier:
            break
        next_frontier = []

        for doc, meta, dist in frontier:
            if meta.get("type") not in ("function", "method"):
                continue

            symbols = extract_called_symbols(doc)
            if not symbols:
                continue

            for sym in symbols:
                try:
                    sym_emb = _EMBEDDER.embed([sym])
                except Exception:
                    continue

                try:
                    raw = collection.query(
                        query_embeddings=sym_emb,
                        n_results=per_symbol,
                        include=["documents", "metadatas", "distances"],
                    )
                except Exception:
                    continue

                docs2 = raw.get("documents", [[]])[0] if raw.get("documents") else []
                metas2 = raw.get("metadatas", [[]])[0] if raw.get("metadatas") else []
                dists2 = raw.get("distances", [[]])[0] if raw.get("distances") else [0.0] * len(docs2)

                for d2, m2, di2 in zip(docs2, metas2, dists2):
                    if repo_name and m2.get("repo") != repo_name:
                        continue
                    if not _meta_matches_symbol(m2, sym):
                        continue
                    k = key_of(m2)
                    if k in seen:
                        continue

                    seen.add(k)
                    expanded.append((d2, m2, float(di2)))
                    next_frontier.append((d2, m2, float(di2)))

        frontier = next_frontier

    return expanded

# ----------------------------------
# RERANKER
# ----------------------------------

from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.device = None
        self.model = None
        self._load()

    def _load(self):
        print("[RERANKER] Initializing cross-encoder...")
        try:
            self.model = CrossEncoder(self.model_name, device="cuda")
            self.device = "cuda"
        except Exception as e:
            print("CUDA load failed, falling back to CPU:", e)
            self.model = CrossEncoder(self.model_name, device="cpu")
            self.device = "cpu"

    def score(self, query, doc):
        return self.score_batch(query, [doc])[0]

    def score_batch(self, query, docs, batch_size=8):
        pairs = [[query, d] for d in docs]
        try:
            return self.model.predict(pairs, batch_size=batch_size)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("CUDA OOM during scoring — switching reranker to CPU")
                self.model = CrossEncoder(self.model_name, device="cpu")
                self.device = "cpu"
                return self.model.predict(pairs, batch_size=batch_size)
            raise e

def _ensure_reranker():
    global _RERANKER
    if _RERANKER is None:
        print("[RERANKER] Creating new reranker instance...")
        _RERANKER = Reranker()
    else:
        print(f"[RERANKER] Reusing existing reranker on { _RERANKER.device }")
    return _RERANKER

# ----------------------------------
# MAIN SEARCH
# ----------------------------------

def search(query, k=5, repo_name=None, synthesize=True, show_chunks=False, use_reranker=True):
    refined = _REWRITER.rewrite(query)

    print("\n--------------------------------------")
    print("Original Query:", query)
    print("Refined Query :", refined)
    print("--------------------------------------\n")

    q_emb = _EMBEDDER.embed([refined])

    client = chromadb.PersistentClient(path=".chromadb")
    col = client.get_collection("kernelmind_index")

    n_candidates = max(k * CANDIDATE_MULTIPLIER, k + 10)
    try:
        raw = col.query(
            query_embeddings=q_emb,
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        print("Chroma dense query failed:", e)
        return None

    docs = raw.get("documents", [[]])[0] if raw.get("documents") else []
    metas = raw.get("metadatas", [[]])[0] if raw.get("metadatas") else []
    dists = raw.get("distances", [[]])[0] if raw.get("distances") else [0.0] * len(docs)

    candidates = []
    for doc, meta, dist in zip(docs, metas, dists):
        if repo_name and meta.get("repo") != repo_name:
            continue
        if not should_allow(meta.get("path", ""), refined):
            continue
        candidates.append({"doc": doc, "meta": meta, "dist": dist})

    if len(candidates) == 0:
        print("No filtered candidates — showing raw top-k.")
        pretty(docs[:k], metas[:k], dists[:k])
        return None

    initial = [(c["doc"], c["meta"], c["dist"]) for c in candidates[:k]]
    expanded = expand_call_chain(initial, repo_name, col, depth=2, per_symbol=6)
    merged = expanded if expanded else initial

    docs2 = [t[0] for t in merged]
    metas2 = [t[1] for t in merged]
    dists2 = [float(t[2]) for t in merged]

    if not docs2:
        print("No documents to rank after expansion.")
        return None

    corpus_tokens = [tokenize(d) for d in docs2]
    bm25 = BM25Okapi(corpus_tokens)
    q_tokens = tokenize(refined)
    bm25_scores = bm25.get_scores(q_tokens)

    if dists2:
        min_d, max_d = min(dists2), max(dists2)
    else:
        min_d, max_d = 0.0, 1.0

    if max_d - min_d < 1e-9:
        chroma_sim = [1.0] * len(dists2)
    else:
        chroma_sim = [(max_d - d) / (max_d - min_d) for d in dists2]

    max_bm = max(bm25_scores) if len(bm25_scores) else 1.0
    bm25_norm = [s / max_bm for s in bm25_scores]

    base_scores = []
    for i, _ in enumerate(merged):
        base = 0.6 * chroma_sim[i] + 0.4 * bm25_norm[i]
        t = metas2[i].get("type")
        boost = TYPE_BOOST.get(t, 0.0)

        path = (metas2[i].get("path") or "").lower()
        domain = 0.0
        if "routing" in path:
            domain += 0.12
        if "applications" in path:
            domain += 0.10
        if "request" in path and path.endswith(".py"):
            domain += 0.08

        score = base + boost + domain
        base_scores.append(score)

    def _normalize_list(values: List[float]) -> List[float]:
        if values is None or len(values) == 0:
            return []
        mn, mx = min(values), max(values)
        if abs(mx - mn) < 1e-9:
            return [1.0] * len(values)
        return [(v - mn) / (mx - mn) for v in values]

    base_norm = _normalize_list(base_scores)

    rerank_scores = None
    if use_reranker:
        try:
            print(f"[RERANKER] Running reranker on {len(docs2)} chunks...")
            rer = _ensure_reranker()
            print(f"[RERANKER] Device = {rer.device}")
            text_chunks = [f"{m.get('qualified_name') or ''} -- {d}" for d, m in zip(docs2, metas2)]
            rerank_scores = rer.score_batch(refined, text_chunks, batch_size=8)
        except Exception as e:
            print("Reranker failed to initialize/score:", e)
            rerank_scores = None

    # Fix: avoid ambiguous truth-value check
    has_rerank = rerank_scores is not None and len(rerank_scores) > 0

    final_scores = []
    if has_rerank:
        rer_norm = _normalize_list(rerank_scores)
        for i in range(len(merged)):
            final = 0.75 * rer_norm[i] + 0.25 * base_norm[i]
            final_scores.append((final, i))
    else:
        for i in range(len(merged)):
            final_scores.append((base_norm[i], i))

    final_scores.sort(key=lambda x: x[0], reverse=True)
    top_indices = [i for (_, i) in final_scores[:k]]

    final_docs = [docs2[i] for i in top_indices]
    final_metas = [metas2[i] for i in top_indices]
    final_dists = [dists2[i] for i in top_indices]

    if show_chunks:
        pretty(final_docs, final_metas, final_dists)

    if not synthesize:
        return None

    chunk_objs = []
    for text, meta in zip(final_docs, final_metas):
        chunk_objs.append({
            "text": text,
            "path": meta.get("path"),
            "start": meta.get("start", None),
            "end": meta.get("end", None),
            "qualified_name": meta.get("qualified_name"),
            "type": meta.get("type"),
        })
    print("[RERANKER] Reranking done - Synthesizing answer")
    answer = synthesize_answer(query, chunk_objs)
    return answer

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Query: ")
    search(q)
