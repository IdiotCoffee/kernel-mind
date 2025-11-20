# kernelmind/search.py
import chromadb
import re
from kernelmind.embeddings.local_backend import LocalEmbeddingBackend
from rank_bm25 import BM25Okapi
from kernelmind.utils.rewriter import QueryRewriter
from kernelmind.synthesis import synthesize_answer

# ----------------------------------
# Init
# ----------------------------------
_EMBEDDER = LocalEmbeddingBackend()
_REWRITER = QueryRewriter()

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

# Find function/method calls like "foo(", "obj.method(" or "A.B.C("
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
    """
    Return a set of simple symbol tokens to search for (last dotted token).
    Example: 'self.prepare_request(' -> 'prepare_request'
    """
    found = set()
    if not text:
        return found
    for match in CALL_PATTERN.finditer(text):
        sym = match.group(1)
        tok = sym.split(".")[-1]
        # exclude common keywords or single-letter tokens
        if tok and tok not in ("if", "for", "while", "return", "class", "with", "async"):
            found.add(tok)
    return found


def _meta_matches_symbol(meta: dict, sym: str):
    """
    Returns True if the metadata corresponds to the symbol `sym`.
    Checks:
      - exact name match
      - qualified_name last segment match
      - qualified_name exact match
    """
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
    # also allow entries that use 'class'+'method' naming pattern
    return False


def expand_call_chain(initial_chunks, repo_name, collection, depth=2, per_symbol=6):
    """
    initial_chunks: list of tuples (doc_text, meta_dict, dist)
    collection: chroma collection object
    Uses _EMBEDDER to embed symbol names so query dims match the collection.
    Returns a list of tuples (doc, meta, dist) that include seeds + discovered call targets.
    """
    seen = set()
    expanded = []

    def key_of(meta):
        return (meta.get("path"), meta.get("qualified_name") or meta.get("name"), meta.get("type"))

    # seed the seen set and expanded list
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
            # only expand from function or method chunks (avoid huge file chunks)
            if meta.get("type") not in ("function", "method"):
                continue

            symbols = extract_called_symbols(doc)
            if not symbols:
                continue

            for sym in symbols:
                # embed symbol using your local embedder so query dims match the collection
                try:
                    sym_emb = _EMBEDDER.embed([sym])
                except Exception:
                    # embedding failed for some reason: skip this symbol
                    continue

                try:
                    raw = collection.query(
                        query_embeddings=sym_emb,
                        n_results=per_symbol,
                        include=["documents", "metadatas", "distances"],
                    )
                except Exception:
                    # If collection API differs or query fails, skip gracefully
                    continue

                # defensive extraction from chroma-like return values
                docs2 = raw.get("documents", [[]])[0] if raw.get("documents") else []
                metas2 = raw.get("metadatas", [[]])[0] if raw.get("metadatas") else []
                dists2 = raw.get("distances", [[]])[0] if raw.get("distances") else [0.0] * len(docs2)

                # iterate through results and accept ones that look like the symbol
                for d2, m2, di2 in zip(docs2, metas2, dists2):
                    # repo filter
                    if repo_name and m2.get("repo") != repo_name:
                        continue

                    # prefer entries that actually match the symbol name/qualified
                    if not _meta_matches_symbol(m2, sym):
                        # skip if it's a file chunk or name mismatch
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
# Main Search
# ----------------------------------
def search(query, k=5, repo_name=None, synthesize=True, show_chunks=False):
    refined = _REWRITER.rewrite(query)

    print("\n--------------------------------------")
    print("Original Query:", query)
    print("Refined Query :", refined)
    print("--------------------------------------\n")

    # Step 1: Dense embedding for the user query
    q_emb = _EMBEDDER.embed([refined])

    client = chromadb.PersistentClient(path=".chromadb")
    col = client.get_collection("kernelmind_index")

    # Step 2: coarse dense retrieval
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

    # defensive extraction
    docs = raw.get("documents", [[]])[0] if raw.get("documents") else []
    metas = raw.get("metadatas", [[]])[0] if raw.get("metadatas") else []
    dists = raw.get("distances", [[]])[0] if raw.get("distances") else [0.0] * len(docs)

    # Step 3: initial filtering
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

    # Step 4: call-chain expansion (use top-k as seeds)
    initial = [(c["doc"], c["meta"], c["dist"]) for c in candidates[:k]]
    expanded = expand_call_chain(initial, repo_name, col, depth=2, per_symbol=6)

    # If expansion returned nothing, fall back to seeds
    merged = expanded if expanded else initial

    # Step 5: re-rank expanded set with BM25 + normalized dense similarity
    docs2 = [t[0] for t in merged]
    metas2 = [t[1] for t in merged]
    dists2 = [float(t[2]) for t in merged]

    if not docs2:
        print("No documents to rank after expansion.")
        return None

    # BM25
    corpus_tokens = [tokenize(d) for d in docs2]
    bm25 = BM25Okapi(corpus_tokens)
    q_tokens = tokenize(refined)
    bm25_scores = bm25.get_scores(q_tokens)

    # Normalize dense similarity
    if dists2:
        min_d, max_d = min(dists2), max(dists2)
    else:
        min_d, max_d = 0.0, 1.0

    if max_d - min_d < 1e-9:
        chroma_sim = [1.0] * len(dists2)
    else:
        chroma_sim = [(max_d - d) / (max_d - min_d) for d in dists2]

    # Normalize BM25 (safe)
    max_bm = max(bm25_scores) if len(bm25_scores) else 1.0
    bm25_norm = [s / max_bm for s in bm25_scores]

    # Combine + type/domain boosts
    combined = []
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
        combined.append((score, i))

    combined.sort(key=lambda x: x[0], reverse=True)
    top_indices = [i for (_, i) in combined[:k]]

    final_docs = [docs2[i] for i in top_indices]
    final_metas = [metas2[i] for i in top_indices]
    final_dists = [dists2[i] for i in top_indices]

    # print("\n=== Retrieved Chunks (Top-k) ===")
    if show_chunks:
        pretty(final_docs, final_metas, final_dists)

    if not synthesize:
        return None

    # Build chunk objects for synthesis
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

    # print("\n=== Synthesizing Answer with Gemma2 ===\n")
    answer = synthesize_answer(query, chunk_objs)
    print(answer)
    return answer


# CLI entrypoint (keeps previous behavior)
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Query: ")
    search(q)
