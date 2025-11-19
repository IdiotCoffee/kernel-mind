
import chromadb
import re
from embeddings.local_backend import LocalEmbeddingBackend
from rank_bm25 import BM25Okapi
from utils.rewriter import QueryRewriter
from synthesis import synthesize_answer   # NEW 🔥

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


# ----------------------------------
# Helper functions
# ----------------------------------
def tokenize(text):
    return token_pattern.findall(text.lower())


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
        dist = indists[i]

        print("\n=== Result", i + 1, "===")
        print("Path   :", meta.get("path"))
        print("Name   :", meta.get("name"))
        print("Type   :", meta.get("type"))
        print("Repo   :", meta.get("repo"))
        print("Score  :", dist)
        print("\nCode:\n")
        print(indoc[i])


# ----------------------------------
# Main Search + Synthesis
# ----------------------------------
def search(query, k=5, repo_name=None, synthesize=True):
    # Step 1: Rewrite query
    refined = _REWRITER.rewrite(query)

    print("\n--------------------------------------")
    print("Original Query:", query)
    print("Refined Query :", refined)
    print("--------------------------------------\n")

    # Step 2: Dense embedding
    q_emb = _EMBEDDER.embed([refined])

    client = chromadb.PersistentClient(path=".chromadb")
    col = client.get_collection("kernelmind_index")

    # Step 3: Dense retrieve
    n_candidates = max(k * CANDIDATE_MULTIPLIER, k + 10)
    raw = col.query(
        query_embeddings=q_emb,
        n_results=n_candidates,
        include=["documents", "metadatas", "distances"],
    )

    docs = raw["documents"][0]
    metas = raw["metadatas"][0]
    dists = raw["distances"][0]

    # Step 4: Filter candidates
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

    # Step 5: BM25 rerank
    corpus_tokens = [tokenize(c["doc"]) for c in candidates]
    bm25 = BM25Okapi(corpus_tokens)
    q_tokens = tokenize(refined)
    bm25_scores = bm25.get_scores(q_tokens)

    # Step 6: Normalize dense similarity
    dense_vals = [c["dist"] for c in candidates]
    min_d, max_d = min(dense_vals), max(dense_vals)
    if max_d - min_d < 1e-9:
        chroma_sim = [1.0] * len(dense_vals)
    else:
        chroma_sim = [(max_d - d) / (max_d - min_d) for d in dense_vals]

    # Step 7: Normalize BM25
    max_bm = max(bm25_scores) if len(bm25_scores) else 1
    bm25_norm = [s / max_bm for s in bm25_scores]

    # Step 8: Combine + type/domain boosts
    combined = []
    for i, c in enumerate(candidates):
        base = 0.6 * chroma_sim[i] + 0.4 * bm25_norm[i]
        t = c["meta"].get("type")
        boost = TYPE_BOOST.get(t, 0.0)

        path = (c["meta"].get("path") or "").lower()
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
    top = [candidates[i] for (_, i) in combined[:k]]

    # Extract for pretty() and synthesis
    final_docs = [t["doc"] for t in top]
    final_metas = [t["meta"] for t in top]
    final_dists = [t["dist"] for t in top]

    print("\n=== Retrieved Chunks (Top-k) ===")
    pretty(final_docs, final_metas, final_dists)

    # Step 9: Synthesis
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
        })

    print("\n=== Synthesizing Answer with Gemma2 ===\n")
    answer = synthesize_answer(query, chunk_objs)
    print(answer)
    return answer


# ----------------------------------
# CLI
# ----------------------------------
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Query: ")
    search(q)

