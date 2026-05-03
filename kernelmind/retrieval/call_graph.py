import re

CALL_PATTERN = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\("
)


def extract_called_symbols(text: str):
    found = set()
    for match in CALL_PATTERN.finditer(text or ""):
        found.add(match.group(1).split(".")[-1])
    return found


def expand_call_chain(initial, repo_name, collection, embedder, depth=2):
    expanded = list(initial)
    seen = set()

    for doc, meta, _ in initial:
        seen.add(meta.get("qualified_name"))

    frontier = initial

    for _ in range(depth):
        next_frontier = []

        for doc, meta, _ in frontier:
            symbols = extract_called_symbols(doc)

            for sym in symbols:
                try:
                    emb = embedder.embed([sym])
                    raw = collection.query(
                        query_embeddings=emb,
                        n_results=5,
                        include=["documents", "metadatas", "distances"],
                    )
                except Exception:
                    continue

                docs = raw["documents"][0]
                metas = raw["metadatas"][0]
                dists = raw["distances"][0]

                for d, m, dist in zip(docs, metas, dists):
                    if repo_name and m.get("repo") != repo_name:
                        continue

                    key = m.get("qualified_name")
                    if key in seen:
                        continue

                    seen.add(key)
                    expanded.append((d, m, dist))
                    next_frontier.append((d, m, dist))

        frontier = next_frontier

    return expanded
