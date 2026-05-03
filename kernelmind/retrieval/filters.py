BLOCKED_FOLDERS = [
    "tests/",
    "docs/",
    "examples/",
    "tutorial/",
    "benchmarks/",
]


def should_allow(path: str, query: str):
    p = (path or "").lower()

    if "test" in query.lower() or "docs" in query.lower():
        return True

    for bad in BLOCKED_FOLDERS:
        if bad in p:
            return False

    return True


def filter_candidates(docs, metas, dists, repo_name, query):
    results = []

    for d, m, dist in zip(docs, metas, dists):
        if repo_name and m.get("repo") != repo_name:
            continue

        if not should_allow(m.get("path", ""), query):
            continue

        results.append((d, m, dist))

    return results
