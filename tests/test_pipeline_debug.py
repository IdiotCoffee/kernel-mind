from indexing.repository_runtime import (
    RepositoryRuntime,
)
from retrieval.expand import expand_context
from retrieval.pipeline import retrieve_context

# =====================================================
# Config
# =====================================================

REPO_ID = "full-stack-fastapi-template"

QUERY = "explain login workflow"

# =====================================================
# Main
# =====================================================


def test_pipeline_debug():

    print("\nLoading runtime...\n")

    runtime = RepositoryRuntime.load(
        repo_id=REPO_ID,
        device="cpu",
    )

    # -------------------------------------------------
    # STEP 1 — Hybrid Retrieval
    # -------------------------------------------------

    print("=" * 80)

    print("\nSTEP 1 — HYBRID RETRIEVAL\n")

    retrieval_results = runtime.hybrid_retriever.search(
        query=QUERY,
        top_k=5,
    )

    for idx, item in enumerate(
        retrieval_results,
        start=1,
    ):
        chunk = item["chunk"]

        print(f"[{idx}] {chunk.fqn}")

        print(f"Score: {round(item['score'], 4)}")

        print()

    # -------------------------------------------------
    # STEP 2 — Graph Expansion
    # -------------------------------------------------

    print("=" * 80)

    print("\nSTEP 2 — GRAPH EXPANSION\n")

    expanded = expand_context(
        seed_results=retrieval_results,
        graph=runtime.graph,
        max_depth=3,
        max_nodes=50,
    )

    for idx, item in enumerate(
        expanded,
        start=1,
    ):
        print(f"[{idx}] {item['fqn']}")

        print(f"Depth: {item['depth']}")

        print(f"Propagated: {round(item['propagated_score'], 4)}")

        if "path" in item:
            print(f"Path: {' -> '.join(item['path'])}")

        print()

    # -------------------------------------------------
    # STEP 3 — Final Pipeline
    # -------------------------------------------------

    print("=" * 80)

    print("\nSTEP 3 — FINAL RANKED RESULTS\n")

    final_results = retrieve_context(
        query=QUERY,
        runtime=runtime,
    )

    for idx, item in enumerate(
        final_results,
        start=1,
    ):
        print(f"[{idx}] {item['fqn']}")

        print(f"Depth: {item['depth']}")

        print(f"Score: {round(item['score'], 4)}")

        print(f"Propagated: {round(item['propagated_score'], 4)}")

        print()

    # -------------------------------------------------
    # Depth Stats
    # -------------------------------------------------

    print("=" * 80)

    print("\nDEPTH ANALYSIS\n")

    depths = {}

    for item in final_results:
        depth = item["depth"]

        depths[depth] = depths.get(depth, 0) + 1

    for depth, count in sorted(depths.items()):
        print(f"Depth {depth}: {count}")

    print()


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    test_pipeline_debug()
