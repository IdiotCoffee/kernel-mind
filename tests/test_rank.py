from indexing.repository_runtime import (
    RepositoryRuntime,
)
from retrieval.expand import (
    expand_context,
)
from retrieval.rank import (
    rank_expansion_results,
)

# =====================================================
# Config
# =====================================================

REPO_ID = "full-stack-fastapi-template"

QUERY = "how is the access token created"

# =====================================================
# Test
# =====================================================


def test_graph_ranking():

    # -------------------------------------------------
    # Load runtime
    # -------------------------------------------------

    print("\nLoading repository runtime...\n")

    runtime = RepositoryRuntime.load(
        repo_id=REPO_ID,
        device="cpu",
    )

    # -------------------------------------------------
    # Hybrid retrieval
    # -------------------------------------------------

    print("\nRunning hybrid retrieval...\n")

    seed_results = runtime.hybrid_retriever.search(
        query=QUERY,
        top_k=5,
    )

    print("\nSEED RESULTS\n")

    for item in seed_results:
        chunk = item["chunk"]

        print(f"{chunk.fqn} (score={round(item['score'], 4)})")

    # -------------------------------------------------
    # Graph expansion
    # -------------------------------------------------

    print("\nRunning graph expansion...\n")

    expanded = expand_context(
        seed_results=seed_results,
        graph=runtime.graph,
        max_depth=3,
        max_nodes=25,
    )

    print(f"\nExpanded nodes: {len(expanded)}")

    # -------------------------------------------------
    # Graph-aware ranking
    # -------------------------------------------------

    print("\nRunning graph-aware ranking...\n")

    ranked = rank_expansion_results(
        expanded_nodes=expanded,
        graph=runtime.graph,
        query=QUERY,
    )

    # -------------------------------------------------
    # Results
    # -------------------------------------------------

    print("\nRANKED RESULTS\n")

    for item in ranked[:20]:
        chunk = runtime.chunk_lookup.get(item["fqn"])

        if not chunk:
            continue

        print("=" * 80)

        print(f"FQN: {chunk.fqn}")

        print(f"TYPE: {chunk.type}")

        print(f"DEPTH: {item['depth']}")

        print(f"PROPAGATED: {round(item['propagated_score'], 4)}")

        print(f"DEGREE: {item['degree']}")

        print(f"SCORE: {round(item['score'], 4)}")

        # ---------------------------------------------
        # Calls
        # ---------------------------------------------

        if item["calls"]:
            print("\nCALLS:\n")

            for edge in item["calls"][:5]:
                print(f"  -> {edge.target} [{edge.edge_type}] (w={edge.weight})")

        # ---------------------------------------------
        # Called by
        # ---------------------------------------------

        if item["called_by"]:
            print("\nCALLED BY:\n")

            for edge in item["called_by"][:5]:
                print(f"  <- {edge.target} [{edge.edge_type}] (w={edge.weight})")

        # ---------------------------------------------
        # Code preview
        # ---------------------------------------------

        print("\nCODE:\n")

        print(chunk.code[:400])

        print()

    # -------------------------------------------------
    # Depth analysis
    # -------------------------------------------------

    print("=" * 80)

    print("\nDEPTH DISTRIBUTION\n")

    depths = {}

    for item in ranked:
        depth = item["depth"]

        depths[depth] = depths.get(depth, 0) + 1

    for depth, count in sorted(depths.items()):
        print(f"Depth {depth}: {count}")

    print()


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    test_graph_ranking()
