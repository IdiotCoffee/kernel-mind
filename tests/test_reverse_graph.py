from graph.reverse_traversal import (
    traverse_callers,
)
from indexing.repository_runtime import (
    RepositoryRuntime,
)

# =====================================================
# Config
# =====================================================

REPO_ID = "full-stack-fastapi-template"

# =====================================================
# Test
# =====================================================


def test_reverse_graph():

    # -------------------------------------------------
    # Load runtime
    # -------------------------------------------------

    print("\nLoading repository runtime...\n")

    runtime = RepositoryRuntime.load(
        repo_id=REPO_ID,
        device="cpu",
    )

    graph = runtime.graph

    print(f"Loaded graph nodes: {len(graph)}")

    # -------------------------------------------------
    # Reverse traversal seed
    # -------------------------------------------------

    start_fqn = "backend.app.core.security.create_access_token"

    print("\nStarting reverse traversal from:\n")

    print(start_fqn)

    # -------------------------------------------------
    # Traverse callers
    # -------------------------------------------------

    results = traverse_callers(
        start_fqn=start_fqn,
        graph=graph,
        max_depth=5,
    )

    # -------------------------------------------------
    # Results
    # -------------------------------------------------

    print(f"\nRESULT COUNT: {len(results)}")

    for item in results:
        print("=" * 80)

        print(f"FQN: {item['fqn']}")

        print(f"Depth: {item['depth']}")

        # ---------------------------------------------
        # Path
        # ---------------------------------------------

        if item.get("path"):
            print("\nPath:\n")

            for p in item["path"]:
                print(f"  -> {p}")

        # ---------------------------------------------
        # Code preview
        # ---------------------------------------------

        chunk = runtime.chunk_lookup.get(item["fqn"])

        if chunk:
            print("\nCODE:\n")

            print(chunk.code[:400])

        print()

    # -------------------------------------------------
    # Depth stats
    # -------------------------------------------------

    print("=" * 80)

    print("\nDEPTH DISTRIBUTION\n")

    depths = {}

    for item in results:
        depth = item["depth"]

        depths[depth] = depths.get(depth, 0) + 1

    for depth, count in sorted(depths.items()):
        print(f"Depth {depth}: {count}")

    print()


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    test_reverse_graph()
