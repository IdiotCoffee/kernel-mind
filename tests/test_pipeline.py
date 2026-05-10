from indexing.repository_runtime import (
    RepositoryRuntime,
)
from retrieval.pipeline import (
    retrieve_context,
)

# =====================================================
# Config
# =====================================================

REPO_ID = "full-stack-fastapi-template"

QUERY = "password reset flow"

# =====================================================
# Test
# =====================================================


def test_pipeline():

    # -------------------------------------------------
    # Load runtime
    # -------------------------------------------------

    print("\nLoading repository runtime...\n")

    runtime = RepositoryRuntime.load(
        repo_id=REPO_ID,
        device="cpu",
    )

    print(f"Loaded chunks: {len(runtime.chunks)}")

    print(f"Loaded graph nodes: {len(runtime.graph)}")

    # -------------------------------------------------
    # Retrieval pipeline
    # -------------------------------------------------

    print("\nRunning retrieval pipeline...\n")

    results = retrieve_context(
        query=QUERY,
        runtime=runtime,
    )

    # -------------------------------------------------
    # Results
    # -------------------------------------------------

    print("\nFINAL RETRIEVAL RESULTS\n")

    for item in results:
        chunk = runtime.chunk_lookup.get(item["fqn"])

        if not chunk:
            continue

        print("=" * 80)

        print(f"FQN: {chunk.fqn}")

        print(f"TYPE: {chunk.type}")

        print(f"DEPTH: {item['depth']}")

        print(f"SCORE: {round(item['score'], 4)}")

        print(f"PROPAGATED: {round(item['propagated_score'], 4)}")

        print(f"DEGREE: {item['degree']}")

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

        print(chunk.code[:500])

        print()

    print()


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    test_pipeline()
