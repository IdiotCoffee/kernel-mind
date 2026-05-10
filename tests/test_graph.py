import os

from download.scan_repo import get_python_files
from graph.build_graph import build_graph
from graph.traversal import traverse_call_graph
from parser.python.parser import (
    parse_python_file,
)

# =====================================================
# Config
# =====================================================

REPO_NAME = "full-stack-fastapi-template"

REPO_PATH = os.path.join(
    "repos",
    REPO_NAME,
)

# =====================================================
# Parse Repository
# =====================================================


def load_chunks(repo_path):

    chunks = []

    files = list(get_python_files(repo_path))

    print(f"\nPython files found: {len(files)}")

    for file_path in files:
        file_chunks = parse_python_file(
            path=file_path,
            repo_path=repo_path,
        )

        chunks.extend(file_chunks)

    return chunks


# =====================================================
# Main Test
# =====================================================


def test_graph_traversal():

    # -----------------------------------
    # Parse Repository
    # -----------------------------------

    print("\nParsing repository...\n")

    chunks = load_chunks(REPO_PATH)

    print(f"Chunks extracted: {len(chunks)}")

    # -----------------------------------
    # Build Graph
    # -----------------------------------

    print("\nBuilding graph...\n")

    graph = build_graph(chunks)

    print(f"Graph nodes: {len(graph)}")

    # -----------------------------------
    # Display Sample FQNs
    # -----------------------------------

    print("\nLoaded FQNs:\n")

    for i, key in enumerate(graph.keys()):
        print(key)

        if i >= 20:
            break

    # -----------------------------------
    # Traversal Seed
    # -----------------------------------

    start_fqn = "backend.app.api.routes.login.login_access_token"

    print("\nStarting traversal from:\n")

    print(start_fqn)

    # -----------------------------------
    # Traversal
    # -----------------------------------

    results = traverse_call_graph(
        start_fqn=start_fqn,
        graph=graph,
        max_depth=4,
    )

    # -----------------------------------
    # Results
    # -----------------------------------

    print("\nTRAVERSAL RESULTS\n")

    for item in results:
        print("=" * 80)

        print(f"FQN: {item['fqn']}")

        print(f"DEPTH: {item['depth']}")

        # -----------------------------------
        # Path
        # -----------------------------------

        print("\nPATH:\n")

        for p in item["path"]:
            print(f"  -> {p}")

        # -----------------------------------
        # Calls
        # -----------------------------------

        if item.get("calls"):
            print("\nCALLS:\n")

            for edge in item["calls"][:10]:
                print(f"  -> {edge.target} [{edge.edge_type}] (w={edge.weight})")

        # -----------------------------------
        # Called By
        # -----------------------------------

        if item.get("called_by"):
            print("\nCALLED BY:\n")

            for edge in item["called_by"][:10]:
                print(f"  <- {edge.target} [{edge.edge_type}] (w={edge.weight})")

        print()

    # -----------------------------------
    # Depth Analysis
    # -----------------------------------

    depths = {}

    for item in results:
        depth = item["depth"]

        depths[depth] = depths.get(depth, 0) + 1

    print("=" * 80)

    print("\nDEPTH DISTRIBUTION\n")

    for depth, count in sorted(depths.items()):
        print(f"Depth {depth}: {count} nodes")

    print()


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    test_graph_traversal()
