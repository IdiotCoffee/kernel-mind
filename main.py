import os
import shutil

from chat.chat_session import (
    start_chat_session,
)
from download.load_repo import (
    clone_repo,
)
from graph.build_graph import (
    build_graph,
)
from indexing.build_repository import (
    build_repository,
)
from utils.tui_helpers import extract_repo_name, load_chunks

# =====================================================
# Config
# =====================================================

DEFAULT_REPO = "https://github.com/fastapi/full-stack-fastapi-template"

DEVICE = "cuda"

REBUILD = True

# =====================================================
# Main
# =====================================================


def main():

    # -------------------------------------------------
    # Repo input
    # -------------------------------------------------

    repo_url = input(
        f"\nEnter repo URL (leave empty for example:\n{DEFAULT_REPO}\n\n> "
    ).strip()

    if not repo_url:
        repo_url = DEFAULT_REPO

    # -------------------------------------------------
    # Repo metadata
    # -------------------------------------------------

    repo_name = extract_repo_name(repo_url)

    repo_storage_path = os.path.join(
        ".kernelmind",
        "repos",
        repo_name,
    )

    # -------------------------------------------------
    # Rebuild cleanup
    # -------------------------------------------------

    if REBUILD:
        if os.path.exists(repo_storage_path):
            print("\nRemoving old repository artifacts...")

            shutil.rmtree(repo_storage_path)

    # -------------------------------------------------
    # Clone repository
    # -------------------------------------------------

    print("\nCloning repository...")

    repo_path = clone_repo(repo_url)

    # -------------------------------------------------
    # Parse repository
    # -------------------------------------------------

    print("\nParsing repository...")

    chunks = load_chunks(repo_path)

    print(f"\nChunks extracted: {len(chunks)}")

    if not chunks:
        print("\nNo chunks extracted.\nStopping build.")

        return

    # -------------------------------------------------
    # Build graph
    # -------------------------------------------------

    print("\nBuilding graph...")

    graph = build_graph(chunks)

    print(f"\nGraph nodes: {len(graph)}")

    # -------------------------------------------------
    # Build repository runtime
    # -------------------------------------------------

    print("\nBuilding repository runtime...")

    result = build_repository(
        repo_id=repo_name,
        chunks=chunks,
        graph=graph,
        device=DEVICE,
    )

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    print("\nBUILD SUMMARY\n")
    summary = result["summary"]

    runtime = result["runtime"]
    for key, value in summary.items():
        print(f"{key}: {value}")

    # -------------------------------------------------
    # Launch chat
    # -------------------------------------------------

    print("\nLaunching repository chat...")

    start_chat_session(
        runtime=runtime,
    )


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    main()
