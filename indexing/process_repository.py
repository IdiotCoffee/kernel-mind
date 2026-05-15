from download.load_repo import clone_repo
from graph.build_graph import build_graph
from utils.tui_helpers import extract_repo_name, load_chunks


def process_repository(repo_url):

    # -----------------------------------
    # Clone
    # -----------------------------------

    repo_path = clone_repo(repo_url)

    # -----------------------------------
    # Repo ID
    # -----------------------------------

    repo_id = extract_repo_name(repo_url)

    # -----------------------------------
    # Parse
    # -----------------------------------

    chunks = load_chunks(repo_path)

    # -----------------------------------
    # Graph
    # -----------------------------------

    graph = build_graph(chunks)

    return {
        "repo_id": repo_id,
        "chunks": chunks,
        "graph": graph,
    }
