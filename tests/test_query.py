from graph.query import (
    find_path,
    get_callees,
    get_callers,
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


def test_graph_queries():

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
    # TEST CALLERS
    # -------------------------------------------------

    fqn = "backend.app.core.security.create_access_token"

    print("\nCALLERS\n")

    callers = get_callers(
        fqn=fqn,
        graph=graph,
    )

    if not callers:
        print("No callers found")

    else:
        for caller in callers:
            print(" <-", caller)

    # -------------------------------------------------
    # TEST CALLEES
    # -------------------------------------------------

    fqn = "backend.app.api.routes.login.login_access_token"

    print("\nCALLEES\n")

    callees = get_callees(
        fqn=fqn,
        graph=graph,
    )

    if not callees:
        print("No callees found")

    else:
        for callee in callees:
            print(" ->", callee)

    # -------------------------------------------------
    # TEST PATHFINDING
    # -------------------------------------------------

    source = "backend.app.api.routes.login.login_access_token"

    target = "backend.app.core.security.verify_password"

    print("\nPATH\n")

    path = find_path(
        source_fqn=source,
        target_fqn=target,
        graph=graph,
    )

    if not path:
        print("NO PATH FOUND")

    else:
        for p in path:
            print(" ->", p)

    print()


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    test_graph_queries()
