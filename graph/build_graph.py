from typing import Dict, List

from db.models import (
    CodeChunk,
    GraphEdge,
    GraphNode,
)
from graph.resolve_calls import (
    resolve_all_calls,
)

# =====================================================
# Edge Weighting
# =====================================================


def classify_edge(call: str):

    edge_weight = 1.0

    edge_type = "call"

    lowered = call.lower()

    # -------------------------------------------------
    # Weak framework / utility patterns
    # -------------------------------------------------

    weak_patterns = [
        "print",
        "logger",
        "logging",
        "json",
        "datetime",
        "typing",
        "pydantic",
        "fastapi",
        "sqlalchemy",
    ]

    for pattern in weak_patterns:
        if pattern in lowered:
            edge_weight = 0.2

            edge_type = "framework"

            break

    # -------------------------------------------------
    # Strong semantic patterns
    # -------------------------------------------------

    strong_patterns = [
        "token",
        "auth",
        "jwt",
        "password",
        "security",
        "login",
        "user",
    ]

    for pattern in strong_patterns:
        if pattern in lowered:
            edge_weight = 1.5

            edge_type = "semantic"

            break

    return edge_type, edge_weight


# =====================================================
# Graph Builder
# =====================================================


def build_graph(
    chunks: List[CodeChunk],
) -> Dict[str, GraphNode]:

    # -------------------------------------------------
    # Resolve Calls FIRST
    # -------------------------------------------------

    print("\nResolving calls...\n")

    chunks = resolve_all_calls(chunks)

    # -------------------------------------------------
    # Build Graph
    # -------------------------------------------------

    graph: Dict[str, GraphNode] = {}

    # -------------------------------------------------
    # PASS 1 — Build nodes
    # -------------------------------------------------

    for chunk in chunks:
        weighted_calls = []

        # ---------------------------------------------
        # Debug visibility
        # ---------------------------------------------

        for call in chunk.calls:
            edge_type, edge_weight = classify_edge(call)

            weighted_calls.append(
                GraphEdge(
                    target=call,
                    edge_type=edge_type,
                    weight=edge_weight,
                )
            )

        graph[chunk.fqn] = GraphNode(
            fqn=chunk.fqn,
            node_type=chunk.type,
            calls=weighted_calls,
        )

    # -------------------------------------------------
    # PASS 2 — Reverse edges
    # -------------------------------------------------

    for node in graph.values():
        for edge in node.calls:
            if edge.target in graph:
                graph[edge.target].called_by.append(
                    GraphEdge(
                        target=node.fqn,
                        edge_type=edge.edge_type,
                        weight=edge.weight,
                    )
                )

    # -------------------------------------------------
    # Debug Check
    # -------------------------------------------------

    # print("\nGRAPH BUILD CHECK\n")

    # found = False

    # for node in graph.values():
    #     if node.calls or node.called_by:
    #         print(f"NODE: {node.fqn}")

    #         print("\nCALLS:")

    #         for edge in node.calls[:10]:
    #             print(f"  -> {edge.target} [{edge.edge_type}] (w={edge.weight})")

    #         print("\nCALLED BY:")

    #         for edge in node.called_by[:10]:
    #             print(f"  <- {edge.target} [{edge.edge_type}] (w={edge.weight})")

    #         found = True

    #         break

    # if not found:
    #     print("No connected graph nodes found.")

    # -------------------------------------------------
    # Connectivity Stats
    # -------------------------------------------------

    connected_nodes = 0

    total_edges = 0

    for node in graph.values():
        edge_count = len(node.calls) + len(node.called_by)

        if edge_count > 0:
            connected_nodes += 1

            total_edges += edge_count

    print("\nGRAPH STATS\n")

    print(f"Total nodes: {len(graph)}")

    print(f"Connected nodes: {connected_nodes}")

    print(f"Total edges: {total_edges}")

    print()

    return graph
