from typing import Dict

from pymongo.collection import Collection

from db.models import GraphEdge, GraphNode


def load_graph(collection: Collection) -> Dict[str, GraphNode]:
    """
    Load parsed chunks from MongoDB
    and construct weighted graph edges.
    """

    graph: Dict[str, GraphNode] = {}

    cursor = collection.find({})

    # -----------------------------------
    # PASS 1: Build graph nodes
    # -----------------------------------

    for doc in cursor:
        fqn = doc.get("fqn")

        if not fqn:
            continue

        raw_calls = doc.get("calls", [])

        weighted_calls = []

        for call in raw_calls:
            edge_weight = 1.0
            edge_type = "call"

            lowered = call.lower()

            # -----------------------------------
            # Penalize framework / utility edges
            # -----------------------------------

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

            # -----------------------------------
            # Boost auth/security edges
            # -----------------------------------

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

            weighted_calls.append(
                GraphEdge(
                    target=call,
                    edge_type=edge_type,
                    weight=edge_weight,
                )
            )

        node = GraphNode(
            fqn=fqn,
            node_type=doc.get("type", "unknown"),
            calls=weighted_calls,
        )

        graph[fqn] = node

    # -----------------------------------
    # PASS 2: Build reverse edges
    # -----------------------------------

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

    # -----------------------------------
    # Debug check
    # -----------------------------------

    print("\nREVERSE EDGE CHECK\n")

    for node in graph.values():
        if node.called_by:
            print(node.fqn)

            print(node.called_by)

            break

    return graph
