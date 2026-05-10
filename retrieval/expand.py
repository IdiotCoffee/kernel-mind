import heapq
import math
from typing import Dict, List, Set

from db.models import GraphNode

DEPTH_DECAY = {
    0: 1.0,
    1: 0.75,
    2: 0.45,
    3: 0.20,
}


def get_decay(depth: int) -> float:
    return DEPTH_DECAY.get(depth, 0.05)


def compute_weighted_connectivity(node) -> float:
    outgoing = sum(edge.weight for edge in node.calls)

    incoming = sum(edge.weight for edge in node.called_by)

    return outgoing + incoming


def expand_context(
    seed_results: List[dict],
    graph: Dict[str, GraphNode],
    max_depth: int = 2,
    max_nodes: int = 25,
) -> List[dict]:
    """
    Relevance-aware graph expansion.

    Expansion score propagates through graph
    with depth-based decay.

    Higher-confidence retrieval seeds influence
    graph expansion more strongly.
    """

    visited: Set[str] = set()

    results: List[dict] = []

    # -----------------------------------
    # Priority queue
    #
    # (-score, depth, fqn)
    # -----------------------------------

    queue = []

    # -----------------------------------
    # Seed queue
    # -----------------------------------

    for item in seed_results:
        chunk = item["chunk"]

        seed_score = item["score"]

        heapq.heappush(
            queue,
            (
                -seed_score,
                0,
                chunk.fqn,
                seed_score,
            ),
        )

    # -----------------------------------
    # Expansion loop
    # -----------------------------------

    while queue and len(results) < max_nodes:
        (
            neg_score,
            depth,
            current_fqn,
            propagated_score,
        ) = heapq.heappop(queue)

        if current_fqn in visited:
            continue

        visited.add(current_fqn)

        node = graph.get(current_fqn)

        if not node:
            continue

        # -----------------------------------
        # Node degree
        # -----------------------------------

        # degree = len(node.calls) + len(node.called_by)
        degree = compute_weighted_connectivity(node)

        # -----------------------------------
        # Store expanded node
        # -----------------------------------

        results.append(
            {
                "fqn": node.fqn,
                "type": node.node_type,
                "depth": depth,
                "calls": node.calls,
                "called_by": node.called_by,
                "propagated_score": round(
                    propagated_score,
                    4,
                ),
                "degree": degree,
            }
        )

        # -----------------------------------
        # Stop expansion
        # -----------------------------------

        if depth >= max_depth:
            continue

        # -----------------------------------
        # Expand callees
        # -----------------------------------

        for edge in node.calls:
            if edge.target not in visited:
                edge_adjusted_score = (
                    propagated_score * edge.weight * get_decay(depth + 1)
                )

                heapq.heappush(
                    queue,
                    (
                        -edge_adjusted_score,
                        depth + 1,
                        edge.target,
                        edge_adjusted_score,
                    ),
                )
        # -----------------------------------
        # Expand callers
        # -----------------------------------

        for edge in node.called_by:
            target_node = graph.get(edge.target)

            if not target_node:
                continue

            connectivity = len(target_node.calls) + len(target_node.called_by)

            hub_penalty = 1 / math.log2(connectivity + 2)

            if edge.target not in visited:
                edge_adjusted_score = (
                    propagated_score * edge.weight * get_decay(depth + 1) * hub_penalty
                )

                heapq.heappush(
                    queue,
                    (
                        -edge_adjusted_score,
                        depth + 1,
                        edge.target,
                        edge_adjusted_score,
                    ),
                )

    return results
