import heapq
import math
from typing import Dict, List

from db.models import GraphNode
from retrieval.rank import compute_query_overlap

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
    query: str,
    max_depth: int = 2,
    max_nodes: int = 25,
) -> List[dict]:
    """
    Query-aware graph expansion.

    Expansion now incorporates:
    - propagation score
    - semantic edge weights
    - depth decay
    - query-conditioned traversal boosts

    This helps preserve:
    - workflow coherence
    - query locality
    - semantic intent alignment
    """

    best_scores = {}

    results: List[dict] = []

    # -------------------------------------------------
    # Priority queue
    #
    # (-score, depth, fqn, propagated_score)
    # -------------------------------------------------

    queue = []

    # -------------------------------------------------
    # Seed queue
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Expansion loop
    # -------------------------------------------------

    while queue and len(results) < max_nodes:
        (
            neg_score,
            depth,
            current_fqn,
            propagated_score,
        ) = heapq.heappop(queue)

        # -------------------------------------------------
        # Strongest-path retention
        # -------------------------------------------------

        existing = best_scores.get(current_fqn)

        if existing is not None and existing >= propagated_score:
            continue

        best_scores[current_fqn] = propagated_score

        node = graph.get(current_fqn)

        if not node:
            continue

        # -------------------------------------------------
        # Node degree
        # -------------------------------------------------

        degree = compute_weighted_connectivity(node)

        # -------------------------------------------------
        # Store expanded node
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Debug
        # -------------------------------------------------

        print(
            f"[EXPAND] "
            f"depth={depth} "
            f"score={round(propagated_score, 4)} "
            f"node={current_fqn}"
        )

        # -------------------------------------------------
        # Stop expansion
        # -------------------------------------------------

        if depth >= max_depth:
            continue

        # =================================================
        # Expand CALLEES
        #
        # Forward execution flow
        # =================================================

        for edge in node.calls:
            # ---------------------------------------------
            # Query-aware traversal boost
            # ---------------------------------------------

            query_overlap = compute_query_overlap(
                query=query,
                fqn=edge.target,
            )

            query_multiplier = 1.0 + min(
                query_overlap,
                1.5,
            )

            # ---------------------------------------------
            # Propagation
            # ---------------------------------------------

            edge_adjusted_score = (
                propagated_score * edge.weight * get_decay(depth + 1) * query_multiplier
            )

            existing = best_scores.get(edge.target)

            if existing is None or edge_adjusted_score > existing:
                heapq.heappush(
                    queue,
                    (
                        -edge_adjusted_score,
                        depth + 1,
                        edge.target,
                        edge_adjusted_score,
                    ),
                )

        # =================================================
        # Expand CALLERS
        #
        # Reverse traversal
        #
        # Caller expansion intentionally weaker
        # to reduce semantic drift.
        # =================================================

        for edge in node.called_by:
            target_node = graph.get(edge.target)

            if not target_node:
                continue

            # ---------------------------------------------
            # Connectivity hub suppression
            # ---------------------------------------------

            connectivity = len(target_node.calls) + len(target_node.called_by)

            hub_penalty = 1 / math.log2(connectivity + 2)

            # ---------------------------------------------
            # Query-aware traversal boost
            # ---------------------------------------------

            query_overlap = compute_query_overlap(
                query=query,
                fqn=edge.target,
            )

            query_multiplier = 1.0 + min(
                query_overlap,
                1.25,
            )

            # ---------------------------------------------
            # Propagation
            # ---------------------------------------------

            edge_adjusted_score = (
                propagated_score
                * edge.weight
                * get_decay(depth + 1)
                * hub_penalty
                * query_multiplier
                * 0.45
            )

            existing = best_scores.get(edge.target)

            if existing is None or edge_adjusted_score > existing:
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
