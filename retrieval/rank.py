import math
from typing import List

from db.models import GraphNode
from retrieval.finetuning import (
    CONNECTIVITY_WEIGHT,
    DEPTH_PENALTY,
    PROPAGATION_WEIGHT,
    PROXIMITY_DEPTH_0_BOOST,
    PROXIMITY_DEPTH_1_BOOST,
    QUERY_MATCH_WEIGHT,
    TYPE_WEIGHTS,
)

# =========================================================
# Tokenization
# =========================================================


def compute_weighted_connectivity(node) -> float:
    outgoing = sum(edge.weight for edge in node.calls)

    incoming = sum(edge.weight for edge in node.called_by)

    return outgoing + incoming


def tokenize(text: str) -> List[str]:
    """
    Simple identifier-aware tokenizer.

    Converts:
        create_access_token
    into:
        ["create", "access", "token"]
    """

    return [token.lower() for token in text.replace("_", " ").split()]


# =========================================================
# Query / Symbol Overlap
# =========================================================


def compute_query_overlap(
    query: str,
    fqn: str,
) -> float:
    """
    Computes lexical overlap between:
    - user query
    - fully-qualified symbol name
    """

    query_tokens = set(tokenize(query))

    symbol_tokens = set(tokenize(fqn))

    overlap = query_tokens.intersection(symbol_tokens)

    return len(overlap) * QUERY_MATCH_WEIGHT


# =========================================================
# Base Graph Node Scoring
# =========================================================


def score_node(
    node: GraphNode,
    depth: int,
    query: str,
) -> float:
    """
    Scores a graph node during traversal.

    Used primarily during:
    - graph expansion
    - propagation
    - traversal prioritization
    """

    score = 0.0

    # -----------------------------------------------------
    # Node Type Weight
    # -----------------------------------------------------

    score += TYPE_WEIGHTS.get(
        node.node_type,
        0.5,
    )

    # -----------------------------------------------------
    # Connectivity Weight
    #
    # LOG-scaled to prevent:
    # - utility domination
    # - framework hubs
    # - graph gravity wells
    # -----------------------------------------------------

    # connectivity = len(node.calls) + len(node.called_by)
    connectivity = compute_weighted_connectivity(node)

    connectivity_score = math.log1p(connectivity)

    score += connectivity_score * CONNECTIVITY_WEIGHT

    # -----------------------------------------------------
    # Query Overlap
    # -----------------------------------------------------

    score += compute_query_overlap(
        query=query,
        fqn=node.fqn,
    )

    # -----------------------------------------------------
    # Depth Penalty
    #
    # Penalize distant traversal nodes
    # -----------------------------------------------------

    score -= depth * DEPTH_PENALTY

    return round(score, 4)


# =========================================================
# Expanded Context Ranking
# =========================================================


def rank_expansion_results(
    expanded_nodes,
    graph,
    query,
):
    """
    Final ranking for graph-expanded retrieval context.

    Combines:
    - propagated relevance
    - traversal locality
    - hub suppression
    - lexical grounding
    """

    ranked = []

    for node in expanded_nodes:
        score = 0.0

        # -------------------------------------------------
        # Propagation Score
        # -------------------------------------------------

        score += node["propagated_score"] * PROPAGATION_WEIGHT

        # -------------------------------------------------
        # Locality / Proximity Boost
        # -------------------------------------------------

        if node["depth"] == 0:
            score += PROXIMITY_DEPTH_0_BOOST

        elif node["depth"] == 1:
            score += PROXIMITY_DEPTH_1_BOOST

        # -------------------------------------------------
        # Connectivity Bonus
        #
        # LOG-scaled hub suppression
        # -------------------------------------------------

        degree = node["degree"]

        connectivity_bonus = math.log1p(degree)

        score += min(connectivity_bonus / 4, 0.35)

        # -------------------------------------------------
        # Query Overlap Boost
        #
        # Re-anchor graph expansion
        # to user intent
        # -------------------------------------------------

        score += compute_query_overlap(
            query=query,
            fqn=node["fqn"],
        )

        ranked.append(
            {
                **node,
                "score": round(score, 4),
            }
        )

    # -----------------------------------------------------
    # Final Sort
    # -----------------------------------------------------

    ranked.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return ranked
