from typing import List


def normalize_score(
    score,
    min_score,
    max_score,
):

    if max_score - min_score == 0:
        return 0.5

    return (score - min_score) / (max_score - min_score)


def compute_confidence(results: List[dict]):
    """
    Lightweight semantic retrieval confidence.

    This estimates:
    - retrieval coherence
    - evidence strength
    - graph consistency
    - query alignment

    NOT probabilistic certainty.
    """

    # =====================================================
    # No Results
    # =====================================================

    if not results:
        return {
            "score": 0.0,
            "label": "LOW",
        }

    # =====================================================
    # Top Result
    # =====================================================

    top = results[0]

    top_score = top.get(
        "final_score",
        top.get("score", 0),
    )

    overlap = top.get("overlap", 0)

    # =====================================================
    # Score Separation
    # =====================================================

    if len(results) > 1:
        second_score = results[1].get(
            "final_score",
            results[1].get("score", 0),
        )

        separation = top_score - second_score

    else:
        separation = top_score

    # =====================================================
    # Overlap Bonus
    # =====================================================

    overlap_bonus = 0.0

    if overlap >= 2.0:
        overlap_bonus += 0.3

    elif overlap >= 1.0:
        overlap_bonus += 0.18

    # =====================================================
    # Graph Coherence
    # =====================================================

    depths = [r.get("depth", 0) for r in results[:6]]

    shallow_nodes = sum(1 for d in depths if d <= 1)

    graph_bonus = (
        min(
            shallow_nodes / 6,
            1.0,
        )
        * 0.3
    )

    # =====================================================
    # Retrieval Strength
    # =====================================================

    # Workflow queries naturally score lower.
    # Use softer scaling.

    retrieval_strength = (
        min(
            top_score / 2.5,
            1.0,
        )
        * 0.25
    )

    # =====================================================
    # Dominance
    # =====================================================

    dominance_bonus = (
        min(
            separation / 1.5,
            1.0,
        )
        * 0.15
    )

    # =====================================================
    # Final Confidence
    # =====================================================

    confidence = overlap_bonus + graph_bonus + retrieval_strength + dominance_bonus

    confidence = min(confidence, 1.0)

    # =====================================================
    # Labels
    # =====================================================

    if confidence >= 0.72:
        label = "HIGH"

    elif confidence >= 0.42:
        label = "MEDIUM"

    else:
        label = "LOW"

    return {
        "score": round(confidence, 4),
        "label": label,
    }
