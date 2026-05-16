from collections import defaultdict


def build_reasoning_trace(results, top_per_depth=3):
    """
    Build lightweight workflow visualization traces.

    This is NOT chain-of-thought.

    It is:
    repository evidence tracing.

    Returns:
        [
            {
                "fqn": "...",
                "depth": 1,
                "score": 0.91,
            }
        ]
    """

    if not results:
        return []

    # =====================================================
    # Group by depth
    # =====================================================

    depth_groups = defaultdict(list)

    for item in results:
        depth = item.get("depth", 0)

        depth_groups[depth].append(item)

    # =====================================================
    # Sort each depth by score
    # =====================================================

    for depth in depth_groups:
        depth_groups[depth] = sorted(
            depth_groups[depth],
            key=lambda x: x.get(
                "final_score",
                x.get("score", 0),
            ),
            reverse=True,
        )

    # =====================================================
    # Build expanded trace
    # =====================================================

    trace = []

    sorted_depths = sorted(depth_groups.keys())

    for depth in sorted_depths:
        top_nodes = depth_groups[depth][:top_per_depth]

        for node in top_nodes:
            trace.append(
                {
                    "fqn": node["fqn"],
                    "depth": depth,
                    "score": round(
                        node.get(
                            "final_score",
                            node.get("score", 0),
                        ),
                        4,
                    ),
                }
            )

    # =====================================================
    # Deduplicate globally
    # =====================================================

    seen = set()

    cleaned = []

    for node in trace:
        if node["fqn"] in seen:
            continue

        seen.add(node["fqn"])

        cleaned.append(node)

    return cleaned
