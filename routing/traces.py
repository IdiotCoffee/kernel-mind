from collections import defaultdict


def build_reasoning_trace(results):
    """
    Build lightweight reasoning traces
    from retrieved graph-expanded results.

    This is NOT chain-of-thought.

    It is:
    repository evidence tracing.
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
    # Build trace
    # =====================================================

    trace = []

    sorted_depths = sorted(depth_groups.keys())

    for depth in sorted_depths:
        if not depth_groups[depth]:
            continue

        best = depth_groups[depth][0]

        trace.append(best["fqn"])

    # =====================================================
    # Deduplicate sequentially
    # =====================================================

    cleaned = []

    prev = None

    for node in trace:
        if node != prev:
            cleaned.append(node)

        prev = node

    return cleaned
