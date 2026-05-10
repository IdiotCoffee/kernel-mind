from collections import deque
from typing import Dict, List, Optional

from db.models import GraphNode


def get_callers(
    fqn: str,
    graph: Dict[str, GraphNode],
) -> List[str]:
    """
    Return all nodes that call this node.
    """

    node = graph.get(fqn)

    if not node:
        return []

    return [edge.target for edge in node.called_by]


def get_callees(
    fqn: str,
    graph: Dict[str, GraphNode],
) -> List[str]:
    """
    Return all nodes called by this node.
    """

    node = graph.get(fqn)

    if not node:
        return []

    return [edge.target for edge in node.calls]


def find_path(
    source_fqn: str,
    target_fqn: str,
    graph: Dict[str, GraphNode],
    max_depth: int = 10,
) -> Optional[List[str]]:
    """
    Find shortest path between two nodes
    using BFS.
    """

    if source_fqn not in graph:
        return None

    if target_fqn not in graph:
        return None

    queue = deque()

    queue.append((source_fqn, [source_fqn]))

    visited = set()

    while queue:
        current_fqn, path = queue.popleft()

        if current_fqn == target_fqn:
            return path

        if current_fqn in visited:
            continue

        visited.add(current_fqn)

        if len(path) > max_depth:
            continue

        node = graph.get(current_fqn)

        if not node:
            continue

        for edge in node.calls:
            callee_fqn = edge.target

            if callee_fqn not in graph:
                continue

            if callee_fqn in visited:
                continue

            queue.append(
                (
                    callee_fqn,
                    path + [callee_fqn],
                )
            )

    return None
