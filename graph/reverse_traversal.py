from typing import Dict, List, Set

from db.models import GraphNode


def traverse_callers(
    start_fqn: str,
    graph: Dict[str, GraphNode],
    max_depth: int = 5,
) -> List[dict]:

    visited: Set[str] = set()

    results: List[dict] = []

    def dfs(
        current_fqn: str,
        depth: int,
        path: List[str],
    ):

        if depth > max_depth:
            return

        if current_fqn in visited:
            return

        visited.add(current_fqn)

        node = graph.get(current_fqn)

        if not node:
            return

        current_path = path + [current_fqn]

        results.append(
            {
                "fqn": current_fqn,
                "depth": depth,
                "path": current_path,
                "type": node.node_type,
            }
        )
        for edge in node.called_by:
            dfs(
                current_fqn=edge.target,
                depth=depth + 1,
                path=current_path,
            )

    dfs(
        current_fqn=start_fqn,
        depth=0,
        path=[],
    )

    return results
