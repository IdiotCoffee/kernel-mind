# import json

from db.models import CodeChunk, GraphEdge, GraphNode


def serialize_chunk(chunk: CodeChunk) -> dict:
    return chunk.to_dict()


def deserialize_chunk(data: dict) -> CodeChunk:
    return CodeChunk(**data)


def serialize_graph(graph):
    serialized = {}

    for fqn, node in graph.items():
        serialized[fqn] = {
            "fqn": node.fqn,
            "node_type": node.node_type,
            "calls": [
                {
                    "target": edge.target,
                    "edge_type": edge.edge_type,
                    "weight": edge.weight,
                }
                for edge in node.calls
            ],
            "called_by": [
                {
                    "target": edge.target,
                    "edge_type": edge.edge_type,
                    "weight": edge.weight,
                }
                for edge in node.called_by
            ],
        }

    return serialized


def deserialize_graph(data):
    graph = {}

    for fqn, node_data in data.items():
        graph[fqn] = GraphNode(
            fqn=node_data["fqn"],
            node_type=node_data["node_type"],
            calls=[GraphEdge(**edge) for edge in node_data["calls"]],
            called_by=[GraphEdge(**edge) for edge in node_data["called_by"]],
        )

    return graph
