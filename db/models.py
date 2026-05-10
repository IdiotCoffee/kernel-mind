from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class CodeChunk:
    id: str

    name: str
    type: str

    fqn: str
    module: str
    file_path: str

    parent_fqn: Optional[str]

    code: str
    docstring: Optional[str]

    calls: List[str]
    imports: dict[str, str]

    start_line: int
    end_line: int

    def to_dict(self):
        return asdict(self)


@dataclass
class GraphEdge:
    target: str

    edge_type: str = "call"

    weight: float = 1.0


@dataclass
class GraphNode:
    fqn: str
    node_type: str

    calls: List[GraphEdge] = field(default_factory=list)

    called_by: List[GraphEdge] = field(default_factory=list)


@dataclass
class RankedChunk:
    chunk_id: str

    embedding_score: float = 0.0
    bm25_score: float = 0.0
    symbol_score: float = 0.0
    graph_score: float = 0.0

    final_score: float = 0.0

    metadata: Dict = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
