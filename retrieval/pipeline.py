from retrieval.expand import expand_context
from retrieval.rank import rank_expansion_results


def retrieve_context(
    query: str,
    runtime,
    retrieval_top_k: int = 5,
    expansion_depth: int = 2,
    expansion_nodes: int = 25,
    final_top_k: int = 15,
):
    """
    Full retrieval pipeline.

    query
      → hybrid retrieval
      → graph expansion
      → graph-aware ranking
    """

    # -----------------------------------
    # Hybrid Retrieval
    # -----------------------------------

    retrieval_results = runtime.hybrid_retriever.search(
        query=query,
        top_k=retrieval_top_k,
    )

    # -----------------------------------
    # Graph Expansion
    # -----------------------------------

    expanded = expand_context(
        seed_results=retrieval_results,
        graph=runtime.graph,
        max_depth=expansion_depth,
        max_nodes=expansion_nodes,
    )

    # -----------------------------------
    # Graph Ranking
    # -----------------------------------

    ranked = rank_expansion_results(
        expanded_nodes=expanded,
        graph=runtime.graph,
        query=query,
    )

    # -----------------------------------
    # Final Top-K
    # -----------------------------------

    return ranked[:final_top_k]
