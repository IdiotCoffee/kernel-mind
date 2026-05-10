from typing import Dict, List

from db.models import CodeChunk

# =====================================================
# Symbol Map
# =====================================================


def build_symbol_map(
    chunks: List[CodeChunk],
) -> Dict[str, str]:
    """
    Maps:

        authenticate
            ->
        backend.app.crud.authenticate
    """

    symbol_map = {}

    for chunk in chunks:
        symbol_map[chunk.name] = chunk.fqn

    return symbol_map


# =====================================================
# Resolve Calls
# =====================================================


def resolve_calls_for_chunk(
    chunk: CodeChunk,
    symbol_map: Dict[str, str],
) -> List[str]:

    resolved = []

    import_map = chunk.imports or {}

    for call in chunk.calls:
        name = call.split(".")[-1]

        # -----------------------------------
        # Local symbol resolution
        # -----------------------------------

        if name in symbol_map:
            resolved.append(symbol_map[name])

            continue

        # -----------------------------------
        # Import resolution
        # -----------------------------------

        if name in import_map:
            resolved.append(import_map[name])

            continue

        # -----------------------------------
        # Fallback
        # -----------------------------------

        resolved.append(call)

    return list(set(resolved))


# =====================================================
# Resolve All Chunks
# =====================================================


def resolve_all_calls(
    chunks: List[CodeChunk],
):

    symbol_map = build_symbol_map(chunks)

    for chunk in chunks:
        chunk.calls = resolve_calls_for_chunk(
            chunk,
            symbol_map,
        )

    return chunks
