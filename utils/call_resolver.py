from db.db import chunks_collection


def build_symbol_map():
    """Builds a symbol map from chunks in the database."""
    symbol_map = {}

    for chunk in chunks_collection.find():
        name = chunk["name"]
        fqn = chunk["fqn"]

        # NOTE: this overwrites duplicates (acceptable for now)
        symbol_map[name] = fqn

    return symbol_map


def build_import_map(imports):
    """
    Converts:
        ["urllib.parse", "requests.utils"]

    Into:
        {
            "urlparse": "urllib.parse",
            "to_native_string": "requests.utils"
        }
    """

    import_map = {}

    for imp in imports:
        parts = imp.split(".")
        if parts:
            import_map[parts[-1]] = imp

    return import_map


def resolve_calls_for_chunk(chunk, symbol_map):
    resolved = []

    import_map = chunk.get("imports", {})

    for call in chunk.get("calls", []):
        name = call.split(".")[-1]

        if name in symbol_map:
            resolved.append(symbol_map[name])
            continue

        if name in import_map:
            resolved.append(import_map[name])
            continue

        resolved.append(call)

    return resolved


def resolve_all_calls():
    print("Building symbol map...")
    symbol_map = build_symbol_map()

    print("Resolving calls...")

    updated = 0

    for chunk in chunks_collection.find():
        resolved_calls = resolve_calls_for_chunk(chunk, symbol_map)

        chunks_collection.update_one(
            {"_id": chunk["_id"]},
            {"$set": {"calls": resolved_calls}},
        )

        updated += 1

    print(f"Resolved calls for {updated} chunks.")
