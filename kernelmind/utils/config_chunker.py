import yaml

def build_config_chunks(config_doc, repo, max_chars=600):
    """
    Builds semantic config chunks from parsed YAML/JSON.

    Each chunk contains:
        - path (REQUIRED for embedding pipeline)
        - repo name
        - type = "config"
        - key_path (dot-notation path inside config: e.g. database.host)
        - text: rendered snippet for embedding
    """

    if not config_doc or "tree" not in config_doc or not config_doc["tree"]:
        return []

    tree = config_doc["tree"]
    file_path = config_doc["file"]          # original path inside repo
    chunks = []

    def walk(node, prefix):
        # prefix = key path like "services.backend.image"
        if isinstance(node, dict):
            for key, value in node.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                walk(value, new_prefix)

        elif isinstance(node, list):
            # Treat entire list as a chunk
            rendered = render_block(node)
            chunks.append(make_chunk(prefix, rendered))

            # Also walk each item
            for item in node:
                walk(item, prefix)

        else:
            # primitive value (str, int, bool…)
            rendered = render_block(node)
            chunks.append(make_chunk(prefix, rendered))

    def make_chunk(key_path, text):
        # Prepare final chunk
        if len(text) > max_chars:
            text = text[:max_chars] + "\n…"

        return {
            "path": file_path,             # ⭐ REQUIRED by embedding pipeline
            "repo": repo,
            "type": "config",
            "key_path": key_path or "",    # always string
            "text": f"{key_path} = {text}".strip(),
        }

    def render_block(value):
        try:
            return yaml.safe_dump(value, sort_keys=False).rstrip()
        except Exception:
            return str(value)

    walk(tree, "")
    return chunks
