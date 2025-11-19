import os

def load_file_lines(absolute_path):
    with open(absolute_path, "r", encoding="utf-8") as f:
        return f.readlines()

def extract_chunk(text_lines, start, end):
    return "".join(text_lines[start-1:end])

def build_text_chunks(context_pack, repo_root):
    file_path = context_pack["file"]["path"]
    absolute = os.path.join(repo_root, file_path)
    repo = context_pack.get("repo", None)

    lines = load_file_lines(absolute)
    chunks = []

    # ----------------------------------
    # File-level chunk
    # ----------------------------------
    chunks.append({
        "type": "file",
        "path": file_path,
        "name": os.path.basename(file_path),
        "repo": repo,
        "text": "".join(lines)
    })

    # ----------------------------------
    # Functions
    # ----------------------------------
    for fn in context_pack["functions"]:
        start = fn["start_line"]
        end = fn["end_line"]
        source = extract_chunk(lines, start, end)

        text = (
            f"# File: {file_path}\n"
            f"# Function: {fn['name']}\n"
            f"# Args: {', '.join(fn['args'])}\n"
            f"# Lines: {start}-{end}\n\n"
            f"{source}"
        )

        chunks.append({
            "type": "function",
            "path": file_path,
            "name": fn["name"],
            "args": fn["args"],
            "repo": repo,
            "text": text.lower()  # optional: improves lexical BM25
        })

    # ----------------------------------
    # Classes
    # ----------------------------------
    for cls in context_pack["classes"]:
        start = cls["start_line"]
        end = cls["end_line"]
        source = extract_chunk(lines, start, end)

        text = (
            f"# File: {file_path}\n"
            f"# Class: {cls['name']}\n"
            f"# Lines: {start}-{end}\n\n"
            f"{source}"
        )

        chunks.append({
            "type": "class",
            "path": file_path,
            "name": cls["name"],
            "repo": repo,
            "text": text.lower()
        })

    # ----------------------------------
    # Methods
    # ----------------------------------
    for m in context_pack["methods"]:
        start = m["start_line"]
        end = m["end_line"]
        source = extract_chunk(lines, start, end)

        text = (
            f"# File: {file_path}\n"
            f"# Class: {m['class']}\n"
            f"# Method: {m['name']}\n"
            f"# Args: {', '.join(m['args'])}\n"
            f"# Lines: {start}-{end}\n\n"
            f"{source}"
        )

        chunks.append({
            "type": "method",
            "path": file_path,
            "name": m["name"],
            "class": m["class"],
            "args": m["args"],
            "repo": repo,
            "text": text.lower()
        })

    return chunks
