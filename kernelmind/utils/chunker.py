import os

def load_file_lines(absolute_path):
    with open(absolute_path, "r", encoding="utf-8") as f:
        return f.readlines()


def extract_chunk(text_lines, start, end):
    return "".join(text_lines[start - 1:end])


def build_text_chunks(context_pack, repo_root):
    file_path = context_pack["file"]["path"]
    absolute = os.path.join(repo_root, file_path)
    repo = context_pack.get("repo", None)

    lines = load_file_lines(absolute)
    chunks = []

    # ----------------------------------
    # Functions
    # ----------------------------------
    for fn in context_pack["functions"]:
        start = fn["start_line"]
        end = fn["end_line"]
        source = extract_chunk(lines, start, end)

        q = fn.get("qualified_name", fn["name"])

        text = (
            f"# file: {file_path}\n"
            f"# function: {fn['name']}\n"
            f"# qualified: {q}\n"
            f"# args: {', '.join(fn.get('args', []))}\n"
            f"# lines: {start}-{end}\n\n"
            f"{source}"
        )

        chunks.append({
            "type": "function",
            "path": file_path,
            "name": fn["name"],
            "qualified_name": q,
            "args": fn.get("args", []),
            "repo": repo,
            "start": start,
            "end": end,
            "text": text
        })

    # ----------------------------------
    # Classes
    # ----------------------------------
    for cls in context_pack["classes"]:
        start = cls["start_line"]
        end = cls["end_line"]
        source = extract_chunk(lines, start, end)

        q = cls.get("qualified_name", cls.get("name", ""))

        text = (
            f"# file: {file_path}\n"
            f"# class: {cls['name']}\n"
            f"# qualified: {q}\n"
            f"# lines: {start}-{end}\n\n"
            f"{source}"
        )

        chunks.append({
            "type": "class",
            "path": file_path,
            "name": cls["name"],
            "qualified_name": q,
            "repo": repo,
            "start": start,
            "end": end,
            "text": text
        })

    # ----------------------------------
    # Methods
    # ----------------------------------
    for m in context_pack["methods"]:
        start = m["start_line"]
        end = m["end_line"]
        source = extract_chunk(lines, start, end)

        q = m.get("qualified_name", f"{m.get('class','')}.{m['name']}")

        text = (
            f"# file: {file_path}\n"
            f"# class: {m.get('class','')}\n"
            f"# method: {m['name']}\n"
            f"# qualified: {q}\n"
            f"# args: {', '.join(m.get('args', []))}\n"
            f"# lines: {start}-{end}\n\n"
            f"{source}"
        )

        chunks.append({
            "type": "method",
            "path": file_path,
            "name": m["name"],
            "qualified_name": q,
            "class": m.get("class", ""),
            "args": m.get("args", []),
            "repo": repo,
            "start": start,
            "end": end,
            "text": text
        })

    return chunks
