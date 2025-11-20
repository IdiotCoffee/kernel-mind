import os

EXCLUDE_DIRS = {"node_modules", "dist", "build", "__pycache__", ".git", ".idea", ".vscode", ".venv",".env"}
INCLUDE_DIRS = {".py", ".js", ",.ts", ".java", ".go", ".json", ".yaml", ".yml", ".toml", ".md"}

def should_ignore(dirname):
    return dirname in EXCLUDE_DIRS
def should_include(filename):
    _, ext = os.path.splitext(filename)
    return ext in INCLUDE_DIRS

def crawl_repo(root_path):
    collected = []
    for current_root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if not should_ignore(d)]
        for file in files:
            if should_include(file):
                full_path = os.path.join(current_root, file)
                collected.append(full_path)
    return collected

