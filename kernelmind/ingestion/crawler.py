import os

EXCLUDE_DIRS = {
    "node_modules",
    "dist",
    "build",
    "out",
    ".next",
    ".nuxt",
    "coverage",
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".idea",
    ".vscode",
    ".venv",
    "env",
    ".env",
    ".pytest_cache",
    "__snapshots__",
    "target",        # Rust
    "bin", "obj",    # C/C++
}

EXCLUDE_FILE_PATTERNS = {
    ".min.js",
    ".map",
    ".lock",
    ".snap",
    ".log",
}

INCLUDE_DIRS = {".py", ".js", ".ts", ".java", ".go", ".json", ".yaml", ".yml", ".toml", ".md"}

def should_ignore(dirname):
    return dirname in EXCLUDE_DIRS
def should_include(filename):
    _, ext = os.path.splitext(filename)
    return ext in INCLUDE_DIRS
def should_exclude_file(filename):
    for pat in EXCLUDE_FILE_PATTERNS:
        if filename.endswith(pat):
            return True
    return False


def crawl_repo(root_path):
    collected = []
    for current_root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if not should_ignore(d)]
        for file in files:
            if should_exclude_file(file):
                continue
            if should_include(file):
                full_path = os.path.join(current_root, file)
                collected.append(full_path)
    return collected

