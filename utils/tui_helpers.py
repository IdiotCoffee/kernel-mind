from download.scan_repo import (
    get_python_files,
)
from parser.python.parser import (
    parse_python_file,
)


def extract_repo_name(
    repo_url: str,
):

    name = repo_url.rstrip("/").split("/")[-1]

    if name.endswith(".git"):
        name = name[:-4]

    return name


def load_chunks(
    repo_path,
):

    chunks = []

    files = list(get_python_files(repo_path))

    print(f"\nPython files found: {len(files)}")

    for file_path in files:
        try:
            file_chunks = parse_python_file(
                path=file_path,
                repo_path=repo_path,
            )

            chunks.extend(file_chunks)

        except Exception as e:
            print("\nFailed parsing:")

            print(file_path)

            print(e)

    return chunks
