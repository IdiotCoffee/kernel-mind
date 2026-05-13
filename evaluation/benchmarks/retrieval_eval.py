import json
from pathlib import Path
from typing import Set

# from runtime.loader import load_runtime
from download.scan_repo import get_python_files
from graph.build_graph import build_graph
from indexing.build_repository import build_repository
from parser.python.parser import parse_python_file
from retrieval.pipeline import retrieve_context

# =========================================================
# CONFIG
# =========================================================

DATASET_PATH = "evaluation/datasets/auth_workflows.json"

REPO_NAME = "full-stack-fastapi-template-master"

TOP_K = 4

# =========================================================
# LOAD RUNTIME
# =========================================================
print("\nLoading repository runtime...\n")


REPO_PATH = "repos/full-stack-fastapi-template"


def load_chunks(repo_path):

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


print("\nBuilding runtime for evaluation...\n")

chunks = load_chunks(REPO_PATH)

graph = build_graph(chunks)

result = build_repository(
    repo_id="eval_repo",
    chunks=chunks,
    graph=graph,
    device="cuda",
)

runtime = result["runtime"]

# print("\nRuntime ready.\n")
# print("\n==============================")
# print("ALL INDEXED CHUNKS")
# print("==============================\n")

# for fqn in sorted(runtime.chunk_lookup.keys()):
#     print(fqn)

# print("\n==============================\n")


# =========================================================
# METRICS
# =========================================================


def compute_precision(
    retrieved: Set[str],
    expected: Set[str],
) -> float:

    if not retrieved:
        return 0.0

    relevant = retrieved.intersection(expected)

    return len(relevant) / len(retrieved)


def compute_recall(
    retrieved: Set[str],
    expected: Set[str],
) -> float:

    if not expected:
        return 0.0

    relevant = retrieved.intersection(expected)

    return len(relevant) / len(expected)


# =========================================================
# RETRIEVAL
# =========================================================


def run_retrieval(
    question: str,
):
    """
    Runs your FULL retrieval pipeline.

    Pipeline:
        hybrid retrieval
        → graph expansion
        → graph ranking
        → cross-encoder reranking
    """

    results = retrieve_context(
        query=question,
        runtime=runtime,
        retrieval_top_k=5,
        expansion_depth=2,
        expansion_nodes=25,
        final_top_k=TOP_K,
    )

    retrieved_fqns = []

    for item in results:
        fqn = item["fqn"]

        retrieved_fqns.append(fqn)

    return retrieved_fqns


# =========================================================
# EVALUATION
# =========================================================


def evaluate_retrieval():

    dataset_path = Path(DATASET_PATH)

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    all_precisions = []
    all_recalls = []

    print("\n======================================")
    print("KERNELMIND RETRIEVAL EVALUATION")
    print("======================================\n")

    for sample in dataset:
        question = sample["question"]

        expected_chunks = set(sample["expected_chunks"])

        retrieved_chunks = set(run_retrieval(question))
        print("\nRETRIEVED FQNS:")
        for chunk in retrieved_chunks:
            print(chunk)

        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------

        precision = compute_precision(
            retrieved_chunks,
            expected_chunks,
        )

        recall = compute_recall(
            retrieved_chunks,
            expected_chunks,
        )

        all_precisions.append(precision)
        all_recalls.append(recall)

        # -------------------------------------------------
        # OUTPUT
        # -------------------------------------------------

        print(f"QUESTION:\n{question}\n")

        print("EXPECTED CHUNKS:")
        for chunk in sorted(expected_chunks):
            print(f"  - {chunk}")

        print("\nRETRIEVED CHUNKS:")
        for chunk in sorted(retrieved_chunks):
            print(f"  - {chunk}")

        print("\nMETRICS:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")

        print("\n--------------------------------------\n")

    # =====================================================
    # FINAL SCORES
    # =====================================================

    avg_precision = sum(all_precisions) / len(all_precisions)

    avg_recall = sum(all_recalls) / len(all_recalls)

    print("\n======================================")
    print("FINAL RESULTS")
    print("======================================\n")

    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average Recall:    {avg_recall:.3f}")

    print("\n======================================\n")


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    evaluate_retrieval()
