# evaluation/ragas/run_ragas_eval.py

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

# =========================================================
# Runtime Construction
# =========================================================
from download.scan_repo import get_python_files

# =========================================================
# RAGAS Helpers
# =========================================================
from evaluation.ragas.build_dataset import (
    build_ragas_dataset,
)
from evaluation.ragas.generate_eval_samples import (
    generate_eval_samples,
)
from evaluation.ragas.provider_factory import (
    build_evaluator,
)
from graph.build_graph import build_graph
from indexing.build_repository import (
    build_repository,
)
from parser.python.parser import (
    parse_python_file,
)

# =========================================================
# CONFIG
# =========================================================

REPO_PATH = "repos/full-stack-fastapi-template"

BENCHMARK_PATH = "evaluation/datasets/auth_workflows.json"

# =========================================================
# Evaluator Selection
# =========================================================

# Options:
# - qwen_local
# - gpt4o_mini
# - sarvam (future)
#
# This controls:
# - RAGAS judge model
# - evaluation backend
# - local/cloud evaluation routing

EVALUATOR_NAME = "qwen_local"

# =========================================================
# LOAD CHUNKS
# =========================================================


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


# =========================================================
# MAIN
# =========================================================


def main():

    # =====================================================
    # Build Runtime
    # =====================================================

    print("\nBuilding runtime for RAGAS...\n")

    chunks = load_chunks(REPO_PATH)

    graph = build_graph(chunks)

    result = build_repository(
        repo_id="ragas_eval_repo",
        chunks=chunks,
        graph=graph,
        device="cuda",
    )

    runtime = result["runtime"]

    print("\nRuntime ready.\n")

    # =====================================================
    # Generate Samples
    # =====================================================

    print("\nGenerating evaluation samples...\n")

    samples = generate_eval_samples(
        evaluation_mode=False,
        runtime=runtime,
        benchmark_path=BENCHMARK_PATH,
    )

    # =====================================================
    # Build Dataset
    # =====================================================

    print("\n======================================")
    print("\nBuilding RAGAS dataset...")

    dataset = build_ragas_dataset(samples)

    # =====================================================
    # Build Evaluator
    # =====================================================

    print(f"\nLoading evaluator: {EVALUATOR_NAME}")

    evaluator_llm = build_evaluator(EVALUATOR_NAME)

    # =====================================================
    # Run RAGAS
    # =====================================================

    print("\nRunning RAGAS evaluation...")

    results = evaluate(
        dataset=dataset,
        run_config=RunConfig(
            timeout=300,
            max_workers=3,
        ),
        metrics=[
            Faithfulness(llm=evaluator_llm),
            # ResponseRelevancy(llm=evaluator_llm),
            # LLMContextPrecisionWithoutReference(llm=evaluator_llm),
            # LLMContextRecall(llm=evaluator_llm),
        ],
    )

    # =====================================================
    # Results
    # =====================================================

    print("\n======================================")

    print("RAGAS RESULTS")

    print("======================================\n")

    print(f"Evaluator: {EVALUATOR_NAME}\n")

    print(results)

    print("\n======================================\n")


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    main()
