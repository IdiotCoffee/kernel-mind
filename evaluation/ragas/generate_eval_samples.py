import json
import os

from generation.answer_generator import (
    AnswerGenerator,
)
from generation.providers.sarvam_provider import (
    SarvamProvider,
)
from retrieval.pipeline import retrieve_context
from routing.modes import QueryMode


def generate_eval_samples(runtime, benchmark_path, evaluation_mode=False):

    # =====================================================
    # Load Benchmarks
    # =====================================================

    with open(benchmark_path, "r") as f:
        benchmarks = json.load(f)

    # =====================================================
    # Provider + Generator
    # =====================================================

    provider = SarvamProvider(
        api_key=os.getenv("SARVAM_API_KEY", ""),
    )

    generator = AnswerGenerator(
        provider=provider,
    )

    samples = []

    # =====================================================
    # Run Evaluation
    # =====================================================

    for benchmark in benchmarks:
        question = benchmark["question"]

        print("\n================================")
        print(f"QUESTION: {question}")
        print("================================\n")

        # -------------------------------------------------
        # Retrieval
        # -------------------------------------------------

        retrieval_results = retrieve_context(
            query=question,
            runtime=runtime,
            retrieval_top_k=5,
            expansion_depth=2,
            expansion_nodes=25,
            final_top_k=4,
        )

        # -------------------------------------------------
        # Contexts
        # -------------------------------------------------

        contexts = []

        for item in retrieval_results:
            chunk = runtime.chunk_lookup.get(item["fqn"])

            if not chunk:
                continue

            contexts.append(chunk.code)

        # -------------------------------------------------
        # Generation
        # -------------------------------------------------

        answer = generator.generate(
            query=question,
            results=retrieval_results,
            runtime=runtime,
            evaluation_mode=evaluation_mode,
            mode=QueryMode.WORKFLOW,
            confidence={
                "label": "HIGH",
                "score": 0.95,
            },
            stream=False,
        )

        # -------------------------------------------------
        # Safety
        # -------------------------------------------------

        if not isinstance(answer, str):
            answer = str(answer)

        print("\nANSWER:\n")
        print(answer)

        # -------------------------------------------------
        # Sample
        # -------------------------------------------------

        samples.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": benchmark["ground_truth"],
            }
        )

    return samples
