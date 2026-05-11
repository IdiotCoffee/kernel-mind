from generation.answer_generator import AnswerGenerator
from retrieval.pipeline import retrieve_context
from routing.confidence import compute_confidence
from routing.modes import QueryMode
from routing.traces import build_reasoning_trace


class QueryExecutor:
    """
    Central orchestration layer.

    Responsible for:
    - routing execution behavior
    - retrieval strategy selection
    - synthesis mode selection
    """

    def __init__(
        self,
        runtime,
        provider,
    ):

        self.runtime = runtime

        self.answer_generator = AnswerGenerator(
            provider=provider,
        )

    # =====================================================
    # Execute Query
    # =====================================================

    def execute(
        self,
        query: str,
        mode: QueryMode,
    ):

        # =================================================
        # CHAT MODE
        # =================================================

        if mode == QueryMode.CHAT:
            print("Hello! Ask me questions about the repository.\n")

            return

        # =================================================
        # WORKFLOW MODE
        #
        # Deep graph traversal
        # =================================================

        elif mode == QueryMode.WORKFLOW:
            results = retrieve_context(
                query=query,
                runtime=self.runtime,
                retrieval_top_k=8,
                expansion_depth=3,
                expansion_nodes=35,
                final_top_k=15,
            )

            self.stream_answer(
                query=query,
                results=results,
                mode=mode,
            )

            return

        # =================================================
        # SYMBOL LOOKUP MODE
        #
        # Minimal graph expansion
        # =================================================

        elif mode == QueryMode.SYMBOL_LOOKUP:
            results = retrieve_context(
                query=query,
                runtime=self.runtime,
                retrieval_top_k=5,
                expansion_depth=1,
                expansion_nodes=10,
                final_top_k=5,
            )

            self.stream_answer(
                query=query,
                results=results,
                mode=mode,
            )

            return

        # =================================================
        # ARCHITECTURE MODE
        #
        # Broader context exploration
        # =================================================

        elif mode == QueryMode.ARCHITECTURE:
            results = retrieve_context(
                query=query,
                runtime=self.runtime,
                retrieval_top_k=10,
                expansion_depth=2,
                expansion_nodes=40,
                final_top_k=20,
            )

            self.stream_answer(
                query=query,
                results=results,
                mode=mode,
            )

            return

        # =================================================
        # EXISTENCE CHECK MODE
        # =================================================

        elif mode == QueryMode.EXISTENCE_CHECK:
            results = retrieve_context(
                query=query,
                runtime=self.runtime,
                retrieval_top_k=10,
                expansion_depth=2,
                expansion_nodes=25,
                final_top_k=10,
            )

            self.stream_answer(
                query=query,
                results=results,
                mode=mode,
            )

            return

        # =================================================
        # GENERAL QA FALLBACK
        # =================================================

        else:
            results = retrieve_context(
                query=query,
                runtime=self.runtime,
                retrieval_top_k=7,
                expansion_depth=2,
                expansion_nodes=20,
                final_top_k=10,
            )

            self.stream_answer(
                query=query,
                results=results,
                mode=mode,
            )

    # =====================================================
    # Streaming Helper
    # =====================================================

    def stream_answer(
        self,
        query,
        results,
        mode,
    ):

        print(f"\nGenerating {mode.value} response...\n")
        confidence = compute_confidence(results)
        trace = build_reasoning_trace(results)

        print(f"[CONFIDENCE] {confidence['label']} ({confidence['score']})\n")
        if trace:
            print("[REASONING TRACE]\n")

            for i, node in enumerate(trace):
                short = node.split(".")[-1]

                if i < len(trace) - 1:
                    print(f"{short} ->")
                else:
                    print(short)

            print()

        # stream = self.answer_generator.generate(
        #     query=query,
        #     results=results,
        #     runtime=self.runtime,
        #     stream=True,
        #     mode=mode,
        # )
        stream = self.answer_generator.generate(
            query=query,
            results=results,
            runtime=self.runtime,
            stream=True,
            mode=mode,
            confidence=confidence,
        )

        for token in stream:
            print(token, end="", flush=True)

        print("\n")
