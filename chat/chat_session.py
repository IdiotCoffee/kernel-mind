import os

from dotenv import load_dotenv

from generation.providers.sarvam_provider import SarvamProvider
from retrieval.pipeline import retrieve_context
from routing.classifier import QueryClassifier
from routing.executor import QueryExecutor

load_dotenv()


def start_chat_session(runtime):

    print(f"\nConnected to: {runtime.repo_id}\nType 'exit' to quit.\n")

    # =====================================================
    # Provider
    # =====================================================

    provider = SarvamProvider(api_key=os.getenv("SARVAM_API_KEY", ""))

    # =====================================================
    # Router
    # =====================================================

    classifier = QueryClassifier()

    # =====================================================
    # Executor
    # =====================================================

    executor = QueryExecutor(
        runtime=runtime,
        provider=provider,
    )

    # =====================================================
    # Chat Loop
    # =====================================================

    while True:
        query = input("\n> ").strip()

        # -------------------------------------------------
        # Exit
        # -------------------------------------------------

        if query.lower() in ["exit", "quit"]:
            print("\nbye bye!\n")

            break

        # -------------------------------------------------
        # Empty Query
        # -------------------------------------------------

        if not query:
            continue

        # -------------------------------------------------
        # Query Classification
        # -------------------------------------------------

        mode = classifier.classify(query)

        print(f"\n[ROUTER] mode = {mode.value}\n")

        # -------------------------------------------------
        # Execute
        # -------------------------------------------------

        executor.execute(
            query=query,
            mode=mode,
        )
