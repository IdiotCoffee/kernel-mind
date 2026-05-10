from retrieval.pipeline import retrieve_context


def start_chat_session(runtime):
    """
    Interactive repository chat session.
    """

    print(f"\nConnected to: {runtime.repo_id}")

    print("Type 'exit' to quit.\n")

    while True:
        query = input("> ").strip()

        if not query:
            continue

        if query.lower() == "exit":
            print("\nbye bye!\n")

            break

        # -----------------------------------
        # Full Retrieval Pipeline
        # -----------------------------------

        results = retrieve_context(
            query=query,
            runtime=runtime,
        )

        print("\nTop Results:\n")

        for idx, item in enumerate(
            results,
            start=1,
        ):
            chunk = runtime.chunk_lookup.get(item["fqn"])

            if not chunk:
                continue

            print("=" * 80)

            print(f"[{idx}] {chunk.fqn}")

            print(f"TYPE: {chunk.type}")

            print(f"SCORE: {round(item['score'], 4)}")

            print(f"DEPTH: {item['depth']}")

            print(f"PROPAGATED: {round(item['propagated_score'], 4)}")

            print(f"PATH: {chunk.file_path}")

            print()

            # -----------------------------------
            # Docstring
            # -----------------------------------

            if chunk.docstring:
                print("DOCSTRING:\n")

                print(chunk.docstring)

                print()

            # -----------------------------------
            # Code Preview
            # -----------------------------------

            print("CODE:\n")

            print(chunk.code[:600])

            print()

            # -----------------------------------
            # Calls
            # -----------------------------------

            if item["calls"]:
                print("CALLS:\n")

                for edge in item["calls"][:5]:
                    print(f"  -> {edge.target} [{edge.edge_type}] (w={edge.weight})")

                print()

            # -----------------------------------
            # Called By
            # -----------------------------------

            if item["called_by"]:
                print("CALLED BY:\n")

                for edge in item["called_by"][:5]:
                    print(f"  <- {edge.target} [{edge.edge_type}] (w={edge.weight})")

                print()

        print()
