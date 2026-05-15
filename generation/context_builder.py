from routing.modes import QueryMode


def workflow_filter(results):
    """
    Filter workflow evidence aggressively.

    Goal:
    preserve causal chains
    suppress neighboring auth workflows.
    """

    filtered = []

    for item in results:
        fqn = item["fqn"].lower()

        depth = item.get("depth", 0)

        overlap = item.get("overlap", 0)

        score = item.get(
            "final_score",
            item.get("score", 0),
        )

        noisy_patterns = [
            "recover_password",
            "reset_password",
            "send_email",
            "email",
            "delete_user",
            "delete_item",
            "test_email",
        ]

        if any(p in fqn for p in noisy_patterns):
            continue

        keep = False

        if overlap >= 1.0:
            keep = True

        elif depth <= 1 and score >= 1.0:
            keep = True

        elif score >= 2.5:
            keep = True

        if keep:
            filtered.append(item)

    return filtered


def build_context(
    results,
    runtime,
    mode=None,
    max_chars: int = 12000,
    evaluation_mode: bool = False,
):
    """
    Build structured repository evidence context.

    evaluation_mode=True:
    - shorter contexts
    - less metadata
    - smaller code windows
    - RAGAS-friendly compression
    """

    # =====================================================
    # Evaluation Compression
    # =====================================================

    if evaluation_mode:
        print("Values being truncated for evaluation")
        max_chars = 5000

    # =====================================================
    # Mode-specific filtering
    # =====================================================

    if mode == QueryMode.WORKFLOW:
        results = workflow_filter(results)

    # =====================================================
    # Sort
    # =====================================================

    results = sorted(
        results,
        key=lambda x: (
            x.get("depth", 0),
            -x.get(
                "final_score",
                x.get("score", 0),
            ),
        ),
    )

    sections = []

    total_chars = 0

    seen = set()

    # =====================================================
    # Build Evidence Cards
    # =====================================================

    for item in results:
        chunk = runtime.chunk_lookup.get(item["fqn"])

        if not chunk:
            continue

        if chunk.fqn in seen:
            continue

        seen.add(chunk.fqn)

        section = []

        # =================================================
        # SYMBOL
        # =================================================

        section.append(f"SYMBOL:\n{chunk.fqn}")

        # =================================================
        # TYPE
        # =================================================

        section.append(f"TYPE:\n{chunk.type}")

        # =================================================
        # FILE
        # =================================================

        section.append(f"FILE:\n{chunk.file_path}")

        # =================================================
        # Evaluation Mode
        #
        # Keep contexts compact
        # =================================================

        if evaluation_mode:
            # ---------------------------------------------
            # Minimal metadata
            # ---------------------------------------------

            if "depth" in item:
                section.append(f"DEPTH:\n{item['depth']}")

            # ---------------------------------------------
            # Shortened code window
            # ---------------------------------------------

            code_limit = 500

            section.append(f"CODE:\n{chunk.code[:code_limit]}")

        # =================================================
        # Normal Interactive Mode
        # =================================================

        else:
            # ---------------------------------------------
            # LINES
            # ---------------------------------------------

            section.append(f"LINES:\n{chunk.start_line}-{chunk.end_line}")

            # ---------------------------------------------
            # MODULE
            # ---------------------------------------------

            section.append(f"MODULE:\n{chunk.module}")

            # ---------------------------------------------
            # PARENT
            # ---------------------------------------------

            if chunk.parent_fqn:
                section.append(f"PARENT:\n{chunk.parent_fqn}")

            # ---------------------------------------------
            # CALLS
            # ---------------------------------------------

            if chunk.calls:
                calls_preview = "\n".join(chunk.calls[:8])

                section.append(f"CALLS:\n{calls_preview}")

            # ---------------------------------------------
            # IMPORTS
            # ---------------------------------------------

            if chunk.imports:
                imports_preview = "\n".join(
                    [f"{k} -> {v}" for k, v in list(chunk.imports.items())[:8]]
                )

                section.append(f"IMPORTS:\n{imports_preview}")

            # ---------------------------------------------
            # DOCSTRING
            # ---------------------------------------------

            if chunk.docstring:
                section.append(f"DOCSTRING:\n{chunk.docstring[:300]}")

            # ---------------------------------------------
            # GRAPH INFO
            # ---------------------------------------------

            if "depth" in item:
                section.append(f"GRAPH_DEPTH:\n{item['depth']}")

            if "final_score" in item:
                section.append(f"RETRIEVAL_SCORE:\n{round(item['final_score'], 4)}")

            # ---------------------------------------------
            # CODE
            # ---------------------------------------------

            code_limit = 1500

            section.append(f"CODE:\n{chunk.code[:code_limit]}")

        # =================================================
        # Assemble
        # =================================================

        text = "\n\n" + ("-" * 80) + "\n\n" + "\n\n".join(section)

        # =================================================
        # Budget
        # =================================================

        if total_chars + len(text) > max_chars:
            break

        sections.append(text)

        total_chars += len(text)

    return "\n".join(sections)
