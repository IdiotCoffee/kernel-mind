from generation.context_builder import build_context
from routing.modes import QueryMode


class AnswerGenerator:
    def __init__(self, provider):

        self.provider = provider

    def append_sources(
        self,
        answer,
        results,
        runtime,
    ):
        """
        Append deterministic source citations.
        """

        seen = set()

        citations = []

        for item in results[:8]:
            chunk = runtime.chunk_lookup.get(item["fqn"])

            if not chunk:
                continue

            key = (
                chunk.file_path,
                chunk.start_line,
            )

            if key in seen:
                continue

            seen.add(key)

            citations.append(
                (
                    f"- "
                    f"[{chunk.file_path}]"
                    f"(source://{chunk.file_path}"
                    f"#L{chunk.start_line})"
                )
            )

        if not citations:
            return answer

        citation_block = "\n".join(citations)

        return f"{answer}\n\n## Sources\n\n{citation_block}"

    # =====================================================
    # Prompt Selection
    # =====================================================

    def build_prompt(
        self,
        query,
        context,
        mode,
        confidence,
    ):

        # =================================================
        # WORKFLOW MODE
        # =================================================

        if mode == QueryMode.WORKFLOW:
            return f"""
            Retrieval Confidence:
            {confidence["label"]}
            ({confidence["score"]})
You are a repository workflow reasoning assistant.

Answer ONLY using the provided repository context.

Your job is to explain:
- execution flow
- causality
- function interactions
- workflow ordering

IMPORTANT RULES:

- Mention exact function names when relevant
- Mention file paths when relevant
- Mention approximate line ranges when relevant
- Explain caller/callee relationships
- Explain what happens first and what happens next
- Stay grounded in retrieved evidence
- Do NOT hallucinate repository structure

If evidence is weak or incomplete,
say so clearly.

Question:
{query}

Repository Context:
{context}

Grounded Workflow Explanation:
"""

        # =================================================
        # SYMBOL LOOKUP MODE
        # =================================================

        elif mode == QueryMode.SYMBOL_LOOKUP:
            return f"""
            Retrieval Confidence:
            {confidence["label"]}
            ({confidence["score"]})
You are a repository symbol lookup assistant.

Answer ONLY using the provided repository context.

Your job is to identify:
- exact implementation locations
- symbol definitions
- file/module placement
- direct references

IMPORTANT RULES:

- Mention exact file paths
- Mention line ranges when available
- Mention direct callers if relevant
- Mention implementation purpose briefly
- Prefer precise grounding over broad explanations
- Do NOT hallucinate symbols or files

Question:
{query}

Repository Context:
{context}

Grounded Symbol Explanation:
"""

        # =================================================
        # EXISTENCE CHECK MODE
        # =================================================

        elif mode == QueryMode.EXISTENCE_CHECK:
            return f"""
            Retrieval Confidence:
            {confidence["label"]}
            ({confidence["score"]})
You are a repository capability verification assistant.

Answer ONLY using the provided repository context.

Your job is to determine whether
a capability appears to exist in the repository.

IMPORTANT RULES:

- Do NOT assume functionality exists
- Distinguish direct evidence from weak evidence
- Mention supporting files/functions when relevant
- Mention uncertainty explicitly when evidence is weak
- Prefer cautious reasoning over speculation
- Do NOT hallucinate integrations or providers

Question:
{query}

Repository Context:
{context}

Grounded Capability Assessment:
"""

        # =================================================
        # ARCHITECTURE MODE
        # =================================================

        elif mode == QueryMode.ARCHITECTURE:
            return f"""
            Retrieval Confidence:
            {confidence["label"]}
            ({confidence["score"]})
You are a repository architecture reasoning assistant.

Answer ONLY using the provided repository context.

Your job is to explain:
- repository structure
- subsystem organization
- module relationships
- component responsibilities

IMPORTANT RULES:

- Focus on modules/packages over individual functions
- Mention important files and directories
- Explain structural relationships
- Group related components together
- Avoid overly detailed workflow narration
- Stay grounded in repository evidence

Question:
{query}

Repository Context:
{context}

Grounded Architecture Explanation:
"""

        # =================================================
        # GENERAL QA
        # =================================================

        return f"""
        Retrieval Confidence:
        {confidence["label"]}
        ({confidence["score"]})
You are a repository reasoning assistant.

Answer ONLY using the provided repository context.

IMPORTANT RULES:

- Stay grounded in retrieved evidence
- Mention files/functions when relevant
- Do NOT hallucinate repository structure
- Say clearly when evidence is insufficient

Question:
{query}

Repository Context:
{context}

Grounded Explanation:
"""

    # =====================================================
    # Generate
    # =====================================================

    def generate(
        self,
        query,
        results,
        runtime,
        mode,
        confidence,
        stream=True,
        evaluation_mode=False,
    ):

        context = build_context(
            results=results, runtime=runtime, mode=mode, evaluation_mode=evaluation_mode
        )

        prompt = self.build_prompt(
            query=query,
            context=context,
            mode=mode,
            confidence=confidence,
        )

        # -------------------------------------------------
        # OPTIONAL DEBUG
        # -------------------------------------------------

        # print("\n========== PROMPT ==========\n")
        # print(prompt[:12000])
        # print("\n============================\n")

        return self.provider.generate(
            prompt=prompt,
            stream=stream,
        )
