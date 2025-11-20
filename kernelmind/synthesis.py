import ollama

# ===============================
#  STRICT TEMPLATE
# ===============================
SYNTHESIS_TEMPLATE = """
You are a strict code-grounded summarizer for senior engineers.
You will produce a short, precise, fully-grounded answer that uses ONLY the code in === SOURCE CODE START === / === SOURCE CODE END ===.

INPUT:
Question:
[QUESTION]

Source chunks:
[CHUNKS]

INVARIANT RULES (must obey exactly):
1) Use only the code in [CHUNKS]. If the answer cannot be found exactly in those chunks, output exactly the single line:
   Not found in retrieved code.
   and stop (no extra text).
2) Every factual statement must end with a citation in this exact format:
   (file: <path>, <symbol>, lines <start>–<end>)
3) Do not use speculative language. Forbidden: likely, probably, might, maybe, generally, usually, often.
4) No repetition. Each sentence must be unique.
5) Never invent code or behavior not explicitly present.
6) Do not quote code unless it appears verbatim in the chunks.
7) Output MUST be plain text. No markdown fences.

OUTPUT FORMAT:
1) One-line summary sentence (ends with citation).
2) Numbered explanation steps (1–N). Each sentence must end with a citation.
3) No conclusion, no extra sections.

Begin now.
"""

# ===============================
#  Chunk formatting utilities
# ===============================

def format_chunks(chunks):
    """
    chunks: list of dicts with fields:
        text, path, start, end
    """
    out = []
    for c in chunks:
        path = c.get("path", "unknown")
        start = c.get("start", "?")
        end = c.get("end", "?")

        header = f"--- File: {path} ({start}–{end}) ---"
        out.append(header)
        out.append(c["text"])
        out.append("")  # blank line for readability

    return "\n".join(out)


def build_prompt(query, chunks):
    """
    Takes user query + chunk list → builds the full prompt sent to Gemma.
    """
    formatted = format_chunks(chunks)
    prompt = SYNTHESIS_TEMPLATE.replace("[QUESTION]", query)
    prompt = prompt.replace("[CHUNKS]", formatted)
    return prompt


# ===============================
#  Synthesis (Gemma)
# ===============================

def synthesize_answer(query: str, chunks: list, model: str = "gemma2:9b"):
    """
    Uses Gemma through Ollama to synthesize a grounded explanation.
    - deterministic output (temperature 0)
    - no hallucinations
    - inline citations enforced
    """
    prompt = build_prompt(query, chunks)

    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={
            "temperature": 0.2,
            "top_p": 1.0,
            "max_tokens": 4096,
        },
    )

    return response.get("response", "").strip()


__all__ = ["synthesize_answer"]
