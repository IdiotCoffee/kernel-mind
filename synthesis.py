import ollama

SYNTHESIS_TEMPLATE = """You are an expert Python developer and software maintainer.

The following question was asked:

[QUESTION]

Below are the retrieved source code chunks related to the question.
Use ONLY this code to answer. Do NOT guess.
If the answer is not in the code, say “Not found in retrieved code”.

=== SOURCE CODE START ===
[CHUNKS]
=== SOURCE CODE END ===

Synthesize a clear, correct explanation grounded strictly in the code above.
When referencing code, cite line numbers or specific functions.
Explain the logic, not just quote it.
"""


def format_chunks(chunks):
    """
    chunks is a list of dicts:
        {
            "text": "...",
            "path": "...",
            "start": int,
            "end": int
        }
    """
    out = []
    for c in chunks:
        header = f"--- File: {c.get('path','unknown')} ({c.get('start')}–{c.get('end')}) ---"
        out.append(header)
        out.append(c["text"])
        out.append("")  # newline
    return "\n".join(out)


def build_prompt(query, chunks):
    formatted = format_chunks(chunks)
    prompt = SYNTHESIS_TEMPLATE.replace("[QUESTION]", query)
    prompt = prompt.replace("[CHUNKS]", formatted)
    return prompt


def synthesize_answer(query: str, chunks: list, model: str = "gemma2:9b"):
    """
    Generate the final synthesized answer using Gemma.
    """
    prompt = build_prompt(query, chunks)

    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 1024,
        }
    )

    # Ollama returns: { "response": "...", ... }
    answer = response.get("response", "").strip()
    return answer


__all__ = ["synthesize_answer"]

