import ollama

SYNTHESIS_TEMPLATE = """
You are a code-understanding model. Your job is to produce a precise, 
grounded explanation using ONLY the retrieved source code chunks.

USER QUESTION:
[QUESTION]

You are given code chunks from a real repository. These chunks are the 
ONLY source of truth. Everything you say MUST be supported by the code below.

=== SOURCE CODE START ===
[CHUNKS]
=== SOURCE CODE END ===

STRICT RULES:

1. Every statement in your answer must be grounded directly in the retrieved 
   code or in a call-chain explicitly visible in the code. 
   Absolutely no external knowledge.

2. NEVER use speculative language.
   Forbidden words: likely, probably, might, maybe, generally, usually, often.

3. When referencing something, ALWAYS cite:
   - the file path
   - the function or class name
   - and line numbers if shown.

4. If the code shows call-chains (A → B → C), explain the flow in that order.
   If only part of the chain is shown, explain only what is visible and 
   logically implied from the calls.

5. NEVER repeat your answer. 
   NEVER restate the same paragraph twice.

6. NEVER describe code that does not appear in the chunks. 
   If something is not present, do NOT invent it.

7. If the retrieved chunks do not contain the answer, say:
      “Not found in retrieved code.”
   and stop.

8. The final answer must be a tightly written explanation for a senior engineer.
   Prefer short, direct paragraphs over long rambling text.

FORMAT TO FOLLOW:

1. One short summary sentence.
2. A step-by-step explanation grounded strictly in the retrieved code.
3. Use bullet points when describing sequences or call-chains.

Begin your answer now.
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
            "max_tokens": 4096,
        }
    )

    # Ollama returns: { "response": "...", ... }
    answer = response.get("response", "").strip()
    return answer


__all__ = ["synthesize_answer"]

