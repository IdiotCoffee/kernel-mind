import ollama

SYNTHESIS_TEMPLATE = """You are a code-understanding model. Your job is to synthesize a clear,
correct answer using ONLY the retrieved code chunks provided to you.
This is the query: [QUESTION]


Below are the retrieved source code chunks related to the question.
Use ONLY this code to answer. Do NOT guess.
If the answer is not in the code, say “Not found in retrieved code”.
=== SOURCE CODE START ===
[CHUNKS]
=== SOURCE CODE END ===
Synthesize a clear, correct explanation grounded strictly in the code above.
When referencing code, ALWAYS cite line numbers or specific functions, along with the file paths.
Explain the logic, not just quote it.
Rules:
1. If the exact function or class is not shown in a chunk, infer it from 
   the call chain shown. For example, if A() calls B(), and B() calls C(),
   and the question refers to behavior caused by C(), you must explain how
   A → B → C leads to that behavior.

2. NEVER say "not shown in the snippet" or "not explicitly provided".
   Always give the best explanation using inference from what IS shown.

3. Avoid hallucinating code that wasn't in the retrieved chunks, but you 
   MAY logically connect the pieces (e.g., “this call implies that X is handled 
   inside Y” if the chain shows that relationship).

4. The final answer should be a concise, direct explanation — not a list 
   of chunks and not a restatement.

5. Treat the user as a senior developer looking for precise understanding 
   of internal mechanics.

You MUST produce a complete explanation even if only part of the chain 
appears in the retrieved chunks."""


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

