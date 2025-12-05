import ollama
import json
import re

# ======================================================
#  CONFIG
# ======================================================
DEFAULT_MODEL = "qwen2.5-coder:14b"

# summary generation constraints
SUMMARY_CHUNK_LIMIT = 30
SUMMARY_MAX_TOKENS = 1024
SYNTHESIS_MAX_TOKENS = 5500
DEFAULT_TEMPERATURE = 0


# ======================================================
#  HELPERS
# ======================================================

def _strip(text):
    """Remove accidental backticks or code fences."""
    if not text:
        return text
    text = re.sub(r"^```(?:\w+)?\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text.strip().strip("`").strip()


# ======================================================
#  PROMPTS
# ======================================================

SUMMARY_PROMPT = """
YOU ARE TRYING TO FIND AND SUMMARIZE INFORMATION IN THE CHUNK PERTAINING TO THIS QUERY: {query}
------------------------------------------------------------
Rules:
- No explanations or interpretation beyond what the chunk literally shows.
- No assumptions about behavior not visible in the snippet.
- Identify the key operations, key functions/methods called, and key data structures touched.
- Keep summary AS SHORT AS YOU CAN WITHOUT REMOVING ANY DETAILS.
- EVEN IF the chunk below is large, mention ALL the functions used, and their usage summary in 2 sentences MINIMUM.
- MENTION ALL THE FUNCTIONS / METHODS / CLASSES that are being used, and the flow that is evident from the given information ONLY.
- CRITICAL: DO NOT make up your own logic for explaining the chunk. What is given in the chunk is your ONE SOURCE OF TRUTH.
- CRITICAL: When you summarize the chunk, use the file and line range format exactly like this: (src/requests/sessions.py:500-591).
Chunk:
path: {path}
qualified: {qualified}
type: {ctype}
lines: {start}-{end}

Code:
{code}"""

SYNTHESIS_PROMPT = """
You are an expert code-reasoning assistant.
Your job is to resolve this query with a precise, technically confident explanation that sounds like someone who has actually traced the code path. The answer should be concise but show real understanding of how the mechanisms work.
QUERY:
{query}

CONTEXT (summaries of relevant code chunks):
{summaries}

RULES:
Use only the RELEVANT information from the summaries - DO NOT ADD THE INFORMATION THAT DOES NOT HELP ANSWER THE QUERY.
ADD INFORMATION THAT ADDS MORE CONTEXT TO THE DIRECT ANSWER, EVEN IF IT DOES NOT DIRECTLY ANSWER THE QUERY.
If the summaries do not contain enough information, say:
The retrieved code does not contain the answer.
CRITICAL: DO NOT make up your own information / nagate the information given in the summarries.
Your answer must follow this structure:
A short, crisp explanation (3–6 sentences) that shows clear understanding of how the code achieves the behavior.
A “Key Points” section with 3–6 bullets. Each bullet must:
Reference the actual mechanism in the summaries
Show priority/order/merge logic when relevant
Whenever you cite support, STRICTLY use the file and line range format exactly like this: (src/requests/sessions.py:500-591).

Tone:
Confident, clear, technically aware.
Not verbose, not hand-wavy.
Assume the reader is preparing for a technical interview."""


# ======================================================
#  CHUNK SUMMARIZATION
# ======================================================

def summarize_chunk(chunk, query, model=DEFAULT_MODEL):
    prompt = SUMMARY_PROMPT.format(
        query=query,
        path=chunk.get("path"),
        qualified=chunk.get("qualified_name"),
        ctype=chunk.get("type"),
        start=chunk.get("start"),
        end=chunk.get("end"),
        code=chunk.get("text"),
    )
    # print(prompt)
    try:
        resp = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0, "max_tokens": SUMMARY_MAX_TOKENS},
        )
        clean = resp.get("response", "").strip()
        # print(clean)
        if clean:
            return clean
    except:
        pass
    return chunk

  
def summarize_chunks(chunks, query, model=DEFAULT_MODEL):
    """Summarize up to SUMMARY_CHUNK_LIMIT chunks."""
    out = []
    for i, c in enumerate(chunks[:SUMMARY_CHUNK_LIMIT], 1):
        s = summarize_chunk(c, query)
        out.append({
            "index": i,
            "summary": s,
            "text": c.get("text"),
            "path": c.get("path"),
            "type": c.get("type"),
            "start": c.get("start"),
            "end": c.get("end"),
            "qualified_name": c.get("qualified_name"),
        })
    # print(out)
    return out


def _summaries_block(sums):
    lines = []
    for s in sums:
        lines.append(
            f"({s['index']}) {s['path']}:{s['start']}-{s['end']} — {s['summary']}"
        )
    return "\n".join(lines)


def _remove_full_duplication(text):
    half = len(text) // 2
    if text[:half].strip() == text[half:].strip():
        return text[:half].strip()
    return text

def _chunks_block(sums):
    out = []
    for s in sums:
        body = s["text"]
        if len(body) > 5000:
            body = body[:5000] + " ...<truncated>..."
        out.append(
            f"({s['index']}) {s['path']}:{s['start']}-{s['end']}\n{body}\n"
        )
    return "\n".join(out)


# ======================================================
#  FINAL SYNTHESIS — NO CLASSIFIER — ONLY SUMMARIES → ANSWER
# ======================================================

def synthesize_answer(query, chunks, model=DEFAULT_MODEL):
    if not chunks:
        return "The retrieved code does not contain the answer."

    # Step 1: summarization
    print(chunks)
    print("-----------------------------------------------------------------")
    summaries = summarize_chunks(chunks, query, model)
    print(summaries)
    print("-----------------------------------------------------------------")

    # Step 2: synthesis
    prompt = SYNTHESIS_PROMPT.format(
        query=query,
        summaries=_summaries_block(summaries),
    )
    #print(prompt)
    resp = ollama.generate(
        model=model,
        prompt=prompt,
        options={"temperature": DEFAULT_TEMPERATURE, "max_tokens": SYNTHESIS_MAX_TOKENS},
    )

    raw = _strip(resp.get("response", ""))
    # run dedupe cleaners
    clean = _remove_full_duplication(raw)
    return clean
