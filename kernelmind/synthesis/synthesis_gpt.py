from openai import OpenAI
import re

client = None

def _print_usage(usage, label=""):
    # Comment out this line for normal use
    print(f"[{label}] prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")


def init_client(api_key):
    global client
    client = OpenAI(api_key=api_key)

def _strip(text):
    if not text:
        return text
    text = re.sub(r"^```(?:\w+)?\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text.strip().strip("`").strip()

def summarize_chunk_gpt(chunk, query, model="gpt-4o-mini"):
    prompt = f"""
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
path: {chunk.get('path')}
qualified: {chunk.get('qualified_name')}
type: {chunk.get('type')}
lines: {chunk.get('start')}-{chunk.get('end')}

Code:
{chunk.get('text')}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    # Print token stats (optional)
    if hasattr(resp, "usage"):
        _print_usage(resp.usage, label="summary")
    return _strip(resp.choices[0].message.content)

def summarize_chunks_gpt(chunks, query, model="gpt-4o-mini"):
    out = []
    for i, c in enumerate(chunks, 1):
        s = summarize_chunk_gpt(c, query, model)
        out.append({
            "index": i,
            "summary": s,
            "text": c["text"],
            "path": c["path"],
            "type": c["type"],
            "start": c["start"],
            "end": c["end"],
            "qualified_name": c.get("qualified_name"),
        })
    return out

def _summaries_block(sums):
    lines = []
    for s in sums:
        lines.append(
            f"({s['index']}) {s['path']}:{s['start']}-{s['end']} — {s['summary']}"
        )
    return "\n".join(lines)

def synthesize_answer_gpt(query, chunks, model="gpt-4o-mini"):
    if not chunks:
        return "The retrieved code does not contain the answer."

    summaries = summarize_chunks_gpt(chunks, query, model)

    synthesis_prompt = f"""
You are an expert code-reasoning assistant.
Your job is to resolve this query with a precise, technically confident explanation that sounds like someone who has actually traced the code path. The answer should be concise but show real understanding of how the mechanisms work.
QUERY:
{query}

CONTEXT (summaries of relevant code chunks):
{_summaries_block(summaries)}

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
Assume the reader is preparing for a technical interview.
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": synthesis_prompt}],
    )

    # Token usage (optional debug)
    if hasattr(resp, "usage"):
        print(f"[synthesis] prompt={resp.usage.prompt_tokens}, completion={resp.usage.completion_tokens}, total={resp.usage.total_tokens}")
    return _strip(resp.choices[0].message.content)
