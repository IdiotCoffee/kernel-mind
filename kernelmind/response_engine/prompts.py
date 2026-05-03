def build_summary_prompt(chunk, query):
    return f"""
TASK:
Extract ONLY relevant behavior from the code chunk for the given query.

QUERY:
{query}

CONSTRAINTS:
- Use ONLY what is explicitly visible in the code.
- NO assumptions. NO inferred behavior.
- NO explanations beyond the code.
- Identify:
  • functions / methods
  • data structures
  • execution flow
- Include ALL referenced functions.
- Minimum 2 sentences if logic exists.

FORMAT:
(file_path:start-end) → summary

CHUNK:
path: {chunk.get("path")}
qualified: {chunk.get("qualified_name")}
type: {chunk.get("type")}
lines: {chunk.get("start")}-{chunk.get("end")}

CODE:
{chunk.get("text")}
"""


def build_synthesis_prompt(query, summaries_block):
    return f"""
ROLE:
You are tracing real code execution paths.

QUERY:
{query}

EVIDENCE:
{summaries_block}

RULES:
- Use ONLY evidence provided
- DO NOT hallucinate
- DO NOT contradict summaries
- If insufficient info → say exactly:
  "The retrieved code does not contain the answer."

THINKING:
Focus on execution flow, function interaction, and data movement.
Avoid general explanation.

OUTPUT STRUCTURE:

Answer:
(3–6 sentences, precise, mechanism-focused)

Key Points:
- Bullet points (3–6)
- Each must reference actual behavior
- Include ordering / flow if present
- MUST cite like: (file:line-line)

STYLE:
Direct. Technical. No filler.
"""
