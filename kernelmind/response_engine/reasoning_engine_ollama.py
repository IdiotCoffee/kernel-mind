# kernelmind/response_engine/reasoning_engine_ollama.py

from concurrent.futures import ThreadPoolExecutor

from kernelmind.response_engine.ollama_client import generate, generate_stream
from kernelmind.response_engine.prompts import (
    build_summary_prompt,
    build_synthesis_prompt,
)

SUMMARY_MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"
SYNTHESIS_MODEL = "qwen2.5-coder:14b-instruct-q4_K_M"

SUMMARY_MAX_TOKENS = 512
SYNTHESIS_MAX_TOKENS = 4096
SUMMARY_CHUNK_LIMIT = 25


def summarize_chunk(chunk, query):
    prompt = build_summary_prompt(chunk, query)
    result = generate(prompt, model=SUMMARY_MODEL, max_tokens=SUMMARY_MAX_TOKENS)
    return result if result else "[EMPTY SUMMARY]"


def summarize_chunks(chunks, query):
    def process(i_c):
        i, c = i_c
        return {
            "index": i + 1,
            "summary": summarize_chunk(c, query),
            "text": c["text"],
            "path": c["path"],
            "type": c["type"],
            "start": c["start"],
            "end": c["end"],
            "qualified_name": c.get("qualified_name"),
        }

    with ThreadPoolExecutor(max_workers=4) as executor:
        return list(
            executor.map(process, list(enumerate(chunks[:SUMMARY_CHUNK_LIMIT])))
        )


def _summaries_block(sums):
    return "\n".join(
        f"({s['index']}) {s['path']}:{s['start']}-{s['end']} — {s['summary']}"
        for s in sums
    )


def synthesize_answer(query, chunks):
    return "".join(synthesize_answer_stream(query, chunks))


def synthesize_answer_stream(query, chunks):
    if not chunks:
        yield "The retrieved code does not contain the answer."
        return

    summaries = summarize_chunks(chunks, query)
    block = _summaries_block(summaries)

    prompt = build_synthesis_prompt(query, block)

    for token in generate_stream(
        prompt,
        model=SYNTHESIS_MODEL,
        max_tokens=SYNTHESIS_MAX_TOKENS,
    ):
        yield token
