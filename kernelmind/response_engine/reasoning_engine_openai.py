# kernelmind/response_engine/reasoning_engine_openai.py

from concurrent.futures import ThreadPoolExecutor

from kernelmind.config import load_config
from kernelmind.response_engine.openai_client import (
    chat,
    chat_stream,
    init_client,
)
from kernelmind.response_engine.prompts import (
    build_summary_prompt,
    build_synthesis_prompt,
)

SUMMARY_CHUNK_LIMIT = 25


def _ensure_client():
    config = load_config() or {}
    inference = config.get("inference", {})
    api_key = inference.get("api_key")
    init_client(api_key)


def summarize_chunk(chunk, query, model="gpt-5-nano"):
    prompt = build_summary_prompt(chunk, query)
    result = chat(
        prompt,
        model=model,
        # temperature=0.0,
        reasoning_effort="medium",
    )
    return result if result else "[EMPTY SUMMARY]"


def summarize_chunks(chunks, query, model="gpt-5-nano"):
    def process(i_c):
        i, c = i_c
        return {
            "index": i + 1,
            "summary": summarize_chunk(c, query, model),
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


def synthesize_answer(query, chunks, model="gpt-5-nano"):
    return "".join(synthesize_answer_stream(query, chunks, model))


def synthesize_answer_stream(query, chunks, model="gpt-5-nano"):
    if not chunks:
        yield "The retrieved code does not contain the answer."
        return

    _ensure_client()

    summaries = summarize_chunks(chunks, query, model)
    block = _summaries_block(summaries)

    prompt = build_synthesis_prompt(query, block)

    for token in chat_stream(
        prompt,
        model=model,
        # temperature=0.0,
        reasoning_effort="medium",
    ):
        yield token
