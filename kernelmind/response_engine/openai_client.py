# kernelmind/response_engine/openai_client.py

import re
from typing import Literal, Optional

from openai import OpenAI

client: Optional[OpenAI] = None


def init_client(api_key):
    global client
    client = OpenAI(api_key=api_key)


def _strip(text):
    if not text:
        return text
    text = re.sub(r"^```(?:\w+)?\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text.strip().strip("`").strip()


def _ensure_client():
    if client is None:
        raise RuntimeError("OpenAI client not initialized")


def chat(
    prompt,
    model="gpt-5-nano",
    # temperature=0.0,
    reasoning_effort: Literal["low", "medium", "high"] = "medium",
):
    _ensure_client()

    resp = client.chat.completions.create(
        model=model,
        # temperature=temperature,
        reasoning_effort=reasoning_effort,
        messages=[{"role": "user", "content": prompt}],
    )
    return _strip(resp.choices[0].message.content)


def chat_stream(
    prompt,
    model="gpt-5-nano",
    # temperature=0.0,
    reasoning_effort: Literal["low", "medium", "high"] = "medium",
):
    _ensure_client()

    stream = client.chat.completions.create(
        model=model,
        # temperature=temperature,
        reasoning_effort=reasoning_effort,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta

        if hasattr(delta, "content") and delta.content:
            yield delta.content
