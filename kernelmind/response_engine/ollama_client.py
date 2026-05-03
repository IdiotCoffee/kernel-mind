# kernelmind/response_engine/ollama_client.py

import re

import ollama

DEFAULT_TEMPERATURE = 0


def _strip(text):
    if not text:
        return text
    text = re.sub(r"^```(?:\w+)?\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text.strip().strip("`").strip()


def generate(prompt, model, max_tokens=1024):
    options = {"temperature": DEFAULT_TEMPERATURE}
    if max_tokens is not None:
        options["num_predict"] = max_tokens

    resp = ollama.generate(
        model=model,
        prompt=prompt,
        options=options,
    )
    return _strip(resp.get("response", ""))


def generate_stream(prompt, model, max_tokens=None):
    options = {"temperature": DEFAULT_TEMPERATURE}
    if max_tokens is not None:
        options["num_predict"] = max_tokens

    stream = ollama.generate(
        model=model,
        prompt=prompt,
        stream=True,
        options=options,
    )

    for chunk in stream:
        token = chunk.get("response", "")
        if token:
            yield token
