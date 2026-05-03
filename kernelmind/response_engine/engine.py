# kernelmind/response_engine/engine.py

from kernelmind.config import load_config
from kernelmind.response_engine.reasoning_engine_ollama import (
    synthesize_answer as ollama_generate,
)
from kernelmind.response_engine.reasoning_engine_ollama import (
    synthesize_answer_stream as ollama_stream,
)
from kernelmind.response_engine.reasoning_engine_openai import (
    synthesize_answer as openai_generate,
)
from kernelmind.response_engine.reasoning_engine_openai import (
    synthesize_answer_stream as openai_stream,
)


class ResponseEngine:
    def __init__(self):
        from kernelmind.response_engine.openai_client import init_client

        config = load_config() or {}
        inference = config.get("inference", {})

        self.mode = inference.get("mode", "local")

        if self.mode == "cloud":
            api_key = inference.get("api_key")

            if not api_key:
                raise RuntimeError("API key not set. Run: km set-api-key")

            init_client(api_key)

    def stream(self, query, chunks):
        if self.mode == "cloud":
            return openai_stream(query, chunks)
        return ollama_stream(query, chunks)

    def generate(self, query, chunks):
        # Prefer direct generate if available (faster than joining stream)
        if self.mode == "cloud":
            return openai_generate(query, chunks)
        return ollama_generate(query, chunks)

    def generate_simple(self, prompt: str) -> str:
        if self.mode == "cloud":
            from kernelmind.response_engine.openai_client import chat

            return chat(prompt)

        else:
            from kernelmind.response_engine.ollama_client import generate

            return generate(prompt, model="qwen2.5-coder:7b")
