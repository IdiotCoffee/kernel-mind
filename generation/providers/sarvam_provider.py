# generation/providers/sarvam_provider.py

from typing import Iterator

from sarvamai import SarvamAI

from generation.providers.base import LLMProvider

SYSTEM_PROMPT = """
You are a repository reasoning assistant.

Answer ONLY using the provided repository context.

Do not hallucinate APIs or workflows.
"""


class SarvamProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "sarvam-m"):

        self.client = SarvamAI(api_subscription_key=api_key)

        self.model = model

    def generate(self, prompt: str, stream: bool = False):

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions(  # type: ignore
            messages=messages, model=self.model, stream=stream, reasoning_effort=None
        )

        if stream:
            return self._stream_response(response)

        return response.choices[0].message.content

    def _stream_response(self, response) -> Iterator[str]:

        for chunk in response:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if not delta:
                continue

            content = delta.content

            if content:
                yield content
