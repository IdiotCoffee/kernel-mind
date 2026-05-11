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


# This is explain workflow flow - a user can ask any type of question, will my architecture hold up for that? I will also need to add something like a "should I evel look at the code for this query?" handler - I dont want to respond to a "hi" with this.
# Also i'll need the confidence stuff...  low confidence actually means the functionality may not exist in the codebase?
