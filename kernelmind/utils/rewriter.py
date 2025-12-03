from ollama import Client


class QueryRewriter:
    def __init__(self):
        self.client = Client(host="http://localhost:11434")
        self.model = "qwen2.5-coder:14b"

    def rewrite(self, query: str) -> str:
        prompt = f"""
Rewrite this query into a precise technical question for source-code retrieval.

Keep it short, keep it focused on relevant functions, modules, or classes. Only return this refined query, nothing else

Query: "{query}"

Refined:
"""
        resp = self.client.generate(
            model=self.model,
            prompt=prompt,
        )
        return resp["response"].strip()
