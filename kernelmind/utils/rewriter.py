class QueryRewriter:
    def __init__(self):
        from kernelmind.response_engine.engine import ResponseEngine

        self.engine = ResponseEngine()

    def rewrite(self, query: str) -> str:
        prompt = f"""
Rewrite this query into a precise technical question for source-code retrieval.

Keep it short and focused on relevant functions, modules, or classes.
Return ONLY the refined query.

Query: "{query}"

Refined:
"""
        result = self.engine.generate_simple(prompt)

        return result.strip() if result else query
