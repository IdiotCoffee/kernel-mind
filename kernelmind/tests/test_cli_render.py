sample_answer = """
**Explanation:**
Requests merges session-level headers using the `merge_setting` function.

**Key Points:**
- Session headers are merged first.
- Request-level headers override them.
- Empty values are removed.

Chunk: src/requests/sessions.py:61-88
"""
sample_chunk = {
    "path": "src/requests/sessions.py",
    "start": 61,
    "end": 88,
    "text": "def merge_setting(requests, session):\n    ...",
}
