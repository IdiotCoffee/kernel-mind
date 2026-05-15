# evaluation/ragas/configs.py

EVAL_CONFIGS = {
    "qwen_local": {
        "provider": "ollama",
        "model": "qwen2.5:7b",
        "temperature": 0,
        "num_ctx": 16384,
        "num_predict": 1024,
    },
    "sarvam": {
        "provider": "sarvam",
        "model": "sarvam-105b",
        "temperature": 0,
        "max_tokens": 2048,
    },
    "gpt4o_mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 1024,
    },
}
