import os

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from evaluation.ragas.config import EVAL_CONFIGS


def build_evaluator(config_name: str):

    config = EVAL_CONFIGS[config_name]

    provider = config["provider"]

    # =====================================================
    # OLLAMA
    # =====================================================

    if provider == "ollama":
        # print("using ollama")
        return ChatOllama(
            model=config["model"],
            temperature=config["temperature"],
            num_ctx=config["num_ctx"],
            num_predict=config["num_predict"],
            format="json",
        )

    # =====================================================
    # OPENAI
    # =====================================================

    elif provider == "openai":
        # print("using openai")
        return ChatOpenAI(
            model=config["model"],
            temperature=config["temperature"],
            # max_tokens=config["max_tokens"],
            max_completion_tokens=config["max_tokens"],
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    # =====================================================
    # SARVAM
    # =====================================================

    elif provider == "sarvam":
        # Placeholder for next step
        raise NotImplementedError("Sarvam evaluator wrapper not implemented yet.")

    else:
        raise ValueError(f"Unknown provider: {provider}")
