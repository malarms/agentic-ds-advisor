
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


def get_llm() -> ChatOpenAI:
    """Create an LLM client using the FreeLLMAPI gateway."""

    api_key = os.getenv("FREELLMAPI_API_KEY")
    base_url = os.getenv(
        "FREELLMAPI_BASE_URL",
        "http://localhost:3001/v1",
    )
    model = os.getenv("FREELLMAPI_MODEL", "auto")

    if not api_key or api_key == "replace_with_your_freellmapi_key":
        raise ValueError(
            "Configure FREELLMAPI_API_KEY in your local .env file."
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
        timeout=60,
        max_retries=2,
    )
