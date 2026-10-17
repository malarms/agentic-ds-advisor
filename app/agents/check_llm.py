
from app.agents.llm_client import get_llm


def main():
    llm = get_llm()

    response = llm.invoke(
        "Reply with exactly: LLM connection successful."
    )

    print("\nLLM response:")
    print(response.content)


if __name__ == "__main__":
    main()
