"""
Example script showing how to invoke vincentpy.langchain helpers.

Set the appropriate environment variables for your LangChain backend
before running (e.g., OPENAI_API_KEY for OpenAI models).
"""

from __future__ import annotations

from vincentpy import langchain as llm


def main() -> None:
    config = llm.LangChainConfig(
        coder="gpt-4o-mini",
        provider="openai",
        temperature=0.2,
        include_raw=True,
    )
    prompt_system = "You are a concise research assistant."
    prompt_human = "List two surprising facts about DuckDB."

    try:
        response = llm.invoke_langchain(prompt_human, prompt_system, config)
    except Exception as exc:  # pragma: no cover - network failure helper
        print("LangChain invocation failed. Ensure credentials are configured.")
        print(exc)
        return

    print("LLM response:")
    print(response)


if __name__ == "__main__":
    main()
