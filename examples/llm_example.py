"""
Example script showing how to invoke vincentpy.langchain helpers with structured output.

Set the appropriate environment variables for your LangChain backend
before running (e.g., OPENAI_API_KEY for OpenAI models, ANTHROPIC_API_KEY for Anthropic).
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from vincentpy import langchain as llm


class FactResponse(BaseModel):
    """Simple structured output for interesting facts."""

    topic: str = Field(description="The topic of the fact")
    fact: str = Field(description="An interesting fact")
    confidence: int = Field(
        description="Confidence level from 1-10 (10=very confident)"
    )


# Single query example
config = llm.LangChainConfig(
    coder="gpt-5-nano",
    provider="openai",
    structured_class=FactResponse,
    temperature=None,
    include_raw=False,
)

prompt_system = "You are a knowledgeable assistant providing interesting facts."
prompt_human = "Tell me an interesting fact about Python programming language."

result = llm.invoke(prompt_human, prompt_system, config)
parsed = result

print(f"Topic: {parsed.topic}")
print(f"Fact: {parsed.fact}")
print(f"Confidence: {parsed.confidence}/10")


# Batch query example
print("\n" + "=" * 80)
print("BATCH EXAMPLE: Facts about 4 programming languages")
print("=" * 80 + "\n")

languages = ["Rust", "Go", "TypeScript", "Julia"]
batch_prompts = [
    f"Tell me an interesting fact about {lang} programming language."
    for lang in languages
]

batch_results = llm.invoke_batch(batch_prompts, prompt_system, config)

for lang, result in zip(languages, batch_results):
    print(f"Language: {lang}")
    print(f"Fact: {result.fact}")
    print(f"Confidence: {result.confidence}/10")
    print()


# Plain text example (no structured output)
print("\n" + "=" * 80)
print("PLAIN TEXT EXAMPLE: No structured output class")
print("=" * 80 + "\n")

plain_config = llm.LangChainConfig(
    coder="gpt-5-nano",
    provider="openai",
    temperature=0.7,
)

prompt_system = "You are a helpful assistant."
prompt_human = "Write a haiku about databases."

plain_result = llm.invoke(prompt_human, prompt_system, plain_config)
print(plain_result.content)
