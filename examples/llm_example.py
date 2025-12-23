"""
Example script showing how to invoke vincentpy.langchain helpers with structured output.

Set the appropriate environment variables for your LangChain backend
before running (e.g., OPENAI_API_KEY for OpenAI models, ANTHROPIC_API_KEY for Anthropic,
FIREWORKS_API_KEY for Fireworks).
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


class ProductivityTip(BaseModel):
    tip: str = Field(description="Concise productivity tip")
    why_it_helps: str = Field(description="Brief rationale for the tip")
    time_to_try: str = Field(description="Rough time commitment to test the tip")


# Single query example
model_kwargs = dict(
    model="gpt-5-nano",
    provider="openai",
    structured_class=FactResponse,
    temperature=None,
    include_raw=False,
)

prompt_system = "You are a knowledgeable assistant providing interesting facts."
prompt_human = "Tell me an interesting fact about Python programming language."

result = llm.invoke(prompt_human, prompt_system, **model_kwargs)
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

batch_results = llm.invoke_batch(batch_prompts, prompt_system, **model_kwargs)

for lang, result in zip(languages, batch_results):
    print(f"Language: {lang}")
    print(f"Fact: {result.fact}")
    print(f"Confidence: {result.confidence}/10")
    print()


# Plain text example (no structured output)
print("\n" + "=" * 80)
print("PLAIN TEXT EXAMPLE: No structured output class")
print("=" * 80 + "\n")

prompt_system = "You are a helpful assistant."
prompt_human = "Write a haiku about databases."

plain_result = llm.invoke(
    prompt_human,
    prompt_system,
    model="gpt-5-nano",
    provider="openai",
    temperature=0.7,
)
print(plain_result.content)


# Fireworks example
print("\n" + "=" * 80)
print("FIREWORKS EXAMPLE: kimi-k2-thinking")
print("=" * 80 + "\n")

fireworks_result = llm.invoke(
    "Give me a concise productivity tip.",
    "You are a concise assistant. Provide a tip, why it helps, and time to try.",
    # Fireworks model ids expect the fully qualified path.
    model="accounts/fireworks/models/kimi-k2-thinking",
    provider="fireworks",
    structured_class=ProductivityTip,
    include_raw=False,  # return parsed Pydantic model instead of a raw dict wrapper
    temperature=0.3,
)
print(f"Tip: {fireworks_result.tip}")
print(f"Why it helps: {fireworks_result.why_it_helps}")
print(f"Time to try: {fireworks_result.time_to_try}")


# GPT-5-mini example without reasoning
print("\n" + "=" * 80)
print("GPT-5-MINI EXAMPLE: Basic usage without reasoning")
print("=" * 80 + "\n")

gpt5_result = llm.invoke(
    prompt_human="What is the capital of France?",
    prompt_system="You are a helpful assistant.",
    model="gpt-5-mini-2025-08-07",
    provider="openai",
)
print(gpt5_result.content)


# GPT-5-mini example with reasoning
print("\n" + "=" * 80)
print("GPT-5-MINI EXAMPLE: Using reasoning with medium effort")
print("=" * 80 + "\n")

gpt5_reasoning_result = llm.invoke(
    prompt_human="Solve this: If a train travels 120 miles in 2 hours, then speeds up and travels 200 miles in the next 2.5 hours, what is the average speed for the entire journey?",
    prompt_system="You are a math tutor. Show your reasoning step by step.",
    model="gpt-5-mini-2025-08-07",
    provider="openai",
    reasoning={"effort": "medium"},
)
print(gpt5_reasoning_result.content)
