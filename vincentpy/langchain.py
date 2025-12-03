"""
Reusable LangChain helpers that wrap common invocation patterns.

The functions in this module keep all project specific policy out of the way
so they can be dropped into any script that wants a structured output from an
LLM.  They purposely avoid references to local configuration files and expect
the caller to provide the minimal information needed to run a chat model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Type, TypeVar

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

T = TypeVar("T")


@dataclass
class LangChainConfig:
    """
    Lightweight configuration for LangChain chat calls.

    Args:
        coder: Model name (e.g. ``gpt-4o-2024-11-20``).
        provider: Model provider understood by LangChain.
        structured_class: Optional Pydantic class describing the response schema.
        temperature: Optional temperature override.
        include_raw: When True, keep the raw LLM output alongside parsed data.
    """

    coder: str
    provider: str
    structured_class: Optional[Type[Any]] = None
    temperature: Optional[float] = None
    include_raw: bool = True

    def build_chat_model(self):
        """Return a LangChain chat model configured with the stored settings."""
        kwargs: dict[str, Any] = {"model_provider": self.provider}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return init_chat_model(self.coder, **kwargs)


def _build_messages(prompt_human: str, prompt_system: str) -> list:
    """Construct the minimal Human/System message list for LangChain."""
    return [
        SystemMessage(content=prompt_system),
        HumanMessage(content=prompt_human),
    ]


def _wrap_with_structure(config: LangChainConfig):
    """
    Convert the base chat model into a structured-output chain when requested.
    """
    llm = config.build_chat_model()
    if config.structured_class is None:
        return llm
    return llm.with_structured_output(
        config.structured_class, include_raw=config.include_raw
    )


def invoke(prompt_human: str, prompt_system: str, config: LangChainConfig) -> T:
    """
    Execute a single LangChain chat call and return the parsed response.
    """
    llm = _wrap_with_structure(config)
    messages = _build_messages(prompt_human, prompt_system)
    return llm.invoke(messages)


def invoke_batch(
    prompt_humans: Sequence[str],
    prompt_system: str,
    config: LangChainConfig,
    *,
    as_completed: bool = False,
) -> Iterable[T] | list[T]:
    """
    Execute a batch of LangChain chats with a shared system prompt.

    The inputs are processed in parallel wherever the LangChain backend allows.

    When ``as_completed`` is True, yield each response as soon as it finishes,
    which lets callers start processing without waiting for the full batch.
    Returns ``(index, output)`` tuples in that mode to preserve input order.
    """
    llm = _wrap_with_structure(config)
    messages_list = [
        _build_messages(prompt_human, prompt_system) for prompt_human in prompt_humans
    ]
    if as_completed:
        return llm.batch_as_completed(messages_list)
    return llm.batch(messages_list)


__all__ = [
    "LangChainConfig",
    "invoke",
    "invoke_batch",
]
