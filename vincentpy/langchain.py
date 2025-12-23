"""
Reusable LangChain helpers that wrap common invocation patterns.

The functions in this module keep all project specific policy out of the way
so they can be dropped into any script that wants a structured output from an
LLM.  They purposely avoid references to local configuration files and expect
the caller to provide the minimal information needed to run a chat model.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence, Type, TypeVar

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

T = TypeVar("T")

def _sanitize_non_empty_str(value: str, field: str) -> str:
    assert isinstance(value, str), f"{field} must be a string"
    value = value.strip()
    assert value, f"{field} cannot be empty"
    return value


def _sanitize_optional_str(value: Optional[str], field: str) -> Optional[str]:
    if value is None:
        return None
    return _sanitize_non_empty_str(value, field)


def _sanitize_temperature(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    assert isinstance(value, (int, float)), "temperature must be a number"
    assert not math.isnan(value), "temperature cannot be NaN"
    return float(value)


def _sanitize_structured_class(structured_class: Optional[Type[Any]]) -> Optional[Type[Any]]:
    if structured_class is None:
        return None
    assert isinstance(structured_class, type), "structured_class must be a class"
    return structured_class


def _sanitize_include_raw(include_raw: bool) -> bool:
    assert isinstance(include_raw, bool), "include_raw must be a boolean"
    return include_raw


def _sanitize_reasoning(reasoning: Optional[dict[str, Any]], provider: str):
    if reasoning is None:
        return None
    assert provider == "openai", "reasoning is only supported for the openai provider"
    assert isinstance(reasoning, dict), "reasoning must be a dict"

    # Validate that only "effort" key is present
    allowed_keys = {"effort"}
    invalid_keys = set(reasoning.keys()) - allowed_keys
    assert not invalid_keys, f"reasoning dict contains invalid keys: {sorted(invalid_keys)}. Only 'effort' is allowed"

    # Validate effort value if present
    if "effort" in reasoning:
        effort = reasoning["effort"]
        assert isinstance(effort, str), "reasoning['effort'] must be a string"
        allowed_efforts = {"low", "medium", "high"}
        assert effort in allowed_efforts, f"reasoning['effort'] must be one of {sorted(allowed_efforts)}"
        reasoning = {"effort": effort}

    return reasoning


def _sanitize_prompts(prompt_human: str, prompt_system: str) -> tuple[str, str]:
    return (
        _sanitize_non_empty_str(prompt_human, "prompt_human"),
        _sanitize_non_empty_str(prompt_system, "prompt_system"),
    )


def _sanitize_prompt_sequence(prompts: Sequence[str]) -> list[str]:
    assert isinstance(prompts, Sequence), "prompt_humans must be a sequence of strings"
    assert not isinstance(
        prompts, (str, bytes)
    ), "prompt_humans must be a sequence of strings, not a single string"
    return [
        _sanitize_non_empty_str(prompt, f"prompt_humans[{idx}]")
        for idx, prompt in enumerate(prompts)
    ]


def _build_chat_model(
    model: str,
    provider: str,
    *,
    temperature: Optional[float] = None,
    service_tier: Optional[str] = None,
    reasoning: Optional[dict[str, Any]] = None,
):
    model = _sanitize_non_empty_str(model, "model")
    provider = _sanitize_non_empty_str(provider, "provider")
    temperature = _sanitize_temperature(temperature)
    service_tier = _sanitize_optional_str(service_tier, "service_tier")
    reasoning = _sanitize_reasoning(reasoning, provider)

    kwargs: dict[str, Any] = {"model_provider": provider}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if provider == "openai" and service_tier is not None:
        kwargs["service_tier"] = service_tier
    if reasoning is not None:
        kwargs["reasoning"] = reasoning
    return init_chat_model(model=model, **kwargs)


def _build_messages(prompt_human: str, prompt_system: str) -> list:
    """Construct the minimal Human/System message list for LangChain."""
    return [
        SystemMessage(content=prompt_system),
        HumanMessage(content=prompt_human),
    ]


def _wrap_with_structure(
    *,
    model: str,
    provider: str,
    structured_class: Optional[Type[Any]] = None,
    temperature: Optional[float] = None,
    service_tier: Optional[str] = None,
    include_raw: bool = True,
    reasoning: Optional[dict[str, Any]] = None,
):
    """Convert the base chat model into a structured-output chain when requested."""
    llm = _build_chat_model(
        model=model,
        provider=provider,
        temperature=temperature,
        service_tier=service_tier,
        reasoning=reasoning,
    )

    structured_class = _sanitize_structured_class(structured_class)
    include_raw = _sanitize_include_raw(include_raw)

    if structured_class is None:
        return llm
    return llm.with_structured_output(structured_class, include_raw=include_raw)


def invoke(
    prompt_human: str,
    prompt_system: str,
    *,
    model: str,
    provider: str,
    structured_class: Optional[Type[Any]] = None,
    temperature: Optional[float] = None,
    service_tier: Optional[str] = None,
    include_raw: bool = True,
    reasoning: Optional[dict[str, Any]] = None,
) -> T:
    """Execute a single LangChain chat call and return the parsed response."""
    prompt_human, prompt_system = _sanitize_prompts(prompt_human, prompt_system)

    llm = _wrap_with_structure(
        model=model,
        provider=provider,
        structured_class=structured_class,
        temperature=temperature,
        service_tier=service_tier,
        include_raw=include_raw,
        reasoning=reasoning,
    )
    messages = _build_messages(prompt_human, prompt_system)
    return llm.invoke(messages)


def invoke_batch(
    prompt_humans: Sequence[str],
    prompt_system: str,
    *,
    model: str,
    provider: str,
    structured_class: Optional[Type[Any]] = None,
    temperature: Optional[float] = None,
    service_tier: Optional[str] = None,
    include_raw: bool = True,
    reasoning: Optional[dict[str, Any]] = None,
    as_completed: bool = False,
) -> Iterable[T] | list[T]:
    """
    Execute a batch of LangChain chats with a shared system prompt.

    The inputs are processed in parallel wherever the LangChain backend allows.

    When ``as_completed`` is True, returns an iterable that yields results as
    they complete (via ``llm.batch_as_completed``). The exact format depends on
    the LangChain implementation; consult the LangChain docs for details.
    """
    prompt_system = _sanitize_non_empty_str(prompt_system, "prompt_system")
    sanitized_prompts = _sanitize_prompt_sequence(prompt_humans)

    llm = _wrap_with_structure(
        model=model,
        provider=provider,
        structured_class=structured_class,
        temperature=temperature,
        service_tier=service_tier,
        include_raw=include_raw,
        reasoning=reasoning,
    )
    messages_list = [_build_messages(prompt, prompt_system) for prompt in sanitized_prompts]
    if as_completed:
        return llm.batch_as_completed(messages_list)
    return llm.batch(messages_list)


__all__ = [
    "invoke",
    "invoke_batch",
]
