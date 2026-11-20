"""Provider-agnostic LLM interface.

Only an Anthropic client is implemented so far; OpenAI and Ollama clients can be
added by implementing :class:`LLMClient`. API keys come from the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the model."""

    name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass(frozen=True)
class LLMResponse:
    """One model turn: free text and/or tool calls."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Message:
    """A conversation message in a provider-neutral shape."""

    role: str  # "user" | "assistant" | "tool"
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    call_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    """Anything that can continue a conversation with optional tool use."""

    def complete(self, system: str, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse: ...


class AnthropicClient:
    """Claude via the Anthropic Messages API (``pip install anthropic``)."""

    def __init__(self, model: str | None = None, max_tokens: int = 2_000) -> None:
        import anthropic  # type: ignore[import-not-found]

        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self._model = model or os.environ.get("SENTINEL_LLM_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self._max_tokens = max_tokens

    def complete(self, system: str, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse:
        payload: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": message.call_id,
                    "content": message.content,
                }
                # Results for one assistant turn must share a single user message.
                if payload and payload[-1]["role"] == "user" and isinstance(payload[-1]["content"], list):
                    payload[-1]["content"].append(block)
                else:
                    payload.append({"role": "user", "content": [block]})
            elif message.role == "assistant":
                blocks: list[dict[str, Any]] = [{"type": "text", "text": message.content}] if message.content else []
                blocks += [
                    {"type": "tool_use", "id": c.call_id, "name": c.name, "input": c.arguments}
                    for c in message.tool_calls
                ]
                payload.append({"role": "assistant", "content": blocks})
            else:
                payload.append({"role": "user", "content": message.content})
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=payload,
            tools=[
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["parameters"],
                }
                for t in tools
            ],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        calls = tuple(
            ToolCall(name=block.name, arguments=dict(block.input), call_id=block.id)
            for block in response.content
            if block.type == "tool_use"
        )
        return LLMResponse(text, calls, response.usage.input_tokens, response.usage.output_tokens)
