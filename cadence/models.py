"""Model implementations.

`ScriptedModel` replays a fixed list of responses (tests, golden trajectories).
`AnthropicModel` talks to the Anthropic Messages API; it needs `pip install .[llm]`
and ANTHROPIC_API_KEY. The message translation is unit-tested with a fake client;
the live path is exercised by `python -m cadence.evals.run --live`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from typing import Any

from .agent import AssistantMsg, Message, ModelResponse, ToolCall, ToolResultMsg, UserMsg
from .tools import ToolSpec


class ScriptedModel:
    """Each step is either a ModelResponse or a function (messages -> ModelResponse)."""

    def __init__(self, steps: Sequence[ModelResponse | Callable[[list[Message]], ModelResponse]]) -> None:
        self._steps = list(steps)
        self._i = 0

    def respond(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelResponse:
        if self._i >= len(self._steps):
            raise AssertionError("scripted model ran out of steps")
        step = self._steps[self._i]
        self._i += 1
        return step(messages) if callable(step) else step


def to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Translate neutral messages to the Messages API shape.

    Consecutive tool results must be merged into ONE user message of tool_result blocks.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, UserMsg):
            out.append({"role": "user", "content": m.text})
        elif isinstance(m, AssistantMsg):
            blocks: list[dict[str, Any]] = []
            if m.text:
                blocks.append({"type": "text", "text": m.text})
            blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.args} for c in m.tool_calls]
            out.append({"role": "assistant", "content": blocks})
        elif isinstance(m, ToolResultMsg):
            block = {
                "type": "tool_result",
                "tool_use_id": m.call_id,
                "content": json.dumps(m.content),
                "is_error": m.is_error,
            }
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out


class AnthropicModel:
    def __init__(self, model: str = "claude-sonnet-5-5", client: Any = None, max_tokens: int = 1024) -> None:
        if client is None:
            import anthropic

            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("set ANTHROPIC_API_KEY to use the live model")
            client = anthropic.Anthropic()
        self._client, self._model, self._max_tokens = client, model, max_tokens

    def respond(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelResponse:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            tools=[{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools],
            messages=to_anthropic_messages(messages),
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = tuple(ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use")
        return ModelResponse(text, calls)
