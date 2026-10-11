"""The agent loop: model proposes tool calls, tools run, results go back, until the model answers.

The model is an interface (`Model`), so the same loop runs against a real LLM, a scripted
stand-in for tests, or a baseline policy in the evaluation harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .domain import State
from .tools import Toolbox, ToolSpec

SYSTEM_PROMPT = """You are Cadence, a scheduling and habit assistant.
Use the tools; never invent calendar contents or streaks. Call `today` before reasoning about relative dates.
Prefer find_free_slots over guessing a time. Never delete an event unless the user asked for that deletion
in this conversation, and only call delete_event with confirm=true after they have confirmed it.
If a tool returns an error, read it, fix the request, and try again; if you cannot, tell the user plainly.
Keep answers short."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ModelResponse:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class UserMsg:
    text: str


@dataclass(frozen=True)
class AssistantMsg:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class ToolResultMsg:
    call_id: str
    content: dict[str, Any]
    is_error: bool


Message = UserMsg | AssistantMsg | ToolResultMsg


class Model(Protocol):
    def respond(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelResponse: ...


@dataclass
class Trace:
    """What happened during one run; the evaluation harness scores this."""

    final_text: str = ""
    calls: list[tuple[str, dict[str, Any], bool]] = field(default_factory=list)  # (tool, args, is_error)
    steps: int = 0
    hit_step_limit: bool = False


class Agent:
    def __init__(self, model: Model, state: State, max_steps: int = 12) -> None:
        self.model, self.state, self.max_steps = model, state, max_steps
        self.tools = Toolbox(state)
        self.history: list[Message] = []

    def run(self, user_text: str) -> Trace:
        trace = Trace()
        self.history.append(UserMsg(user_text))
        specs = self.tools.specs()
        for _ in range(self.max_steps):
            trace.steps += 1
            resp = self.model.respond(SYSTEM_PROMPT, self.history, specs)
            self.history.append(AssistantMsg(resp.text, resp.tool_calls))
            if not resp.tool_calls:
                trace.final_text = resp.text
                return trace
            for call in resp.tool_calls:
                result, is_error = self.tools.call(call.name, call.args)
                trace.calls.append((call.name, call.args, is_error))
                self.history.append(ToolResultMsg(call.id, result, is_error))
        trace.hit_step_limit = True
        trace.final_text = "I could not finish this within the step limit."
        return trace
