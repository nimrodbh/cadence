"""Watch the agent loop work, step by step, with no API key.

    python -m cadence.demo

Replays the first evaluation scenario with a scripted stand-in for the model, printing every
message that crosses the loop: what the user said, which tools the "model" asked for, and what
the tools answered. With a real model the loop is identical; only the decisions come from the LLM.
"""

from __future__ import annotations

import json
from datetime import timedelta

from .agent import Agent, AssistantMsg, ToolResultMsg, UserMsg
from .domain import State
from .evals.scenarios import SCENARIOS
from .models import ScriptedModel


def main() -> None:
    sc = SCENARIOS[0]
    state = State()
    sc.setup(state)
    print("Calendar before:")
    for e in state.calendar.between(state.today, state.today + timedelta(days=5)):
        print(f"  {e.start:%a %H:%M}-{e.end:%H:%M}  {e.title}")
    agent = Agent(ScriptedModel(sc.golden), state)
    agent.run(sc.turns[0])
    print()
    for m in agent.history:
        if isinstance(m, UserMsg):
            print(f"USER      {m.text}")
        elif isinstance(m, AssistantMsg):
            for c in m.tool_calls:
                print(f"MODEL ->  call {c.name}({json.dumps(c.args)})")
            if m.text:
                print(f"MODEL     {m.text}")
        elif isinstance(m, ToolResultMsg):
            tag = "ERROR" if m.is_error else "ok"
            print(f"TOOL {tag:5s} {json.dumps(m.content)}")
    print("\nCalendar after:")
    for e in state.calendar.between(state.today, state.today + timedelta(days=5)):
        print(f"  {e.start:%a %H:%M}-{e.end:%H:%M}  {e.title}")


if __name__ == "__main__":
    main()
