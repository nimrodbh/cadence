# cadence

A scheduling and habit agent, built to be inspected. The agent loop, the tool layer, an MCP server and an evaluation harness are about 700 lines of Python with no agent framework. Calendar and habit data is synthetic; nothing here talks to a real calendar.

```
user ──> Agent loop ──> Model (interface) ──> tool calls ──> Toolbox ──> calendar / habits
                                                                 ▲
                                          MCP server (same tools) ┘
```

## Why it is built this way

- **One source of truth for tools.** `Toolbox` methods are typed Python functions. Their signatures generate the JSON schemas the model sees, and the same methods are registered on the MCP server, so the two interfaces cannot drift (`tests/test_mcp.py` checks the tool lists match).
- **Errors are data, not crashes.** A bad date, a double booking or a missing argument returns a message the model can read and fix. Nothing raises out of `Toolbox.call`.
- **Destructive actions need consent in the tool, not only in the prompt.** `delete_event` refuses without `confirm=true`, and a truthy string like `"yes"` does not count.
- **The model is an interface.** The loop runs against a real LLM (`AnthropicModel`), a scripted replay, or anything else with a `respond` method.
- **Domain logic is pure.** Overlap detection, free-slot search and streak counting have no I/O; a property test checks that returned free slots never overlap an event.

## Evaluation harness

Five scenarios score the agent by final state and tool-call trace, never by wording:

| Scenario | What it catches |
|---|---|
| `schedule_in_real_free_time` | Guessing a time instead of searching for a gap that fits |
| `recover_from_conflict` | Giving up after the first tool error |
| `confirm_before_deleting` | Deleting before the user confirmed (two-turn scenario) |
| `log_habit_and_report_streak` | Claiming a streak without logging, or reporting the wrong number |
| `do_not_invent_events` | Answering about the calendar without looking, or making things up |

```bash
python -m cadence.evals.run            # offline
python -m cadence.evals.run --live     # real model: pip install ".[llm]" and set ANTHROPIC_API_KEY
```

Offline, every scenario runs a golden trajectory (must pass) and a deliberately bad one (must be caught), so a check that accepts everything fails the build. **The `--live` mode is implemented but has not been run yet**; no results from a real model are claimed here.

## Run it

```bash
pip install -e ".[dev]"
pytest -q                              # 31 tests
python -m cadence.mcp_server           # MCP server on stdio
```

To use it from an MCP client such as Claude Desktop, add a server entry that runs `python -m cadence.mcp_server`. State lives in memory for the life of the process.

## Not done yet

- Live evaluation results against a real model.
- Persistence (state resets when the process ends).
- A command-line chat front end.
