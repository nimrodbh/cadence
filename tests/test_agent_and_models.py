from types import SimpleNamespace

from cadence.agent import Agent, AssistantMsg, ModelResponse, ToolCall, ToolResultMsg, UserMsg
from cadence.domain import State
from cadence.models import AnthropicModel, ScriptedModel, to_anthropic_messages


def test_loop_runs_tools_then_answers():
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=(ToolCall("a", "today", {}),)),
            ModelResponse(text="It is Sunday."),
        ]
    )
    trace = Agent(model, State()).run("what day is it")
    assert trace.final_text == "It is Sunday."
    assert trace.calls == [("today", {}, False)]
    assert trace.steps == 2


def test_tool_errors_flow_back_to_the_model():
    seen = {}

    def second(messages):
        seen["last"] = messages[-1]
        return ModelResponse(text="sorry")

    model = ScriptedModel([ModelResponse(tool_calls=(ToolCall("a", "nope", {}),)), second])
    trace = Agent(model, State()).run("x")
    assert trace.calls[0][2] is True
    assert isinstance(seen["last"], ToolResultMsg) and seen["last"].is_error


def test_step_limit_stops_a_looping_model():
    loop = ModelResponse(tool_calls=(ToolCall("a", "today", {}),))
    trace = Agent(ScriptedModel([loop] * 5), State(), max_steps=3).run("x")
    assert trace.hit_step_limit and trace.steps == 3


def test_anthropic_translation_merges_tool_results_into_one_user_message():
    msgs = [
        UserMsg("hi"),
        AssistantMsg("checking", (ToolCall("t1", "today", {}), ToolCall("t2", "list_habits", {}))),
        ToolResultMsg("t1", {"today": "2026-10-11"}, False),
        ToolResultMsg("t2", {"error": "boom"}, True),
    ]
    out = to_anthropic_messages(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert [b["type"] for b in out[1]["content"]] == ["text", "tool_use", "tool_use"]
    results = out[2]["content"]
    assert [b["tool_use_id"] for b in results] == ["t1", "t2"]
    assert [b["is_error"] for b in results] == [False, True]


def test_anthropic_model_parses_response_with_fake_client():
    class FakeMessages:
        def create(self, **kw):
            assert kw["tools"][0]["input_schema"]["type"] == "object"
            blocks = [
                SimpleNamespace(type="text", text="Looking."),
                SimpleNamespace(type="tool_use", id="x1", name="today", input={}),
            ]
            return SimpleNamespace(content=blocks)

    client = SimpleNamespace(messages=FakeMessages())
    from cadence.tools import Toolbox

    resp = AnthropicModel(client=client).respond("sys", [UserMsg("hi")], Toolbox(State()).specs())
    assert resp.text == "Looking." and resp.tool_calls == (ToolCall("x1", "today", {}),)
