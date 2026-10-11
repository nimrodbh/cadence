import json
from datetime import date

from mcp.shared.memory import create_connected_server_and_client_session

from cadence.domain import State
from cadence.mcp_server import build_server
from cadence.tools import Toolbox


async def test_server_lists_exactly_the_toolbox_tools():
    names = set(Toolbox(State()).tool_names())
    server = build_server(State(today=date(2026, 10, 11)))
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        listed = await client.list_tools()
        assert {t.name for t in listed.tools} == names
        sched = next(t for t in listed.tools if t.name == "schedule_event")
        assert set(sched.inputSchema["required"]) == {"title", "start", "duration_min"}


async def test_calling_a_tool_through_mcp_changes_state_and_errors_are_flagged():
    state = State(today=date(2026, 10, 11))
    server = build_server(state)
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        ok = await client.call_tool("schedule_event", {"title": "Focus", "start": "2026-10-12T09:00", "duration_min": 60})
        assert not ok.isError
        assert json.loads(ok.content[0].text)["created"]["title"] == "Focus"
        clash = await client.call_tool("schedule_event", {"title": "X", "start": "2026-10-12T09:30", "duration_min": 30})
        assert clash.isError
    assert len(state.calendar.events) == 1
