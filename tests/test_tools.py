from datetime import date

from cadence.domain import State
from cadence.tools import Toolbox


def box():
    return Toolbox(State(today=date(2026, 10, 11)))


def test_schemas_are_generated_from_signatures():
    specs = {s.name: s for s in box().specs()}
    sch = specs["schedule_event"].input_schema
    assert sch["required"] == ["title", "start", "duration_min"]
    assert sch["properties"]["duration_min"] == {"type": "integer"}
    assert specs["find_free_slots"].input_schema["required"] == ["day", "duration_min"]
    assert specs["delete_event"].input_schema["properties"]["confirm"] == {"type": "boolean"}
    assert "Destructive" in specs["delete_event"].description
    assert "call" not in specs and "specs" not in specs


def test_happy_path_schedule_and_list():
    tb = box()
    res, err = tb.call("schedule_event", {"title": "Focus", "start": "2026-10-12T09:00", "duration_min": 45})
    assert not err and res["created"]["end"] == "2026-10-12T09:45"
    res, err = tb.call("list_events", {"first_day": "2026-10-12", "last_day": "2026-10-12"})
    assert [e["title"] for e in res["events"]] == ["Focus"]


def test_errors_are_returned_not_raised():
    tb = box()
    tb.call("schedule_event", {"title": "A", "start": "2026-10-12T09:00", "duration_min": 60})
    cases = [
        ("schedule_event", {"title": "B", "start": "2026-10-12T09:30", "duration_min": 30}, "conflicts"),
        ("schedule_event", {"title": "B", "start": "tomorrow 9am", "duration_min": 30}, "YYYY-MM-DDTHH:MM"),
        ("schedule_event", {"title": "B", "start": "2026-10-12T11:00"}, "missing"),
        ("schedule_event", {"title": "B", "start": "2026-10-12T11:00", "duration_min": "30"}, "integer"),
        ("schedule_event", {"title": "B", "start": "2026-10-12T11:00", "duration_min": 30, "room": "x"}, "unexpected"),
        ("nope", {}, "unknown tool"),
        ("move_event", {"event_id": 99, "new_start": "2026-10-12T11:00"}, "no event"),
        ("log_habit", {"name": "ghost"}, "no habit"),
    ]
    for name, args, needle in cases:
        res, err = tb.call(name, args)
        assert err and needle in res["error"], (name, res)


def test_delete_needs_explicit_confirm():
    tb = box()
    tb.call("schedule_event", {"title": "A", "start": "2026-10-12T09:00", "duration_min": 60})
    res, err = tb.call("delete_event", {"event_id": 1})
    assert err and "confirm=true" in res["error"]
    assert len(tb.state.calendar.events) == 1
    res, err = tb.call("delete_event", {"event_id": 1, "confirm": "yes"})  # truthy string is not consent
    assert err and len(tb.state.calendar.events) == 1
    res, err = tb.call("delete_event", {"event_id": 1, "confirm": True})
    assert not err and not tb.state.calendar.events


def test_habit_flow_and_future_day_guard():
    tb = box()
    tb.call("add_habit", {"name": "read", "target_per_week": 5})
    res, err = tb.call("log_habit", {"name": "read"})
    assert not err and res["streak_days"] == 1 and res["this_week"] == "1/5"
    res, err = tb.call("log_habit", {"name": "read", "day": "2026-10-20"})
    assert err and "future" in res["error"]
