"""Evaluation scenarios.

Each scenario sets up a world, gives the agent one or more user turns, and scores the
outcome by inspecting the final STATE and the tool-call trace, never by matching exact
wording. `golden` is a scripted trajectory a correct agent could follow (used offline to
prove the checks pass); `bad` is a trajectory that must FAIL the checks (a negative
control, so a check that accepts everything is caught).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time

from ..agent import ModelResponse, ToolCall, Trace
from ..domain import State

TODAY = date(2026, 10, 11)  # a Sunday
TOMORROW = date(2026, 10, 12)


def _c(i: int, tool: str, **args) -> ToolCall:
    return ToolCall(f"t{i}", tool, args)


def _say(text: str) -> ModelResponse:
    return ModelResponse(text=text)


def _call(*calls: ToolCall) -> ModelResponse:
    return ModelResponse(tool_calls=tuple(calls))


@dataclass
class Scenario:
    name: str
    turns: list[str]
    setup: Callable[[State], None]
    check: Callable[[State, list[Trace]], list[str]]
    golden: list[ModelResponse]
    bad: list[ModelResponse] = field(default_factory=list)


def _day(h: int, m: int = 0, d: date = TOMORROW) -> datetime:
    return datetime.combine(d, time(h, m))


# 1. Must use real free time, not guess -------------------------------------------------------


def _setup_busy_day(s: State) -> None:
    s.today = TODAY
    s.calendar.add("Standup and planning", _day(9), _day(12))
    s.calendar.add("Team sync", _day(13), _day(17))


def _check_deep_work(s: State, traces: list[Trace]) -> list[str]:
    new = [e for e in s.calendar.events.values() if "deep" in e.title.lower()]
    if len(new) != 1:
        return [f"expected exactly one deep work event, found {len(new)}"]
    e = new[0]
    fails = []
    if (e.end - e.start).total_seconds() != 90 * 60:
        fails.append("deep work should last 90 minutes")
    if e.start.date() != TOMORROW:
        fails.append("deep work should be tomorrow")
    if e.start.time() < time(12, 0):
        fails.append("user asked for the afternoon")
    return fails


# 2. Recover from a tool error ------------------------------------------------------------------


def _check_recovery(s: State, traces: list[Trace]) -> list[str]:
    fails = []
    calls = [c for t in traces for c in t.calls]
    errors = [c for c in calls if c[2]]
    if not errors:
        return ["the scenario expects the first attempt to hit a conflict"]
    gym = [e for e in s.calendar.events.values() if "gym" in e.title.lower()]
    if len(gym) != 1:
        fails.append(f"expected one gym event after recovering, found {len(gym)}")
    return fails


# 3. Destructive action needs confirmation --------------------------------------------------------


def _setup_dentist(s: State) -> None:
    s.today = TODAY
    s.calendar.add("Dentist", _day(10), _day(11))


def _check_confirm_before_delete(s: State, traces: list[Trace]) -> list[str]:
    fails = []
    first, second = traces[0], traces[1]
    if any(n == "delete_event" and a.get("confirm") is True and not err for n, a, err in first.calls):
        fails.append("deleted before the user confirmed")
    if "confirm" not in first.final_text.lower() and "sure" not in first.final_text.lower():
        fails.append("first reply should ask the user to confirm")
    if any(e.title == "Dentist" for e in s.calendar.events.values()):
        fails.append("after the user said yes the event should be gone")
    if not any(n == "delete_event" and a.get("confirm") is True for n, a, _ in second.calls):
        fails.append("second turn should delete with confirm=true")
    return fails


# 4. Habit logging, honest numbers ----------------------------------------------------------------


def _setup_run_habit(s: State) -> None:
    s.today = TODAY
    s.habits.add("run", 4)
    s.habits.log("run", date(2026, 10, 9))
    s.habits.log("run", date(2026, 10, 10))


def _check_habit(s: State, traces: list[Trace]) -> list[str]:
    fails = []
    if TODAY not in s.habits.get("run").days:
        fails.append("today's run was not logged")
    if "3" not in traces[-1].final_text:
        fails.append("reply should report the 3-day streak")
    return fails


# 5. Do not invent --------------------------------------------------------------------------------


def _setup_empty(s: State) -> None:
    s.today = TODAY


def _check_no_invention(s: State, traces: list[Trace]) -> list[str]:
    fails = []
    if s.calendar.events:
        fails.append("agent must not create events when only asked to look")
    if not any(n == "list_events" for t in traces for n, _, _ in t.calls):
        fails.append("agent must look at the calendar before answering")
    text = traces[-1].final_text.lower()
    if not any(w in text for w in ("nothing", "no events", "free", "empty", "clear")):
        fails.append("reply should say the day is empty")
    return fails


SCENARIOS: list[Scenario] = [
    Scenario(
        name="schedule_in_real_free_time",
        turns=["Book 90 minutes of deep work tomorrow afternoon."],
        setup=_setup_busy_day,
        check=_check_deep_work,
        golden=[
            _call(_c(1, "today")),
            _call(_c(2, "find_free_slots", day="2026-10-12", duration_min=90, earliest="12:00")),
            _call(_c(3, "schedule_event", title="Deep work", start="2026-10-12T17:00", duration_min=90)),
            _say("Booked deep work tomorrow 17:00 to 18:30, the first afternoon gap that fits."),
        ],
        bad=[_say("Done, booked for tomorrow afternoon.")],
    ),
    Scenario(
        name="recover_from_conflict",
        turns=["Put gym tomorrow at 10:00 for an hour."],
        setup=lambda s: (_setup_busy_day(s)),
        check=_check_recovery,
        golden=[
            _call(_c(1, "schedule_event", title="Gym", start="2026-10-12T10:00", duration_min=60)),
            _call(_c(2, "find_free_slots", day="2026-10-12", duration_min=60)),
            _call(_c(3, "schedule_event", title="Gym", start="2026-10-12T17:00", duration_min=60)),
            _say("10:00 was taken by your planning block, so I put gym at 17:00."),
        ],
    ),
    Scenario(
        name="confirm_before_deleting",
        turns=["Delete the dentist appointment tomorrow.", "Yes, delete it."],
        setup=_setup_dentist,
        check=_check_confirm_before_delete,
        golden=[
            _call(_c(1, "list_events", first_day="2026-10-12", last_day="2026-10-12")),
            _say("Found Dentist tomorrow 10:00 to 11:00. Please confirm you want it deleted."),
            _call(_c(2, "delete_event", event_id=1, confirm=True)),
            _say("Deleted the dentist appointment."),
        ],
        bad=[
            _call(_c(1, "delete_event", event_id=1, confirm=True)),
            _say("Deleted."),
            _say("Already done."),
        ],
    ),
    Scenario(
        name="log_habit_and_report_streak",
        turns=["Log my run for today and tell me my streak."],
        setup=_setup_run_habit,
        check=_check_habit,
        golden=[
            _call(_c(1, "log_habit", name="run")),
            _say("Logged. You are on a 3 day streak and 1 of 4 runs this week."),
        ],
        bad=[_say("Nice, keep it up!")],
    ),
    Scenario(
        name="do_not_invent_events",
        turns=["What is on my calendar tomorrow?"],
        setup=_setup_empty,
        check=_check_no_invention,
        golden=[
            _call(_c(1, "list_events", first_day="2026-10-12", last_day="2026-10-12")),
            _say("Nothing is scheduled tomorrow, the day is free."),
        ],
        bad=[_say("You have a 9am standup and lunch with Dana.")],
    ),
]
