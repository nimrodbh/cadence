"""The tool layer: typed functions an agent (or an MCP client) may call.

`Toolbox` methods are the single source of truth. Their signatures and docstrings
generate the JSON schemas for the agent loop, and the same methods are registered
on the MCP server, so the two interfaces cannot drift apart.

Every failure becomes a `ToolError` with a message the model can act on; nothing
raises out of `Toolbox.call`.
"""

from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

from .domain import DomainError, State


class ToolError(Exception):
    pass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


def _d(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ToolError(f"'{value}' is not a date; use YYYY-MM-DD") from None


def _dt(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise ToolError(f"'{value}' is not a date and time; use YYYY-MM-DDTHH:MM") from None


def _t(value: str) -> time:
    try:
        return time.fromisoformat(value)
    except ValueError:
        raise ToolError(f"'{value}' is not a time; use HH:MM") from None


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M")


def _event(e: Any) -> dict[str, Any]:
    return {"id": e.id, "title": e.title, "start": _fmt(e.start), "end": _fmt(e.end)}


class Toolbox:
    """Tools bound to one `State`. Public methods that do not start with an underscore are tools."""

    def __init__(self, state: State) -> None:
        self.state = state

    # --- calendar ---------------------------------------------------------------------------

    def list_events(self, first_day: str, last_day: str) -> dict[str, Any]:
        """List calendar events from first_day to last_day inclusive (dates as YYYY-MM-DD)."""
        events = self.state.calendar.between(_d(first_day), _d(last_day))
        return {"events": [_event(e) for e in events]}

    def find_free_slots(
        self, day: str, duration_min: int, earliest: str = "08:00", latest: str = "22:00"
    ) -> dict[str, Any]:
        """Find free windows on one day (YYYY-MM-DD) that fit duration_min minutes, between earliest and latest (HH:MM)."""
        slots = self.state.calendar.free_slots(_d(day), duration_min, _t(earliest), _t(latest))
        return {"slots": [{"start": _fmt(a), "end": _fmt(b)} for a, b in slots]}

    def schedule_event(self, title: str, start: str, duration_min: int) -> dict[str, Any]:
        """Create an event starting at start (YYYY-MM-DDTHH:MM) lasting duration_min minutes. Fails if it overlaps another event."""
        if duration_min <= 0:
            raise ToolError("duration_min must be positive")
        s = _dt(start)
        ev = self.state.calendar.add(title, s, s + timedelta(minutes=duration_min))
        return {"created": _event(ev)}

    def move_event(self, event_id: int, new_start: str) -> dict[str, Any]:
        """Move an event to start at new_start (YYYY-MM-DDTHH:MM), keeping its length. Fails on overlap."""
        return {"moved": _event(self.state.calendar.move(event_id, _dt(new_start)))}

    def delete_event(self, event_id: int, confirm: bool = False) -> dict[str, Any]:
        """Delete an event. Destructive: only call with confirm=true after the user has clearly asked for the deletion."""
        ev = self.state.calendar.get(event_id)
        if not confirm:
            raise ToolError(
                f"deleting '{ev.title}' needs confirm=true; ask the user to confirm first, then call again with confirm=true"
            )
        self.state.calendar.remove(event_id)
        return {"deleted": _event(ev)}

    # --- habits -----------------------------------------------------------------------------

    def add_habit(self, name: str, target_per_week: int) -> dict[str, Any]:
        """Start tracking a habit with a weekly target of 1 to 7 days."""
        h = self.state.habits.add(name, target_per_week)
        return {"habit": h.name, "target_per_week": h.target_per_week}

    def log_habit(self, name: str, day: str = "") -> dict[str, Any]:
        """Record that a habit was done on a day (YYYY-MM-DD). Leave day empty for today."""
        d = _d(day) if day else self.state.today
        if d > self.state.today:
            raise ToolError("cannot log a habit for a future day")
        self.state.habits.log(name, d)
        return self.habit_status(name)

    def habit_status(self, name: str) -> dict[str, Any]:
        """Current streak in days and progress this ISO week against the weekly target."""
        today = self.state.today
        done, target = self.state.habits.week_progress(name, today)
        return {
            "habit": self.state.habits.get(name).name,
            "streak_days": self.state.habits.streak(name, today),
            "this_week": f"{done}/{target}",
        }

    def list_habits(self) -> dict[str, Any]:
        """List every tracked habit with its weekly target."""
        return {"habits": [{"habit": h.name, "target_per_week": h.target_per_week} for h in self.state.habits.habits.values()]}

    def today(self) -> dict[str, Any]:
        """Today's date (YYYY-MM-DD) and weekday, as the agent should understand it."""
        t = self.state.today
        return {"today": t.isoformat(), "weekday": t.strftime("%A")}

    # --- plumbing ---------------------------------------------------------------------------

    def tool_names(self) -> list[str]:
        return [n for n, m in inspect.getmembers(self, inspect.ismethod) if not n.startswith("_") and n not in _NOT_TOOLS]

    def specs(self) -> list[ToolSpec]:
        return [_spec(n, getattr(self, n)) for n in self.tool_names()]

    def call(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Run a tool. Returns (result, is_error). Never raises."""
        if name not in self.tool_names():
            return {"error": f"unknown tool '{name}'; available: {', '.join(self.tool_names())}"}, True
        fn = getattr(self, name)
        try:
            kwargs = _coerce(fn, args)
            return fn(**kwargs), False
        except (ToolError, DomainError) as exc:
            return {"error": str(exc)}, True


_NOT_TOOLS = {"call", "specs", "tool_names"}

_JSON_TYPES = {str: "string", int: "integer", bool: "boolean", float: "number"}


def _spec(name: str, fn: Any) -> ToolSpec:
    hints = typing.get_type_hints(fn)
    props: dict[str, Any] = {}
    required: list[str] = []
    for pname, p in inspect.signature(fn).parameters.items():
        props[pname] = {"type": _JSON_TYPES[hints[pname]]}
        if p.default is inspect.Parameter.empty:
            required.append(pname)
    return ToolSpec(
        name=name,
        description=inspect.getdoc(fn) or "",
        input_schema={"type": "object", "properties": props, "required": required, "additionalProperties": False},
    )


def _coerce(fn: Any, args: dict[str, Any]) -> dict[str, Any]:
    hints = typing.get_type_hints(fn)
    params = inspect.signature(fn).parameters
    unknown = set(args) - set(params)
    if unknown:
        raise ToolError(f"unexpected argument(s): {', '.join(sorted(unknown))}")
    missing = [n for n, p in params.items() if p.default is inspect.Parameter.empty and n not in args]
    if missing:
        raise ToolError(f"missing required argument(s): {', '.join(missing)}")
    out: dict[str, Any] = {}
    for k, v in args.items():
        want = hints[k]
        if want is bool:
            if not isinstance(v, bool):
                raise ToolError(f"{k} must be true or false")
        elif want is int:
            if isinstance(v, bool) or not isinstance(v, int):
                raise ToolError(f"{k} must be an integer")
        elif want is str and not isinstance(v, str):
            raise ToolError(f"{k} must be a string")
        out[k] = v
    return out
