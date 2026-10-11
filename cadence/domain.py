"""Domain model: a calendar that refuses double bookings, and habits with streaks.

Everything here is plain Python with no I/O, so it is easy to test exhaustively.
Times are naive local datetimes at minute precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta


class DomainError(Exception):
    """A request that cannot be honoured (conflict, unknown id, bad range)."""


@dataclass(frozen=True)
class Event:
    id: int
    title: str
    start: datetime
    end: datetime

    def overlaps(self, start: datetime, end: datetime) -> bool:
        return self.start < end and start < self.end


@dataclass
class Calendar:
    events: dict[int, Event] = field(default_factory=dict)
    _next_id: int = 1

    def add(self, title: str, start: datetime, end: datetime) -> Event:
        if not title.strip():
            raise DomainError("title must not be empty")
        if end <= start:
            raise DomainError("end must be after start")
        if start.date() != end.date():
            raise DomainError("events cannot span midnight; split them per day")
        clash = self._clash(start, end)
        if clash:
            raise DomainError(f"conflicts with '{clash.title}' ({clash.start:%H:%M}-{clash.end:%H:%M}, id {clash.id})")
        ev = Event(self._next_id, title.strip(), start, end)
        self.events[ev.id] = ev
        self._next_id += 1
        return ev

    def get(self, event_id: int) -> Event:
        try:
            return self.events[event_id]
        except KeyError:
            raise DomainError(f"no event with id {event_id}") from None

    def remove(self, event_id: int) -> Event:
        ev = self.get(event_id)
        del self.events[event_id]
        return ev

    def move(self, event_id: int, new_start: datetime) -> Event:
        ev = self.get(event_id)
        new_end = new_start + (ev.end - ev.start)
        if new_start.date() != new_end.date():
            raise DomainError("move would cross midnight")
        clash = self._clash(new_start, new_end, ignore=event_id)
        if clash:
            raise DomainError(f"conflicts with '{clash.title}' ({clash.start:%H:%M}-{clash.end:%H:%M}, id {clash.id})")
        moved = Event(ev.id, ev.title, new_start, new_end)
        self.events[ev.id] = moved
        return moved

    def between(self, first: date, last: date) -> list[Event]:
        if last < first:
            raise DomainError("last day is before first day")
        return sorted(
            (e for e in self.events.values() if first <= e.start.date() <= last),
            key=lambda e: e.start,
        )

    def free_slots(
        self, day: date, duration_min: int, earliest: time = time(8, 0), latest: time = time(22, 0)
    ) -> list[tuple[datetime, datetime]]:
        """Maximal free windows on `day` inside [earliest, latest] that fit `duration_min`."""
        if duration_min <= 0:
            raise DomainError("duration must be positive")
        cursor = datetime.combine(day, earliest)
        limit = datetime.combine(day, latest)
        if limit <= cursor:
            raise DomainError("latest must be after earliest")
        need = timedelta(minutes=duration_min)
        slots: list[tuple[datetime, datetime]] = []
        for e in self.between(day, day):
            if e.end <= cursor:
                continue
            if e.start >= limit:
                break
            if e.start - cursor >= need:
                slots.append((cursor, e.start))
            cursor = max(cursor, e.end)
        if limit - cursor >= need:
            slots.append((cursor, limit))
        return slots

    def _clash(self, start: datetime, end: datetime, ignore: int | None = None) -> Event | None:
        for e in self.events.values():
            if e.id != ignore and e.overlaps(start, end):
                return e
        return None


@dataclass
class Habit:
    name: str
    target_per_week: int
    days: set[date] = field(default_factory=set)


@dataclass
class Habits:
    habits: dict[str, Habit] = field(default_factory=dict)

    def add(self, name: str, target_per_week: int) -> Habit:
        key = name.strip().lower()
        if not key:
            raise DomainError("habit name must not be empty")
        if not 1 <= target_per_week <= 7:
            raise DomainError("target_per_week must be between 1 and 7")
        if key in self.habits:
            raise DomainError(f"habit '{key}' already exists")
        self.habits[key] = Habit(key, target_per_week)
        return self.habits[key]

    def get(self, name: str) -> Habit:
        try:
            return self.habits[name.strip().lower()]
        except KeyError:
            raise DomainError(f"no habit called '{name}'") from None

    def log(self, name: str, day: date) -> Habit:
        h = self.get(name)
        h.days.add(day)
        return h

    def streak(self, name: str, today: date) -> int:
        """Consecutive logged days ending today, or ending yesterday if today is not logged yet."""
        h = self.get(name)
        d = today if today in h.days else today - timedelta(days=1)
        n = 0
        while d in h.days:
            n += 1
            d -= timedelta(days=1)
        return n

    def week_progress(self, name: str, today: date) -> tuple[int, int]:
        """(days logged in the ISO week containing `today`, weekly target)."""
        h = self.get(name)
        monday = today - timedelta(days=today.weekday())
        done = sum(1 for d in h.days if monday <= d <= monday + timedelta(days=6))
        return done, h.target_per_week


@dataclass
class State:
    calendar: Calendar = field(default_factory=Calendar)
    habits: Habits = field(default_factory=Habits)
    today: date = field(default_factory=date.today)
