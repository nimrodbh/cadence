from datetime import date, datetime, time, timedelta

import pytest

from cadence.domain import Calendar, DomainError, Habits

D = date(2026, 10, 12)


def at(h, m=0, d=D):
    return datetime.combine(d, time(h, m))


def test_overlap_is_rejected_but_touching_is_fine():
    cal = Calendar()
    cal.add("a", at(9), at(10))
    cal.add("b", at(10), at(11))  # back to back is allowed
    with pytest.raises(DomainError, match="conflicts with 'a'"):
        cal.add("c", at(9, 30), at(10, 30))


def test_rejects_bad_ranges():
    cal = Calendar()
    with pytest.raises(DomainError):
        cal.add("x", at(10), at(10))
    with pytest.raises(DomainError):
        cal.add("x", at(23), at(23) + timedelta(hours=2))
    with pytest.raises(DomainError):
        cal.add("  ", at(9), at(10))


def test_move_ignores_itself_but_not_others():
    cal = Calendar()
    a = cal.add("a", at(9), at(10))
    cal.add("b", at(11), at(12))
    assert cal.move(a.id, at(9, 30)).start == at(9, 30)  # overlaps its old self only
    with pytest.raises(DomainError):
        cal.move(a.id, at(10, 30))
    assert cal.get(a.id).start == at(9, 30)  # failed move changes nothing


def test_free_slots_between_events():
    cal = Calendar()
    cal.add("m1", at(9), at(12))
    cal.add("m2", at(13), at(17))
    assert cal.free_slots(D, 60) == [(at(8), at(9)), (at(12), at(13)), (at(17), at(22))]
    assert cal.free_slots(D, 90) == [(at(17), at(22))]
    assert cal.free_slots(D, 600) == []


def test_free_slots_respects_window_and_empty_day():
    cal = Calendar()
    assert cal.free_slots(D, 30, time(10), time(11)) == [(at(10), at(11))]
    cal.add("early", at(6), at(9))  # starts before the window
    assert cal.free_slots(D, 30)[0] == (at(9), at(22))


def test_free_slots_never_overlap_events_property():
    import random

    rng = random.Random(1)
    for _ in range(200):
        cal = Calendar()
        for _ in range(rng.randint(0, 6)):
            h = rng.randint(6, 20)
            try:
                cal.add("e", at(h, rng.choice([0, 15, 30])), at(h + 1, 0))
            except DomainError:
                pass
        dur = rng.choice([15, 30, 60, 90])
        for a, b in cal.free_slots(D, dur):
            assert b - a >= timedelta(minutes=dur)
            assert not any(e.overlaps(a, b) for e in cal.events.values())


def test_streak_counts_back_from_today_or_yesterday():
    h = Habits()
    h.add("Run", 4)
    for day in (7, 8, 9, 10):
        h.log("run", date(2026, 10, day))
    assert h.streak("run", date(2026, 10, 10)) == 4
    assert h.streak("run", date(2026, 10, 11)) == 4  # today not logged yet: streak is still alive
    assert h.streak("run", date(2026, 10, 12)) == 0  # missed a full day


def test_week_progress_uses_iso_week():
    h = Habits()
    h.add("read", 3)
    h.log("read", date(2026, 10, 11))  # Sunday of the previous ISO week
    h.log("read", date(2026, 10, 12))  # Monday
    assert h.week_progress("read", date(2026, 10, 14)) == (1, 3)


def test_habit_validation():
    h = Habits()
    with pytest.raises(DomainError):
        h.add("x", 0)
    h.add("x", 3)
    with pytest.raises(DomainError):
        h.add("X", 3)
    with pytest.raises(DomainError):
        h.get("nope")
