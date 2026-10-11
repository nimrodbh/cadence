import pytest

from cadence.evals.run import run_offline, run_scenario
from cadence.evals.scenarios import SCENARIOS
from cadence.models import ScriptedModel


@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda s: s.name)
def test_golden_trajectory_passes(sc):
    problems, _ = run_scenario(sc, ScriptedModel(sc.golden))
    assert problems == []


@pytest.mark.parametrize("sc", [s for s in SCENARIOS if s.bad], ids=lambda s: s.name)
def test_bad_trajectory_is_caught(sc):
    problems, _ = run_scenario(sc, ScriptedModel(sc.bad))
    assert problems, "the checks accepted a clearly wrong run"


def test_offline_runner_exit_status(capsys):
    assert run_offline() == 0
    assert "MISSED" not in capsys.readouterr().out
