"""Run the evaluation scenarios.

    python -m cadence.evals.run            # offline: golden trajectories must pass, bad ones must fail
    python -m cadence.evals.run --live     # a real model (needs ANTHROPIC_API_KEY and `pip install .[llm]`)
"""

from __future__ import annotations

import argparse
import sys

from ..agent import Agent, Model, ModelResponse, Trace
from ..domain import State
from ..models import ScriptedModel
from .scenarios import SCENARIOS, Scenario


def run_scenario(sc: Scenario, model: Model) -> tuple[list[str], list[Trace]]:
    state = State()
    sc.setup(state)
    agent = Agent(model, state)
    traces = [agent.run(turn) for turn in sc.turns]
    return sc.check(state, traces), traces


def _scripted(steps: list[ModelResponse]) -> ScriptedModel:
    return ScriptedModel(steps)


def run_offline() -> int:
    failures = 0
    for sc in SCENARIOS:
        problems, _ = run_scenario(sc, _scripted(sc.golden))
        status = "pass" if not problems else "FAIL"
        print(f"golden  {sc.name:34s} {status}")
        for p in problems:
            print(f"        - {p}")
        failures += bool(problems)
        if sc.bad:
            bad_problems, _ = run_scenario(sc, _scripted(sc.bad))
            caught = bool(bad_problems)
            print(f"control {sc.name:34s} {'caught' if caught else 'MISSED (check accepts a bad run)'}")
            failures += not caught
    return failures


def run_live(model_name: str) -> int:
    from ..models import AnthropicModel

    model = AnthropicModel(model_name)
    failures = 0
    for sc in SCENARIOS:
        problems, traces = run_scenario(sc, model)
        steps = sum(t.steps for t in traces)
        print(f"live    {sc.name:34s} {'pass' if not problems else 'FAIL'}  ({steps} model calls)")
        for p in problems:
            print(f"        - {p}")
        failures += bool(problems)
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    args = ap.parse_args()
    failures = run_live(args.model) if args.live else run_offline()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
