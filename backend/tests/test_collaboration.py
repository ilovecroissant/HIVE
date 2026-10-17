"""Deterministic conflict, coordination, and execution regressions."""
import asyncio
import json
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from actions import VALID_ACTIONS, describe_action, detect_conflicts, evaluate_plan
from graph import build_graph
from simulator import Simulator
from tests.test_graph import make_colony, make_initial_state


def balances(**updates):
    return {"oxygen": 50., "power": 50., "water": 50., "food": 50., "fuel": 50., **updates}


def test_fuel_competition_and_combined_budget():
    colony = make_colony()
    resources = balances(fuel=4)
    assert evaluate_plan(["conserve_power"], resources, colony.modules).feasible
    assert evaluate_plan(["repair_module"], resources, colony.modules).feasible
    report = evaluate_plan(["conserve_power", "repair_module"], resources, colony.modules)
    assert not report.feasible
    conflict = next(c for c in report.conflicts if c.constraint == "budget:fuel")
    assert conflict.actions == ["conserve_power", "repair_module"]
    assert "5.0 exceeds available 4.00" in conflict.explanation
    assert report.effects == {}


def test_power_competition():
    report = evaluate_plan(["boost_oxygen", "greenhouse_optimize"], balances(power=1.5), [])
    assert not report.feasible
    assert any(c.constraint == "budget:power" for c in report.conflicts)


def test_three_way_budget_conflict_when_every_pair_fits():
    actions = ["conserve_power", "repair_module", "emergency_power"]
    modules = make_colony().modules
    resources = balances(fuel=6)
    for pair in ((actions[0], actions[1]), (actions[0], actions[2]), (actions[1], actions[2])):
        assert evaluate_plan(list(pair), resources, modules).feasible
    conflicts = detect_conflicts(actions, resources, modules)
    assert any(c.constraint == "budget:fuel" and len(c.actions) == 3 for c in conflicts)


def test_backup_controller_modes_are_incompatible_and_execution_enforces_it():
    simulator = Simulator()
    before = simulator.snapshot().model_dump()
    report = evaluate_plan(["battery_reserves", "emergency_power"], balances(), [])
    assert any(c.constraint == "backup_controller_mode" for c in report.conflicts)
    with pytest.raises(ValueError, match="alternative backup-controller modes"):
        simulator.execute_actions(["battery_reserves", "emergency_power"])
    assert simulator.snapshot().model_dump() == before


@pytest.mark.parametrize("actions,resources", [
    (["boost_oxygen"], balances(water=3.99)),
    (["boost_water_recycling"], balances(power=1.49)),
    (["launch_rocket"], balances()),
    (["conserve_power", "boost_oxygen"], balances(power=0)),
])
def test_unsafe_actions_have_no_executable_effects(actions, resources):
    report = evaluate_plan(actions, resources, [])
    assert not report.feasible
    assert report.effects == {}
    assert report.executions == []


def test_duplicate_and_saturated_actions_are_redundant():
    assert any(c.constraint == "duplicate_action" for c in detect_conflicts(
        ["conserve_power", "conserve_power"], balances(), [],
    ))
    report = evaluate_plan(["conserve_power"], balances(power=100), [])
    assert any(c.constraint == "no_benefit" for c in report.conflicts)
    assert not evaluate_plan(["idle", "ration_food"], balances(), []).feasible
    assert evaluate_plan(["idle"], balances(), []).feasible


@pytest.mark.parametrize("action", list(VALID_ACTIONS))
def test_every_catalog_action_has_matching_preview_and_execution(action):
    simulator = Simulator()
    before = simulator.snapshot()
    expected = evaluate_plan([action], dict(simulator._resources), before.modules)
    actual = simulator.execute_actions([action])
    after = simulator.snapshot()
    assert actual == expected
    for resource, delta in actual.effects.items():
        assert after.resources[resource].value - before.resources[resource].value == pytest.approx(delta)
    for execution in actual.executions:
        for change in execution.module_changes:
            assert next(m.health for m in after.modules if m.id == change.id) == change.after
    if action == "idle":
        assert after == before
    else:
        assert after != before


def test_repair_caps_health_and_rejects_when_no_target_remains():
    simulator = Simulator()
    for module in simulator._modules:
        module.health = 100
    simulator._modules[0].health = 98
    report = simulator.execute_actions(["repair_module"])
    assert report.executions[0].module_changes[0].after == 100
    before = simulator.snapshot().model_dump()
    with pytest.raises(ValueError, match="No damaged module"):
        simulator.execute_actions(["repair_module"])
    assert simulator.snapshot().model_dump() == before


def test_invalid_combined_plan_preserves_all_simulator_state():
    simulator = Simulator()
    simulator._resources["fuel"] = 4
    before = simulator.snapshot().model_dump()
    with pytest.raises(ValueError, match="Combined fuel cost"):
        simulator.execute_actions(["conserve_power", "repair_module"])
    assert simulator.snapshot().model_dump() == before


def coordinated_graph(proposals, decisions, captured):
    """Route test doubles by specialist identity; record what Commander sees."""
    responses = iter(decisions)

    def factory():
        async def invoke(messages):
            prompt = messages[0].content
            if "Commander Hayes" in prompt:
                captured.append(messages)
                return MagicMock(content=json.dumps(next(responses)))
            agent = next(a for a, name in {"energy": "ARIA", "engineering": "NEXUS", "life_support": "LYRA"}.items() if f"You are {name}" in prompt)
            return MagicMock(content=json.dumps({
                "agent_id": agent, "action": proposals[agent], "rationale": "Protect the colony.",
                "priority": 5 if agent == "energy" else 3,
                "assessment": {"resource_cost": {"fuel": -999}},
            }))
        return MagicMock(ainvoke=invoke)

    return build_graph(factory)


def test_commander_receives_real_conflict_and_selects_valid_subset():
    captured = []
    graph = coordinated_graph(
        {"energy": "conserve_power", "engineering": "repair_module", "life_support": "boost_oxygen"},
        [{"approved_actions": ["conserve_power", "boost_oxygen"],
          "rationale": "Protect power and oxygen; defer maintenance.",
          "rejection_reasons": {"repair_module": "Reserve scarce fuel for power."}}], captured,
    )
    colony = make_colony(resources={"fuel": 4, "power": 20, "oxygen": 25}, crisis="dust_storm")
    result = asyncio.run(graph.ainvoke(make_initial_state(colony)))
    assert result["approved_actions"] == ["conserve_power", "boost_oxygen"]
    assert result["approved_effects"]["fuel"] == -2
    prompt = captured[0][0].content
    assert "budget:fuel" in prompt and "Feasible subsets" in prompt
    assert "Module health" in prompt and "survival_delta" in prompt
    assert len(captured) == 1
    assert any("Combined fuel cost" in e.message for e in result["events"] if "Rejects: repair_module" in e.title)
    assert any(e.title.startswith("Conflict:") for e in result["events"])
    for proposal in result["proposals"]:
        assert proposal.assessment == describe_action(proposal.action, {k: r.value for k, r in colony.resources.items()}, colony.modules)


def test_infeasible_commander_plan_gets_one_revision_then_valid_subset():
    captured = []
    graph = coordinated_graph(
        {"energy": "conserve_power", "engineering": "repair_module", "life_support": "idle"},
        [{"approved_actions": ["conserve_power", "repair_module"], "rationale": "Both."},
         {"approved_actions": ["conserve_power"], "rationale": "Power is urgent.",
          "rejection_reasons": {"repair_module": "Fuel budget."}}], captured,
    )
    result = asyncio.run(graph.ainvoke(make_initial_state(make_colony(resources={"fuel": 4}))))
    assert len(captured) == 2
    assert "Combined fuel cost" in captured[1][1].content
    assert result["approved_actions"] == ["conserve_power"]
    assert not result["errors"]
    assert any(e.title == "Commander Plan Requires Revision" for e in result["events"])


def test_repeated_infeasible_output_is_filtered_without_inventing_actions():
    captured = []
    decision = {"approved_actions": ["conserve_power", "repair_module"], "rationale": "Both."}
    graph = coordinated_graph(
        {"energy": "conserve_power", "engineering": "repair_module", "life_support": "idle"},
        [decision, decision], captured,
    )
    result = asyncio.run(graph.ainvoke(make_initial_state(make_colony(resources={"fuel": 4}))))
    assert len(captured) == 2
    assert result["approved_actions"] == ["conserve_power"]
    assert any("validation removed" in e.message for e in result["events"])
