"""Fairness and bookkeeping tests; no model endpoint is called."""
import asyncio
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from langchain_core.messages import AIMessage, SystemMessage

import benchmark
from actions import VALID_ACTIONS
from graph import build_graph
from single_agent import CONTEXT_MARKER, decide_generalist, decision_context
from simulator import sim
from tests.test_graph import make_colony, make_initial_state, make_stub_factory


def config(**updates):
    return SimpleNamespace(
        **(dict(steps=3, model="same-qwen", base_url="http://unused/v1", temperature=.2,
                planner_tokens=500, specialist_tokens=300, timeout=12) | updates)
    )


def test_matched_initial_states_and_crisis_sequences_are_isolated():
    live_before = sim.snapshot().model_dump()
    for scenario in benchmark.SCENARIOS:
        first = benchmark.new_simulator(scenario, 123)
        second = benchmark.new_simulator(scenario, 123)
        assert first.snapshot() == second.snapshot()
        first.trigger_crisis(scenario.crisis)
        second.trigger_crisis(scenario.crisis)
        for _ in range(3):
            first.advance_tick()
            second.advance_tick()
            assert first.snapshot() == second.snapshot()
        first.execute_actions(["ration_food"])
        assert first.snapshot() != second.snapshot()
    assert sim.snapshot().model_dump() == live_before


def test_shared_context_contains_full_catalog_hazards_and_objective():
    colony = make_colony(crisis="power_failure")
    context = decision_context(colony, 3)
    data = json.loads(context.split("\n", 1)[1])
    assert data["action_resource_effects"] == VALID_ACTIONS
    assert data["state"]["resources"] == colony.model_dump()["resources"]
    assert data["additional_crisis_drain"]["power_failure"]["power"] == -5
    assert data["power_failure_disables_power_regeneration"] is True
    assert set(data["individual_feasibility"]) == set(VALID_ACTIONS)
    assert data["remaining_decision_ticks_including_this_one"] == 3


def test_generalist_can_choose_any_feasible_catalog_combination():
    colony = make_colony()
    actions = ["boost_water_recycling", "ration_food", "greenhouse_optimize", "repair_module"]
    llm = AsyncMock()
    llm.ainvoke.return_value = AIMessage(content=json.dumps({"approved_actions": actions, "rationale": "Joint plan."}))
    result = asyncio.run(decide_generalist(colony, lambda: llm, decision_context(colony, 3)))
    assert result["approved_actions"] == actions
    assert llm.ainvoke.call_count == 1


def test_generalist_has_one_repair_and_same_ordered_safety_filter():
    colony = make_colony(resources={"fuel": 4})
    content = json.dumps({"approved_actions": ["conserve_power", "repair_module"], "rationale": "Both."})
    llm = AsyncMock()
    llm.ainvoke.return_value = AIMessage(content=content)
    result = asyncio.run(decide_generalist(colony, lambda: llm, decision_context(colony, 3)))
    assert result["approved_actions"] == ["conserve_power"]
    assert llm.ainvoke.call_count == 2


def test_generalist_bad_json_never_becomes_successful_recovery():
    colony = make_colony()
    llm = AsyncMock()
    llm.ainvoke.return_value = AIMessage(content="no JSON")
    result = asyncio.run(decide_generalist(colony, lambda: llm, decision_context(colony, 3)))
    assert result["approved_actions"] == []
    assert result["errors"]
    assert llm.ainvoke.call_count == 2


def test_raw_output_audit_counts_bad_plans_retries_and_duplicates():
    colony = make_colony(resources={"fuel": 4})
    def record(content):
        return {"role": "generalist", "content": content, "error": None}
    records = [
        record('{"approved_actions":["repair_module","conserve_power"],"rationale":"Both"}'),
        record('{"approved_actions":["launch_rocket"],"rationale":"Bad"}'),
        record('{"approved_actions":["conserve_power","conserve_power"],"rationale":"Repeat"}'),
        record("malformed"),
    ]
    counts = benchmark.audit_calls(records, colony, ["conserve_power"])
    assert counts["invalid_proposals"] == 1
    assert counts["invalid_responses"] == 1
    assert counts["infeasible_plans"] == 2
    assert counts["rejected_actions"] == 3


def test_sampling_budgets_usage_and_context_match_across_roles(monkeypatch):
    clients = []
    def client(**kwargs):
        clients.append(kwargs)
        response = AIMessage(content='{"approved_actions":[],"rationale":"Hold"}', usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        return SimpleNamespace(ainvoke=AsyncMock(return_value=response))
    monkeypatch.setattr(benchmark, "ChatOpenAI", client)
    context = decision_context(make_colony(), 3)
    records = []
    async def check():
        for prompt in ("You are generalist", "You are Commander Hayes", "You are ARIA"):
            await benchmark.MeasuredFactory(config(), context, 42, records).ainvoke([SystemMessage(content=prompt)])
    asyncio.run(check())
    assert [c["max_tokens"] for c in clients] == [500, 500, 300]
    assert all(c["model"] == "same-qwen" and c["seed"] == 42 and c["temperature"] == .2 and c["max_retries"] == 0 for c in clients)
    assert all(r["messages"][0]["content"].endswith(context) for r in records)
    assert all(r["messages"][0]["content"].count(CONTEXT_MARKER) == 1 for r in records)
    assert all(r["total_tokens"] == 15 for r in records)


def test_recovery_definition_accounts_for_exhaustion():
    good = dict(oxygen=40, power=30, water=20, food=20, fuel=0)
    assert benchmark.recovery_met(good, good)
    assert not benchmark.recovery_met(good, good | {"power": 0})
    assert not benchmark.recovery_met(good | {"oxygen": 39.9}, good)


def test_scarce_fuel_recovery_is_physically_attainable():
    simulator = benchmark.new_simulator(benchmark.SCENARIOS[-1], 123)
    simulator.trigger_crisis("dust_storm")
    for step in range(3):
        simulator.advance_tick()
        simulator.execute_actions(["emergency_power", "conserve_power", "boost_oxygen"] if step == 0 else [])
    assert benchmark.recovery_met(simulator._resources, simulator._resources)


def test_failed_episode_is_retained_and_passive_minima_are_measured():
    class FailedFactory:
        def __init__(self, args, context, seed, records):
            self.records = records
        def __call__(self):
            return self
        async def ainvoke(self, messages):
            self.records.append({"role": "generalist", "error": "unavailable", "input_characters": 10, "input_tokens": None, "output_tokens": None, "total_tokens": None})
            raise RuntimeError("unavailable")
    row, trace = asyncio.run(benchmark.run_episode(benchmark.SCENARIOS[1], "single", 1, 7, config(), FailedFactory))
    assert row["run_failed"] and not row["recovery_success"]
    assert row["model_calls"] == 3 and row["transport_failures"] == 3
    assert row["total_tokens"] is None and not row["token_usage_complete"]
    assert row["executed_actions"] == 0
    assert row["min_power"] == pytest.approx(6.8)
    assert len(trace["steps"]) == 3
    assert len(json.loads(row["errors"])) == 3


def test_benchmark_does_not_change_existing_graph_factory_behavior():
    result = asyncio.run(build_graph(make_stub_factory()).ainvoke(make_initial_state()))
    assert result["approved_actions"] == ["conserve_power"]
