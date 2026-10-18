"""Independent routing, prompt, planning, physics, and legacy-replay regressions."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from langchain_core.messages import AIMessage

from actions import SCORE_WEIGHTS, evaluate_plan
from decision_policy import candidate_plans, projected_state, specialist_routing
from efficient_graph import build_efficient_graph
from physics_context import shared_context
from single_agent import decide_generalist, decision_context
from simulator import Simulator
from tests.test_graph import make_colony, make_initial_state

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "benchmark_results" / "qwen_20261017"


def nominal_colony():
    return Simulator().snapshot()


def test_stage_one_nominal_hold_needs_no_agents():
    assert not any(item["active"] for item in specialist_routing(nominal_colony()).values())


@pytest.mark.parametrize("crisis,required", [
    ("dust_storm", "energy"), ("power_failure", "energy"), ("oxygen_leak", "life_support"),
])
def test_stage_one_crisis_expertise_is_never_skipped(crisis, required):
    routing = specialist_routing(make_colony(crisis=crisis))
    assert routing[required]["active"]
    assert routing[required]["reasons"]


@pytest.mark.parametrize("resources,required", [
    ({"oxygen": 10}, "life_support"), ({"water": 10}, "life_support"),
    ({"food": 10}, "life_support"), ({"power": 10}, "energy"), ({"fuel": 1}, "energy"),
])
def test_stage_one_cross_crisis_critical_reserves_are_never_skipped(resources, required):
    assert specialist_routing(make_colony(resources=resources, crisis="dust_storm"))[required]["active"]


def test_stage_one_real_damage_and_deferral_are_distinguished():
    colony = make_colony(resources={"power": 20}, crisis="dust_storm")
    colony.modules[0].health = 30
    assert specialist_routing(colony)["engineering"]["active"]
    colony.modules[0].health = 65
    result = specialist_routing(colony)["engineering"]
    assert not result["active"] and "Defer noncritical" in result["reasons"][0]
    colony.modules[0].type = "solar"
    assert specialist_routing(colony, physics_version="v2")["engineering"]["active"]


def test_stage_three_v1_replays_all_old_benchmark_plans_exactly():
    from benchmark import SCENARIOS, new_simulator
    for line in (OLD / "traces.jsonl").read_text().splitlines():
        trace = json.loads(line)
        row = trace["result"]
        simulator = new_simulator(next(s for s in SCENARIOS if s.name == row["scenario"]), row["seed"])
        simulator.trigger_crisis(simulator.snapshot().active_crisis or next(s.crisis for s in SCENARIOS if s.name == row["scenario"]))
        for step in trace["steps"]:
            simulator.advance_tick()
            assert simulator.snapshot().model_dump() == step["state_before_decision"]
            assert simulator.execute_actions(step["approved_actions"]).model_dump() == step["execution"]
        assert simulator._compute_score() == pytest.approx(row["final_survival_score"])


def test_baseline_and_legacy_graph_sources_are_unchanged():
    manifest = json.loads((OLD / "manifest.json").read_text())
    for name in ("single_agent.py", "graph.py", "actions.py"):
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == manifest["source_sha256"][name]


def test_stage_two_baseline_v1_first_prompt_is_byte_identical_to_old_run():
    from benchmark import SCENARIOS, new_simulator
    trace = next(json.loads(line) for line in (OLD / "traces.jsonl").read_text().splitlines() if json.loads(line)["result"]["approach"] == "single")
    scenario = next(s for s in SCENARIOS if s.name == trace["result"]["scenario"])
    simulator = new_simulator(scenario, trace["result"]["seed"])
    simulator.trigger_crisis(scenario.crisis)
    simulator.advance_tick()
    colony = simulator.snapshot().model_copy(deep=True)
    context = shared_context(colony, 3, "v1")
    assert context == decision_context(colony, 3)
    llm = AsyncMock()
    llm.ainvoke.return_value = AIMessage(content='{"approved_actions":[],"rationale":"hold"}')
    asyncio.run(decide_generalist(colony, lambda: llm, context))
    assert llm.ainvoke.call_args.args[0][0].content == trace["calls"][0]["messages"][0]["content"]


def test_stage_three_health_derates_existing_production_without_score_bonus():
    simulator = Simulator(physics_version="v2")
    for module in simulator._modules:
        module.health = 100
    assert simulator.regeneration("power") == pytest.approx(.3)
    assert simulator.regeneration("food") == .1
    assert simulator.regeneration("water") == .05
    solar = next(m for m in simulator._modules if m.type == "solar")
    solar.health = 20
    before = simulator.regeneration("power")
    report = simulator.execute_actions(["repair_module"])
    assert solar.health == 20  # execution atomically replaced the module collection
    assert simulator.regeneration("power") - before == pytest.approx(.012)
    assert report.effects == {"fuel": -3}
    assert simulator._compute_score() == pytest.approx(sum(simulator._resources[r] * w for r, w in SCORE_WEIGHTS.items()))
    simulator._active_crisis = "power_failure"
    old_power = simulator._resources["power"]
    simulator.advance_tick()
    assert simulator._resources["power"] == pytest.approx(old_power - 5.8)


@pytest.mark.parametrize("physics", ["v1", "v2"])
def test_stage_three_predictions_match_simulator_execution_and_future_drain(physics):
    colony = make_colony(crisis="oxygen_leak")
    predicted, _, _ = projected_state(colony, ["boost_oxygen", "repair_module"], 3, physics)
    actual = Simulator(physics)
    actual._resources = {r: v.value for r, v in colony.resources.items()}
    actual._modules = [m.model_copy(deep=True) for m in colony.modules]
    actual._active_crisis = colony.active_crisis
    actual.execute_actions(["boost_oxygen", "repair_module"])
    actual.advance_tick()
    actual.advance_tick()
    assert actual._resources == predicted._resources
    assert actual._modules == predicted._modules


def test_stage_three_v2_environment_is_accurate_and_shared():
    context = shared_context(make_colony(), 3, "v2")
    data = json.loads(context.split("\n", 1)[1])
    assert data["physics_version"] == "v2"
    assert "power=0.20*solar_health/100" in data["rules"]
    assert "Module health does not affect" not in data["rules"]


def test_stage_four_shortlist_is_feasible_and_threshold_aware():
    # Same physically attainable scarce-fuel case as the frozen benchmark.
    from benchmark import SCENARIOS, new_simulator
    simulator = new_simulator(SCENARIOS[-1], 42)
    simulator.trigger_crisis("dust_storm")
    simulator.advance_tick()
    colony = simulator.snapshot()
    plans = candidate_plans(colony, 3)
    assert plans[0]["threshold_deficit"] == 0
    assert not plans[0]["essential_exhaustion"]
    assert "emergency_power" in plans[0]["actions"] and "conserve_power" in plans[0]["actions"]
    assert "repair_module" not in plans[0]["actions"]
    assert all(p["threshold_deficit"] == 0 for p in plans)
    for plan in plans:
        assert evaluate_plan(plan["actions"], dict(simulator._resources), colony.modules).feasible


def test_stage_four_does_not_offer_oxygen_oversupply_that_breaks_power_threshold():
    colony = make_colony(resources={"oxygen": 45, "power": 32.5, "fuel": .1}, crisis="dust_storm")
    plans = candidate_plans(colony, horizon=2)
    assert plans and all(p["threshold_deficit"] == 0 for p in plans)
    assert all("boost_oxygen" not in p["actions"] for p in plans)


def test_routing_does_not_call_unaffordable_noncritical_maintenance():
    colony = make_colony(resources={"fuel": .1})
    colony.modules[0].health = 45
    routing = specialist_routing(colony)
    assert not routing["engineering"]["active"]
    assert "unaffordable" in routing["engineering"]["reasons"][0]
    colony.modules[0].health = 30
    assert specialist_routing(colony)["engineering"]["active"]


def efficient_factory(calls, *, bad_commander=False, no_usage=False):
    def factory():
        async def invoke(messages):
            prompt = messages[0].content
            calls.append(messages)
            if "You are Commander Hayes" in prompt:
                if bad_commander:
                    content = "not JSON"
                else:
                    candidates = json.loads(prompt.split("survival index: ", 1)[1].split("\nPredictions", 1)[0])
                    content = json.dumps({"approved_actions": candidates[0]["actions"], "rationale": "Protect essential reserves."})
            else:
                action = "conserve_power" if "You are ARIA" in prompt else "boost_oxygen" if "You are LYRA" in prompt else "repair_module"
                content = json.dumps({"action": action, "priority": 5, "rationale": "Protect this subsystem."})
            return AIMessage(content=content, usage_metadata=None if no_usage else {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        return type("LLM", (), {"ainvoke": staticmethod(invoke)})()
    return factory


def test_stage_two_nominal_hold_has_no_invented_inference():
    calls = []
    result = asyncio.run(build_efficient_graph(efficient_factory(calls)).ainvoke(make_initial_state(nominal_colony())))
    assert not calls and not result["approved_actions"] and not result["proposals"]
    assert any(e.title == "No Specialist Intervention Needed" for e in result["events"])


def test_stages_two_and_four_routed_fan_in_preserves_proposals_and_coordination():
    calls = []
    colony = make_colony(resources={"power": 20, "oxygen": 25}, crisis="dust_storm")
    result = asyncio.run(build_efficient_graph(efficient_factory(calls)).ainvoke(make_initial_state(colony)))
    assert {p.agent_id for p in result["proposals"]} == {"energy", "life_support"}
    assert sum("You are Commander Hayes" in c[0].content for c in calls) == 1
    assert len(calls) == 3
    assert not result["activation"]["engineering"]["active"]
    assert all(c[0].content.count("BENCHMARK ENVIRONMENT AND OBJECTIVE:") == 1 for c in calls)
    metrics = json.loads(next(e.message for e in result["events"] if e.title == "Decision Inference Metrics"))
    assert metrics["logical_model_calls"] == 3 and metrics["total_tokens"] == 45
    assert result["approved_actions"]


def test_routing_can_be_disabled_for_independent_ablation():
    calls = []
    result = asyncio.run(build_efficient_graph(efficient_factory(calls), routing=False).ainvoke(make_initial_state()))
    assert len(result["proposals"]) == 3 and len(calls) == 4


def test_stage_four_invalid_commander_is_bounded_and_never_fakes_success():
    calls = []
    result = asyncio.run(build_efficient_graph(efficient_factory(calls, bad_commander=True)).ainvoke(make_initial_state(make_colony(crisis="dust_storm"))))
    assert sum("You are Commander Hayes" in c[0].content for c in calls) == 2
    assert result["errors"] and not result["approved_actions"]


def test_stage_two_missing_usage_is_unavailable_not_zero():
    result = asyncio.run(build_efficient_graph(efficient_factory([], no_usage=True)).ainvoke(make_initial_state(make_colony(crisis="dust_storm"))))
    metrics = json.loads(next(e.message for e in result["events"] if e.title == "Decision Inference Metrics"))
    assert metrics["total_tokens"] is None


def test_malformed_specialist_json_blocks_coordination_without_fake_approval():
    calls = []
    base = efficient_factory(calls)
    def factory():
        llm = base()
        original = llm.ainvoke
        async def invoke(messages):
            if "You are LYRA" in messages[0].content:
                calls.append(messages)
                return AIMessage(content='{"action":"boost_oxygen","priority":1,"rationale":"Protect oxygen.")}')
            return await original(messages)
        return type("LLM", (), {"ainvoke": staticmethod(invoke)})()
    colony = make_colony(crisis="oxygen_leak")
    result = asyncio.run(build_efficient_graph(factory).ainvoke(make_initial_state(colony)))
    assert result["errors"] and result["approved_actions"] == []
    assert not any("You are Commander Hayes" in call[0].content for call in calls)
    assert any(e.title == "Commander Blocked" for e in result["events"])


def test_v2_status_and_lock_cleanup_on_cancel(monkeypatch):
    import main
    simulator = Simulator()
    simulator._resources.update(power=20, oxygen=25)
    simulator._active_crisis = "dust_storm"
    monkeypatch.setattr(main, "sim", simulator)
    monkeypatch.setattr(main, "LIVE_AGENTS", True)
    monkeypatch.setattr(main, "GRAPH_VERSION", "v2")
    async def check():
        started = asyncio.Event()
        async def blocked(messages):
            started.set()
            await asyncio.Event().wait()
        llm = AsyncMock()
        llm.ainvoke.side_effect = blocked
        monkeypatch.setattr(main, "default_graph", build_efficient_graph(lambda: llm, status_hook=simulator._set_status), raising=False)
        task = asyncio.create_task(main.agents_decide())
        await asyncio.wait_for(started.wait(), 2)
        statuses = {a.id: a.status for a in simulator.snapshot().agents}
        assert statuses["energy"] == "analyzing"
        assert statuses["engineering"] == "idle"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert all(a.status == "idle" for a in simulator.snapshot().agents)
        assert main.reset().state.tick == 0
    asyncio.run(check())
