"""Live route regression tests; injected models never change production configuration."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException

import main
from graph import build_graph
from simulator import Simulator
from tests.test_graph import make_stub_factory


@pytest.fixture
def live(monkeypatch):
    simulator = Simulator()
    monkeypatch.setattr(main, "sim", simulator)
    monkeypatch.setattr(main, "LIVE_AGENTS", True)
    graph = build_graph(make_stub_factory())
    monkeypatch.setattr(main, "default_graph", graph, raising=False)
    return TestClient(main.app), simulator


def test_crisis_does_not_start_inference(live):
    client, _ = live
    response = client.post("/simulation/crisis", json={"crisis": "dust_storm"})
    assert response.status_code == 200
    assert all(a["status"] == "idle" for a in response.json()["state"]["agents"])


def test_repeated_live_decisions_apply_once_per_tick(live):
    client, simulator = live
    for tick in (1, 2, 3):
        response = client.post("/agents/decide")
        assert response.status_code == 200
        body = response.json()
        assert body["state"]["tick"] == tick
        assert body["state"]["active_crisis"] is None
        assert body["state"]["resources"]["power"]["value"] == 80 + 4.5 * tick
        assert body["state"]["resources"]["fuel"]["value"] == 60 - 2.1 * tick
        assert all(a["status"] == "idle" for a in body["state"]["agents"])
        ids = [event.id for event in simulator._events]
        assert len(ids) == len(set(ids))


@pytest.mark.parametrize("failure", ["specialist", "commander", "graph"])
def test_live_failure_resets_statuses_without_action_effects(live, monkeypatch, failure):
    client, simulator = live
    modules_before = [m.model_dump() for m in simulator.snapshot().modules]
    if failure == "graph":
        graph = AsyncMock()
        graph.ainvoke.side_effect = RuntimeError("Graph failed")
    else:
        graph = build_graph(make_stub_factory(
            fail=failure == "specialist", fail_decision=failure == "commander",
        ))
    monkeypatch.setattr(main, "default_graph", graph)
    body = client.post("/agents/decide").json()
    # A failed attempt still advances the clock and incurs passive drain.
    assert body["state"]["tick"] == 1
    assert body["state"]["resources"]["power"]["value"] == 79.5
    assert body["state"]["resources"]["fuel"]["value"] == 59.9
    assert any(e["type"] == "error" for e in body["events"])
    assert all(a.status == "idle" for a in simulator.snapshot().agents)
    assert body["state"]["modules"] == modules_before


def test_live_status_visible_during_inference_and_reset_after_cancel(live, monkeypatch):
    _, simulator = live

    async def check():
        started = asyncio.Event()

        async def blocked(_):
            started.set()
            await asyncio.Event().wait()

        graph = AsyncMock()
        graph.ainvoke.side_effect = blocked
        monkeypatch.setattr(main, "default_graph", graph)
        task = asyncio.create_task(main.agents_decide())
        await asyncio.wait_for(started.wait(), timeout=2)
        active = {a.id: a.status for a in simulator.snapshot().agents}
        assert active["science"] == "idle"
        assert all(active[aid] == "analyzing" for aid in (
            "energy", "life_support", "engineering", "commander",
        ))
        # Concurrent mutations must not advance/reset the running simulation.
        with pytest.raises(HTTPException) as conflict:
            await main.agents_decide()
        assert conflict.value.status_code == 409
        for mutation in (
            main.reset,
            lambda: main.trigger_crisis(main.CrisisRequest(crisis="dust_storm")),
            lambda: main.commander_command(main.CommanderRequest(instruction="all clear")),
        ):
            with pytest.raises(HTTPException) as conflict:
                mutation()
            assert conflict.value.status_code == 409
        assert main.get_state().state.tick == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert all(a.status == "idle" for a in simulator.snapshot().agents)
        # Cancellation releases the mutation guard as well as agent statuses.
        assert main.reset().state.tick == 0

    asyncio.run(check())


def test_outcome_reports_clamped_delta(live, monkeypatch):
    client, simulator = live
    simulator._resources["power"] = 99
    graph = AsyncMock()
    graph.ainvoke.return_value = {
        "approved_effects": {"power": 50}, "approved_actions": ["emergency_power"],
    }
    monkeypatch.setattr(main, "default_graph", graph)
    body = client.post("/agents/decide").json()
    assert body["state"]["resources"]["power"]["value"] == 100
    outcome = next(e for e in body["events"] if e["title"] == "Colony Resources Updated")
    assert "power: +1.50" in outcome["message"]
    assert "fuel: -3.00" in outcome["message"]
    # Execution used the catalog and clamp, not the forged +50 graph delta.
    assert body["state"]["resources"]["fuel"]["value"] == 56.9


def test_repair_execution_and_timeline_match(live, monkeypatch):
    client, simulator = live
    graph = AsyncMock()
    graph.ainvoke.return_value = {"approved_actions": ["repair_module"]}
    monkeypatch.setattr(main, "default_graph", graph)
    body = client.post("/agents/decide").json()
    comms = next(m for m in body["state"]["modules"] if m["id"] == "comms")
    assert comms["health"] == 91
    event = next(e for e in body["events"] if e["title"] == "Executed: repair_module")
    assert "Communications health: 85 → 91 (+6)" in event["message"]
    assert "fuel: -3.00" in event["message"]
    assert body["state"]["resources"]["fuel"]["value"] == 56.9


def test_execution_rechecks_exact_balances_atomically(live, monkeypatch):
    client, simulator = live
    simulator._resources["fuel"] = 2
    modules = [m.model_dump() for m in simulator.snapshot().modules]
    graph = AsyncMock()
    graph.ainvoke.return_value = {"approved_actions": ["ration_food", "repair_module"]}
    monkeypatch.setattr(main, "default_graph", graph)
    body = client.post("/agents/decide").json()
    assert body["state"]["resources"]["food"]["value"] == 69.9  # passive drain only
    assert body["state"]["resources"]["fuel"]["value"] == 1.9
    assert body["state"]["modules"] == modules
    assert not any(e["title"].startswith("Executed:") for e in body["events"])
    assert any("Plan rejected at execution" in e["message"] for e in body["events"])
    assert all(a["status"] == "idle" for a in body["state"]["agents"])
