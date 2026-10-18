"""
Pytest tests for the HIVE FastAPI backend.
Run from backend/ with:  pytest -v
"""
import sys
import os

# Ensure backend/ is on sys.path when tests are run from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from main import app
from simulator import sim

client = TestClient(app)

RESOURCES = ["oxygen", "power", "water", "food", "fuel"]


# ── helpers ───────────────────────────────────────────────────────────────────

def fresh() -> None:
    """Reset simulator state before each test."""
    sim.reset()


def get_resources(data: dict) -> dict:
    return data["state"]["resources"]


# ── health ────────────────────────────────────────────────────────────────────

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── reset ─────────────────────────────────────────────────────────────────────

def test_reset_returns_valid_envelope():
    resp = client.post("/simulation/reset")
    assert resp.status_code == 200
    data = resp.json()
    assert "state" in data
    assert "events" in data


def test_reset_tick_is_zero():
    resp = client.post("/simulation/reset")
    assert resp.json()["state"]["tick"] == 0


def test_reset_resources_in_range():
    resp = client.post("/simulation/reset")
    for key in RESOURCES:
        val = resp.json()["state"]["resources"][key]["value"]
        assert 0 <= val <= 100, f"{key}={val} out of bounds after reset"


def test_reset_no_active_crisis():
    # Trigger a crisis first, then reset
    client.post("/simulation/crisis", json={"crisis": "dust_storm"})
    resp = client.post("/simulation/reset")
    assert resp.json()["state"]["active_crisis"] is None


def test_reset_event_present():
    resp = client.post("/simulation/reset")
    events = resp.json()["events"]
    assert len(events) >= 1


# ── get state ─────────────────────────────────────────────────────────────────

def test_get_state_returns_envelope():
    fresh()
    resp = client.get("/simulation/state")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) >= {"state", "events"}


def test_get_state_has_all_resources():
    fresh()
    resp = client.get("/simulation/state")
    resources = get_resources(resp.json())
    for key in RESOURCES:
        assert key in resources, f"Missing resource: {key}"


def test_get_state_has_agents():
    fresh()
    resp = client.get("/simulation/state")
    agents = resp.json()["state"]["agents"]
    agent_ids = {a["id"] for a in agents}
    assert {"commander", "energy", "life_support", "engineering", "science"} == agent_ids


def test_get_state_has_modules():
    fresh()
    resp = client.get("/simulation/state")
    modules = resp.json()["state"]["modules"]
    assert len(modules) >= 4


# ── crisis ────────────────────────────────────────────────────────────────────

def test_dust_storm_sets_active_crisis():
    fresh()
    resp = client.post("/simulation/crisis", json={"crisis": "dust_storm"})
    assert resp.json()["state"]["active_crisis"] == "dust_storm"


def test_oxygen_leak_sets_active_crisis():
    fresh()
    resp = client.post("/simulation/crisis", json={"crisis": "oxygen_leak"})
    assert resp.json()["state"]["active_crisis"] == "oxygen_leak"


def test_power_failure_sets_active_crisis():
    fresh()
    resp = client.post("/simulation/crisis", json={"crisis": "power_failure"})
    assert resp.json()["state"]["active_crisis"] == "power_failure"


def test_dust_storm_drains_power_over_ticks():
    """After a dust storm, sustained ticks should lower power vs a control run."""
    # Control: reset + 5 decide ticks, no crisis
    sim.reset()
    for _ in range(5):
        client.post("/agents/decide")
    power_no_crisis = client.get("/simulation/state").json()["state"]["resources"]["power"]["value"]

    # Crisis run: reset + storm + 5 decide ticks
    sim.reset()
    client.post("/simulation/crisis", json={"crisis": "dust_storm"})
    for _ in range(5):
        client.post("/agents/decide")
    power_with_storm = client.get("/simulation/state").json()["state"]["resources"]["power"]["value"]

    assert power_with_storm < power_no_crisis, (
        f"Dust storm should reduce power: {power_with_storm} < {power_no_crisis}"
    )


def test_oxygen_leak_drains_oxygen_over_ticks():
    sim.reset()
    baseline = client.get("/simulation/state").json()["state"]["resources"]["oxygen"]["value"]

    sim.reset()
    client.post("/simulation/crisis", json={"crisis": "oxygen_leak"})
    for _ in range(3):
        client.post("/agents/decide")
    after = client.get("/simulation/state").json()["state"]["resources"]["oxygen"]["value"]

    assert after < baseline, f"Oxygen leak should drain oxygen: {after} < {baseline}"


# ── resource boundaries ───────────────────────────────────────────────────────

def test_resources_bounded_after_many_ticks():
    fresh()
    for _ in range(30):
        client.post("/agents/decide")
    resp = client.get("/simulation/state")
    for key in RESOURCES:
        val = resp.json()["state"]["resources"][key]["value"]
        assert 0 <= val <= 100, f"{key}={val} out of [0,100] after 30 ticks"


def test_survival_score_bounded():
    fresh()
    for _ in range(20):
        client.post("/agents/decide")
    score = client.get("/simulation/state").json()["state"]["survival_score"]
    assert 0 <= score <= 100, f"survival_score={score} out of bounds"


# ── agents/decide ─────────────────────────────────────────────────────────────

def test_agents_decide_returns_events():
    fresh()
    resp = client.post("/agents/decide")
    assert resp.status_code == 200
    assert len(resp.json()["events"]) > 0


def test_agents_decide_increments_tick():
    fresh()
    tick_before = client.get("/simulation/state").json()["state"]["tick"]
    client.post("/agents/decide")
    tick_after = client.get("/simulation/state").json()["state"]["tick"]
    assert tick_after == tick_before + 1


# ── commander/command ─────────────────────────────────────────────────────────

def test_commander_command_ok():
    fresh()
    resp = client.post("/commander/command", json={"instruction": "Ration food supplies now"})
    assert resp.status_code == 200


def test_commander_command_produces_commander_event():
    fresh()
    resp = client.post("/commander/command", json={"instruction": "Ration food supplies now"})
    events = resp.json()["events"]
    assert any(e["type"] == "commander" for e in events)


def test_commander_clear_removes_crisis():
    fresh()
    client.post("/simulation/crisis", json={"crisis": "dust_storm"})
    client.post("/commander/command", json={"instruction": "all clear"})
    state = client.get("/simulation/state").json()["state"]
    assert state["active_crisis"] is None


# ── event schema ──────────────────────────────────────────────────────────────

def test_event_ids_stable_across_gets():
    fresh()
    ids1 = {e["id"] for e in client.get("/simulation/state").json()["events"]}
    ids2 = {e["id"] for e in client.get("/simulation/state").json()["events"]}
    assert ids1 == ids2


def test_event_timestamps_are_utc_iso():
    fresh()
    for evt in client.get("/simulation/state").json()["events"]:
        ts: str = evt["timestamp"]
        assert "T" in ts, f"Timestamp missing 'T': {ts}"
        # Ends with Z or has +/-offset after date
        assert ts.endswith("Z") or "+" in ts[10:] or ts[19] == "-", \
            f"Timestamp not UTC ISO: {ts}"


def test_event_has_required_fields():
    fresh()
    for evt in client.get("/simulation/state").json()["events"]:
        for field in ("id", "tick", "timestamp", "type", "title", "message"):
            assert field in evt, f"Event missing field '{field}': {evt}"


# ── mode field ────────────────────────────────────────────────────────────────

def test_state_mode_is_live():
    fresh()
    assert client.get("/simulation/state").json()["state"]["mode"] == "live"


def test_health_reports_actual_judging_configuration(monkeypatch):
    import main
    monkeypatch.setattr(main, "LIVE_AGENTS", True)
    monkeypatch.setattr(main, "GRAPH_VERSION", "v2")
    monkeypatch.setenv("HIVE_SPECIALIST_ROUTING", "true")
    monkeypatch.setenv("HIVE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    config = client.get("/health").json()
    assert config["selection_policy"] == "v2.1"
    assert config["specialist_routing"] is True
    assert config["physics_version"] == sim.physics_version
    assert config["model"] == "Qwen/Qwen2.5-7B-Instruct"
    monkeypatch.setenv("HIVE_SPECIALIST_ROUTING", "false")
    assert client.get("/health").json()["specialist_routing"] is False
    monkeypatch.setattr(main, "LIVE_AGENTS", False)
    config = client.get("/health").json()
    assert config["model"] is None
    assert config["specialist_routing"] is False
