"""
Graph tests using stub LLMs — no real model calls, no network.

Coverage:
- validate_action catalog and resource bounds
- Shared state: proposals from concurrent nodes merged via operator.add
- Proposal handoff: commander receives all proposals
- Commander approval + event types
- Disagreement events for rejected proposals
- Validation rejects unknown / unaffordable actions
- Model failure → error events, no resource changes
- Commander blocked when specialist fails
- JSON extraction helper (_extract_json)
- Live API status cleanup is covered separately in test_live_api.py
"""
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import AsyncMock, MagicMock

import pytest

from graph import (
    AgentGraphState,
    CommanderDecision,
    Proposal,
    VALID_ACTIONS,
    build_graph,
    validate_action,
    _extract_json,
)
from models import Agent, AgentEvent, ColonyState, Module, Resource


# ── helpers ───────────────────────────────────────────────────────────────────

def make_colony(
    resources: dict[str, float] | None = None,
    crisis: str | None = None,
) -> ColonyState:
    defaults = {"oxygen": 75.0, "power": 70.0, "water": 65.0, "food": 60.0, "fuel": 55.0}
    r_vals = {**defaults, **(resources or {})}
    return ColonyState(
        simulation_id="test-sim",
        tick=5,
        sol=0,
        hour=5,
        survival_score=70.0,
        resources={k: Resource(value=v, trend=0.0) for k, v in r_vals.items()},
        modules=[Module(id="habitat", label="Habitat Alpha", type="habitat", health=90.0)],
        agents=[
            Agent(id="energy",       name="ARIA",  role="Energy",       status="idle"),
            Agent(id="life_support", name="LYRA",  role="Life Support", status="idle"),
            Agent(id="engineering",  name="NEXUS", role="Engineering",  status="idle"),
            Agent(id="commander",    name="Hayes", role="Commander",    status="idle"),
        ],
        active_crisis=crisis,  # type: ignore[arg-type]
        mode="live",
    )


def make_initial_state(colony: ColonyState | None = None) -> AgentGraphState:
    return {
        "colony":           colony or make_colony(),
        "proposals":        [],
        "events":           [],
        "errors":           [],
        "approved_effects": {},
        "approved_actions": [],
    }


def make_stub_factory(
    proposal_json: str | None = None,
    decision_json: str | None = None,
    fail: bool = False,
    fail_decision: bool = False,
) -> object:
    """
    Returns a factory callable that produces mock ChatOpenAI instances.
    Routing is done by inspecting the system-message content so the stub
    is robust to asyncio concurrency ordering:
      • System message contains "Commander Hayes" → commander call → decision JSON
      • Otherwise → specialist call → proposal JSON
    """
    _prop_json = proposal_json or (
        '{"agent_id":"energy","action":"conserve_power",'
        '"rationale":"Power is low.","priority":3}'
    )
    _dec_json = decision_json or (
        '{"approved_actions":["conserve_power"],'
        '"rationale":"Energy conservation is critical.","disagreements":[]}'
    )

    def factory():
        mock_llm = MagicMock()

        if fail:
            # All calls fail (specialist failure tests)
            mock_llm.ainvoke = AsyncMock(side_effect=Exception("Simulated model failure"))
        else:
            async def ainvoke_side_effect(messages):
                system_content = messages[0].content if messages else ""
                is_commander = "Commander Hayes" in system_content
                if is_commander and fail_decision:
                    raise Exception("Simulated commander failure")
                content = _dec_json if is_commander else _prop_json
                resp = MagicMock()
                resp.content = content
                return resp

            mock_llm.ainvoke = ainvoke_side_effect

        return mock_llm

    return factory


def run_graph(state: AgentGraphState, factory) -> AgentGraphState:
    """Synchronous helper that runs the async graph in a fresh event loop."""
    graph = build_graph(make_llm_fn=factory)
    return asyncio.run(graph.ainvoke(state))


# ── _extract_json ─────────────────────────────────────────────────────────────

def test_extract_json_plain():
    assert _extract_json('{"a":1}') == {"a": 1}

def test_extract_json_fenced():
    assert _extract_json('```json\n{"a":1}\n```') == {"a": 1}

def test_extract_json_with_prose():
    assert _extract_json('Here is the result: {"a":1} done.') == {"a": 1}

def test_extract_json_none():
    assert _extract_json("no json here") is None

def test_extract_json_empty():
    assert _extract_json("") is None


# ── validate_action ────────────────────────────────────────────────────────────

def test_validate_known_action_ok():
    ok, reason, effects = validate_action("conserve_power", {"power": 50.0, "fuel": 30.0})
    assert ok is True
    assert "power" in effects

def test_validate_unknown_action_rejected():
    ok, reason, effects = validate_action("launch_rocket", {"fuel": 100.0})
    assert ok is False
    assert "Unknown" in reason
    assert effects == {}

def test_validate_insufficient_resource_rejected():
    ok, reason, effects = validate_action("repair_module", {"fuel": 1.0})
    assert ok is False
    assert "fuel" in reason.lower()

def test_validate_idle_always_ok():
    ok, _, effects = validate_action("idle", {})
    assert ok is True
    assert effects == {}

def test_validate_exact_resource_boundary():
    ok, _, _ = validate_action("boost_oxygen", {"oxygen": 50.0, "water": 4.0, "power": 10.0})
    assert ok is True

def test_validate_just_below_boundary():
    ok, _, _ = validate_action("boost_oxygen", {"oxygen": 50.0, "water": 3.99, "power": 10.0})
    assert ok is False


# ── shared state: proposals merged from concurrent nodes ──────────────────────

def test_proposals_from_all_specialists_merged():
    result = run_graph(make_initial_state(), make_stub_factory())
    assert len(result["proposals"]) == 3

def test_proposals_have_correct_agent_ids():
    result = run_graph(make_initial_state(), make_stub_factory())
    ids = {p.agent_id for p in result["proposals"]}
    assert ids == {"energy", "life_support", "engineering"}

def test_events_accumulated_from_all_nodes():
    result = run_graph(make_initial_state(), make_stub_factory())
    # 3 specialists × ≥2 events + commander events
    assert len(result["events"]) >= 6


# ── proposal handoff to commander ─────────────────────────────────────────────

def test_commander_receives_proposals():
    result = run_graph(make_initial_state(), make_stub_factory())
    decision_events = [e for e in result["events"] if e.type == "decision"]
    assert len(decision_events) >= 1

def test_commander_decision_event_mentions_approved_action():
    result = run_graph(make_initial_state(), make_stub_factory())
    decision_events = [e for e in result["events"] if e.type == "decision"]
    titles = " ".join(e.title for e in decision_events)
    assert "conserve_power" in titles or "Approves" in titles


# ── validation inside commander ───────────────────────────────────────────────

def test_commander_rejects_unknown_action_via_validation():
    factory = make_stub_factory(
        decision_json='{"approved_actions":["launch_rocket"],'
                      '"rationale":"Go big.","disagreements":[]}'
    )
    result = run_graph(make_initial_state(), factory)
    assert "launch_rocket" not in result["approved_actions"]

def test_commander_rejects_unaffordable_action():
    factory = make_stub_factory(
        decision_json='{"approved_actions":["repair_module"],'
                      '"rationale":"Fix it.","disagreements":[]}'
    )
    colony = make_colony(resources={"fuel": 1.0})  # repair_module needs 3
    result = run_graph(make_initial_state(colony), factory)
    assert "repair_module" not in result["approved_actions"]

def test_approved_effects_match_catalog():
    factory = make_stub_factory(
        decision_json='{"approved_actions":["conserve_power"],'
                      '"rationale":"OK.","disagreements":[]}'
    )
    result = run_graph(make_initial_state(), factory)
    if "conserve_power" in result["approved_actions"]:
        expected = VALID_ACTIONS["conserve_power"]
        for res, delta in expected.items():
            assert result["approved_effects"].get(res) == pytest.approx(delta)


# ── model failure handling ─────────────────────────────────────────────────────

def test_specialist_failure_produces_error_event():
    result = run_graph(make_initial_state(), make_stub_factory(fail=True))
    error_events = [e for e in result["events"] if e.type == "error"]
    assert len(error_events) >= 1

def test_specialist_failure_no_resource_changes():
    """Bug regression: stuck ANALYZING — approved_effects must be empty on failure."""
    result = run_graph(make_initial_state(), make_stub_factory(fail=True))
    assert result["errors"], "Expected errors in state"
    assert result["approved_effects"] == {}, \
        "No resource changes must occur when model fails"

def test_specialist_failure_blocks_commander():
    result = run_graph(make_initial_state(), make_stub_factory(fail=True))
    blocked = [e for e in result["events"] if "Blocked" in e.title or e.type == "error"]
    assert len(blocked) >= 1

def test_commander_failure_no_resource_changes():
    """Commander model failure must not apply any resource effects."""
    factory = make_stub_factory(fail_decision=True)
    result = run_graph(make_initial_state(), factory)
    # Specialists succeed, commander fails → error event, no effects
    assert result["approved_effects"] == {}
    assert result["approved_actions"] == []


# ── event schema ──────────────────────────────────────────────────────────────

def test_all_events_have_required_fields():
    result = run_graph(make_initial_state(), make_stub_factory())
    for evt in result["events"]:
        assert evt.id
        assert evt.tick == 5
        assert evt.timestamp
        assert evt.type
        assert evt.title
        assert evt.message

def test_event_types_are_valid():
    valid_types = {"system","proposal","disagreement","decision","outcome","error","commander"}
    result = run_graph(make_initial_state(), make_stub_factory())
    for evt in result["events"]:
        assert evt.type in valid_types, f"Invalid type: {evt.type}"

def test_event_ids_are_unique():
    result = run_graph(make_initial_state(), make_stub_factory())
    ids = [e.id for e in result["events"]]
    assert len(ids) == len(set(ids)), f"Duplicate IDs: {ids}"


# ── commander JSON repair ──────────────────────────────────────────────────────

def test_commander_retries_on_malformed_json():
    """
    When the commander's first response is not valid JSON, the node must retry
    once and succeed if the second response is valid.
    Commander call count must be exactly 2.
    """
    _prop_json = (
        '{"agent_id":"energy","action":"conserve_power",'
        '"rationale":"Low power.","priority":3}'
    )
    _dec_json = (
        '{"approved_actions":["conserve_power"],'
        '"rationale":"Energy conservation is critical.","disagreements":[]}'
    )
    commander_calls = [0]

    def factory():
        mock_llm = MagicMock()

        async def ainvoke(messages):
            system_content = messages[0].content if messages else ""
            is_commander = "Commander Hayes" in system_content
            if is_commander:
                commander_calls[0] += 1
                resp = MagicMock()
                # First call → unparseable prose; second (retry) → valid JSON
                resp.content = "I approve the proposals." if commander_calls[0] == 1 else _dec_json
                return resp
            resp = MagicMock()
            resp.content = _prop_json
            return resp

        mock_llm.ainvoke = ainvoke
        return mock_llm

    result = run_graph(make_initial_state(), factory)
    assert "conserve_power" in result["approved_actions"], "Expected approval after retry"
    assert commander_calls[0] == 2, f"Expected 2 commander calls, got {commander_calls[0]}"


def test_commander_approved_actions_as_string_normalized():
    """
    When the model returns approved_actions as a bare string instead of a list,
    _normalize_commander_data must coerce it so the pipeline still succeeds.
    """
    factory = make_stub_factory(
        decision_json=(
            '{"approved_actions":"conserve_power",'
            '"rationale":"OK.","disagreements":[]}'
        )
    )
    result = run_graph(make_initial_state(), factory)
    assert "conserve_power" in result["approved_actions"]


@pytest.mark.parametrize("bad_content", ["", "[]", '{"rationale":42}'])
def test_commander_repairs_empty_nonobject_and_invalid_schema(bad_content):
    calls = []
    base_factory = make_stub_factory()

    def factory():
        llm = base_factory()
        original = llm.ainvoke

        async def invoke(messages):
            if "Commander Hayes" in messages[0].content:
                calls.append(messages)
                if len(calls) == 1:
                    return MagicMock(content=bad_content)
            return await original(messages)

        llm.ainvoke = invoke
        return llm

    result = run_graph(make_initial_state(), factory)
    assert result["approved_actions"] == ["conserve_power"]
    assert len(calls) == 2


def test_commander_duplicate_action_applied_once():
    factory = make_stub_factory(decision_json=(
        '{"approved_actions":["conserve_power","conserve_power"],'
        '"rationale":"Preserve power."}'
    ))
    result = run_graph(make_initial_state(), factory)
    assert result["approved_actions"] == ["conserve_power"]
    assert result["approved_effects"] == VALID_ACTIONS["conserve_power"]
    assert any(c.constraint == "duplicate_action" for c in result["conflicts"])


def test_commander_checks_combined_resource_cost():
    factory = make_stub_factory(decision_json=(
        '{"approved_actions":["conserve_power","emergency_power"],'
        '"rationale":"Preserve power."}'
    ))
    result = run_graph(make_initial_state(make_colony(resources={"fuel": 4})), factory)
    assert result["approved_actions"] == ["conserve_power"]
    assert result["approved_effects"]["fuel"] == -2


def test_commander_stops_after_one_unsuccessful_repair():
    calls = []
    base_factory = make_stub_factory()

    def factory():
        llm = base_factory()
        original = llm.ainvoke

        async def invoke(messages):
            if "Commander Hayes" in messages[0].content:
                calls.append(messages)
                return MagicMock(content="not JSON")
            return await original(messages)

        llm.ainvoke = invoke
        return llm

    result = run_graph(make_initial_state(), factory)
    assert len(calls) == 2
    assert result["approved_actions"] == []
    assert result["approved_effects"] == {}
    assert result["errors"]
