import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import AgentEvent, CrisisKind, StateResponse
from simulator import sim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("hive.api")

# The simulator is process-local. Keep its mutations exclusive while inference
# awaits the model; reads remain available for status polling.
_mutation_lock = Lock()


@contextmanager
def _mutation_guard():
    if not _mutation_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="A colony action is already running. Try again after it completes.",
        )
    try:
        yield
    finally:
        _mutation_lock.release()


# Set HIVE_LIVE_AGENTS=true to route /agents/decide through the LangGraph pipeline.
# Default (false) uses the deterministic scripted path — safe for tests and demo.
LIVE_AGENTS = os.getenv("HIVE_LIVE_AGENTS", "false").lower() == "true"
GRAPH_VERSION = os.getenv("HIVE_GRAPH_VERSION", "v1")
if GRAPH_VERSION not in ("v1", "v2"):
    raise ValueError(f"Unknown graph version: {GRAPH_VERSION}")

if LIVE_AGENTS:
    from graph import default_graph, AgentGraphState
    if GRAPH_VERSION == "v2":
        from efficient_graph import build_efficient_graph
        default_graph = build_efficient_graph(
            routing=os.getenv("HIVE_SPECIALIST_ROUTING", "true").lower() == "true",
            horizon=int(os.getenv("HIVE_PLANNING_HORIZON", "3")),
            physics_version=sim.physics_version, status_hook=sim._set_status,
        )
    log.info("Live agent mode enabled. Graph loaded.")

app = FastAPI(title="HIVE Mars Colony API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ── health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    # Configuration only: this is not a claim that the model endpoint is healthy.
    return {
        "status": "ok", "live_agents": LIVE_AGENTS,
        "graph_version": GRAPH_VERSION,
        "selection_policy": "v2.1" if GRAPH_VERSION == "v2" else "v1",
        "specialist_routing": LIVE_AGENTS and GRAPH_VERSION == "v2"
            and os.getenv("HIVE_SPECIALIST_ROUTING", "true").lower() == "true",
        "physics_version": sim.physics_version,
        "model": os.getenv("HIVE_MODEL", "gpt-4o-mini") if LIVE_AGENTS else None,
    }


# ── simulation ────────────────────────────────────────────────────────────────

@app.get("/simulation/state", response_model=StateResponse)
def get_state() -> StateResponse:
    state, events = sim.get_state()
    return StateResponse(state=state, events=events)


@app.post("/simulation/reset", response_model=StateResponse)
def reset() -> StateResponse:
    with _mutation_guard():
        state, events = sim.reset()
        return StateResponse(state=state, events=events)


class CrisisRequest(BaseModel):
    crisis: CrisisKind


@app.post("/simulation/crisis", response_model=StateResponse)
def trigger_crisis(req: CrisisRequest) -> StateResponse:
    with _mutation_guard():
        state, events = sim.trigger_crisis(req.crisis)
        return StateResponse(state=state, events=events)


# ── agents ────────────────────────────────────────────────────────────────────

@app.post("/agents/decide", response_model=StateResponse)
async def agents_decide() -> StateResponse:
    with _mutation_guard():
        return await _run_agents_decide()


async def _run_agents_decide() -> StateResponse:
    """
    Scripted mode (HIVE_LIVE_AGENTS=false):  deterministic simulator decision.
    Live mode   (HIVE_LIVE_AGENTS=true):      LangGraph multi-agent pipeline.

    CRITICAL: agent statuses are ALWAYS reset in a finally block so that no
    agent can be left permanently stuck on "analyzing" if the request fails
    or the client times out.
    """
    if not LIVE_AGENTS:
        state, events = sim.agents_decide()
        return StateResponse(state=state, events=events)

    # ── live path ─────────────────────────────────────────────────────────────

    # Advance sim clock and expose current colony snapshot to graph
    sim.advance_tick()
    colony = sim.snapshot()

    # Mark all specialist agents as "analyzing" immediately so the UI's polling
    # sees activity before the graph completes (FastAPI runs this in an async
    # context so GET /simulation/state can be served concurrently).
    for aid in ("energy", "life_support", "engineering", "commander"):
        sim._set_status(aid, "analyzing" if GRAPH_VERSION == "v1" else "idle")  # type: ignore[attr-defined]

    log.info("agents_decide START tick=%d crisis=%s", colony.tick, colony.active_crisis)

    initial: AgentGraphState = {
        "colony":           colony,
        "proposals":        [],
        "events":           [],
        "errors":           [],
        "approved_effects": {},
        "approved_actions": [],
    }

    try:
        # ainvoke runs specialist nodes concurrently (asyncio.gather internally)
        result = await default_graph.ainvoke(initial)

        sim.push_events(result.get("events") or [])
        # Execute action names through the same rules used for feasibility.
        # Do not trust model/graph-provided numeric effects as executable input.
        actions = result.get("approved_actions") or []
        report = sim.execute_actions(actions)
        actual_effects = report.effects
        for index, execution in enumerate(report.executions):
            changes = [f"{r}: {d:+.2f}" for r, d in sorted(execution.resource_changes.items())]
            changes.extend(
                f"{m.label} health: {m.before:g} → {m.after:g} (+{m.after - m.before:g})"
                for m in execution.module_changes
            )
            sim.push_events([AgentEvent(
                id=f"evt-{colony.tick:05d}-executed-{index}", tick=colony.tick,
                timestamp=datetime.now(timezone.utc).isoformat(), type="outcome",
                title=f"Executed: {execution.action}",
                message="; ".join(changes) or "Held position; no resources spent or modules changed.",
            )])

        # Emit a deterministic outcome event so the UI can confirm that the
        # simulator actually applied the commander-approved effects.
        if actual_effects:
            parts = [f"{k}: {d:+.2f}" for k, d in sorted(actual_effects.items())]
            sim.push_events([AgentEvent(
                id=f"evt-{colony.tick:05d}-sim-outcome",
                tick=colony.tick,
                timestamp=datetime.now(timezone.utc).isoformat(),
                type="outcome",
                title="Colony Resources Updated",
                message=f"Simulator applied: {', '.join(parts)}",
            )])

        errors = result.get("errors") or []
        log.info(
            "agents_decide DONE tick=%d actions=%s errors=%d",
            colony.tick, actions, len(errors),
        )

    except Exception as exc:
        # Unexpected graph-level failure (should be rare — nodes catch internally)
        log.exception("agents_decide graph raised unexpectedly: %s", exc)
        err_evt = AgentEvent(
            id=f"evt-{colony.tick:05d}-graph-crash",
            tick=colony.tick,
            timestamp=datetime.now(timezone.utc).isoformat(),
            type="error",
            title="Decision or Execution Failed",
            message=f"Unexpected error: {type(exc).__name__}: {exc}",
        )
        sim.push_events([err_evt])

    finally:
        # ALWAYS reset agent statuses — this is the primary fix for stuck ANALYZING
        sim.reset_all_statuses()

    return StateResponse(state=sim.snapshot(), events=sim.recent_events())


# ── commander ─────────────────────────────────────────────────────────────────

class CommanderRequest(BaseModel):
    instruction: str


@app.post("/commander/command", response_model=StateResponse)
def commander_command(req: CommanderRequest) -> StateResponse:
    with _mutation_guard():
        state, events = sim.commander_command(req.instruction)
        return StateResponse(state=state, events=events)
