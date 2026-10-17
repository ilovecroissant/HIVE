import logging
import os
from datetime import datetime, timezone

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

# Set HIVE_LIVE_AGENTS=true to route /agents/decide through the LangGraph pipeline.
# Default (false) uses the deterministic scripted path — safe for tests and demo.
LIVE_AGENTS = os.getenv("HIVE_LIVE_AGENTS", "false").lower() == "true"

if LIVE_AGENTS:
    from graph import default_graph, AgentGraphState
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
    return {"status": "ok", "live_agents": LIVE_AGENTS}


# ── simulation ────────────────────────────────────────────────────────────────

@app.get("/simulation/state", response_model=StateResponse)
def get_state() -> StateResponse:
    state, events = sim.get_state()
    return StateResponse(state=state, events=events)


@app.post("/simulation/reset", response_model=StateResponse)
def reset() -> StateResponse:
    state, events = sim.reset()
    return StateResponse(state=state, events=events)


class CrisisRequest(BaseModel):
    crisis: CrisisKind


@app.post("/simulation/crisis", response_model=StateResponse)
def trigger_crisis(req: CrisisRequest) -> StateResponse:
    state, events = sim.trigger_crisis(req.crisis)
    return StateResponse(state=state, events=events)


# ── agents ────────────────────────────────────────────────────────────────────

@app.post("/agents/decide", response_model=StateResponse)
async def agents_decide() -> StateResponse:
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
        sim._set_status(aid, "analyzing")  # type: ignore[attr-defined]

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

        approved_effects = result.get("approved_effects") or {}
        sim.apply_effects(approved_effects)
        sim.push_events(result.get("events") or [])

        # Emit a deterministic outcome event so the UI can confirm that the
        # simulator actually applied the commander-approved effects.
        if approved_effects:
            parts = [f"{k}: {d:+.1f}" for k, d in sorted(approved_effects.items())]
            sim.push_events([AgentEvent(
                id=f"evt-{colony.tick:05d}-sim-outcome",
                tick=colony.tick,
                timestamp=datetime.now(timezone.utc).isoformat(),
                type="outcome",
                title="Colony Resources Updated",
                message=f"Simulator applied: {', '.join(parts)}",
            )])

        errors = result.get("errors") or []
        actions = result.get("approved_actions") or []
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
            title="Graph Execution Failed",
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
    state, events = sim.commander_command(req.instruction)
    return StateResponse(state=state, events=events)
