"""
HIVE multi-agent LangGraph implementation.

Graph flow (async fan-out / fan-in):
  START → [energy, life_support, engineering]  ← run concurrently via ainvoke
         ↓         ↓              ↓
        commander (fan-in, runs after all specialists)
         ↓
        END

Key design decisions
────────────────────
• Specialist nodes are async → LangGraph parallelises them with asyncio.gather
  when the graph is invoked via ainvoke().
• We call llm.ainvoke() directly and parse JSON manually instead of using
  with_structured_output(), because json_mode + explicit schema in the prompt
  is the most compatible path for vLLM-served smaller models (Qwen, Mistral, etc.).
• TIMEOUT defaults to 12 s per model request; SDK transport retries may extend it.
• Commander gets one JSON repair retry; failures emit error events.
• All exceptions are caught inside node functions; the graph always completes.
"""
from __future__ import annotations

import json
import logging
import operator
import os
import re
import time
from datetime import datetime, timezone
from itertools import combinations
from typing import Annotated, Callable, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import TypedDict

from models import AgentEvent, AgentId, ColonyState
from actions import (
    VALID_ACTIONS, VALID_ACTION_LIST, validate_action,
    Conflict, SCORE_WEIGHTS, describe_action, detect_conflicts, evaluate_plan, format_assessment,
)

log = logging.getLogger("hive.graph")

# ── config (all read at import time from env) ─────────────────────────────────

OPENAI_API_BASE  = os.getenv("OPENAI_API_BASE",     "http://localhost:11434/v1")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY",      "ollama")
HIVE_MODEL       = os.getenv("HIVE_MODEL",           "gpt-4o-mini")
MAX_TOKENS       = int(os.getenv("HIVE_MAX_TOKENS",     "300"))
CMD_MAX_TOKENS   = int(os.getenv("HIVE_CMD_MAX_TOKENS", "500"))  # higher budget for commander
TIMEOUT          = float(os.getenv("HIVE_TIMEOUT", "12"))

# ── validated action catalog ──────────────────────────────────────────────────

# ── Pydantic schemas (for validation only — LLM output is parsed manually) ───

class Proposal(BaseModel):
    agent_id: str  = Field(description="ID of the proposing agent")
    action:   str  = Field(description=f"Exactly one of: {VALID_ACTION_LIST}")
    rationale:str  = Field(description="1-2 sentence explanation")
    priority: int  = Field(ge=1, le=5, description="1=low … 5=critical")
    # Filled from simulator rules after parsing, never from LLM estimates.
    assessment: dict = Field(default_factory=dict)


class CommanderDecision(BaseModel):
    approved_actions: list[str] = Field(default_factory=list, max_length=len(VALID_ACTIONS))
    rationale:        str       = Field(description="Why these actions were chosen")
    disagreements:    list[str] = Field(default_factory=list)
    rejection_reasons: dict[str, str] = Field(default_factory=dict)


# ── LangGraph shared state ────────────────────────────────────────────────────

class AgentGraphState(TypedDict):
    colony:           ColonyState
    proposals:        Annotated[list[Proposal],   operator.add]
    events:           Annotated[list[AgentEvent], operator.add]
    errors:           Annotated[list[str],         operator.add]
    # Written once by commander; no reducer needed
    approved_effects: dict[str, float]
    approved_actions: list[str]
    conflicts: list[Conflict]


# ── helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _evt_id(tick: int, tag: str) -> str:
    return f"evt-{tick:05d}-{tag}"


def _event(
    tick:     int,
    tag:      str,
    evt_type: str,
    title:    str,
    message:  str,
    agent_id: Optional[AgentId] = None,
) -> AgentEvent:
    return AgentEvent(
        id=_evt_id(tick, tag),
        tick=tick,
        timestamp=_now_iso(),
        type=evt_type,       # type: ignore[arg-type]
        agent_id=agent_id,
        title=title,
        message=message,
    )


def _resource_summary(colony: ColonyState) -> str:
    lines = []
    for k, r in colony.resources.items():
        trend = f" (trend {r.trend:+.1f})" if r.trend is not None else ""
        lines.append(f"  {k}: {r.value:.1f}{trend}")
    return "\n".join(lines)


def _action_context(colony: ColonyState, actions: list[str]) -> str:
    resources = {k: r.value for k, r in colony.resources.items()}
    return json.dumps({a: describe_action(a, resources, colony.modules) for a in actions}, separators=(",", ":"))


def _make_llm(max_tokens: int | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=HIVE_MODEL,
        base_url=OPENAI_API_BASE,
        api_key=OPENAI_API_KEY,
        max_tokens=max_tokens or MAX_TOKENS,
        timeout=TIMEOUT,
    )


def _extract_json(text: str) -> dict | None:
    """
    Robustly extract the first JSON object from model output.
    Handles: plain JSON, ```json blocks, JSON surrounded by prose.
    """
    text = text.strip()
    # 1. Direct parse
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    # 2. Code-fenced block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 3. First {...} in the text (greedy but usually fine for single objects)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _normalize_commander_data(data: dict) -> dict:
    """
    Fix common model JSON quirks in commander output before Pydantic validation.
    Handles: approved_actions as bare string, null approved_actions, missing fields.
    """
    aa = data.get("approved_actions")
    if isinstance(aa, str):
        # e.g. "approved_actions": "conserve_power"  (string, not list)
        data["approved_actions"] = [aa.strip()] if aa.strip() else []
    elif not isinstance(aa, list):
        data["approved_actions"] = []
    # Drop blank or non-string entries
    data["approved_actions"] = [
        str(a).strip() for a in data["approved_actions"]
        if a and str(a).strip()
    ]
    if not data.get("rationale"):
        data["rationale"] = "No rationale provided."
    if not isinstance(data.get("disagreements"), list):
        data["disagreements"] = []
    if data.get("rejection_reasons") is None:
        data["rejection_reasons"] = {}
    return data


# ── specialist configs ────────────────────────────────────────────────────────

SPECIALIST_CFG: dict[str, dict] = {
    "energy": {
        "name": "ARIA",
        "role": "Energy Systems Agent",
        "focus": "power reserves, solar efficiency, fuel cells",
        "preferred": ["conserve_power", "battery_reserves", "emergency_power"],
    },
    "life_support": {
        "name": "LYRA",
        "role": "Life Support Agent",
        "focus": "oxygen recycling, water processing, atmospheric stability",
        "preferred": ["boost_oxygen", "boost_water_recycling"],
    },
    "engineering": {
        "name": "NEXUS",
        "role": "Engineering Agent",
        "focus": "module health, structural repairs, maintenance",
        "preferred": ["repair_module"],
    },
}


# ── specialist node (async) ───────────────────────────────────────────────────

async def _run_specialist_async(
    agent_id:      str,
    state:         AgentGraphState,
    make_llm_fn:   Callable,
) -> dict:
    colony     = state["colony"]
    tick       = colony.tick
    cfg        = SPECIALIST_CFG[agent_id]
    events_out: list[AgentEvent] = []
    t0 = time.monotonic()

    log.info("[%s] node START tick=%d crisis=%s", agent_id, tick, colony.active_crisis)

    events_out.append(_event(
        tick, f"analyzing-{agent_id}", "proposal",
        f"{cfg['name']} Analyzing",
        f"{cfg['name']} ({cfg['role']}) assessing colony status.",
        agent_id=agent_id,  # type: ignore[arg-type]
    ))

    system_prompt = (
        f"You are {cfg['name']}, the {cfg['role']} for HIVE Mars colony.\n"
        f"Responsibility: {cfg['focus']}.\n\n"
        f"Colony status:\n{_resource_summary(colony)}\n"
        f"Active crisis: {colony.active_crisis or 'none'}\n"
        f"Survival score: {colony.survival_score}\n\n"
        f"Module health: {', '.join(f'{m.label}={m.health:g}' for m in colony.modules)}\n"
        f"Preferred actions for your role: {', '.join(cfg['preferred'])}\n\n"
        f"Simulator estimates for your actions and idle: {_action_context(colony, cfg['preferred'] + ['idle'])}\n"
        "Choose a feasible action for current needs; idle if no useful work is safe. "
        "Explain the benefit and tradeoff briefly. Do not invent effects or costs.\n"
        "RESPOND WITH ONLY VALID JSON — no markdown, no explanation, nothing else.\n"
        "Required format (replace values appropriately):\n"
        '{"agent_id":"' + agent_id + '",'
        '"action":"<one of: ' + VALID_ACTION_LIST + '>",'
        '"rationale":"<1-2 sentence reason>",'
        '"priority":<integer 1-5>}'
    )

    try:
        llm = make_llm_fn()
        log.info("[%s] inference START", agent_id)
        inf_t0 = time.monotonic()

        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="Propose your best action to protect the colony."),
        ])

        elapsed = time.monotonic() - inf_t0
        content = (response.content or "").strip()
        log.info("[%s] inference DONE %.2fs content_len=%d", agent_id, elapsed, len(content))

        if not content:
            raise ValueError("Model returned empty response")

        data = _extract_json(content)
        if data is None:
            raise ValueError(f"Cannot parse JSON from: {content[:300]}")

        # Force the correct agent_id regardless of model output
        data["agent_id"] = agent_id
        proposal = Proposal.model_validate(data)

        # Clamp action to catalog (model might hallucinate a close name)
        if proposal.action not in VALID_ACTIONS:
            log.warning("[%s] unknown action '%s', falling back to idle", agent_id, proposal.action)
            proposal.action = "idle"
            proposal.rationale += " [action corrected to idle — unknown value returned]"

        proposal.assessment = describe_action(
            proposal.action, {k: r.value for k, r in colony.resources.items()}, colony.modules,
        )

        events_out.append(_event(
            tick, f"proposal-{agent_id}", "proposal",
            f"{cfg['name']} Proposes: {proposal.action}",
            f"Priority {proposal.priority}/5. {proposal.rationale} Simulator estimate: " + format_assessment(proposal.assessment),
            agent_id=agent_id,  # type: ignore[arg-type]
        ))
        log.info("[%s] node DONE %.2fs action=%s priority=%d",
                 agent_id, time.monotonic() - t0, proposal.action, proposal.priority)
        return {"proposals": [proposal], "events": events_out, "errors": []}

    except Exception as exc:
        elapsed = time.monotonic() - t0
        err_msg = f"{cfg['name']} ({agent_id}) error after {elapsed:.1f}s: {type(exc).__name__}: {exc}"
        log.error("[%s] node ERROR %.2fs %s", agent_id, elapsed, exc)
        events_out.append(_event(
            tick, f"error-{agent_id}", "error",
            f"{cfg['name']} Model Error",
            err_msg,
            agent_id=agent_id,  # type: ignore[arg-type]
        ))
        return {"proposals": [], "events": events_out, "errors": [err_msg]}


# ── commander node (async) ────────────────────────────────────────────────────

async def _run_commander_async(
    state:       AgentGraphState,
    make_llm_fn: Callable,
) -> dict:
    colony    = state["colony"]
    tick      = colony.tick
    proposals = state.get("proposals", [])
    errors    = state.get("errors", [])
    events_out: list[AgentEvent] = []
    t0 = time.monotonic()

    log.info("[commander] node START proposals=%d errors=%d", len(proposals), len(errors))

    # Block if any specialist failed
    if errors:
        msg = f"Cannot coordinate — {len(errors)} specialist error(s): {'; '.join(errors)}"
        log.warning("[commander] BLOCKED: %s", msg)
        events_out.append(_event(
            tick, "commander-blocked", "error",
            "Commander Blocked: Specialist Failure", msg,
            agent_id="commander",
        ))
        return {"events": events_out, "approved_effects": {}, "approved_actions": []}

    if not proposals:
        events_out.append(_event(
            tick, "commander-noop", "outcome",
            "No Proposals Received",
            "Commander received no proposals. No actions taken this tick.",
            agent_id="commander",
        ))
        return {"events": events_out, "approved_effects": {}, "approved_actions": []}

    resources = {k: r.value for k, r in colony.resources.items()}
    proposed = list(dict.fromkeys(p.action for p in proposals))
    conflicts = detect_conflicts([p.action for p in proposals], resources, colony.modules)
    for index, conflict in enumerate(conflicts):
        events_out.append(_event(
            tick, f"conflict-{index}", "disagreement", f"Conflict: {', '.join(conflict.actions)}",
            f"Constraint {conflict.constraint}: {conflict.explanation}",
        ))
    feasible_subsets = []
    for size in range(1, len(proposed) + 1):
        for subset in combinations(proposed, size):
            preview = evaluate_plan(list(subset), resources, colony.modules)
            if preview.feasible:
                feasible_subsets.append({
                    "actions": subset,
                    "survival_delta": round(sum(SCORE_WEIGHTS[r] * d for r, d in preview.effects.items()), 2),
                    "lowest_essential_reserve": round(min(
                        resources[r] + preview.effects.get(r, 0) for r in ("oxygen", "power", "water", "food")
                    ), 2),
                })
    summary = "\n".join(
        f"- {p.agent_id}: action={p.action}, priority={p.priority}, rationale={p.rationale}"
        for p in proposals
    )
    system_prompt = (
        "You are Commander Hayes, Mission Commander of HIVE Mars colony.\n"
        "Coordinate specialist proposals into a coherent action plan.\n\n"
        f"Colony status:\n{_resource_summary(colony)}\n"
        f"Active crisis: {colony.active_crisis or 'none'}\n"
        f"Survival score: {colony.survival_score}\n\n"
        f"Module health: {', '.join(f'{m.label}={m.health:g}' for m in colony.modules)}\n"
        f"Specialist proposals:\n{summary}\n\n"
        f"Simulator estimates: {_action_context(colony, proposed)}\n"
        f"Detected conflicts: {json.dumps([c.model_dump() for c in conflicts], separators=(',', ':'))}\n"
        f"Feasible subsets: {json.dumps(feasible_subsets, separators=(',', ':'))}\n"
        "Choose ONLY proposed actions. Compare urgency, lowest life-critical reserves, "
        "crisis, module health and survival-score impact; do not merely approve every proposal. "
        "Stabilize the lowest essential reserves before optional maintenance; compare the feasible subsets. "
        "Do not call a rejected action unaffordable if it fits alongside your selected actions. "
        "Reserve all costs from starting balances; gains cannot fund this cycle. "
        "Battery and emergency power are exclusive backup-controller modes. "
        "Select idle alone, or [] to hold. Explain each rejection with an action-keyed reason. "
        "Use short public decision summaries, not step-by-step private reasoning.\n\n"
        "RESPOND WITH ONLY VALID JSON — no markdown, no explanation, nothing else.\n"
        "Required format:\n"
        '{"approved_actions":["<action1>","<action2>"],'
        '"rationale":"<1-2 sentence reason>",'
        '"rejection_reasons":{"<rejected_action>":"<short reason>"},'
        '"disagreements":[]}'
        "\napproved_actions may be empty [] if no action is safe to take."
    )

    events_out.append(_event(
        tick, "commander-reviewing", "decision",
        f"Commander Reviewing {len(proposals)} Proposals",
        "Commander Hayes coordinating specialist recommendations.",
        agent_id="commander",
    ))

    decision: CommanderDecision | None = None
    try:
        llm = make_llm_fn()
        log.info("[commander] inference START")
        inf_t0 = time.monotonic()

        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="Issue your coordinated action plan."),
        ])

        elapsed = time.monotonic() - inf_t0
        content = (response.content or "").strip()
        log.info("[commander] inference DONE %.2fs content_len=%d", elapsed, len(content))

        # One repair attempt for empty, malformed, or schema-invalid JSON.
        for attempt in range(2):
            try:
                data = _extract_json(content)
                if data is None:
                    raise ValueError(f"Expected a JSON object: {content[:200]}")
                decision = CommanderDecision.model_validate(_normalize_commander_data(data))
                selected = list(dict.fromkeys(decision.approved_actions))
                unproposed = [a for a in selected if a not in proposed]
                preview = evaluate_plan(selected, resources, colony.modules)
                if not attempt and (unproposed or not preview.feasible):
                    reasons = [c.explanation for c in preview.conflicts]
                    if unproposed:
                        reasons.append(f"Actions not proposed: {unproposed}")
                    raise ValueError("Infeasible plan: " + "; ".join(reasons))
                break
            except (ValueError, TypeError) as exc:
                if attempt:
                    raise
                log.warning("[commander] Invalid decision — retrying with stricter prompt")
                events_out.append(_event(
                    tick, "commander-retry", "system", "Commander Plan Requires Revision",
                    f"Validation rejected the first response: {str(exc)[:500]}. Requesting one corrected plan.",
                    agent_id="commander",
                ))
                retry_response = await llm.ainvoke([
                    SystemMessage(
                        content=system_prompt
                        + "\n\nCRITICAL: Your ENTIRE response must be ONLY the JSON object."
                        " Follow the required field types exactly. No explanation, no markdown."
                        " Begin immediately with '{' and end with '}'."
                    ),
                    HumanMessage(content=f"Correct this validation failure: {str(exc)[:600]}. Output a feasible subset as raw JSON only."),
                ])
                content = (retry_response.content or "").strip()

    except Exception as exc:
        elapsed = time.monotonic() - t0
        err_msg = f"Commander error after {elapsed:.1f}s: {type(exc).__name__}: {exc}"
        log.error("[commander] node ERROR %.2fs %s", elapsed, exc)
        events_out.append(_event(
            tick, "commander-error", "error",
            "Commander Model Error", err_msg,
            agent_id="commander",
        ))
        return {
            "events": events_out, "errors": [err_msg],
            "approved_effects": {}, "approved_actions": [], "conflicts": conflicts,
        }

    # Validate approved actions against deterministic simulation rules
    applied:  list[str]        = []
    rejected: dict[str, str]   = {}

    # A plan selects actions; repeating a name must not multiply its effects.
    for action in dict.fromkeys(decision.approved_actions):
        preview = evaluate_plan(applied + [action], resources, colony.modules)
        if action not in proposed:
            rejected[action] = "Action was not proposed by a specialist."
        elif preview.feasible:
            applied.append(action)
        else:
            rejected[action] = "; ".join(c.explanation for c in preview.conflicts)

    # Disagreement events for proposals not approved
    approved_set = set(applied)
    for p in proposals:
        if p.action not in approved_set:
            feasibility = evaluate_plan(applied + [p.action], resources, colony.modules)
            constraint_reasons = "; ".join(c.explanation for c in feasibility.conflicts)
            reject_reason = rejected.get(p.action) or constraint_reasons
            if not reject_reason:
                stated_reason = decision.rejection_reasons.get(p.action) or decision.rationale
                reject_reason = (
                    "Deterministic check: feasible with the chosen plan; Commander chose not to select it. "
                    f"Commander's stated preference: {stated_reason}"
                )
            events_out.append(_event(
                tick, f"disagree-{p.agent_id}", "disagreement",
                f"Commander Rejects: {p.action}", reject_reason,
                agent_id=p.agent_id,  # type: ignore[arg-type]
            ))

    if rejected:
        events_out.append(_event(
            tick, "validation-reject", "system",
            f"{len(rejected)} Action(s) Failed Validation",
            "; ".join(f"{action}: {reason}" for action, reason in rejected.items()),
        ))

    if applied:
        events_out.append(_event(
            tick, "commander-decision", "decision",
            f"Commander Approves: {', '.join(applied)}",
            decision.rationale + (
                " Deterministic validation removed unsafe/unproposed actions: " + ", ".join(rejected)
                if rejected else " All selected actions passed deterministic validation."
            ),
            agent_id="commander",
        ))
        log.info("[commander] node DONE %.2fs approved=%s",
                 time.monotonic() - t0, applied)
    else:
        events_out.append(_event(
            tick, "commander-no-approve", "outcome",
            "Commander: No Actions Approved",
            "All proposals failed validation or Commander chose not to act.",
            agent_id="commander",
        ))
        log.info("[commander] node DONE %.2fs no approvals", time.monotonic() - t0)

    return {
        "events": events_out,
        "errors": [],
        "approved_effects": evaluate_plan(applied, resources, colony.modules).effects,
        "approved_actions": applied,
        "conflicts": conflicts,
    }


# ── graph builder ─────────────────────────────────────────────────────────────

LLMFactory = Callable[[], ChatOpenAI]


def build_graph(make_llm_fn: Optional[LLMFactory] = None):
    """
    Build and compile the HIVE agent graph.
    Inject make_llm_fn to replace the real LLM in tests.
    Use graph.ainvoke() (not invoke) to parallelise specialist nodes.
    In production, commander gets a higher max_tokens budget than specialists.
    """
    _specialist_lm = make_llm_fn or _make_llm
    # Commander needs more tokens for rationale + action list; use CMD_MAX_TOKENS
    # unless a test stub is provided (stubs ignore max_tokens anyway).
    _commander_lm = make_llm_fn or (lambda: _make_llm(max_tokens=CMD_MAX_TOKENS))

    async def energy_node(state: AgentGraphState) -> dict:
        return await _run_specialist_async("energy", state, _specialist_lm)

    async def life_support_node(state: AgentGraphState) -> dict:
        return await _run_specialist_async("life_support", state, _specialist_lm)

    async def engineering_node(state: AgentGraphState) -> dict:
        return await _run_specialist_async("engineering", state, _specialist_lm)

    async def commander_node(state: AgentGraphState) -> dict:
        return await _run_commander_async(state, _commander_lm)

    workflow = StateGraph(AgentGraphState)
    workflow.add_node("energy",       energy_node)
    workflow.add_node("life_support", life_support_node)
    workflow.add_node("engineering",  engineering_node)
    workflow.add_node("commander",    commander_node)

    # Fan-out: specialists receive same colony snapshot, run concurrently
    workflow.add_edge(START,         "energy")
    workflow.add_edge(START,         "life_support")
    workflow.add_edge(START,         "engineering")

    # Fan-in: commander runs after all specialists complete
    workflow.add_edge("energy",       "commander")
    workflow.add_edge("life_support", "commander")
    workflow.add_edge("engineering",  "commander")

    workflow.add_edge("commander", END)

    return workflow.compile()


# Default compiled graph (uses real LLM)
default_graph = build_graph()
