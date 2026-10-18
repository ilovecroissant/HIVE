"""Opt-in HIVE v2: routed specialists, one environment copy, validated choices."""
import json
import operator
import time
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from actions import describe_action, detect_conflicts, evaluate_plan, format_assessment
from decision_policy import candidate_plans, specialist_routing
from graph import (
    AgentGraphState, CMD_MAX_TOKENS, Proposal, SPECIALIST_CFG,
    _event, _extract_json, _make_llm,
)
from single_agent import decision_context


class EfficientState(AgentGraphState):
    activation: dict
    usage_records: Annotated[list[dict], operator.add]
    started_at: float


def build_efficient_graph(make_llm_fn=None, *, routing=True, horizon=3, physics_version="v1", context=None, status_hook=None):
    if horizon < 1:
        raise ValueError("Planning horizon must be positive")

    def status(aid, value):
        if status_hook:
            status_hook(aid, value)

    def environment(colony):
        if context is not None:
            return context
        # Lazy import prevents any changes to the preserved generalist prompt.
        from physics_context import shared_context
        return shared_context(colony, horizon, physics_version)

    async def invoke(role, messages, usage):
        started = time.monotonic()
        record = {"role": role, "input_tokens": None, "output_tokens": None, "total_tokens": None}
        usage.append(record)
        try:
            llm = make_llm_fn() if make_llm_fn else _make_llm(CMD_MAX_TOKENS if role == "commander" else None)
            response = await llm.ainvoke(messages)
            record.update(response.usage_metadata or {})
            return response
        finally:
            record["seconds"] = time.monotonic() - started

    async def route(state):
        colony = state["colony"]
        activation = specialist_routing(colony, horizon, physics_version) if routing else {
            aid: {"active": True, "reasons": ["Specialist routing disabled"]} for aid in SPECIALIST_CFG
        }
        events = [_event(colony.tick, "routing", "system", "Specialist Activation",
            "; ".join(f"{aid}: {'ACTIVE' if item['active'] else 'SKIPPED'} — {', '.join(item['reasons'])}" for aid, item in activation.items()))]
        return {"activation": activation, "events": events, "started_at": time.monotonic()}

    def specialist(aid):
        async def run(state):
            colony, usage = state["colony"], []
            cfg = SPECIALIST_CFG[aid]
            status(aid, "analyzing")
            prompt = (
                f"You are {cfg['name']}, HIVE's {cfg['role']}. Focus: {cfg['focus']}. "
                f"Preferred actions: {', '.join(cfg['preferred'])}. You may recommend any catalog action. "
                "Use the environment's exact costs and hazards. Prioritize essential reserves over optional work. "
                "Return only JSON: {\"action\":\"catalog_action\",\"priority\":1,\"rationale\":\"one short public reason\"}. "
                "priority is 1–5. Choose idle if no useful affordable action exists.\n" + environment(colony)
            )
            try:
                response = await invoke(aid, [SystemMessage(content=prompt), HumanMessage(content="Propose your best action.")], usage)
                data = _extract_json((response.content or "").strip())
                if data is None:
                    raise ValueError("Expected a proposal JSON object")
                proposal = Proposal.model_validate({**data, "agent_id": aid})
                proposal.assessment = describe_action(proposal.action, {r: v.value for r, v in colony.resources.items()}, colony.modules)
                status(aid, "proposing")
                return {"proposals": [proposal], "usage_records": usage, "events": [_event(
                    colony.tick, f"proposal-{aid}", "proposal", f"{cfg['name']} Proposes: {proposal.action}",
                    f"Priority {proposal.priority}/5. {proposal.rationale} " + format_assessment(proposal.assessment), aid,
                )]}
            except Exception as exc:
                status(aid, "error")
                error = f"{aid}: {type(exc).__name__}: {exc}"
                return {"errors": [error], "usage_records": usage, "events": [_event(colony.tick, f"error-{aid}", "error", f"{cfg['name']} Model Error", error, aid)]}
        return run

    async def commander(state):
        colony, usage, events = state["colony"], [], []
        if state.get("errors"):
            return {"approved_actions": [], "approved_effects": {}, "events": [_event(colony.tick, "blocked", "error", "Commander Blocked", "Active specialist inference failed; no plan executed.", "commander")]}
        status("commander", "negotiating")
        resources = {r: v.value for r, v in colony.resources.items()}
        proposals = state.get("proposals", [])
        conflicts = detect_conflicts([p.action for p in proposals], resources, colony.modules)
        for index, conflict in enumerate(conflicts):
            events.append(_event(colony.tick, f"conflict-{index}", "disagreement", f"Conflict: {', '.join(conflict.actions)}", conflict.explanation))
        candidates = candidate_plans(colony, horizon, physics_version)
        public_choices = [{k: v for k, v in c.items() if k != "essential_exhaustion"} for c in candidates]
        prompt = (
            "You are Commander Hayes, HIVE's Mission Commander. Coordinate these real specialist recommendations: "
            + json.dumps([{k: p.model_dump()[k] for k in ("agent_id", "action", "priority", "rationale")} for p in proposals], separators=(",", ":"))
            + "\nValidated candidate plans, ordered by projected exhaustion, threshold deficit, minimum reserve ratio, then survival index: "
            + json.dumps(public_choices, separators=(",", ":"))
            + "\nPredictions assume this plan followed by holding for the remaining horizon, not future AI decisions. "
            "If any threshold-preserving plans exist, avoidable projected threshold breaches have been excluded. "
            "Choose one listed plan's exact approved_actions. Compare hazards, resource costs, thresholds and specialist advice. "
            "Supporting catalog actions may complement proposals; never attribute them to a specialist who did not propose them. "
            "Return only JSON: {\"approved_actions\":[\"action\"],\"rationale\":\"one short public decision summary\"}. "
            "Use [] only if a hold plan is listed. Non-listed selections never execute. "
            "Do not provide private step-by-step reasoning.\n" + environment(colony)
        )
        try:
            for attempt in range(2):
                response = await invoke("commander", [SystemMessage(content=prompt), HumanMessage(content="Select your coordinated plan." if not attempt else "Previous response was invalid. Return only a listed action array and short rationale as JSON.")], usage)
                data = _extract_json((response.content or "").strip())
                if isinstance(data, dict) and isinstance(data.get("approved_actions"), list) and isinstance(data.get("rationale"), str):
                    approved = data["approved_actions"]
                    chosen = next((c for c in candidates if c["actions"] == approved), None)
                    if chosen is not None and evaluate_plan(approved, resources, colony.modules).feasible:
                        break
                if attempt:
                    raise ValueError("Commander did not return a valid listed plan after one correction")
            selected = set(approved)
            for proposal in proposals:
                if proposal.action not in selected:
                    conflicts_with_plan = evaluate_plan(approved + [proposal.action], resources, colony.modules).conflicts
                    reason = "; ".join(c.explanation for c in conflicts_with_plan) or "Feasible alternative deferred by Commander in favor of the selected reserve plan."
                    events.append(_event(colony.tick, f"reject-{proposal.agent_id}", "disagreement", f"Commander Rejects: {proposal.action}", reason, proposal.agent_id))
            proposed_actions = {p.action for p in proposals}
            support = [a for a in approved if a not in proposed_actions]
            events.append(_event(colony.tick, "commander-decision", "decision", f"Commander Approves: {', '.join(approved) or 'hold'}",
                data["rationale"] + f" Validated forecast if holding: {chosen['projected_resources_if_hold']}. "
                + (f"Supporting catalog actions selected by Commander: {', '.join(support)}." if support else ""), "commander"))
            return {"approved_actions": approved, "approved_effects": evaluate_plan(approved, resources, colony.modules).effects,
                    "conflicts": conflicts, "events": events, "usage_records": usage}
        except Exception as exc:
            error = f"Commander: {type(exc).__name__}: {exc}"
            events.append(_event(colony.tick, "commander-error", "error", "Commander Model Error", error, "commander"))
            return {"approved_actions": [], "approved_effects": {}, "errors": [error], "events": events, "usage_records": usage}

    async def hold(state):
        return {"approved_actions": [], "approved_effects": {}, "events": [_event(state["colony"].tick, "hold", "system", "No Specialist Intervention Needed", "All routing checks passed; deterministic hold with no invented AI proposals.")]}

    async def metrics(state):
        records = state.get("usage_records", [])
        totals = {}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            totals[key] = sum(r[key] for r in records) if all(isinstance(r.get(key), int) for r in records) else None
        return {"events": [_event(state["colony"].tick, "inference-metrics", "system", "Decision Inference Metrics", json.dumps({"logical_model_calls": len(records), "seconds": round(time.monotonic() - state["started_at"], 3), **totals}))]}

    graph = StateGraph(EfficientState)
    graph.add_node("route", route)
    graph.add_node("hold", hold)
    graph.add_node("commander", commander)
    graph.add_node("metrics", metrics)
    for aid in SPECIALIST_CFG:
        graph.add_node(aid, specialist(aid))
        graph.add_edge(aid, "commander")
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", lambda s: [aid for aid, item in s["activation"].items() if item["active"]] or ["hold"], list(SPECIALIST_CFG) + ["hold"])
    graph.add_edge("hold", "metrics")
    graph.add_edge("commander", "metrics")
    graph.add_edge("metrics", END)
    return graph.compile()
