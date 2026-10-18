"""Generalist baseline: one planner, the same action rules, one bounded repair."""
import json

from langchain_core.messages import HumanMessage, SystemMessage

from actions import VALID_ACTIONS, describe_action, evaluate_plan
from graph import CommanderDecision, _extract_json, _normalize_commander_data
from models import ColonyState
from simulator import BASE_DRAIN, BASE_REGEN, CRISIS_DRAIN

CONTEXT_MARKER = "BENCHMARK ENVIRONMENT AND OBJECTIVE:"
RECOVERY_THRESHOLDS = {"oxygen": 40., "power": 30., "water": 20., "food": 20.}
RULES = (
    "Reserve total costs from starting balances; this tick's gains cannot fund other actions. "
    "Resources and module health cap at 100; no negative balances. Each action once per tick. "
    "battery_reserves and emergency_power are exclusive backup-controller modes; conserve_power can coexist. "
    "repair_module spends 3 fuel and restores up to 6 health to the lowest-health damaged module; ties by ID. "
    "Reject saturated resource gains and repairs without damaged targets. idle alone or [] holds position. "
    "Crises persist throughout the horizon; repair does not stop crisis drains. "
    "Module health does not affect resource production or the resource-only survival index."
)


def decision_context(colony: ColonyState, remaining_steps: int) -> str:
    """Exactly the same supplementary state, dynamics, and catalog for both systems."""
    resources = {r: v.value for r, v in colony.resources.items()}
    state = colony.model_dump(exclude={"agents", "mode"})
    # Raw catalog numbers plus individual feasibility are compact enough for 4K context.
    feasibility = {}
    for action in VALID_ACTIONS:
        assessment = describe_action(action, resources, colony.modules)
        feasibility[action] = {"feasible": assessment["feasible"], "constraints": assessment["constraints"]}
    return CONTEXT_MARKER + "\n" + json.dumps({
        "state": state, "action_resource_effects": VALID_ACTIONS,
        "individual_feasibility": feasibility, "rules": RULES,
        "passive_drain_per_tick": BASE_DRAIN, "passive_regeneration": BASE_REGEN,
        "additional_crisis_drain": CRISIS_DRAIN,
        "power_failure_disables_power_regeneration": True,
        "remaining_decision_ticks_including_this_one": remaining_steps,
        "objective": "Stabilize essential reserves over the remaining horizon, avoid exhaustion, then maximize final survival index. Compare costs and competing needs; do not optimize only immediate score.",
        "final_recovery_thresholds": RECOVERY_THRESHOLDS,
        "recovery_also_requires": "No oxygen, power, water or food exhaustion at any observation, nonnegative final fuel, and no terminal inference/execution failures.",
    }, separators=(",", ":"))


async def decide_generalist(colony: ColonyState, make_llm_fn, context: str) -> dict:
    """Full catalog plan, not restricted to a specialist's preferred action list."""
    prompt = (
        "You are HIVE's generalist Mars colony recovery agent. Plan jointly for energy, "
        "engineering, oxygen, water, food and fuel using the supplied state, costs and hazards. "
        "You may choose any feasible combination from the full catalog. Prioritize urgent "
        "essential reserves, consider future crisis drains, and conserve scarce resources. "
        "Return only JSON, with short public summaries, in this schema: "
        '{"approved_actions":["action"],"rationale":"short reason",'
        '"rejection_reasons":{},"disagreements":[]}. Empty actions means hold.\n'
        + context
    )
    resources = {r: v.value for r, v in colony.resources.items()}
    try:
        llm = make_llm_fn()
        response = await llm.ainvoke([SystemMessage(content=prompt), HumanMessage(content="Issue your recovery plan.")])
        for attempt in range(2):
            try:
                data = _extract_json((response.content or "").strip())
                if data is None:
                    raise ValueError("Expected a JSON object")
                decision = CommanderDecision.model_validate(_normalize_commander_data(data))
                selected = list(dict.fromkeys(decision.approved_actions))
                preview = evaluate_plan(selected, resources, colony.modules)
                if not attempt and not preview.feasible:
                    raise ValueError("; ".join(c.explanation for c in preview.conflicts))
                break
            except (ValueError, TypeError) as exc:
                if attempt:
                    raise
                response = await llm.ainvoke([
                    SystemMessage(content=prompt + "\nCRITICAL: Return only schema-valid JSON and a feasible plan."),
                    HumanMessage(content=f"Correct this validation failure: {str(exc)[:600]}"),
                ])
        # Same ordered safety filter as Commander after the bounded correction.
        approved = []
        for action in selected:
            if evaluate_plan(approved + [action], resources, colony.modules).feasible:
                approved.append(action)
        return {"approved_actions": approved, "errors": []}
    except Exception as exc:
        return {"approved_actions": [], "errors": [f"{type(exc).__name__}: {exc}"]}
