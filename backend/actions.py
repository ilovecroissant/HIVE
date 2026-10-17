"""Shared deterministic rules for proposal evaluation and atomic execution.

Costs are reserved from the start-of-cycle balances: speculative gains cannot
finance another operation in the same plan. Battery dispatch and emergency
generation are alternative settings of the single backup-power controller.
Resource safety means no balance may become negative; gains cap at 100.
"""
from collections import Counter
from itertools import combinations

from pydantic import BaseModel, Field

from models import Module

VALID_ACTIONS: dict[str, dict[str, float]] = {
    "boost_oxygen": {"oxygen": 10.0, "water": -4.0, "power": -1.0},
    "conserve_power": {"power": 5.0, "fuel": -2.0},
    "battery_reserves": {"power": 1.0, "fuel": -0.5},
    "boost_water_recycling": {"water": 5.0, "power": -1.5},
    "ration_food": {"food": 5.0},
    "repair_module": {"fuel": -3.0},
    "emergency_power": {"power": 6.0, "fuel": -3.0},
    "greenhouse_optimize": {"food": 4.0, "power": -1.0},
    "idle": {},
}
VALID_ACTION_LIST = ", ".join(VALID_ACTIONS)
REPAIR_HEALTH = 6.0
SCORE_WEIGHTS = {"oxygen": .35, "power": .25, "water": .20, "food": .10, "fuel": .10}
BACKUP_MODES = {"battery_reserves", "emergency_power"}


class Conflict(BaseModel):
    actions: list[str]
    constraint: str
    explanation: str


class ModuleChange(BaseModel):
    id: str
    label: str
    before: float
    after: float


class ActionExecution(BaseModel):
    action: str
    resource_changes: dict[str, float]
    module_changes: list[ModuleChange] = Field(default_factory=list)


class PlanEvaluation(BaseModel):
    feasible: bool
    conflicts: list[Conflict]
    effects: dict[str, float] = Field(default_factory=dict)
    executions: list[ActionExecution] = Field(default_factory=list)


def validate_action(action: str, resources: dict[str, float]) -> tuple[bool, str, dict[str, float]]:
    """Existing single-action catalog/resource validator, shared by all paths."""
    if action not in VALID_ACTIONS:
        return False, f"Unknown action '{action}' — not in catalog", {}
    for resource, delta in VALID_ACTIONS[action].items():
        if delta < 0 and resources.get(resource, 0) + delta < -1e-9:
            return False, f"Insufficient {resource} for '{action}' (needs {-delta:.1f})", {}
    return True, "ok", dict(VALID_ACTIONS[action])


def repair_target(modules: list[Module]) -> Module | None:
    damaged = [m for m in modules if m.health < 100]
    return min(damaged, key=lambda m: (m.health, m.id)) if damaged else None


def evaluate_plan(actions: list[str], resources: dict[str, float], modules: list[Module]) -> PlanEvaluation:
    """Pure, order-preserving dry run. Infeasible plans have no execution effects."""
    conflicts: list[Conflict] = []

    def conflict(involved, constraint, explanation):
        conflicts.append(Conflict(actions=list(involved), constraint=constraint, explanation=explanation))

    for action, count in Counter(actions).items():
        if count > 1:
            conflict([action], "duplicate_action", f"{action} appears {count} times; execute it at most once per cycle.")
        ok, reason, _ = validate_action(action, resources)
        if not ok:
            conflict([action], "action_affordability" if action in VALID_ACTIONS else "unknown_action", reason)
    unique = list(dict.fromkeys(actions))
    modes = [a for a in unique if a in BACKUP_MODES]
    if len(modes) > 1:
        conflict(modes, "backup_controller_mode", "Battery dispatch and emergency generation are alternative backup-controller modes; select one per cycle.")
    for resource in resources:
        spenders = [a for a in unique if VALID_ACTIONS.get(a, {}).get(resource, 0) < 0]
        cost = sum(-VALID_ACTIONS[a][resource] for a in spenders)
        if len(spenders) > 1 and cost > resources[resource] + 1e-9:
            conflict(spenders, f"budget:{resource}", f"Combined {resource} cost {cost:.1f} exceeds available {resources[resource]:.2f}; gains are not spendable until the next cycle.")
    if "idle" in unique and len(unique) > 1:
        conflict(unique, "redundant_idle", "Idle is a hold-only plan; omit it when executing other operations.")
    if conflicts:
        return PlanEvaluation(feasible=False, conflicts=conflicts)

    balances = dict(resources)
    health = [m.model_copy(deep=True) for m in modules]
    executions = []
    for action in unique:
        changes = []
        before = dict(balances)
        if action == "repair_module":
            target = repair_target(health)
            if target is None:
                conflict([action], "repair_target", "No damaged module exists; spending repair fuel would be redundant.")
                continue
            after = min(100.0, target.health + REPAIR_HEALTH)
            changes = [ModuleChange(id=target.id, label=target.label, before=target.health, after=after)]
            target.health = after
        for resource, delta in VALID_ACTIONS[action].items():
            balances[resource] = min(100.0, max(0.0, balances[resource] + delta))
        deltas = {r: round(balances[r] - before[r], 10) for r in VALID_ACTIONS[action]}
        if action not in ("idle", "repair_module") and not any(d > 0 for d in deltas.values()):
            conflict([action], "no_benefit", f"{action} restores no resource at current levels; its gains are already saturated.")
        executions.append(ActionExecution(action=action, resource_changes=deltas, module_changes=changes))
    if conflicts:
        return PlanEvaluation(feasible=False, conflicts=conflicts)
    touched = {r for a in unique for r in VALID_ACTIONS[a]}
    return PlanEvaluation(
        feasible=True, conflicts=[], executions=executions,
        effects={r: round(balances[r] - resources[r], 10) for r in sorted(touched)},
    )


def detect_conflicts(actions: list[str], resources: dict[str, float], modules: list[Module]) -> list[Conflict]:
    """Evaluate single actions and every proposal subset (at most three specialists)."""
    found = {}
    for size in range(1, len(actions) + 1):
        for subset in combinations(actions, size):
            for item in evaluate_plan(list(subset), resources, modules).conflicts:
                found[(tuple(item.actions), item.constraint)] = item
    return list(found.values())


def describe_action(action: str, resources: dict[str, float], modules: list[Module]) -> dict:
    """Catalog-derived benefits/costs; model-supplied numbers are never trusted."""
    effects = VALID_ACTIONS.get(action, {})
    benefits = {r: min(d, 100 - resources[r]) for r, d in effects.items() if d > 0}
    costs = {r: -d for r, d in effects.items() if d < 0}
    repair = repair_target(modules) if action == "repair_module" else None
    result = evaluate_plan([action], resources, modules)
    return {
        "resource_cost": costs,
        "expected_benefit": benefits,
        "repair": ({"module": repair.label, "health_gain": min(REPAIR_HEALTH, 100 - repair.health)} if repair else None),
        "survival_delta": round(sum(SCORE_WEIGHTS[r] * (benefits.get(r, 0) - costs.get(r, 0)) for r in SCORE_WEIGHTS), 2),
        "risk_tradeoff": (
            "; ".join(f"Consumes {v:g} {r}" for r, v in costs.items()) or "No resource cost"
        ) + ("; excludes the other backup-controller mode" if action in BACKUP_MODES else ""),
        "feasible": result.feasible,
        "constraints": [c.explanation for c in result.conflicts],
    }


def format_assessment(assessment: dict) -> str:
    gains = [f"{r} +{v:g}" for r, v in assessment["expected_benefit"].items()]
    repair = assessment["repair"]
    if repair:
        gains.append(f"{repair['module']} health +{repair['health_gain']:g}")
    costs = [f"{r} {v:g}" for r, v in assessment["resource_cost"].items()]
    text = f"Expected benefit: {', '.join(gains) or 'hold position'}. Cost: {', '.join(costs) or 'none'}. Tradeoff: {assessment['risk_tradeoff']}."
    if not assessment["feasible"]:
        text += " Not feasible: " + "; ".join(assessment["constraints"])
    return text
