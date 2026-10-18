"""Versioned routing and deterministic predictions; never chooses an AI approval."""
from itertools import combinations

from actions import VALID_ACTIONS, evaluate_plan
from models import ColonyState
from simulator import Simulator

THRESHOLDS = {"oxygen": 40., "power": 30., "water": 20., "food": 20.}
POLICY_VERSION = "v2.1"


def projected_state(colony: ColonyState, actions: list[str], horizon: int, physics_version="v1"):
    simulator = Simulator(physics_version=physics_version)
    simulator._resources = {r: value.value for r, value in colony.resources.items()}
    simulator._modules = [m.model_copy(deep=True) for m in colony.modules]
    simulator._active_crisis = colony.active_crisis
    report = simulator.execute_actions(actions)
    exhausted = any(simulator._resources[r] <= 0 for r in THRESHOLDS)
    for _ in range(max(0, horizon - 1)):
        simulator.advance_tick()
        exhausted |= any(simulator._resources[r] <= 0 for r in THRESHOLDS)
    return simulator, report, exhausted


def specialist_routing(colony: ColonyState, horizon=3, physics_version="v1") -> dict[str, dict]:
    """Operational intervention thresholds, independent of scenario identity."""
    r = {name: item.value for name, item in colony.resources.items()}
    future, _, _ = projected_state(colony, [], horizon, physics_version)
    reasons = {"energy": [], "life_support": [], "engineering": []}
    if colony.active_crisis in ("dust_storm", "power_failure"):
        reasons["energy"].append(f"Power hazard: {colony.active_crisis}")
    if r["power"] < 45 or future._resources["power"] < THRESHOLDS["power"]:
        reasons["energy"].append("Current power below intervention reserve 45 or projected below 30")
    if r["fuel"] < 10:
        reasons["energy"].append("Scarce fuel below 10 requires energy tradeoff review")
    if colony.active_crisis == "oxygen_leak":
        reasons["life_support"].append("Active oxygen leak")
    for resource, intervention in (("oxygen", 60), ("water", 40), ("food", 45)):
        if r[resource] < intervention or future._resources[resource] < THRESHOLDS[resource]:
            reasons["life_support"].append(f"{resource} below intervention reserve {intervention} or projected safety threshold")
    threatened = any(future._resources[r] < threshold for r, threshold in THRESHOLDS.items())
    damaged = [m for m in colony.modules if m.health < 70]
    critical = [m for m in colony.modules if m.health < 40]
    if critical or (damaged and not threatened and r["fuel"] >= 3):
        reasons["engineering"].append("Actual damage needs review: " + ", ".join(f"{m.label} {m.health:g}%" for m in (critical or damaged)))
    # A low-output producer can be relevant before the generic maintenance threshold.
    for module in colony.modules:
        output = {"solar": "power", "reactor": "power", "life_support": "water", "greenhouse": "food"}.get(module.type)
        if physics_version == "v2" and output and module.health < 85 and future._resources[output] < THRESHOLDS[output]:
            reasons["engineering"].append(f"Damaged {module.label} supplies threatened {output}")
    decisions = {aid: {"active": bool(why), "reasons": why or ["No crisis, reserve or damage trigger for this expertise"]} for aid, why in reasons.items()}
    if damaged and threatened and not reasons["engineering"]:
        decisions["engineering"]["reasons"] = ["Defer noncritical damage during reserve emergency; repair has no threatened-production benefit under current physics"]
    elif damaged and r["fuel"] < 3 and not reasons["engineering"]:
        decisions["engineering"]["reasons"] = ["Noncritical repair is unaffordable; critical-damage checks remain active"]
    return decisions


def candidate_plans(colony: ColonyState, horizon=3, physics_version="v1", limit=6) -> list[dict]:
    """Evaluate catalog combinations, rank predictions, offer distinct feasible choices.

    No health-score bonus is added. v2 health affects only real regeneration.
    Predictions assume the current plan followed by holding for the remaining
    horizon; they are explicitly not forecasts of future model decisions.
    """
    resources = {r: value.value for r, value in colony.resources.items()}
    choices = [a for a in VALID_ACTIONS if a != "idle"]
    candidates = []
    for size in range(len(choices) + 1):
        for actions in combinations(choices, size):
            if not evaluate_plan(list(actions), resources, colony.modules).feasible:
                continue
            future, report, exhausted = projected_state(colony, list(actions), horizon, physics_version)
            deficits = sum(max(0, threshold - future._resources[r]) / threshold for r, threshold in THRESHOLDS.items())
            minimum_ratio = min(future._resources[r] / threshold for r, threshold in THRESHOLDS.items())
            costs = {r: sum(-VALID_ACTIONS[a].get(r, 0) for a in actions if VALID_ACTIONS[a].get(r, 0) < 0) for r in resources}
            candidates.append({
                "actions": list(actions), "costs": {r: n for r, n in costs.items() if n},
                "effects_now": report.effects,
                "projected_resources_if_hold": {r: round(v, 3) for r, v in future._resources.items()},
                "projected_score": round(future._compute_score(), 3),
                "threshold_deficit": round(deficits, 6), "essential_exhaustion": exhausted,
                "rank": (int(exhausted), round(deficits, 9), -minimum_ratio, -future._compute_score(), costs["fuel"], len(actions)),
            })
    candidates.sort(key=lambda item: item["rank"])
    # Minimum-reserve policy is a selection guard, not a changed benchmark metric.
    # Do not offer avoidable forecast breaches alongside threshold-preserving plans.
    preserving = [c for c in candidates if c["rank"][1] <= 1e-8 and not c["essential_exhaustion"]]
    if preserving:
        candidates = preserving
    shortlisted = candidates[:limit]
    # Retain hold when it passes the same forecast policy as active plans.
    hold = next((c for c in candidates if not c["actions"]), None)
    if hold is not None and hold not in shortlisted:
        shortlisted = shortlisted[:max(0, limit - 1)] + [hold]
    for index, item in enumerate(shortlisted):
        item.pop("rank")
        item["id"] = index
    return shortlisted
