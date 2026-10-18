"""Versioned environment data; generalist planner instructions remain untouched."""
import json

from single_agent import CONTEXT_MARKER, decision_context


def shared_context(colony, horizon, physics_version="v1"):
    original = decision_context(colony, horizon)
    if physics_version == "v1":
        return original  # byte-identical original baseline input
    if physics_version != "v2":
        raise ValueError(f"Unknown physics version: {physics_version}")
    data = json.loads(original.split("\n", 1)[1])
    data["rules"] = data["rules"].replace(
        "Module health does not affect resource production or the resource-only survival index.",
        "Module health linearly derates passive production: power=0.20*solar_health/100+0.10*reactor_health/100; water=0.05*life_support_health/100; food=0.10*greenhouse_health/100. Power-failure suppression still applies. Repair improves only the repaired module's future passive production. Survival index weights are unchanged. Active action effects are unchanged.",
    )
    data["physics_version"] = "v2"
    return CONTEXT_MARKER + "\n" + json.dumps(data, separators=(",", ":"))
