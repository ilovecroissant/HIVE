"""
Deterministic Mars colony simulator.
All resource values are bounded to [0, 100].
State is process-local (in-memory singleton).
"""
from __future__ import annotations

import os
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Optional

from models import (
    Agent, AgentEvent, AgentId, AgentStatus, ColonyState,
    CrisisKind, Module, Resource,
)
from actions import PlanEvaluation, REPAIR_HEALTH, SCORE_WEIGHTS, evaluate_plan

# ── constants ─────────────────────────────────────────────────────────────────

# Passive drain applied every tick (1 tick = 1 Martian hour)
BASE_DRAIN: dict[str, float] = {
    "oxygen": -0.5,
    "power": -0.8,
    "water": -0.3,
    "food": -0.2,
    "fuel": -0.1,
}

# Natural passive regen per tick (greenhouse, reactor, recyclers)
BASE_REGEN: dict[str, float] = {
    "food": 0.10,
    "water": 0.05,
    "power": 0.30,   # solar + reactor baseline; zeroed during power_failure
}

# Extra drain applied on top of BASE_DRAIN while a crisis is active
CRISIS_DRAIN: dict[CrisisKind, dict[str, float]] = {
    "dust_storm": {"power": -2.0},      # solar panels blocked
    "oxygen_leak": {"oxygen": -3.0},    # hull breach
    "power_failure": {"power": -5.0},   # grid collapse
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _evt_id(tick: int, tag: str) -> str:
    return f"evt-{tick:05d}-{tag}"


# ── simulator ─────────────────────────────────────────────────────────────────

class Simulator:
    def __init__(self, physics_version: str = "v1") -> None:
        if physics_version not in ("v1", "v2"):
            raise ValueError(f"Unknown physics version: {physics_version}")
        self.physics_version = physics_version
        self.score_version = "resource_v1"
        self.reset()

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self) -> tuple[ColonyState, list[AgentEvent]]:
        self._sim_id = str(uuid.uuid4())
        self._tick = 0
        self._active_crisis: Optional[CrisisKind] = None
        self._resources: dict[str, float] = {
            "oxygen": 85.0,
            "power": 80.0,
            "water": 75.0,
            "food": 70.0,
            "fuel": 60.0,
        }
        self._prev_resources: dict[str, float] = deepcopy(self._resources)
        self._modules: list[Module] = [
            Module(id="habitat",      label="Habitat Alpha",   type="habitat",       health=95.0),
            Module(id="solar",        label="Solar Array",     type="solar",         health=90.0),
            Module(id="life_support", label="Life Support",    type="life_support",  health=92.0),
            Module(id="greenhouse",   label="Greenhouse",      type="greenhouse",    health=88.0),
            Module(id="reactor",      label="Nuclear Reactor", type="reactor",       health=97.0),
            Module(id="comms",        label="Communications",  type="communications",health=85.0),
        ]
        self._agents: list[Agent] = [
            Agent(id="commander",    name="Commander Hayes", role="Mission Commander",    status="idle"),
            Agent(id="energy",       name="ARIA",            role="Energy Systems Agent", status="idle"),
            Agent(id="life_support", name="LYRA",            role="Life Support Agent",   status="idle"),
            Agent(id="engineering",  name="NEXUS",           role="Engineering Agent",    status="idle"),
            Agent(id="science",      name="COSMOS",          role="Science Agent",        status="idle"),
        ]
        self._events: list[AgentEvent] = []

        evt = AgentEvent(
            id=_evt_id(0, "reset"),
            tick=0,
            timestamp=_now_iso(),
            type="system",
            title="Colony Initialized",
            message=(
                "HIVE colony systems initialized. All modules nominal. "
                "Resources at starting levels. Awaiting simulation start."
            ),
        )
        self._events.append(evt)
        return self._snapshot(), [evt]

    def get_state(self) -> tuple[ColonyState, list[AgentEvent]]:
        return self._snapshot(), self._recent_events()

    def trigger_crisis(self, kind: CrisisKind) -> tuple[ColonyState, list[AgentEvent]]:
        self._active_crisis = kind
        self._advance_tick()
        # Do NOT set any agent to "analyzing" here — agents only enter
        # "analyzing" when inference actually starts (in agents_decide / live
        # graph path).  Setting it here caused the stuck-ANALYZING bug.

        titles = {
            "dust_storm":    "ALERT: Dust Storm Detected",
            "oxygen_leak":   "ALERT: Oxygen Leak Detected",
            "power_failure": "ALERT: Power Failure",
        }
        messages = {
            "dust_storm": (
                "A severe Martian dust storm is reducing solar-panel efficiency by ~70%. "
                "Power generation critically impaired. All agents mobilized."
            ),
            "oxygen_leak": (
                "Breach detected in Habitat Alpha oxygen recycling loop. "
                "Atmospheric processors at emergency rate."
            ),
            "power_failure": (
                "Critical power-grid failure in reactor subsystem. "
                "Emergency battery protocols activated. Non-essential systems offline."
            ),
        }

        evt = self._push_event(_evt_id(self._tick, f"crisis-{kind}"), "system",
                               titles[kind], messages[kind])
        return self._snapshot(), self._recent_events()

    def agents_decide(self) -> tuple[ColonyState, list[AgentEvent]]:
        self._advance_tick()

        for agent_id, decision in self._build_decisions().items():
            status_mid, evt_type, title, message, effects = decision
            self._set_status(agent_id, status_mid)
            self._apply_deltas(effects)
            self._push_event(_evt_id(self._tick, f"decide-{agent_id}"),
                             evt_type, title, message, agent_id=agent_id)
            self._set_status(agent_id, "idle")

        return self._snapshot(), self._recent_events()

    def commander_command(self, instruction: str) -> tuple[ColonyState, list[AgentEvent]]:
        self._advance_tick()
        self._set_status("commander", "executing")

        effects: dict[str, float] = {}
        low = instruction.lower()

        if "ration" in low or "food" in low:
            effects["food"] = 5.0
            msg = f'Commander order: "{instruction}". Food rationing activated — consumption reduced 20 %.'
        elif "power" in low or "energy" in low:
            effects["power"] = 5.0
            msg = f'Commander order: "{instruction}". Energy conservation mode engaged across all sectors.'
        elif "oxygen" in low or "air" in low:
            effects["oxygen"] = 5.0
            msg = f'Commander order: "{instruction}". Oxygen production prioritised — electrolysis rate increased.'
        elif "water" in low:
            effects["water"] = 5.0
            msg = f'Commander order: "{instruction}". Water-recycling priority elevated.'
        elif "fuel" in low:
            effects["fuel"] = 3.0
            msg = f'Commander order: "{instruction}". Fuel reserves consolidated and locked.'
        elif any(kw in low for kw in ("all clear", "stand down", "clear", "resume")):
            self._active_crisis = None
            self._reset_all_statuses()
            msg = f'Commander order: "{instruction}". Crisis protocols deactivated — normal operations resumed.'
        else:
            msg = f'Commander order acknowledged: "{instruction}". Agents processing directive.'

        self._apply_deltas(effects)
        self._push_event(_evt_id(self._tick, "commander"), "commander",
                         "Commander Directive", msg, agent_id="commander")
        self._set_status("commander", "idle")
        return self._snapshot(), self._recent_events()

    # ── internals ─────────────────────────────────────────────────────────────

    def _advance_tick(self) -> None:
        self._prev_resources = deepcopy(self._resources)
        self._tick += 1

        # Passive drain
        for k, d in BASE_DRAIN.items():
            self._resources[k] = _clamp(self._resources[k] + d)

        # Crisis extra drain
        if self._active_crisis:
            for k, d in CRISIS_DRAIN.get(self._active_crisis, {}).items():
                self._resources[k] = _clamp(self._resources[k] + d)

        # Natural regen (solar suppressed during power_failure)
        for k, r in BASE_REGEN.items():
            if k == "power" and self._active_crisis == "power_failure":
                continue
            self._resources[k] = _clamp(self._resources[k] + self.regeneration(k))

    def regeneration(self, resource: str) -> float:
        """v2 derates existing production; v1 preserves the original arithmetic."""
        if self.physics_version == "v1":
            return BASE_REGEN.get(resource, 0.0)
        health = {m.type: m.health / 100 for m in self._modules}
        if resource == "power":
            # Nominal 0.30 units/hour: solar 0.20, reactor 0.10.
            return .20 * health.get("solar", 0) + .10 * health.get("reactor", 0)
        module_type = {"water": "life_support", "food": "greenhouse"}.get(resource)
        return BASE_REGEN.get(resource, 0.0) * health.get(module_type, 0)

    def _apply_deltas(self, deltas: dict[str, float]) -> None:
        for k, d in deltas.items():
            self._resources[k] = _clamp(self._resources[k] + d)

    def _snapshot(self) -> ColonyState:
        resources = {
            k: Resource(
                value=round(v, 1),
                trend=round(v - self._prev_resources.get(k, v), 2),
            )
            for k, v in self._resources.items()
        }
        sol = self._tick // 24
        hour = self._tick % 24
        return ColonyState(
            simulation_id=self._sim_id,
            tick=self._tick,
            sol=sol,
            hour=hour,
            survival_score=round(self._compute_score(), 1),
            resources=resources,  # type: ignore[arg-type]
            modules=self._modules,
            agents=self._agents,
            active_crisis=self._active_crisis,
            mode="live",
        )

    def _compute_score(self) -> float:
        return _clamp(sum(self._resources[k] * w for k, w in SCORE_WEIGHTS.items()))

    def _build_decisions(self) -> dict[str, tuple]:
        r = self._resources
        decisions: dict[str, tuple] = {}

        # Energy agent
        if r["power"] < 30:
            decisions["energy"] = (
                "executing", "decision",
                "Emergency Power Conservation",
                "Activating emergency power conservation. Non-essential systems offline. Fuel cells engaged.",
                {"power": 6.0, "fuel": -3.0},
            )
        elif self._active_crisis == "dust_storm":
            decisions["energy"] = (
                "proposing", "proposal",
                "Dust Storm Response: Battery Reserves",
                "Switching to battery reserves — solar efficiency severely reduced. Estimated 4-sol duration.",
                {"power": 1.0, "fuel": -0.5},
            )
        else:
            decisions["energy"] = (
                "analyzing", "outcome",
                "Power Grid Nominal",
                "All power systems within normal parameters. Solar array at 94 % efficiency.",
                {},
            )

        # Life-support agent
        if r["oxygen"] < 40:
            decisions["life_support"] = (
                "executing", "decision",
                "Oxygen Production Boost",
                "Increasing electrolysis rate 40 %. Water consumption rising. Oxygen levels stabilising.",
                {"oxygen": 10.0, "water": -4.0},
            )
        elif r["water"] < 35:
            decisions["life_support"] = (
                "proposing", "proposal",
                "Water Recycling Enhancement",
                "Proposing 20 % increase in water-recycler throughput. Minor power draw required.",
                {"water": 5.0, "power": -1.5},
            )
        else:
            decisions["life_support"] = (
                "analyzing", "outcome",
                "Life Support Stable",
                "Atmospheric processors nominal. CO₂ scrubbers at 87 % capacity.",
                {},
            )

        # Engineering agent — repair lowest-health module
        worst = min(self._modules, key=lambda m: m.health)
        if worst.health < 85:
            decisions["engineering"] = (
                "executing", "decision",
                f"Repairing {worst.label}",
                f"Dispatching repair drone to {worst.label} (health {worst.health:.0f}%). ETA 2 h. Fuel cost: 3 units.",
                {"fuel": -3.0},
            )
            worst.health = _clamp(worst.health + REPAIR_HEALTH)
        else:
            decisions["engineering"] = (
                "analyzing", "outcome",
                "All Modules Nominal",
                "Structural integrity within acceptable limits. No critical maintenance required.",
                {},
            )

        # Science agent
        if r["food"] < 45:
            decisions["science"] = (
                "proposing", "proposal",
                "Greenhouse Yield Optimisation",
                "Proposing UV-spectrum adjustment to boost crop yield 15 %. Minor power overhead.",
                {"food": 4.0, "power": -1.0},
            )
        else:
            decisions["science"] = (
                "analyzing", "outcome",
                "Research Nominal",
                "Soil-sample analysis complete. Iron-oxide composition confirms volcanic origin.",
                {},
            )

        return decisions

    def _push_event(
        self,
        event_id: str,
        evt_type: str,
        title: str,
        message: str,
        agent_id: Optional[AgentId] = None,
    ) -> AgentEvent:
        evt = AgentEvent(
            id=event_id,
            tick=self._tick,
            timestamp=_now_iso(),
            type=evt_type,  # type: ignore[arg-type]
            agent_id=agent_id,
            title=title,
            message=message,
        )
        self._events.append(evt)
        return evt

    def _recent_events(self, n: int = 100) -> list[AgentEvent]:
        return self._events[-n:]

    def _set_status(self, agent_id: AgentId, status: AgentStatus) -> None:
        for a in self._agents:
            if a.id == agent_id:
                a.status = status
                return

    def _reset_all_statuses(self) -> None:
        for a in self._agents:
            a.status = "idle"

    # ── public hooks for graph integration ────────────────────────────────────

    def advance_tick(self) -> None:
        self._advance_tick()

    def snapshot(self) -> ColonyState:
        return self._snapshot()

    def apply_effects(self, effects: dict[str, float]) -> dict[str, float]:
        before = dict(self._resources)
        self._apply_deltas(effects)
        return {key: self._resources[key] - before[key] for key in effects}

    def execute_actions(self, actions: list[str]) -> PlanEvaluation:
        """Revalidate against exact balances, then commit all changes atomically."""
        report = evaluate_plan(actions, self._resources, self._modules)
        if not report.feasible:
            raise ValueError("Plan rejected at execution: " + "; ".join(c.explanation for c in report.conflicts))
        resources = dict(self._resources)
        modules = [m.model_copy(deep=True) for m in self._modules]
        for execution in report.executions:
            for resource, delta in execution.resource_changes.items():
                resources[resource] = _clamp(resources[resource] + delta)
            for change in execution.module_changes:
                next(m for m in modules if m.id == change.id).health = change.after
        self._resources = resources
        self._modules = modules
        return report

    def push_events(self, events: list[AgentEvent]) -> None:
        self._events.extend(events)

    def recent_events(self) -> list[AgentEvent]:
        return self._recent_events()

    def reset_all_statuses(self) -> None:
        self._reset_all_statuses()


# Process-local singleton
sim = Simulator(physics_version=os.getenv("HIVE_PHYSICS_VERSION", "v1"))
