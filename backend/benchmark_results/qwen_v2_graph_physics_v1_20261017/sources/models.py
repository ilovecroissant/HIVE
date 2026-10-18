from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel

ResourceKey = Literal["oxygen", "power", "water", "food", "fuel"]
CrisisKind = Literal["dust_storm", "oxygen_leak", "power_failure"]
AgentId = Literal["commander", "energy", "life_support", "engineering", "science"]
AgentStatus = Literal["idle", "analyzing", "proposing", "negotiating", "executing", "error"]
EventType = Literal["system", "proposal", "disagreement", "decision", "outcome", "error", "commander"]
ModuleType = Literal["habitat", "solar", "life_support", "greenhouse", "reactor", "communications"]


class Resource(BaseModel):
    value: float
    trend: Optional[float] = None


class Module(BaseModel):
    id: str
    label: str
    type: ModuleType
    health: float


class Agent(BaseModel):
    id: AgentId
    name: str
    role: str
    status: AgentStatus


class ColonyState(BaseModel):
    simulation_id: str
    tick: int
    sol: int
    hour: int
    survival_score: float
    resources: dict[ResourceKey, Resource]
    modules: list[Module]
    agents: list[Agent]
    active_crisis: Optional[CrisisKind]
    mode: Literal["demo", "live"]


class AgentEvent(BaseModel):
    id: str
    tick: int
    timestamp: str
    type: EventType
    agent_id: Optional[AgentId] = None
    title: str
    message: str


class StateResponse(BaseModel):
    state: ColonyState
    events: list[AgentEvent]
