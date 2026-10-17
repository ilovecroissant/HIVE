export type ResourceKey = "oxygen" | "power" | "water" | "food" | "fuel";
export type CrisisKind = "dust_storm" | "oxygen_leak" | "power_failure";
export type AgentId = "commander" | "energy" | "life_support" | "engineering" | "science";
export type AgentStatus = "idle" | "analyzing" | "proposing" | "negotiating" | "executing" | "error";
export type EventType = "system" | "proposal" | "disagreement" | "decision" | "outcome" | "error" | "commander";
export interface Resource { value: number; trend?: number; }
export interface Module { id: string; label: string; type: "habitat" | "solar" | "life_support" | "greenhouse" | "reactor" | "communications"; health: number; }
export interface Agent { id: AgentId; name: string; role: string; status: AgentStatus; }
export interface ColonyState {
  simulation_id: string; tick: number; sol: number; hour: number; survival_score: number;
  resources: Record<ResourceKey, Resource>; modules: Module[]; agents: Agent[];
  active_crisis: CrisisKind | null; mode: "demo" | "live";
}
export interface AgentEvent { id: string; tick: number; timestamp: string; type: EventType; agent_id?: AgentId; title: string; message: string; }
export interface StateResponse { state: ColonyState; events: AgentEvent[]; }
export interface ActionResponse { state: ColonyState; events: AgentEvent[]; }
export interface CommanderRequest { instruction: string; }
export interface ApiErrorDetail { detail?: string | { message?: string }; }
