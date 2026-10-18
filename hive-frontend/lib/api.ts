import type { ActionResponse, CommanderRequest, CrisisKind, StateResponse } from "./types";
import { demoApi } from "./demo";
export const IS_DEMO = process.env.NEXT_PUBLIC_DEMO_MODE !== "false";
const BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
// timeoutMs defaults to 15 s; the decide endpoint uses 90 s because the
// LangGraph pipeline runs three concurrent LLM calls then a commander call.
async function request<T>(path:string, method:"GET"|"POST"="GET",body?:unknown,timeoutMs=15000):Promise<T>{
  const controller=new AbortController();const timeout=setTimeout(()=>controller.abort(),timeoutMs);
  try {
    const response=await fetch(`${BASE}${path}`,{method,headers:{"Content-Type":"application/json"},body:body===undefined?undefined:JSON.stringify(body),signal:controller.signal,cache:"no-store"});
    if(!response.ok){ const payload:unknown=await response.json().catch(()=>null);const detail=payload&&typeof payload==="object"&&"detail" in payload?String((payload as {detail:unknown}).detail):response.statusText;throw new Error(`Backend ${response.status}: ${detail}`);}
    return await response.json() as T;
  } catch(error) { if(error instanceof Error&&error.name==="AbortError") throw new Error(`Backend request timed out (${timeoutMs/1000} s).`); throw error; }
  finally{clearTimeout(timeout);}
}
// Contract: GET state returns {state, events}; action endpoints return same envelope.
// Edit only this adapter if your existing FastAPI routes or response shapes differ.
export interface BackendHealth { live_agents: boolean; graph_version: string; selection_policy: string; specialist_routing: boolean; physics_version: string; model: string | null; }
export const api={
 health:():Promise<BackendHealth>=>request("/health"),
 getState:():Promise<StateResponse>=>IS_DEMO?demoApi.getState():request("/simulation/state"),
 reset:():Promise<ActionResponse>=>IS_DEMO?demoApi.reset():request("/simulation/reset","POST",{}),
 crisis:(kind:CrisisKind):Promise<ActionResponse>=>IS_DEMO?demoApi.crisis(kind):request("/simulation/crisis","POST",{crisis:kind}),
 decide:():Promise<ActionResponse>=>IS_DEMO?demoApi.decide():request("/agents/decide","POST",{},90000),
 command:(instruction:string):Promise<ActionResponse>=>IS_DEMO?demoApi.command(instruction):request("/commander/command","POST",{instruction} satisfies CommanderRequest)
};
