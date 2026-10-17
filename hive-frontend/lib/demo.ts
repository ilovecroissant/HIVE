import type { ActionResponse, AgentEvent, AgentId, ColonyState, CrisisKind, ResourceKey, StateResponse } from "./types";
const agentRows: ColonyState["agents"] = [
  { id: "commander", name: "COMMANDER", role: "Coordinator / safety", status: "idle" },
  { id: "energy", name: "ENERGY", role: "Power & generation", status: "idle" },
  { id: "life_support", name: "LIFE SUPPORT", role: "Crew survival", status: "idle" },
  { id: "engineering", name: "ENGINEERING", role: "Repairs & systems", status: "idle" },
  { id: "science", name: "SCIENCE", role: "Risk forecasting", status: "idle" }
];
export const baseState = (): ColonyState => ({
  simulation_id: "demo-olympus", tick: 0, sol: 37, hour: 14, survival_score: 92,
  resources: { oxygen: {value:88,trend:1},power:{value:76,trend:2},water:{value:91,trend:0},food:{value:83,trend:-1},fuel:{value:68,trend:0} },
  modules: [
    {id:"hab1",label:"HABITAT A",type:"habitat",health:100},
    {id:"solar",label:"SOLAR ARRAY",type:"solar",health:96},
    {id:"life",label:"LIFE SUPPORT",type:"life_support",health:99},
    {id:"green",label:"HYDROPONICS",type:"greenhouse",health:92},
    {id:"reactor",label:"POWER CORE",type:"reactor",health:97},
    {id:"comms",label:"UPLINK",type:"communications",health:100}
  ], agents:agentRows.map(x=>({...x})), active_crisis:null,mode:"demo"
});
let demoState = baseState();
let demoEvents: AgentEvent[] = [];
let sequence = 0;
function event(type:AgentEvent["type"],title:string,message:string,agent_id?:AgentId):AgentEvent {
  sequence++; return {id:`demo-${sequence}`,tick:demoState.tick,timestamp:new Date().toISOString(),type,title,message,agent_id};
}
function setResource(k: ResourceKey, delta:number) { demoState.resources[k] = {...demoState.resources[k],value:Math.max(0,Math.min(100,demoState.resources[k].value+delta)),trend:delta}; }
function snapshot(): StateResponse { return {state:structuredClone(demoState),events:structuredClone(demoEvents)}; }
export const demoApi = {
  getState: async():Promise<StateResponse>=>snapshot(),
  reset: async():Promise<ActionResponse>=>{demoState=baseState();demoEvents=[];return snapshot();},
  crisis: async(kind:CrisisKind):Promise<ActionResponse>=>{
    demoState.tick++;demoState.hour=(demoState.hour+1)%24;demoState.active_crisis=kind;
    if(kind==="dust_storm"){setResource("power",-33);setResource("oxygen",-9);demoState.survival_score-=12;}
    if(kind==="oxygen_leak"){setResource("oxygen",-38);demoState.survival_score-=18;}
    if(kind==="power_failure"){setResource("power",-48);demoState.survival_score-=19;}
    demoState.agents=demoState.agents.map(a=>({...a,status:"analyzing"}));
    demoEvents.push(event("system","CRISIS DETECTED",`${kind.replace(/_/g," ").toUpperCase()} — scenario injected into local simulation.`));
    return snapshot();
  },
  decide: async():Promise<ActionResponse>=>{
    if(!demoState.active_crisis) throw new Error("Trigger a crisis first.");
    const crisis=demoState.active_crisis;
    // These are explicitly scripted DEMO events, not model-generated output.
    demoEvents.push(event("proposal","POWER ALLOCATION","Propose isolating nonessential systems to preserve critical power.","energy"));
    demoEvents.push(event("proposal","CREW PRIORITY","Reserve available resources for life support and crew habitats.","life_support"));
    demoEvents.push(event("disagreement","TRADEOFF FLAGGED","Science favors preserving research uptime; Energy prioritizes emergency reserves.","science"));
    demoEvents.push(event("proposal","RECOVERY ROUTE","Engineering recommends staged repair and backup activation.","engineering"));
    demoEvents.push(event("decision","SIMULATED COORDINATION","Scripted example: prioritize crew and power restoration. No language model was called.","commander"));
    if(crisis==="dust_storm"){setResource("power",21);setResource("oxygen",6);demoState.survival_score+=8;}
    if(crisis==="oxygen_leak"){setResource("oxygen",27);setResource("fuel",-6);demoState.survival_score+=11;}
    if(crisis==="power_failure"){setResource("power",32);setResource("fuel",-13);demoState.survival_score+=10;}
    demoState.tick++;demoState.hour=(demoState.hour+1)%24;demoState.survival_score=Math.min(100,demoState.survival_score);
    demoState.active_crisis=null;demoState.agents=demoState.agents.map(a=>({...a,status:"idle"}));
    demoEvents.push(event("outcome","SIMULATION UPDATED","Predefined recovery effects applied by demo simulator. Results are not AI-generated."));
    return snapshot();
  },
  command: async(instruction:string):Promise<ActionResponse>=>{
    demoEvents.push(event("commander","HUMAN COMMAND RECEIVED",instruction));
    demoEvents.push(event("system","DEMO MODE","Command recorded for display only. No AI evaluation or physical action performed."));
    return snapshot();
  }
};
