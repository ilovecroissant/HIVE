import type { AgentEvent } from "./types";

// Parse existing public routing events; never infer activation from the crisis.
export function routingRecords(events: AgentEvent[]) {
  const event = [...events].reverse().find(e => e.title === "Specialist Activation");
  if (!event) return null;
  const records = [...event.message.matchAll(/(energy|life_support|engineering): (ACTIVE|SKIPPED) — (.*?)(?=; (?:energy|life_support|engineering): |$)/g)]
    .map(match => ({ id: match[1], status: match[2], reason: match[3] }));
  return records.length ? { tick: event.tick, records } : null;
}

export function eventLabel(event: AgentEvent) {
  if (event.title === "Specialist Activation") return "ROUTING";
  if (event.title.startsWith("Conflict:")) return "CONFLICT";
  if (event.title.startsWith("Commander Rejects:")) return "REJECTED";
  if (event.title.startsWith("Executed:")) return "EXECUTED";
  return event.type.toUpperCase();
}
