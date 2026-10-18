"""Paired real-inference benchmark. Never imports the API or calls its endpoints."""
import argparse
import asyncio
from collections import Counter
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

import httpx
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from actions import VALID_ACTIONS, evaluate_plan
from graph import (
    HIVE_MODEL, OPENAI_API_BASE, OPENAI_API_KEY, SPECIALIST_CFG, Proposal,
    CommanderDecision, _extract_json, _normalize_commander_data, build_graph,
)
from simulator import Simulator
from single_agent import CONTEXT_MARKER, RECOVERY_THRESHOLDS, decide_generalist, decision_context
from physics_context import shared_context


@dataclass(frozen=True)
class Scenario:
    name: str
    crisis: str
    resources: dict[str, float]
    damaged_module: str
    health: float


SCENARIOS = (
    Scenario("dust_storm", "dust_storm", dict(oxygen=55, power=28, water=55, food=60, fuel=18), "solar", 65),
    Scenario("power_failure", "power_failure", dict(oxygen=60, power=30, water=55, food=60, fuel=20), "reactor", 65),
    Scenario("oxygen_leak", "oxygen_leak", dict(oxygen=28, power=50, water=35, food=60, fuel=18), "life_support", 65),
    Scenario("scarce_fuel", "dust_storm", dict(oxygen=36, power=30, water=40, food=60, fuel=5.4), "comms", 45),
)


def new_simulator(scenario: Scenario, seed: int, physics_version="v1") -> Simulator:
    simulator = Simulator(physics_version=physics_version)
    simulator._sim_id = f"benchmark-{scenario.name}-{seed}"
    simulator._resources = dict(scenario.resources)
    simulator._prev_resources = dict(scenario.resources)
    next(m for m in simulator._modules if m.id == scenario.damaged_module).health = scenario.health
    return simulator


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def role_for(messages) -> str:
    prompt = messages[0].content.split(CONTEXT_MARKER)[0]
    if "Commander Hayes" in prompt:
        return "commander"
    for agent, config in SPECIALIST_CFG.items():
        if f"You are {config['name']}" in prompt:
            return agent
    return "generalist"


class MeasuredFactory:
    """Same sampling/client configuration for both paths, per-call usage auditing."""
    def __init__(self, args, context, seed, records):
        self.args, self.context, self.seed, self.records = args, context, seed, records
        self.role_calls = Counter()

    def __call__(self):
        return self

    async def ainvoke(self, messages):
        role = role_for(messages)
        budget = self.args.specialist_tokens if role in SPECIALIST_CFG else self.args.planner_tokens
        call_seed = self.seed + self.role_calls[role]
        self.role_calls[role] += 1
        messages = list(messages)
        if CONTEXT_MARKER not in messages[0].content:
            messages[0] = SystemMessage(content=messages[0].content + "\n" + self.context)
        record = {
            "role": role, "seed": call_seed, "max_output_tokens": budget,
            "messages": [{"type": m.type, "content": m.content} for m in messages],
            "input_characters": sum(len(m.content) for m in messages),
            "input_tokens": None, "output_tokens": None, "total_tokens": None,
            "content": None, "error": None,
        }
        self.records.append(record)
        started = time.perf_counter()
        try:
            client = ChatOpenAI(
                model=self.args.model, base_url=self.args.base_url, api_key=OPENAI_API_KEY,
                temperature=self.args.temperature, seed=call_seed, max_tokens=budget,
                timeout=self.args.timeout, max_retries=0,
            )
            response = await client.ainvoke(messages)
            record["content"] = response.content
            usage = response.usage_metadata or {}
            raw_usage = response.response_metadata.get("token_usage") or {}
            for key, raw_key in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"), ("total_tokens", "total_tokens")):
                record[key] = usage.get(key, raw_usage.get(raw_key))
            record["finish_reason"] = response.response_metadata.get("finish_reason")
            record["system_fingerprint"] = response.response_metadata.get("system_fingerprint")
            return response
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["seconds"] = time.perf_counter() - started


def audit_calls(records, colony, executed):
    """Raw-output counts include retries; no invalid output is silently corrected away."""
    counts = Counter()
    resources = {r: v.value for r, v in colony.resources.items()}
    for record in records:
        if record["error"]:
            continue  # transport errors are counted separately, not invented proposals
        try:
            data = _extract_json(record["content"] or "")
            if data is None:
                raise ValueError("No JSON object")
            if record["role"] in SPECIALIST_CFG:
                proposal = Proposal.model_validate({**data, "agent_id": record["role"]})
                actions = [proposal.action]
            else:
                plan = CommanderDecision.model_validate(_normalize_commander_data(data))
                actions = plan.approved_actions
                if not evaluate_plan(list(dict.fromkeys(actions)), resources, colony.modules).feasible:
                    counts["infeasible_plans"] += 1
            for action in actions:
                if not evaluate_plan([action], resources, colony.modules).feasible:
                    counts["invalid_proposals"] += 1
            for action, occurrences in Counter(actions).items():
                counts["rejected_actions"] += max(0, occurrences - int(action in executed))
        except (ValueError, TypeError, AttributeError):
            counts["invalid_responses"] += 1
    return counts


def resource_sample(simulator, phase):
    return {"phase": phase, "tick": simulator.snapshot().tick, "resources": dict(simulator._resources)}


def recovery_met(final_resources, minima):
    return (
        all(final_resources[r] >= threshold - 1e-8 for r, threshold in RECOVERY_THRESHOLDS.items())
        and all(minima[r] > 0 for r in RECOVERY_THRESHOLDS)
        and final_resources["fuel"] >= 0
    )


async def run_episode(scenario, approach, repetition, seed, args, factory_class=MeasuredFactory):
    physics_version = getattr(args, "physics_version", "v1")
    graph_version = getattr(args, "graph_version", "v1")
    simulator = new_simulator(scenario, seed, physics_version)
    initial = simulator.snapshot().model_dump()
    samples = [resource_sample(simulator, "initial")]
    simulator.trigger_crisis(scenario.crisis)
    samples.append(resource_sample(simulator, "crisis"))
    calls, steps, errors = [], [], []
    counts = Counter()
    latency = 0.
    for step in range(args.steps):
        simulator.advance_tick()
        samples.append(resource_sample(simulator, f"passive-{step + 1}"))
        colony = simulator.snapshot().model_copy(deep=True)
        step_calls = []
        context = shared_context(colony, args.steps - step, physics_version)
        factory = factory_class(args, context, seed + step * 10, step_calls)
        started = time.perf_counter()
        approved, executed, step_errors = [], [], []
        execution = None
        result = {}
        try:
            if approach == "single":
                result = await decide_generalist(colony, factory, context)
            else:
                if graph_version == "v2":
                    from efficient_graph import build_efficient_graph
                    pipeline = build_efficient_graph(
                        factory, routing=not getattr(args, "disable_specialist_routing", False),
                        horizon=args.steps - step, physics_version=physics_version, context=context,
                    )
                else:
                    pipeline = build_graph(factory)
                result = await pipeline.ainvoke({
                    "colony": colony, "proposals": [], "events": [], "errors": [],
                    "approved_actions": [], "approved_effects": {}, "conflicts": [],
                })
            step_errors.extend(result.get("errors") or [])
            approved = result.get("approved_actions") or []
            execution = simulator.execute_actions(approved)
            executed = [item.action for item in execution.executions]
        except Exception as exc:
            step_errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            simulator.reset_all_statuses()
        elapsed = time.perf_counter() - started
        latency += elapsed
        errors.extend(f"step {step + 1}: {error}" for error in step_errors)
        counts.update(audit_calls(step_calls, colony, executed))
        counts["executed_actions"] += sum(a != "idle" for a in executed)
        counts["executed_holds"] += sum(a == "idle" for a in executed)
        for call in step_calls:
            call["step"] = step + 1
        calls.extend(step_calls)
        samples.append(resource_sample(simulator, f"executed-{step + 1}"))
        steps.append({
            "step": step + 1, "state_before_decision": colony.model_dump(),
            "approved_actions": approved, "executed_actions": executed,
            "execution": execution.model_dump() if execution else None,
            "activation": result.get("activation"),
            "errors": step_errors, "seconds": elapsed,
        })
    minima = {r: min(sample["resources"][r] for sample in samples) for r in simulator._resources}
    row = {
        "scenario": scenario.name, "approach": approach, "repetition": repetition,
        "graph_version": graph_version if approach == "multi" else "generalist_v1",
        "physics_version": physics_version, "score_version": simulator.score_version,
        "seed": seed, "initial_state_sha256": digest(initial), "decision_ticks": args.steps,
        "final_survival_score": round(simulator._compute_score(), 6),
        **{f"min_{r}": round(v, 6) for r, v in minima.items()},
        **{f"final_{r}": round(v, 6) for r, v in simulator._resources.items()},
        **{name: counts[name] for name in ("invalid_proposals", "invalid_responses", "infeasible_plans", "rejected_actions", "executed_actions", "executed_holds")},
        "state_recovery_met": recovery_met(simulator._resources, minima),
        "recovery_success": recovery_met(simulator._resources, minima) and not errors,
        "run_failed": bool(errors), "errors": json.dumps(errors),
        "decision_latency_seconds": round(latency, 6), "model_calls": len(calls),
        "transport_failures": sum(bool(c["error"]) for c in calls),
        "mean_input_characters": round(statistics.mean(c["input_characters"] for c in calls), 2) if calls else None,
    }
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        row[key] = sum(c[key] for c in calls) if calls and all(isinstance(c[key], int) for c in calls) else None
    row["token_usage_complete"] = all(row[k] is not None for k in ("input_tokens", "output_tokens", "total_tokens"))
    trace = {"initial_state": initial, "samples": samples, "steps": steps, "calls": calls}
    return row, trace


def write_summary(rows, output, manifest):
    def mean(group, key):
        values = [r[key] for r in group]
        return f"{statistics.mean(values):.2f}" if all(v is not None for v in values) else "unavailable"

    lines = [
        "# HIVE paired benchmark", "", f"Command: `{manifest['command']}`", "",
        f"Model: `{manifest['config']['model']}`. {len(rows)} of {len(SCENARIOS) * manifest['config']['repetitions'] * 2} planned episodes recorded; all failed episodes retained.", "",
        f"Versions: HIVE graph {manifest['config'].get('graph_version', 'v1')}, physics {manifest['config'].get('physics_version', 'v1')}, unchanged resource_v1 score and generalist_v1 planner.", "",
        "| Scenario | Approach | n | Score mean ± SD | Recovery | Failures | Latency s | Calls | Input tokens | Output tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        for approach in ("single", "multi"):
            group = [r for r in rows if r["scenario"] == scenario.name and r["approach"] == approach]
            if not group:
                continue
            sd = statistics.stdev(r["final_survival_score"] for r in group) if len(group) > 1 else 0.
            lines.append(f"| {scenario.name} | {approach} | {len(group)} | {mean(group, 'final_survival_score')} ± {sd:.2f} | {sum(r['recovery_success'] for r in group)}/{len(group)} | {sum(r['run_failed'] for r in group)} | {mean(group, 'decision_latency_seconds')} | {mean(group, 'model_calls')} | {mean(group, 'input_tokens')} | {mean(group, 'output_tokens')} |")
    lines += ["", "## Overall (all scenarios equally represented)", "",
              "| Approach | Mean score | Recovery | Mean latency s | Total calls | Total tokens | Invalid proposals | Invalid responses | Infeasible plans | Rejected occurrences | Executed actions |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for approach in ("single", "multi"):
        group = [r for r in rows if r["approach"] == approach]
        if not group:
            continue
        tokens = sum(r["total_tokens"] for r in group) if all(r["total_tokens"] is not None for r in group) else "unavailable"
        lines.append(f"| {approach} | {mean(group, 'final_survival_score')} | {sum(r['recovery_success'] for r in group)}/{len(group)} | {mean(group, 'decision_latency_seconds')} | {sum(r['model_calls'] for r in group)} | {tokens} | {sum(r['invalid_proposals'] for r in group)} | {sum(r['invalid_responses'] for r in group)} | {sum(r['infeasible_plans'] for r in group)} | {sum(r['rejected_actions'] for r in group)} | {sum(r['executed_actions'] for r in group)} |")
    lines += ["", "## Interpretation and limitations", "",
        "Recovery means final oxygen ≥40, power ≥30, water ≥20, food ≥20, fuel ≥0; no essential resource hit zero at any sampled point and no terminal inference/execution failure occurred. It means resource stabilization, not clearing the persistent crisis flag.", "",
        "Each episode has one crisis injection (which advances a tick), then the configured number of decision ticks. Minima include initial state, crisis effects, every passive-drain boundary, and every atomic action-plan completion. They exclude internal operation ordering within an atomic plan.", "",
        "Both systems receive the same state, full catalog, hazards, horizon, objectives and constraints. Generalist instructions and capabilities are unchanged. HIVE v1 adds role guidance and proposal-subset analysis; opt-in HIVE v2 uses routed compact specialist prompts and Commander selection from a deterministically ranked catalog-plan shortlist. Supporting actions are labeled as Commander selections, not invented specialist proposals. This is an engineered-system comparison, not equal compute or isolated proof of multi-agent superiority.", "",
        "Generalist and Commander each receive 500 output tokens by default; each specialist gets 300. Both use one bounded plan/JSON correction and the same ordered feasibility filter. Transport retries are disabled for auditable calls. Actual input/output tokens and prompts are saved; unavailable usage stays null/blank, never zero-imputed.", "",
        "Invalid proposals count individually infeasible raw action occurrences across all model replies, including Commander and repair replies. Invalid responses count unparsable/schema-invalid replies separately. Infeasible plans count raw combined-plan failures after deduplication. Rejected occurrences count raw action occurrences absent from execution, plus within-reply duplicates. An action can therefore be counted at multiple stages; these are process metrics, not directly normalized error rates. Executed actions exclude idle holds.", "",
        "Paired scenario/model seeds and alternating approach order reduce confounding. There is no simulation randomness; repeated scenarios use identical physical initial states. vLLM seed determinism is best effort, including concurrent specialist calls. Hardware load and prefix caching are uncontrolled. Three repetitions per scenario are exploratory, with no significance claim.", "",
        "Scores and recovery thresholds are unchanged. Physics v1 ignores module health in production; opt-in v2 scales existing passive production linearly by the producing modules' health. No direct health-score bonus is added. Free rationing and abstract catalog gains remain limitations. Source hashes, configuration and endpoint model metadata are in manifest.json. traces.jsonl preserves responses, prompts, activation reasons, errors and trajectories. Warm-up is recorded separately and excluded from scored costs.",
    ]
    output.write_text("\n".join(lines) + "\n")


async def main(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(), "command": shlex.join([sys.executable, *sys.argv]),
        "config": vars(args), "scenarios": [asdict(s) for s in SCENARIOS],
        "recovery_thresholds": RECOVERY_THRESHOLDS, "python": sys.version,
        "package_versions": {p: version(p) for p in ("langgraph", "langchain-openai", "openai", "pydantic")},
        "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in ("benchmark.py", "single_agent.py", "actions.py", "graph.py", "efficient_graph.py", "decision_policy.py", "physics_context.py", "simulator.py", "models.py")},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "schedule": [], "warmup": [],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        response = await client.get(args.base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {OPENAI_API_KEY}"})
        response.raise_for_status()
        manifest["endpoint_models"] = response.json()
        if args.model not in [m["id"] for m in response.json()["data"]]:
            raise ValueError(f"Requested checkpoint {args.model!r} is not served by endpoint")
    # One unscored request to reduce first-request loading effects.
    warmup = MeasuredFactory(args, "", args.seed, manifest["warmup"])
    try:
        await warmup.ainvoke([SystemMessage(content='Return only {"approved_actions":[],"rationale":"ready"}.')])
    except Exception as exc:
        manifest["warmup_error"] = str(exc)
    rows = []
    with (output / "results.csv").open("w", newline="") as csv_file, (output / "traces.jsonl").open("w") as trace_file:
        writer = None
        for index, scenario in enumerate(SCENARIOS):
            for repetition in range(1, args.repetitions + 1):
                seed = args.seed + index * 1000 + repetition * 100
                order = ("single", "multi") if (index + repetition) % 2 else ("multi", "single")
                paired_hash = None
                for approach in order:
                    manifest["schedule"].append({"scenario": scenario.name, "repetition": repetition, "seed": seed, "approach": approach})
                    row, trace = await run_episode(scenario, approach, repetition, seed, args)
                    if paired_hash is not None and paired_hash != row["initial_state_sha256"]:
                        raise AssertionError("Paired initial states differ")
                    paired_hash = row["initial_state_sha256"]
                    rows.append(row)
                    if writer is None:
                        writer = csv.DictWriter(csv_file, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    csv_file.flush()
                    trace_file.write(json.dumps({"result": row, **trace}) + "\n")
                    trace_file.flush()
                    manifest_path.write_text(json.dumps(manifest, indent=2))
                    write_summary(rows, output / "summary.md", manifest)
                    print(f"{scenario.name} rep={repetition} {approach}: score={row['final_survival_score']:.2f} recovery={row['recovery_success']} latency={row['decision_latency_seconds']:.2f}s calls={row['model_calls']} errors={row['errors']}", flush=True)
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved {len(rows)} episodes to {output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=HIVE_MODEL)
    parser.add_argument("--base-url", default=OPENAI_API_BASE)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--temperature", type=float, default=.2)
    parser.add_argument("--planner-tokens", type=int, default=500)
    parser.add_argument("--specialist-tokens", type=int, default=300)
    parser.add_argument("--timeout", type=float, default=12.)
    parser.add_argument("--graph-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--physics-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--disable-specialist-routing", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if min(args.steps, args.repetitions, args.planner_tokens, args.specialist_tokens) < 1:
        parser.error("Steps, repetitions and token limits must be positive")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main(parse_args()))
