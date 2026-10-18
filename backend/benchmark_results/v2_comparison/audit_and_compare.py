"""Reproduce the failure-inclusive comparison and replay all recorded episodes."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[1]
sys.path.insert(0, str(BACKEND))
from benchmark import SCENARIOS, audit_calls, digest, new_simulator, recovery_met
from models import ColonyState

DATASETS = {
    "Original v1": "qwen_20261017",
    "Intermediate v2.0 pilot": "qwen_v2_graph_physics_v1_20261017",
    "Final v2.1 / physics v1": "qwen_v21_physics_v1_20261017",
    "Final v2.1 / physics v2": "qwen_v21_physics_v2_20261017",
}


def main():
    aggregate, audited, loaded = [], {}, {}
    saved = json.loads((HERE / "original_files_sha256.json").read_text())
    for name, expected in saved.items():
        assert hashlib.sha256((BACKEND / "benchmark_results/qwen_20261017" / name).read_bytes()).hexdigest() == expected
    audited["original_files_preserved_byte_for_byte"] = True
    original_manifest = json.loads((BACKEND / "benchmark_results/qwen_20261017/manifest.json").read_text())
    for label, directory in DATASETS.items():
        path = BACKEND / "benchmark_results" / directory
        manifest = json.loads((path / "manifest.json").read_text())
        assert manifest.get("completed_at"), f"Incomplete batch: {directory}"
        rows = list(csv.DictReader((path / "results.csv").open()))
        traces = [json.loads(line) for line in (path / "traces.jsonl").read_text().splitlines()]
        assert len(rows) == len(traces) == len(manifest["schedule"]) == 24
        assert manifest["source_sha256"]["single_agent.py"] == original_manifest["source_sha256"]["single_agent.py"]
        if (path / "sources").exists():
            for name, expected in manifest["source_sha256"].items():
                assert hashlib.sha256((path / "sources" / name).read_bytes()).hexdigest() == expected
        pairs = {}
        for row, trace in zip(rows, traces):
            assert all(row[key] == str(trace["result"][key]) if trace["result"][key] is not None else row[key] == "" for key in row)
            assert digest(trace["initial_state"]) == row["initial_state_sha256"]
            pairs.setdefault((row["scenario"], row["repetition"]), []).append(row["initial_state_sha256"])
            scenario = next(s for s in SCENARIOS if s.name == row["scenario"])
            simulator = new_simulator(scenario, int(row["seed"]), row.get("physics_version", "v1"))
            simulator.trigger_crisis(scenario.crisis)
            counts = {}
            for step in trace["steps"]:
                simulator.advance_tick()
                assert simulator.snapshot().model_dump() == step["state_before_decision"]
                if step["execution"] is not None:
                    report = simulator.execute_actions(step["approved_actions"])
                    assert report.model_dump() == step["execution"]
                audited_calls = audit_calls([c for c in trace["calls"] if c["step"] == step["step"]], ColonyState.model_validate(step["state_before_decision"]), step["executed_actions"])
                for key, value in audited_calls.items():
                    counts[key] = counts.get(key, 0) + value
            assert abs(simulator._compute_score() - float(row["final_survival_score"])) < 1e-6
            minima = {r: min(s["resources"][r] for s in trace["samples"]) for r in simulator._resources}
            for r in minima:
                assert abs(minima[r] - float(row["min_" + r])) < 1e-6
                assert abs(simulator._resources[r] - float(row["final_" + r])) < 1e-6
            assert str(recovery_met(simulator._resources, minima) and not json.loads(row["errors"])) == row["recovery_success"]
            for key in ("invalid_proposals", "invalid_responses", "infeasible_plans", "rejected_actions"):
                assert counts.get(key, 0) == int(row[key])
            assert len(trace["calls"]) == int(row["model_calls"])
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                values = [call[key] for call in trace["calls"]]
                if all(isinstance(value, int) for value in values):
                    assert sum(values) == int(row[key])
                else:
                    assert row[key] == ""
        assert len(pairs) == 12 and all(len(values) == 2 and values[0] == values[1] for values in pairs.values())
        audited[directory] = {"episodes": 24, "matched_pairs": 12, "deterministic_replay": True,
            "token_and_csv_trace_accounting": True, "unchanged_generalist_source": True,
            "terminal_failed_episodes": sum(r["run_failed"] == "True" for r in rows)}
        loaded[label] = (rows, traces)
        for approach in ("single", "multi"):
            group = [r for r in rows if r["approach"] == approach]
            successful = [r for r in group if r["run_failed"] == "False"]
            aggregate.append({
                "version": label, "approach": approach, "episodes": len(group),
                "mean_score": statistics.mean(float(r["final_survival_score"]) for r in group),
                "recoveries": sum(r["recovery_success"] == "True" for r in group),
                "model_calls": sum(int(r["model_calls"]) for r in group),
                "total_tokens": sum(int(r["total_tokens"]) for r in group),
                "input_tokens": sum(int(r["input_tokens"]) for r in group),
                "output_tokens": sum(int(r["output_tokens"]) for r in group),
                "mean_episode_latency": statistics.mean(float(r["decision_latency_seconds"]) for r in group),
                "mean_nonfailed_episode_latency": statistics.mean(float(r["decision_latency_seconds"]) for r in successful),
                "failed_episodes": sum(r["run_failed"] == "True" for r in group),
            })
    # Actual first generalist request equality, not merely a source-code assertion.
    original = loaded["Original v1"][1]
    final = loaded["Final v2.1 / physics v1"][1]
    matching = 0
    for trace in final:
        if trace["result"]["approach"] != "single":
            continue
        previous = next(t for t in original if all(t["result"][k] == trace["result"][k] for k in ("scenario", "repetition", "approach")))
        assert previous["calls"][0]["messages"] == trace["calls"][0]["messages"]
        assert previous["calls"][0]["seed"] == trace["calls"][0]["seed"]
        matching += 1
    audited["first_baseline_requests_unchanged_v1"] = matching
    (HERE / "audit.json").write_text(json.dumps(audited, indent=2) + "\n")
    with (HERE / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    lines = ["# HIVE version comparison", "", "All means include failed episodes. Latency is per three-decision-tick episode. Each row contains 12 episodes; no failed runs were removed.", "",
        "| Version / physics | Approach | Mean score | Recovery | Calls | Total tokens | Mean latency s | Failed episodes |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate:
        lines.append(f"| {row['version']} | {row['approach']} | {row['mean_score']:.2f} | {row['recoveries']}/12 | {row['model_calls']} | {row['total_tokens']:,} | {row['mean_episode_latency']:.2f} | {row['failed_episodes']} |")
    lines += ["", "## Per-scenario final comparisons", "",
        "| Version | Scenario | Single score | HIVE score | Single recovery | HIVE recovery |",
        "|---|---|---:|---:|---:|---:|"]
    for label in ("Final v2.1 / physics v1", "Final v2.1 / physics v2"):
        rows = loaded[label][0]
        for scenario in SCENARIOS:
            single = [r for r in rows if r["scenario"] == scenario.name and r["approach"] == "single"]
            multi = [r for r in rows if r["scenario"] == scenario.name and r["approach"] == "multi"]
            lines.append(f"| {label} | {scenario.name} | {statistics.mean(float(r['final_survival_score']) for r in single):.2f} | {statistics.mean(float(r['final_survival_score']) for r in multi):.2f} | {sum(r['recovery_success']=='True' for r in single)}/3 | {sum(r['recovery_success']=='True' for r in multi)}/3 |")
    lines += ["", "## Interpretation", "",
        "The intermediate v2.0 pilot is retained, including its exact executed sources. Its avoidable oxygen/power tradeoff motivated the separately tested v2.1 forecast-threshold guard. This development pilot is not an independent confirmation sample.", "",
        "The original-physics comparison isolates graph changes. The health-physics comparison changes production equally for both systems and must not be interpreted as a pure graph improvement. Score weights, recovery thresholds, scenarios and baseline capabilities stay fixed. All 12 first baseline requests in the final original-physics run match the original prompts and seeds byte-for-byte; later trajectories can diverge with best-effort GPU sampling.", "",
        "HIVE v2.1 includes deterministic catalog-plan enumeration and forecast selection in addition to specialist routing and model coordination. It is not an equal-compute or isolated test of agent collaboration. More stable reserves can reduce the resource-weighted score. Module health receives no direct score bonus. Forecasts assume holding after the current plan and can miss multi-step opportunities.", "",
        "All failures are retained in primary means. Latency on nonfailed episodes is available separately in comparison.csv so failures that skip Commander inference cannot masquerade as the entire efficiency improvement. No significance claim is made from three repetitions per scenario. Host load, caching and GPU sampling remain uncontrolled.", "",
        "See each run's results.csv for resource minima and every individual result, traces.jsonl for raw replies/usage/activation reasons, and sources/ plus manifest.json for the actual executed version. The original result files are unchanged.",
    ]
    (HERE / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
