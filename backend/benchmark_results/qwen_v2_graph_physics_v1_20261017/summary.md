# HIVE paired benchmark

Command: `/home/jasonnguyen/projects/hive/backend/.venv/bin/python backend/benchmark.py --base-url http://127.0.0.1:8001/v1 --model Qwen/Qwen2.5-7B-Instruct --steps 3 --repetitions 3 --seed 20261017 --graph-version v2 --physics-version v1 --output-dir backend/benchmark_results/qwen_v2_graph_physics_v1_20261017`

Model: `Qwen/Qwen2.5-7B-Instruct`. 24 of 24 planned episodes recorded; all failed episodes retained.

Versions: HIVE graph v2, physics v1, unchanged resource_v1 score and generalist_v1 planner.

| Scenario | Approach | n | Score mean ± SD | Recovery | Failures | Latency s | Calls | Input tokens | Output tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dust_storm | single | 3 | 52.25 ± 0.39 | 0/3 | 0 | 5.07 | 3.00 | 3016.33 | 233.67 |
| dust_storm | multi | 3 | 54.22 ± 0.00 | 3/3 | 0 | 6.59 | 8.00 | 10307.67 | 315.67 |
| power_failure | single | 3 | 53.77 ± 1.05 | 2/3 | 0 | 6.67 | 3.00 | 3002.67 | 315.33 |
| power_failure | multi | 3 | 50.42 ± 0.00 | 3/3 | 0 | 6.08 | 7.00 | 9154.67 | 264.33 |
| oxygen_leak | single | 3 | 41.92 ± 0.00 | 3/3 | 0 | 3.90 | 3.00 | 3015.00 | 177.33 |
| oxygen_leak | multi | 3 | 49.05 ± 0.00 | 3/3 | 0 | 6.76 | 7.00 | 9616.00 | 302.67 |
| scarce_fuel | single | 3 | 40.81 ± 0.00 | 0/3 | 0 | 11.63 | 3.67 | 3873.67 | 514.33 |
| scarce_fuel | multi | 3 | 42.09 ± 0.29 | 0/3 | 0 | 5.93 | 10.00 | 12728.33 | 335.33 |

## Overall (all scenarios equally represented)

| Approach | Mean score | Recovery | Mean latency s | Total calls | Total tokens | Invalid proposals | Invalid responses | Infeasible plans | Rejected occurrences | Executed actions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 47.19 | 5/12 | 6.82 | 38 | 42445 | 0 | 0 | 2 | 2 | 93 |
| multi | 48.94 | 9/12 | 6.34 | 96 | 129074 | 6 | 0 | 0 | 18 | 125 |

## Interpretation and limitations

Recovery means final oxygen ≥40, power ≥30, water ≥20, food ≥20, fuel ≥0; no essential resource hit zero at any sampled point and no terminal inference/execution failure occurred. It means resource stabilization, not clearing the persistent crisis flag.

Each episode has one crisis injection (which advances a tick), then the configured number of decision ticks. Minima include initial state, crisis effects, every passive-drain boundary, and every atomic action-plan completion. They exclude internal operation ordering within an atomic plan.

Both systems receive the same state, full catalog, hazards, horizon, objectives and constraints. Generalist instructions and capabilities are unchanged. HIVE v1 adds role guidance and proposal-subset analysis; opt-in HIVE v2 uses routed compact specialist prompts and Commander selection from a deterministically ranked catalog-plan shortlist. Supporting actions are labeled as Commander selections, not invented specialist proposals. This is an engineered-system comparison, not equal compute or isolated proof of multi-agent superiority.

Intermediate v2.0 pilot: Generalist and Commander each receive 500 output tokens; specialists get 300. Each planner has one bounded correction. The generalist keeps its original ordered feasibility filter; pilot Commander instead requires a validated listed plan. This pilot predates the v2.1 forecast-threshold guard. Actual executed sources are archived in sources/. Transport retries are disabled and missing usage is never zero-imputed.

Invalid proposals count individually infeasible raw action occurrences across all model replies, including Commander and repair replies. Invalid responses count unparsable/schema-invalid replies separately. Infeasible plans count raw combined-plan failures after deduplication. Rejected occurrences count raw action occurrences absent from execution, plus within-reply duplicates. An action can therefore be counted at multiple stages; these are process metrics, not directly normalized error rates. Executed actions exclude idle holds.

Paired scenario/model seeds and alternating approach order reduce confounding. There is no simulation randomness; repeated scenarios use identical physical initial states. vLLM seed determinism is best effort, including concurrent specialist calls. Hardware load and prefix caching are uncontrolled. Three repetitions per scenario are exploratory, with no significance claim.

Scores and recovery thresholds are unchanged. Physics v1 ignores module health in production; opt-in v2 scales existing passive production linearly by the producing modules' health. No direct health-score bonus is added. Free rationing and abstract catalog gains remain limitations. Source hashes, configuration and endpoint model metadata are in manifest.json. traces.jsonl preserves responses, prompts, activation reasons, errors and trajectories. Warm-up is recorded separately and excluded from scored costs.
