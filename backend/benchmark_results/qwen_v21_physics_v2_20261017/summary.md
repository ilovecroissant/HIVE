# HIVE paired benchmark

Command: `/home/jasonnguyen/projects/hive/backend/.venv/bin/python backend/benchmark.py --base-url http://127.0.0.1:8001/v1 --model Qwen/Qwen2.5-7B-Instruct --steps 3 --repetitions 3 --seed 20261017 --graph-version v2 --physics-version v2 --output-dir backend/benchmark_results/qwen_v21_physics_v2_20261017`

Model: `Qwen/Qwen2.5-7B-Instruct`. 24 of 24 planned episodes recorded; all failed episodes retained.

Versions: HIVE graph v2, physics v2, unchanged resource_v1 score and generalist_v1 planner.

| Scenario | Approach | n | Score mean ± SD | Recovery | Failures | Latency s | Calls | Input tokens | Output tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dust_storm | single | 3 | 52.21 ± 0.47 | 0/3 | 0 | 4.88 | 3.00 | 3283.33 | 200.67 |
| dust_storm | multi | 3 | 47.16 ± 1.53 | 2/3 | 1 | 7.13 | 10.67 | 14514.33 | 425.33 |
| power_failure | single | 3 | 52.01 ± 0.61 | 0/3 | 0 | 4.80 | 3.00 | 3267.00 | 215.33 |
| power_failure | multi | 3 | 49.68 ± 0.75 | 3/3 | 0 | 7.38 | 10.67 | 14037.33 | 415.33 |
| oxygen_leak | single | 3 | 41.88 ± 0.00 | 3/3 | 0 | 4.42 | 3.00 | 3282.00 | 173.00 |
| oxygen_leak | multi | 3 | 47.01 ± 3.45 | 2/3 | 1 | 6.27 | 6.67 | 9762.33 | 280.00 |
| scarce_fuel | single | 3 | 40.27 ± 0.81 | 0/3 | 0 | 14.34 | 4.67 | 5385.00 | 664.00 |
| scarce_fuel | multi | 3 | 42.06 ± 0.29 | 0/3 | 0 | 6.30 | 9.00 | 12621.67 | 334.00 |

## Overall (all scenarios equally represented)

| Approach | Mean score | Recovery | Mean latency s | Total calls | Total tokens | Invalid proposals | Invalid responses | Infeasible plans | Rejected occurrences | Executed actions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 46.59 | 3/12 | 7.11 | 41 | 49411 | 0 | 0 | 5 | 8 | 83 |
| multi | 46.48 | 7/12 | 6.77 | 111 | 157171 | 6 | 2 | 0 | 28 | 115 |

## Interpretation and limitations

Recovery means final oxygen ≥40, power ≥30, water ≥20, food ≥20, fuel ≥0; no essential resource hit zero at any sampled point and no terminal inference/execution failure occurred. It means resource stabilization, not clearing the persistent crisis flag.

Each episode has one crisis injection (which advances a tick), then the configured number of decision ticks. Minima include initial state, crisis effects, every passive-drain boundary, and every atomic action-plan completion. They exclude internal operation ordering within an atomic plan.

Both systems receive the same state, full catalog, hazards, horizon, objectives and constraints. Generalist instructions and capabilities are unchanged. HIVE v1 adds role guidance and proposal-subset analysis; opt-in HIVE v2 uses routed compact specialist prompts and Commander selection from a deterministically ranked catalog-plan shortlist. Supporting actions are labeled as Commander selections, not invented specialist proposals. This is an engineered-system comparison, not equal compute or isolated proof of multi-agent superiority.

Generalist and Commander each receive 500 output tokens by default; specialists get 300. Each planner has one bounded correction. Generalist and legacy Commander retain the original ordered feasibility filter. HIVE v2 instead requires a validated listed plan after correction, otherwise it fails safely. Its deterministic selection policy excludes avoidable forecast threshold breaches when threshold-preserving plans exist. Transport retries are disabled; usage is never zero-imputed when missing.

Invalid proposals count individually infeasible raw action occurrences across all model replies, including Commander and repair replies. Invalid responses count unparsable/schema-invalid replies separately. Infeasible plans count raw combined-plan failures after deduplication. Rejected occurrences count raw action occurrences absent from execution, plus within-reply duplicates. An action can therefore be counted at multiple stages; these are process metrics, not directly normalized error rates. Executed actions exclude idle holds.

Paired scenario/model seeds and alternating approach order reduce confounding. There is no simulation randomness; repeated scenarios use identical physical initial states. vLLM seed determinism is best effort, including concurrent specialist calls. Hardware load and prefix caching are uncontrolled. Three repetitions per scenario are exploratory, with no significance claim.

Scores and recovery thresholds are unchanged. Physics v1 ignores module health in production; opt-in v2 scales existing passive production linearly by the producing modules' health. No direct health-score bonus is added. Free rationing and abstract catalog gains remain limitations. Source hashes, configuration and endpoint model metadata are in manifest.json. traces.jsonl preserves responses, prompts, activation reasons, errors and trajectories. Warm-up is recorded separately and excluded from scored costs.
