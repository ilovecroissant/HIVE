# HIVE paired benchmark

Command: `/home/jasonnguyen/projects/hive/backend/.venv/bin/python backend/benchmark.py --base-url http://127.0.0.1:8001/v1 --model Qwen/Qwen2.5-7B-Instruct --steps 3 --repetitions 3 --seed 20261017 --output-dir backend/benchmark_results/qwen_20261017`

Model: `Qwen/Qwen2.5-7B-Instruct`. 24 of 24 planned episodes recorded; all failed episodes retained.

| Scenario | Approach | n | Score mean ± SD | Recovery | Failures | Latency s | Calls | Input tokens | Output tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dust_storm | single | 3 | 53.34 ± 1.40 | 1/3 | 0 | 5.26 | 3.00 | 3015.00 | 239.00 |
| dust_storm | multi | 3 | 44.64 ± 1.41 | 3/3 | 0 | 9.97 | 12.00 | 17725.00 | 594.33 |
| power_failure | single | 3 | 53.77 ± 1.05 | 2/3 | 0 | 6.25 | 3.00 | 3002.67 | 289.33 |
| power_failure | multi | 3 | 48.19 ± 0.79 | 0/3 | 0 | 11.00 | 12.00 | 17690.67 | 687.33 |
| oxygen_leak | single | 3 | 41.92 ± 0.00 | 3/3 | 0 | 3.96 | 3.00 | 3015.00 | 177.33 |
| oxygen_leak | multi | 3 | 41.00 ± 1.59 | 2/3 | 0 | 12.20 | 12.67 | 18914.00 | 731.00 |
| scarce_fuel | single | 3 | 40.81 ± 0.00 | 0/3 | 0 | 11.05 | 3.67 | 3873.67 | 520.67 |
| scarce_fuel | multi | 3 | 38.98 ± 0.00 | 0/3 | 0 | 9.67 | 12.00 | 17766.67 | 615.33 |

## Overall (all scenarios equally represented)

| Approach | Mean score | Recovery | Mean latency s | Total calls | Total tokens | Invalid proposals | Invalid responses | Infeasible plans | Rejected occurrences | Executed actions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 47.46 | 6/12 | 6.63 | 38 | 42398 | 0 | 0 | 2 | 2 | 95 |
| multi | 43.20 | 5/12 | 10.71 | 146 | 224173 | 0 | 2 | 0 | 32 | 73 |

## Interpretation and limitations

Recovery means final oxygen ≥40, power ≥30, water ≥20, food ≥20, fuel ≥0; no essential resource hit zero at any sampled point and no terminal inference/execution failure occurred. It means resource stabilization, not clearing the persistent crisis flag.

Each episode has one crisis injection (which advances a tick), then the configured number of decision ticks. Minima include initial state, crisis effects, every passive-drain boundary, and every atomic action-plan completion. They exclude internal operation ordering within an atomic plan.

The same full catalog, state, crisis dynamics, horizon, objective and constraints are appended to every model call. Existing HIVE prompts additionally include role guidance and Commander proposal-subset analysis. The generalist can combine any catalog actions; HIVE Commander can only select specialist proposals. These are architectural differences, not equal prompt lengths or equal compute.

Generalist and Commander each receive 500 output tokens by default; each specialist gets 300. Both use one bounded plan/JSON correction and the same ordered feasibility filter. Transport retries are disabled for auditable calls. Actual input/output tokens and prompts are saved; unavailable usage stays null/blank, never zero-imputed.

Invalid proposals count individually infeasible raw action occurrences across all model replies, including Commander and repair replies. Invalid responses count unparsable/schema-invalid replies separately. Infeasible plans count raw combined-plan failures after deduplication. Rejected occurrences count raw action occurrences absent from execution, plus within-reply duplicates. An action can therefore be counted at multiple stages; these are process metrics, not directly normalized error rates. Executed actions exclude idle holds.

Paired scenario/model seeds and alternating approach order reduce confounding. There is no simulation randomness; repeated scenarios use identical physical initial states. vLLM seed determinism is best effort, including concurrent specialist calls. Hardware load and prefix caching are uncontrolled. Three repetitions per scenario are exploratory, with no significance claim.

Scores are the simulator's resource-weighted index, not survival probabilities. Module repairs do not affect resource production or score, and existing free rationing/abstract recovery effects can favor plans optimizing the simplified catalog. Full action costs, scenario definitions, source hashes, configuration and endpoint model metadata are in manifest.json. traces.jsonl preserves every response, prompt, error and state trajectory. Warm-up is recorded separately and excluded from scored cost/latency.
