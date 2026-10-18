# HIVE version comparison

All means include failed episodes. Latency is per three-decision-tick episode. Each row contains 12 episodes; no failed runs were removed.

| Version / physics | Approach | Mean score | Recovery | Calls | Total tokens | Mean latency s | Failed episodes |
|---|---|---:|---:|---:|---:|---:|---:|
| Original v1 | single | 47.46 | 6/12 | 38 | 42,398 | 6.63 | 0 |
| Original v1 | multi | 43.20 | 5/12 | 146 | 224,173 | 10.71 | 0 |
| Intermediate v2.0 pilot | single | 47.19 | 5/12 | 38 | 42,445 | 6.82 | 0 |
| Intermediate v2.0 pilot | multi | 48.94 | 9/12 | 96 | 129,074 | 6.34 | 0 |
| Final v2.1 / physics v1 | single | 47.19 | 5/12 | 38 | 42,569 | 6.79 | 0 |
| Final v2.1 / physics v1 | multi | 46.69 | 11/12 | 90 | 118,253 | 6.09 | 1 |
| Final v2.1 / physics v2 | single | 46.59 | 3/12 | 41 | 49,411 | 7.11 | 0 |
| Final v2.1 / physics v2 | multi | 46.48 | 7/12 | 111 | 157,171 | 6.77 | 2 |

## Per-scenario final comparisons

| Version | Scenario | Single score | HIVE score | Single recovery | HIVE recovery |
|---|---|---:|---:|---:|---:|
| Final v2.1 / physics v1 | dust_storm | 52.25 | 54.39 | 0/3 | 3/3 |
| Final v2.1 / physics v1 | power_failure | 53.77 | 50.42 | 2/3 | 3/3 |
| Final v2.1 / physics v1 | oxygen_leak | 41.92 | 45.11 | 3/3 | 2/3 |
| Final v2.1 / physics v1 | scarce_fuel | 40.81 | 36.86 | 0/3 | 3/3 |
| Final v2.1 / physics v2 | dust_storm | 52.21 | 47.16 | 0/3 | 2/3 |
| Final v2.1 / physics v2 | power_failure | 52.01 | 49.68 | 0/3 | 3/3 |
| Final v2.1 / physics v2 | oxygen_leak | 41.88 | 47.01 | 3/3 | 2/3 |
| Final v2.1 / physics v2 | scarce_fuel | 40.27 | 42.06 | 0/3 | 0/3 |

## Interpretation

The intermediate v2.0 pilot is retained, including its exact executed sources. Its avoidable oxygen/power tradeoff motivated the separately tested v2.1 forecast-threshold guard. This development pilot is not an independent confirmation sample.

The original-physics comparison isolates graph changes. The health-physics comparison changes production equally for both systems and must not be interpreted as a pure graph improvement. Score weights, recovery thresholds, scenarios and baseline capabilities stay fixed. All 12 first baseline requests in the final original-physics run match the original prompts and seeds byte-for-byte; later trajectories can diverge with best-effort GPU sampling.

HIVE v2.1 includes deterministic catalog-plan enumeration and forecast selection in addition to specialist routing and model coordination. It is not an equal-compute or isolated test of agent collaboration. More stable reserves can reduce the resource-weighted score. Module health receives no direct score bonus. Forecasts assume holding after the current plan and can miss multi-step opportunities.

All failures are retained in primary means. Latency on nonfailed episodes is available separately in comparison.csv so failures that skip Commander inference cannot masquerade as the entire efficiency improvement. No significance claim is made from three repetitions per scenario. Host load, caching and GPU sampling remain uncontrolled.

See each run's results.csv for resource minima and every individual result, traces.jsonl for raw replies/usage/activation reasons, and sources/ plus manifest.json for the actual executed version. The original result files are unchanged.
