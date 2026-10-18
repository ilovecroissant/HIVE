# Single-agent versus HIVE benchmark

Run from the repository root against the existing vLLM server:

```sh
backend/.venv/bin/python backend/benchmark.py \
  --base-url http://127.0.0.1:8001/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --steps 3 --repetitions 3 --seed 20261017 \
  --output-dir backend/benchmark_results/qwen_20261017
```

The output directory must be new; previous runs are never overwritten. Choose a
different directory for a rerun. No API server, frontend colony, or infrastructure
is changed. The runner creates a fresh `Simulator` per episode and uses the real
model endpoint. Test doubles exist only in tests.

## Initial real-inference results

The [24-episode run](benchmark_results/qwen_20261017/summary.md) completed with no
terminal failures. Generalist/HIVE mean scores were **47.46/43.20**, recoveries
**6/12 versus 5/12**, and mean three-tick episode latencies **6.63/10.71 seconds**.
Scored inference consumed **42,398/224,173 tokens** over **38/146 calls**. HIVE won
dust-storm recovery (**3/3 versus 1/3**); the generalist recovered more often in
power failure and oxygen leak. Neither recovered the scarce-fuel case. This pilot
does not establish general superiority for either architecture.

The [CSV](benchmark_results/qwen_20261017/results.csv) contains every individual
result, including all resource minima. The [audit](benchmark_results/qwen_20261017/audit.json)
verified all 12 matched state pairs, source hashes, reported token totals, and
deterministic execution replay of every saved plan. Every call reported usage,
and no response hit its output-token limit. One unscored warm-up used 32 tokens.

HIVE restored 90 total module-health points versus the generalist's 24 across
these episodes. The resource-only survival index does not reward that difference;
it is a substantive limitation of using this simulator to compare architectures.

## Protocol fixed before measurement

- Four scenarios: dust storm, power failure, oxygen leak, and a scarce-fuel dust
  storm. Each has three paired repetitions and three decision ticks: 24 episodes.
- One crisis injection advances the clock once, followed by three passive-drain
  and decision/execution steps. Crises stay active throughout. Runs continue after
  failed decisions so subsequent passive deterioration is measured.
- Every pair starts from byte-identical serialized state, including deterministic
  simulation ID. Scenario resources and damaged-module health are explicit in
  `SCENARIOS`; the manifest preserves them. There is no random physics. Repetition
  seeds vary inference sampling, not physical initial conditions.
- Identical checkpoint ID and endpoint; temperature 0.2 and matched per-step
  inference seeds. Seeds are best effort with vLLM parallel scheduling, not a
  promise of bitwise-repeatable text. A seed may produce different responses in
  different roles because their prompts differ.
- Generalist and Commander use 500 maximum output tokens; each specialist uses
  300. One bounded JSON/plan correction is allowed for both planners. SDK transport
  retries are disabled so attempted model calls are auditable.
- Every call receives the same supplemental full state, complete catalog,
  individual feasibility checks, actual passive/crisis rules, remaining horizon,
  objective and recovery thresholds. HIVE retains its normal specialist prompts,
  proposal merging, conflict evaluation, and Commander subset analysis. No new
  specialist is added and no production graph code is changed.
- The generalist can choose any catalog combination, including more than three
  actions. Commander remains limited to specialist-proposed actions. This is an
  end-to-end architecture comparison, not equal compute or identical prompts.
- Both paths use `evaluate_plan`, the same ordered safety filtering after bounded
  correction, and atomic `Simulator.execute_actions`. Exact balances are rechecked
  before execution. No model-provided numeric effects are applied.
- Approach order alternates across pairs, giving six pairs in each order.
  Episodes run sequentially to avoid cross-approach GPU contention; HIVE retains
  its three concurrent specialist calls. One unscored warm-up call is recorded.

## Recovery and metrics

Recovery means **resource stabilization**: final oxygen ≥40, power ≥30, water ≥20,
food ≥20, fuel ≥0, no oxygen/power/water/food exhaustion at any sampled state, and
no terminal inference or execution failure. It does not mean the crisis flag was
cleared. These thresholds are evaluation criteria, not new simulator safety rules.

Minima include the initial state, crisis injection, every passive-drain boundary,
and every completed atomic plan. They exclude unobservable intermediate action
ordering within an atomic plan. Final score uses the simulator's resource-weighted
index. Latency sums complete decision/execution wall times, including graph setup
and bounded corrections; it does not sum concurrent specialist latencies.

`results.csv` has every episode, including failures: final score, all five resource
minima/final levels, invalid proposals/responses/plans, rejected occurrences,
executed actions/holds, recovery, failures, latency, model calls and tokens.

- **Invalid proposals:** individually infeasible raw action occurrences in all
  replies, including Commander and correction replies.
- **Invalid responses:** unparsable or schema-invalid replies, counted separately.
- **Infeasible plans:** raw combined plans failing validation after deduplication.
- **Rejected occurrences:** raw action occurrences absent from the executed plan,
  plus excess duplicates within a reply. The same action can be counted at multiple
  pipeline stages. These process counts are not normalized cross-architecture error
  rates; HIVE naturally emits more intermediate recommendations.
- **Executed actions:** actually committed non-idle operations; holds are separate.
- **Usage:** reported prompt/completion/total tokens. Missing metadata produces
  blank CSV values and null trace values, with `token_usage_complete=false`.

`summary.md` aggregates all recorded episodes without dropping failures.
`traces.jsonl` preserves prompts, raw replies, per-call usage/seeds, approvals,
execution reports, sampled states and errors. `manifest.json` preserves parameters,
scenario definitions, source hashes, dependency versions, endpoint model metadata,
ordering, warm-up usage and completion time. Interrupted runs lack completion time
and explicitly show fewer recorded episodes than planned.

## Limits on conclusions

Three repetitions and short horizons support a hackathon pilot, not a statistical
claim that one architecture is generally superior. Different prompt lengths and
model-call counts are measured costs. Server load and caching are uncontrolled;
the served model ID is verified, but the weight revision/GPU software stack is not
independently hashed. No uncertainty beyond descriptive sample standard deviations
is claimed.

The simulator is intentionally simple: module health does not affect production
or survival score, rationing adds food without a modeled crew penalty, and crises
cannot be repaired away. A generalist using broader action combinations may score
better; those results must be reported. The scarce-fuel scenario's recovery target
is physically attainable under the catalog rules, as checked by a deterministic
test, but optional repairs compete directly with the recovery fuel budget.
