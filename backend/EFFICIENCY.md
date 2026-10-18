# HIVE efficiency and decision-quality versions

The generalist source, action catalog, legacy graph, survival weights, scenario
definitions, recovery thresholds, repetitions, sampling and token budgets remain
unchanged. `resource_v1` is still the sole survival score; no health bonus exists.
All output directories are new and the original benchmark files are preserved.

## Independently tested stages

1. **Routing:** `decision_policy.specialist_routing` activates Energy for power
   crises, power below 45, projected power below 30, or fuel below 10. Life Support
   activates for oxygen leaks, oxygen below 60, water below 40, food below 45, or
   projected essential-reserve breaches. Engineering always reviews damage below
   40 health. Noncritical maintenance below 70 is deferred during resource
   emergencies, unless v2 production damage threatens the associated resource.
   Unaffordable noncritical maintenance is skipped. These are explicit operational
   intervention policies, not new physics constraints. Every activation/skip and
   reason is recorded. Routing can be disabled for an ablation.
2. **Context and calls:** compact specialist and Commander prompts carry exactly
   one shared environment copy per call. Real specialist proposals still fan out
   and join at one Commander. No triggered expertise means an explicitly logged
   deterministic hold, with no fabricated inference. Existing events report usage
   and latency; benchmark traces retain the measured calls and activation records.
3. **Optional physics v2:** passive power production is
   `0.20*solar_health/100 + 0.10*reactor_health/100`; passive water regeneration is
   `0.05*life_support_health/100`, food `0.10*greenhouse_health/100`. The nominal
   0.30 power split is an explicit modeling assumption, not a calibrated hardware
   measurement. Power failure still suppresses power regeneration. Repair improves
   the repaired module's subsequent passive production. Active action deltas,
   consumption, crisis drains, communications/habitat effects, and score weights
   are unchanged. v1 remains the default and exactly replays the original episodes.
4. **Commander selection:** evaluate catalog combinations with the real validator
   and simulator, including predicted costs, resource changes and future passive
   drains. Rank by essential exhaustion, normalized threshold deficit, lowest
   essential reserve ratio, then the original score. Show up to six alternatives
   to Commander together with the genuine specialist proposals. Supporting catalog
   actions are attributed to Commander. This fixes the old restriction that one
   Energy proposal could never be combined with another useful power operation.
   After one correction, non-listed/malformed plans safely fail; there is no
   invented substitute approval. Execution still revalidates atomically.

Predictions assume this tick's plan followed by holding for the rest of the
planning horizon. They are not a full multi-step optimizer and do not assume
future AI decisions. The shortlist uses deterministic classical computation in
addition to model inference. Any outcome improvement measures the combined
engineered system, not an isolated causal effect of agent collaboration.

## Preserved pilot and selection correction

The first v2 pilot is retained in
`benchmark_results/qwen_v2_graph_physics_v1_20261017/`, including exact source
snapshots. It exposed avoidable power-threshold breaches: Commander selected extra
oxygen production even when a forecast threshold-preserving option existed.

Selection policy **v2.1** therefore excludes avoidable projected threshold breaches
when at least one threshold-preserving plan exists. If none exists, feasible
partial-recovery plans and hold remain eligible. Hold is otherwise offered only
when it meets the same forecast policy. This is an explicit HIVE selection guard;
the benchmark's recovery metrics and generalist planner are unchanged. A separate
regression reproduces the observed oxygen/power tradeoff. The guard is deliberately
conservative and may reject plans that could be rescued by later interventions.

## Benchmark protocol

Run from the repository root; directories must not already exist:

```sh
backend/.venv/bin/python backend/benchmark.py \
  --base-url http://127.0.0.1:8001/v1 --model Qwen/Qwen2.5-7B-Instruct \
  --steps 3 --repetitions 3 --seed 20261017 \
  --graph-version v2 --physics-version v1 \
  --output-dir backend/benchmark_results/qwen_v21_physics_v1_20261017

backend/.venv/bin/python backend/benchmark.py \
  --base-url http://127.0.0.1:8001/v1 --model Qwen/Qwen2.5-7B-Instruct \
  --steps 3 --repetitions 3 --seed 20261017 \
  --graph-version v2 --physics-version v2 \
  --output-dir backend/benchmark_results/qwen_v21_physics_v2_20261017
```

The first batch isolates graph changes under original physics. The second applies
the new physical production rule equally to both systems. Generalist instructions
and capabilities never change; physics v1 shared context is byte-identical to the
original builder. Physics v2 changes only shared environment data so neither
system receives false hazard/production information. Source hashes and snapshots
record the actual executed components. No old CSV or summary is overwritten.

Use `--graph-version v1 --physics-version v1` for the legacy protocol, and
`--disable-specialist-routing` to isolate routing within the v2 graph. No live
frontend colony is accessed. Small repeated samples and best-effort GPU seeds do
not support significance claims; compare the freshly rerun baseline as well as
the archived baseline. Different prompts, classical shortlist computation and
model-call counts are part of the engineering tradeoff, not equal-compute results.

## Optional application activation

`HIVE_GRAPH_VERSION=v2` opts into the new graph; `HIVE_SPECIALIST_ROUTING=false`
disables routing. `HIVE_PLANNING_HORIZON` defaults to 3.
`HIVE_PHYSICS_VERSION=v2` separately opts into production derating. Both versions
default to v1 to preserve existing application behavior. No server or GPU instance
is restarted by this implementation. Frontend and response contracts are unchanged.

## Measured outcome

The [failure-inclusive comparison](benchmark_results/v2_comparison/summary.md)
includes the original run, the preserved v2.0 development pilot, and both final
v2.1 batches. Under original physics, HIVE changed from score **43.20**, recovery
**5/12**, **146 calls**, **224,173 tokens**, and **10.71 seconds/episode**, to
**46.69**, **11/12**, **90 calls**, **118,253 tokens**, and **6.09 seconds**.

The freshly rerun generalist scored **47.19**, recovered **5/12**, used **38 calls**
and **42,569 tokens**, and averaged **6.79 seconds**. HIVE therefore still costs
about **2.78 times** the tokens and has a slightly lower mean score. In particular,
HIVE scores lower on power failure and scarce fuel; it also loses one oxygen-leak
recovery to malformed specialist JSON. All 12 first generalist requests match the
original prompts and seeds exactly. Later stochastic trajectories can differ.

With physics v2 applied equally, generalist/HIVE scores were **46.59/46.48**,
recoveries **3/12 versus 7/12**, calls **41/111**, tokens **49,411/157,171**, and
latencies **7.11/6.77 seconds**. HIVE had two failed episodes in that batch: one
specialist JSON failure and one Commander listed-plan validation failure. The
original-physics batch had one failed episode. Every failure is retained.

The development pilot scored higher than final v2.1 (**48.94 versus 46.69**) but
recovered fewer episodes (**9/12 versus 11/12**). Protecting threshold reserves can
reduce the weighted index; no metric was changed to hide that tradeoff. Exact
sources and pilot results remain available. The physics extension remains opt-in;
it did not demonstrate a uniform quality benefit and needs longer-horizon tests.

The comparison audit replays all **96 episodes** and verifies paired starting
states, unchanged baseline source, archived source hashes, final scores, resource
minima, execution reports, token totals and CSV/trace agreement. It also checks
that all original result files remain byte-identical. Run it again with:

```sh
backend/.venv/bin/python backend/benchmark_results/v2_comparison/audit_and_compare.py
```

## Final judging launcher

From the repository root, `bash scripts/demo-backend.sh` explicitly enables live
Qwen, graph v2 / policy v2.1, routing, horizon 3 and original physics v1. It preflights
the existing model endpoint and starts only a local single-worker API. `/health`
reports the selected configuration; it does not test model availability. See the
[root README](../README.md) for frontend build configuration and demo instructions.
