# HIVE v2.1: stronger recovery, higher inference cost

**Recorded exploratory evaluation · Qwen2.5-7B-Instruct · original physics**

| System | Recovery | Mean score /100 | Model calls, total | Tokens, total | Seconds / episode |
|---|---:|---:|---:|---:|---:|
| Original HIVE (historical reference) | 5/12 | 43.20 | 146 | 224,173 | 10.71 |
| Fresh single agent (paired comparison) | 5/12 | **47.19** | **38** | **42,569** | 6.79 |
| HIVE v2.1 (paired comparison) | **11/12** | 46.69 | 90 | 118,253 | **6.09** |

HIVE v2.1 used about **47% fewer tokens and 43% less latency than original HIVE**,
but still **2.78× the tokens of the fresh single agent**. Historical and fresh
batches are distinct; bold indicates a metric, not statistical significance.

| Crisis | Single recovery | HIVE v2.1 recovery | Single / HIVE mean score |
|---|---:|---:|---:|
| Dust storm | 0/3 | 3/3 | 52.25 / 54.39 |
| Power failure | 2/3 | 3/3 | 53.77 / 50.42 |
| Oxygen leak | 3/3 | 2/3 | 41.92 / 45.11 |
| Scarce fuel | 0/3 | 3/3 | 40.81 / 36.86 |

**Where single wins:** total tokens/calls, overall mean score, power/scarce-fuel
scores, and oxygen recovery. One HIVE JSON-failure episode is retained; the fresh
single agent had no failed episodes. Latencies include failures and are not SLAs.

Recovery: final oxygen≥40, power≥30, water≥20, food≥20, fuel≥0, no sampled essential
resource exhaustion, no terminal failure. Each episode: one crisis + three decisions.
All totals include all 12 episodes per system. Four scenarios, three repetitions;
no statistical significance claim. Same model/rules/states, different prompts and
compute. HIVE includes deterministic plan search; this is a whole-system comparison.

**30-second explanation:** “HIVE routes only relevant specialists, then a Commander
chooses among simulator-validated plans. In our small paired test it stabilized
11 of 12 crises versus 5 for one generalist. It still costs nearly three times the
tokens and scores slightly lower on the weighted resource index. We show every
failure, and every executed resource change comes from deterministic rules.”

Sources: [fresh paired CSV](../backend/benchmark_results/qwen_v21_physics_v1_20261017/results.csv),
[fresh full summary](../backend/benchmark_results/qwen_v21_physics_v1_20261017/summary.md),
[historical summary](../backend/benchmark_results/qwen_20261017/summary.md).
