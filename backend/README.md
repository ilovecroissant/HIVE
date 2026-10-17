# HIVE decision pipeline

Three concurrent Qwen specialist calls feed one Commander call. The Commander
receives the proposals, module health, catalog-derived costs and benefits,
structured conflicts, and feasible subsets with resource-survival-index impact
and lowest essential reserve. Summaries are public model explanations, not
private reasoning traces. No extra agents or iterative negotiation loop are used.

`actions.py` is the shared source of truth for previews and live execution:

- Reserve the total cost of a plan from its starting balances. Gains are usable
  next cycle, so a projected gain cannot finance another operation this cycle.
- No resource may go negative; gains and module health cap at 100.
- Battery dispatch and emergency generation are alternative modes of the single
  backup-power controller. Select at most one per cycle. Conservation can coexist
  with either when resources permit.
- Execute each action at most once per cycle. Repeated specialist proposals are
  reported as redundant and coalesced into one candidate action.
- Reject resource-producing actions whose benefits are entirely saturated.
- Repair costs three fuel and restores up to six health to the lowest-health
  damaged module (ties break by ID). Reject repair when no damaged module exists.
- `idle` is an explicit hold with no effects or cost; select it alone or return
  an empty plan. All other actions have costs/effects in `VALID_ACTIONS`.

Commander JSON or plan validation gets at most one correction request. After
that, malformed output yields an error and no actions. For parseable but still
infeasible output, validation retains only the feasible actions actually selected
by Commander, in its requested order, and explicitly logs every removed action.
It never invents substitute approvals. Only specialist-proposed actions may pass.

`Simulator.execute_actions` rechecks the entire selected plan against exact
current balances before atomically committing resource and health changes.
Model-supplied numeric effects are never executed. Events distinguish simulator
constraints from the Commander's discretionary preferences, and log per-action
execution and actual aggregate resource changes. Existing API fields/event types
are preserved. The response retains the latest 100 events so a full decision
cycle remains visible.

The graph is stateless between calls. The colony remains process-local and
requires one API worker. Concurrent mutations return HTTP 409 while reads remain
available. Each decision attempt still advances one tick and incurs passive
drain, even if inference fails; failed actions do not mutate resources or modules.
Module health affects repair decisions but does not yet change resource production
or the resource-only survival index. Crisis damage/automatic resolution are not
modeled. Rationing and recycling retain the existing abstract resource effects.

## Verification

From the repository root:

```sh
backend/.venv/bin/python -m pytest -q
cd hive-frontend && npm run build
```

Root and backend pytest configurations disable unrelated ROS launch plugins.
Tests inject model doubles; production live inference still uses `ChatOpenAI`.
To enable the existing vLLM endpoint, the API process must receive:

```sh
HIVE_LIVE_AGENTS=true
OPENAI_API_BASE=http://127.0.0.1:8001/v1
HIVE_MODEL=Qwen/Qwen2.5-7B-Instruct
```

These are environment settings, not commands to restart infrastructure. Commander
defaults to 500 output tokens and one bounded JSON/plan repair; specialists use
300. SDK transport retries may extend wall time beyond the per-call timeout.
