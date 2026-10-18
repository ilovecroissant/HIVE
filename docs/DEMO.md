# Three-minute judging demonstration

Prepare the [README configuration](../README.md) and open mission control plus the
[recorded comparison card](BENCHMARK_CARD.md). This is a live model demonstration,
not a guaranteed replay. Inspect actual events before describing any decision.

**0:00–0:25 — Problem and deployment.** “On Mars, power, oxygen and fuel decisions
compete. HIVE uses Qwen2.5-7B-Instruct served through vLLM on NVIDIA Brev L40S.
LangGraph coordinates three specialists and a Commander.” Show the v2.1/routed/
physics-v1 identity. Science is not connected to the inference graph.

**0:25–0:50 — Real starting state.** Reset the demo colony, select Dust Storm and
trigger it. Point out actual power loss and the tick change. “This is a deterministic
simulator. Polling does not spend colony time; injecting a crisis does.”

**0:50–1:40 — Run one HIVE decision.** Click Run HIVE Decision once. Show actual
statuses during inference and recorded ACTIVE/SKIPPED routing reasons afterward.
Read the latest-tick timeline in chronological order: specialist recommendations,
real conflicts if present, Commander approval/rejection, execution, measured usage.
“No conflict” is a valid outcome. Do not imply conflict detection always fires.
Explanations are public decision summaries, not private model reasoning.

**1:40–2:10 — Verify execution.** Point to an Executed event and its actual resource
deltas, then corresponding resource cards. If repair runs, show module health.
Original physics does not translate that health increase into extra production.
A crisis flag persists; one successful action is not the benchmark recovery test.
If a model call fails, show the error; do not call it successful or silently switch
to scripted mode. Explain bounded validation and safe nonexecution of invalid plans.

**2:10–2:45 — Honest comparison.** Open the comparison card. “11/12 versus 5/12
recovered; mean score 46.69 versus 47.19; 118,253 versus 42,569 tokens. Routing
cut tokens about 47% and latency 43% relative to original HIVE. The single agent
still wins cost and some scenarios.” This is a small exploratory system comparison.

**2:45–3:00 — Close.** “Every plan is revalidated atomically. Our contribution is
resource-aware coordination with an auditable timeline, not a claim that more agents
always win.” Show the repository's reproduction command and retained failure traces.

## Submission checklist

- [ ] Confirm hackathon eligibility, required NVIDIA/Red Hat evidence, deadline,
  licensing and submission fields against the actual rules; do not imply an
  unimplemented Red Hat deployment or NVIDIA-owned model.
- [ ] Existing Brev endpoint returns the exact Qwen checkpoint; no GPU restart needed.
- [ ] Start API with `bash scripts/demo-backend.sh`; verify `/health` configuration.
- [ ] Build frontend with live API values; browser shows confirmed v2.1 identity.
- [ ] Backend tests, timeline tests and production build pass on the submission tree.
- [ ] Perform one real HTTP crisis→decision run; inspect errors, approvals, executed
  deltas and final idle statuses. Record date/configuration and actual outcome.
- [ ] Rehearse three minutes; zoom text for judges and leave latest-tick view enabled.
- [ ] Keep archived CSVs, summaries, traces and manifests; no cherry-picked reruns.
- [ ] Explain score/recovery distinction, costs, failure episode and small sample.
- [ ] Provide repository URL, reproducibility docs, demo video/link and hardware
  evidence as required; check access from a signed-out browser.
- [ ] Exclude credentials/private environment files; do not expose this unauthenticated
  local demo publicly without appropriate access controls.
- [ ] If inference is unavailable, label recorded evidence as recorded; never pass
  scripted mode or an old run off as a successful new live verification.
