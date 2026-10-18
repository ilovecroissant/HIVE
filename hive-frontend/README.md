# HIVE mission control frontend

Next.js + TypeScript dashboard for the FastAPI colony simulator. See the
[root README](../README.md) for the complete architecture, verified v2.1 launch
configuration, model details and benchmark limitations.

For judging, build with `NEXT_PUBLIC_DEMO_MODE=false` and
`NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`, then `npm run start`.
These public values are embedded at build time. The API must independently enable
live inference and v2 routing; the dashboard checks `/health` to label the mode.

The network has three model specialists and one Commander. Science is displayed
as not connected. Activation/skip reasons are recorded backend events, agent
statuses are polled during decisions, and the latest tick timeline displays real
proposals, conflicts (when present), decisions, rejections and execution deltas.
“All ticks” retains the previous events. No hidden chain-of-thought is displayed.
Human directive input invokes keyword simulator operations, not a model prompt.

`NEXT_PUBLIC_DEMO_MODE` values other than `false` enable the explicitly labeled
scripted frontend demo; this does not demonstrate Qwen inference.

Checks: `node --test tests/event-presentation.test.cjs` and `npm run build`.
