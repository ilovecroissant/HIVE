# HIVE — Mars Mission Control frontend

Cinematic Next.js + TypeScript + Tailwind command center with an SVG Mars map, animated telemetry, colony modules, crisis controls, five specialist agent statuses, chronological event log, and Human Commander input.

## Running in Windows + WSL + VS Code

1. Open **Ubuntu/WSL terminal**, keep the project under `~/projects/hive-frontend` rather than `/mnt/c/...` for faster file watching.
2. Run `code .` from the project directory (requires VS Code Remote - WSL extension).
3. Install Node.js 20+ (Node.js 22 LTS recommended), then run `npm install`.
4. Copy `.env.example` to `.env.local` and set `NEXT_PUBLIC_DEMO_MODE=true` for the first run.
5. Run `npm run dev` and open http://localhost:3000 in Windows browser.
6. Run `npm run lint` and `npm run build` before committing.

## Important: scripted demo vs live AI

**Default DEMO MODE is clearly labeled.** It supplies deterministic resource values and scripted example event messages. No live AI model is called, and Human Commander instructions are merely recorded.

For live FastAPI responses, set:

```env
NEXT_PUBLIC_DEMO_MODE=false
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

Restart `npm run dev` after editing `.env.local`. FastAPI must enable CORS for `http://localhost:3000`. For example:

```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["GET","POST"], allow_headers=["Content-Type"])
```

### Typed backend contract (adjust adapter if existing API differs)

Defined in `lib/types.ts` and called via `lib/api.ts`:

| Method | Endpoint | Request | Response |
|---|---|---|---|
| GET | `/simulation/state` | — | `{ "state": ColonyState, "events": AgentEvent[] }` |
| POST | `/simulation/reset` | `{}` | same |
| POST | `/simulation/crisis` | `{ "crisis": "dust_storm" \| "oxygen_leak" \| "power_failure" }` | same |
| POST | `/agents/decide` | `{}` | same |
| POST | `/commander/command` | `{ "instruction": "..." }` | same |

`ColonyState.resources` should contain oxygen, power, water, food and fuel values on a **0–100** scale. Agent events should have stable IDs, ISO timestamps, event types, and titles/messages. Route assumptions are based on the earlier project plan; your actual FastAPI routes were not provided. Adapt `lib/api.ts` (and optionally normalize payload shapes) to your real backend.

### Development milestones

- [x] Static responsive dashboard and interactive SVG colony map
- [x] Typed backend adapter, error handling, timeout, offline state
- [x] Crisis and decision actions
- [x] Agent timeline rendering and Commander control
- [x] Explicit scripted demo mode
- [ ] Connect to user's *actual* FastAPI route definitions and validate real data
- [ ] Test successful production deployment with real NVIDIA/Red Hat model inference

### Notes

- No fake **live** AI events are generated: demo events are scripted and explicitly labeled.
- React rendering does not claim that Human Commander commands were applied.
- The frontend does not implement deterministic simulation logic for live mode. That belongs in FastAPI.
