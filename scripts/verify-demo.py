"""Real HTTP-path smoke test in an isolated API process; never touches live colony.
Run: backend/.venv/bin/python scripts/verify-demo.py
No mock models, infrastructure operations, or fallback responses.
"""
import json
import os
from pathlib import Path
import sys
from urllib.request import urlopen

os.environ.update(HIVE_LIVE_AGENTS="true", HIVE_GRAPH_VERSION="v2",
                  HIVE_SPECIALIST_ROUTING="true", HIVE_PLANNING_HORIZON="3",
                  HIVE_PHYSICS_VERSION="v1", HIVE_MODEL="Qwen/Qwen2.5-7B-Instruct")
os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:8001/v1")
os.environ.setdefault("OPENAI_API_KEY", "ollama")
try:
    with urlopen(os.environ["OPENAI_API_BASE"].rstrip("/") + "/models", timeout=5) as response:
        models = json.load(response)
    assert any(model["id"] == os.environ["HIVE_MODEL"] for model in models["data"]), models
except Exception as exc:
    print(f"LIVE VERIFICATION BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(2)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from fastapi.testclient import TestClient
from main import app
with TestClient(app) as client:
    config = client.get("/health").json()
    assert config["live_agents"] and config["specialist_routing"]
    assert config["selection_policy"] == "v2.1" and config["physics_version"] == "v1"
    client.post("/simulation/reset").raise_for_status()
    before = client.post("/simulation/crisis", json={"crisis": "dust_storm"})
    before.raise_for_status()
    response = client.post("/agents/decide")
    response.raise_for_status()
    result = response.json()
    tick = result["state"]["tick"]
    events = [event for event in result["events"] if event["tick"] == tick]
    print(json.dumps({"configuration": config, "before": before.json()["state"]["resources"],
                      "after": result["state"]["resources"], "events": events}, indent=2))
    assert not any(event["type"] == "error" for event in events), "Real inference failed"
    assert any(event["title"] == "Specialist Activation" for event in events)
    assert any(event["title"].startswith("Commander Approves:") for event in events)
    assert any(event["title"].startswith("Executed:") for event in events)
    assert all(agent["status"] == "idle" for agent in result["state"]["agents"])
    print("PASS: real Qwen crisis, routing, Commander, execution and status cleanup")
