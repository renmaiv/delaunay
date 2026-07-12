#!/usr/bin/env bash
# End-to-end smoke test: build the frontend, boot the server with the mock
# judge, and exercise the API + SPA. Requires core Python deps + node only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PORT=8765
CONFIG=/tmp/smoke-config.yaml

echo "==> building frontend"
(cd frontend && npm ci --silent && npm run build >/dev/null)

echo "==> writing mock-provider config"
python3 - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open("config.yaml"))
cfg["judge"]["provider"] = "mock"
yaml.safe_dump(cfg, open(sys.argv[1], "w"))
PY

echo "==> starting server on :$PORT"
python3 run.py --config "$CONFIG" --port "$PORT" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo "==> waiting for readiness"
for i in $(seq 1 30); do
  if curl -sf "http://localhost:$PORT/api/health" >/tmp/smoke-health.json 2>/dev/null; then
    break
  fi
  sleep 1
done
python3 - <<'PY'
import json
h = json.load(open("/tmp/smoke-health.json"))
assert h["status"] == "ok", h
assert h["judge"]["provider"] == "mock", h
print("   health ok:", h["judge"])
PY

echo "==> uploading a fixture"
curl -sf -F "file=@tests/fixtures/basic.json" \
  "http://localhost:$PORT/api/analyze" >/tmp/smoke-analyze.json
python3 - <<'PY'
import json, time, urllib.request
body = json.load(open("/tmp/smoke-analyze.json"))
if body["status"] != "completed":
    aid = body["analysis_id"]
    for _ in range(30):
        body = json.load(urllib.request.urlopen(f"http://localhost:8765/api/analysis/{aid}"))
        if body["status"] == "completed":
            break
        time.sleep(1)
assert body["status"] == "completed", body
assert body["result"]["turns"], "no turns in result"
print("   analyze ok:", len(body["result"]["turns"]), "turns")
PY

echo "==> uploading synthetic jsonl (detections expected)"
curl -sf -F "file=@data/ground_truth/synthetic_v2.jsonl" \
  "http://localhost:$PORT/api/analyze" >/tmp/smoke-syn.json
python3 - <<'PY'
import json, time, urllib.request
body = json.load(open("/tmp/smoke-syn.json"))
if body["status"] != "completed":
    aid = body["analysis_id"]
    for _ in range(60):
        body = json.load(urllib.request.urlopen(f"http://localhost:8765/api/analysis/{aid}"))
        if body["status"] == "completed":
            break
        time.sleep(1)
assert body["status"] == "completed", body
n = sum(len(t["detections"]) for t in body["result"]["turns"])
assert n > 0, "expected at least one detection"
print("   synthetic ok:", n, "detections")
PY

echo "==> fetching SPA"
curl -sf "http://localhost:$PORT/" >/tmp/smoke-index.html
python3 - <<'PY'
html = open("/tmp/smoke-index.html").read().lower()
assert "<!doctype html" in html or "<html" in html, "SPA not served"
print("   SPA ok")
PY

echo "SMOKE OK"
