#!/usr/bin/env bash
# Restart the Perplexity -> OpenAI proxy (uvicorn server:app, port 64130).
# Kills any running instance, starts a fresh one in the background,
# and prints the exact tail -f command to follow the log.
set -euo pipefail

cd "$(dirname "$0")"          # run from repo root regardless of cwd
PORT=64130
LOG="$(pwd)/server.log"

# --- 1. Kill existing instance (match this app only) -----------------------
PIDS="$(pgrep -f 'uvicorn.*server:app' || true)"
if [ -n "$PIDS" ]; then
  echo "Stopping existing server: $(echo "$PIDS" | tr '\n' ' ')"
  kill $PIDS 2>/dev/null || true
  for p in $PIDS; do
    for _ in $(seq 1 20); do
      kill -0 "$p" 2>/dev/null || break
      sleep 0.25
    done
    if kill -0 "$p" 2>/dev/null; then
      echo "pid $p did not exit cleanly, sending SIGKILL"
      kill -9 "$p" 2>/dev/null || true
    fi
  done
  sleep 1   # let the port release
fi

# --- 2. Start in background, log to server.log ------------------------------
if [ -x ".venv/bin/uvicorn" ]; then
  UVICORN=".venv/bin/uvicorn"
elif command -v uvicorn >/dev/null 2>&1; then
  UVICORN="uvicorn"
else
  echo "ERROR: uvicorn not found. Create a venv per README or install uvicorn." >&2
  exit 1
fi

echo "Starting: $UVICORN server:app --host 0.0.0.0 --port $PORT"
if command -v setsid >/dev/null 2>&1; then
  nohup setsid "$UVICORN" server:app --host 0.0.0.0 --port "$PORT" </dev/null >> "$LOG" 2>&1 &
else
  nohup "$UVICORN" server:app --host 0.0.0.0 --port "$PORT" </dev/null >> "$LOG" 2>&1 &
fi
echo "Started in background (pid $!), log: $LOG"

# --- 3. Wait for readiness --------------------------------------------------
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
    echo "Server is up: http://127.0.0.1:$PORT/healthz"
    UP=1
    break
  fi
  sleep 1
done
if [ -z "${UP:-}" ]; then
  echo "WARNING: not responding yet after 30s — check the log below."
  exit 1
fi

# --- 4. Copy-paste tail command ---------------------------------------------
echo
echo "tail -f $LOG"