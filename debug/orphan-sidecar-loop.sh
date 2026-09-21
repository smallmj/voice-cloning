#!/bin/bash
# Feedback loop for the orphan-sidecar bug (issue #29, fix 3e3cb33).
# RED  = kill uv only (old behavior) -> python grandchild survives = orphan
# GREEN= kill whole process group (new behavior) -> python grandchild also dies
set -u
cd "$(dirname "$0")/.."
PROJ="$PWD/sidecar"
AUDIO="$(mktemp -d)"
LOG="$(mktemp)"

start_sidecar() { # $1 = extra setsid? prints uv pid
  python3 -c 'import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
    uv run --project "$PROJ" python debug/sidecar_with_worker.py --port 0 --audio-dir "$AUDIO" >"$LOG" 2>&1 &
  echo $!
}

wait_ready() { # $1 = uv pid
  for i in $(seq 1 100); do
    grep -q '"event": *"ready"' "$LOG" 2>/dev/null && return 0
    grep -q "ready" "$LOG" 2>/dev/null && return 0
    sleep 0.2
  done
  echo "sidecar failed to become ready; log tail:"; tail -5 "$LOG"; return 1
}

group_survivors() { # $1 = pgid -> print all alive processes still in that group
  ps -eo pid,pgid,comm 2>/dev/null | awk -v g="$1" '$2==g && $1!=1 {print $1" "$3}'
}

py_grandchildren() { # $1 = uv pid -> print python child pids still alive
  pgrep -P "$1" 2>/dev/null | while read -r p; do
    [[ "$(ps -o comm= -p "$p" 2>/dev/null)" == *python* ]] && echo "$p"
  done
}

cleanup_all() { # kill any surviving tree for a uv pid
  local pid="$1"
  kill -- -"$pid" 2>/dev/null
  for p in $(py_grandchildren "$pid"); do kill -9 "$p" 2>/dev/null; done
  kill -9 "$pid" 2>/dev/null
}

FAIL=0

echo "=== Scenario A (old behavior): kill uv only ==="
UVPID=$(start_sidecar); wait_ready "$UVPID" || exit 1
PYPIDS=$(py_grandchildren "$UVPID")
echo "uv=$UVPID python_children=[$PYPIDS]"
[[ -n "$PYPIDS" ]] || { echo "LOOP BROKEN: no python grandchild found"; exit 2; }
kill "$UVPID"; sleep 2
SURV=$(group_survivors "$UVPID")
if [[ -n "$SURV" ]]; then
  echo "RED confirmed: process-group members survived kill of uv:"; echo "$SURV"
  cleanup_all "$UVPID"
else
  echo "uv 0.12.10 forwards SIGTERM to its direct python child; no group survivor"
  FAIL=2
fi

echo "=== Scenario B (new behavior): kill process group ==="
UVPID=$(start_sidecar); wait_ready "$UVPID" || exit 1
PYPIDS=$(py_grandchildren "$UVPID")
echo "uv=$UVPID python_children=[$PYPIDS]"
kill -- -"$UVPID"; sleep 2
SURV=$(group_survivors "$UVPID")
if [[ -z "$SURV" ]]; then
  echo "GREEN confirmed: whole tree terminated by group kill"
  kill -9 "$UVPID" 2>/dev/null
else
  echo "FAIL: python survived group kill: [$SURV]"; cleanup_all "$UVPID"; FAIL=1
fi

rm -rf "$AUDIO" "$LOG"
exit $FAIL
