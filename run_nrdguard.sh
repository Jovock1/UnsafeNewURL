#!/usr/bin/env bash
# Runs nrdguard.py with the correct interpreter (.venv, which has the
# `ollama` package that the system python3 lacks) and a lock so a slow
# run (LLM classification over ~600 batches) can't overlap with the next
# scheduled one.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$REPO_DIR/logs/cron"
LOCK_FILE="$REPO_DIR/.nrdguard.lock"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%F_%H-%M-%S).log"

cd "$REPO_DIR"

exec {lock_fd}>"$LOCK_FILE"
if ! flock -n "$lock_fd"; then
    echo "$(date -Is) Another nrdguard run is already in progress; skipping." >>"$LOG_FILE"
    exit 0
fi

"$REPO_DIR/.venv/bin/python3" "$REPO_DIR/nrdguard.py" "$@" >>"$LOG_FILE" 2>&1
