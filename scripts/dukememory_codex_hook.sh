#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${DUKEMEMORY_BIN:-"$ROOT/target/debug/dukememory"}"
EVENT="${DUKEMEMORY_EVENT:-codex_hook}"
MAX_CANDIDATES="${DUKEMEMORY_MAX_CANDIDATES:-8}"
PAYLOAD="$(cat || true)"

if [[ -z "${PAYLOAD//[[:space:]]/}" ]]; then
  exit 0
fi

PROJECT_ARGS=()
if [[ -n "${DUKEMEMORY_PROJECT:-}" ]]; then
  PROJECT_ARGS=(--project "$DUKEMEMORY_PROJECT")
fi

printf '%s' "$PAYLOAD" | "$BIN" dukememory-extract \
  "${PROJECT_ARGS[@]}" \
  --source "dukememory_hook:${EVENT}" \
  --max-candidates "$MAX_CANDIDATES"
