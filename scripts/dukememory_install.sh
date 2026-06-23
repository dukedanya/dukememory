#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${PREFIX:-$HOME/.local}"
BINDIR="${BINDIR:-$PREFIX/bin}"
PROFILE="${PROFILE:-release}"
VERIFY="${VERIFY:-doctor}"

mkdir -p "$BINDIR"

if [[ "$PROFILE" == "release" ]]; then
  cargo build --manifest-path "$ROOT/Cargo.toml" --release
  BIN="$ROOT/target/release/dukememory"
else
  cargo build --manifest-path "$ROOT/Cargo.toml"
  BIN="$ROOT/target/debug/dukememory"
fi

install -m 0755 "$BIN" "$BINDIR/dukememory"

echo "installed: $BINDIR/dukememory"
echo "version: $("$BINDIR/dukememory" --version)"

case "$VERIFY" in
  0|false|no|none)
    echo "verify: skipped"
    ;;
  doctor)
    "$BINDIR/dukememory" doctor --json >/dev/null
    echo "verify: doctor --json passed"
    ;;
  deep)
    "$BINDIR/dukememory" doctor --deep
    echo "verify: doctor --deep passed"
    ;;
  *)
    echo "unknown VERIFY=$VERIFY; use doctor, deep, or 0" >&2
    exit 2
    ;;
esac

echo
echo "next:"
echo "  export PATH=\"$BINDIR:\$PATH\""
echo "  scripts/dukememory_postgres.sh migrate"
echo "  dukememory doctor --deep"
echo "  dukememory codex-config --install --force"
