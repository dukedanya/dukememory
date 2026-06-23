#!/usr/bin/env bash
set -euo pipefail

PG_HOME="${PG_HOME:-/opt/homebrew/opt/postgresql@17}"
if [[ ! -x "$PG_HOME/bin/postgres" ]]; then
  PG_HOME="/opt/homebrew/Cellar/postgresql@17/17.8"
fi

PG_BIN="${PG_BIN:-$PG_HOME/bin}"
PGDATA="${DUKEMEMORY_PGDATA:-$HOME/.dukememory/postgres}"
PGSOCKET="${DUKEMEMORY_PGSOCKET:-$HOME/.dukememory/postgres-socket}"
PGPORT="${DUKEMEMORY_PGPORT:-55432}"
PGDATABASE="${DUKEMEMORY_PGDATABASE:-dukememory}"
PGUSER_NAME="${DUKEMEMORY_PGUSER:-$USER}"
LOG_FILE="${DUKEMEMORY_PGLOG:-$HOME/.dukememory/postgres.log}"

export PATH="$PG_BIN:$PATH"

usage() {
  cat <<EOF
Usage: $0 <init|start|stop|restart|status|createdb|migrate|psql|url>

Environment:
  DUKEMEMORY_PGDATA=$PGDATA
  DUKEMEMORY_PGSOCKET=$PGSOCKET
  DUKEMEMORY_PGPORT=$PGPORT
  DUKEMEMORY_PGDATABASE=$PGDATABASE
  DUKEMEMORY_PGUSER=$PGUSER_NAME
EOF
}

require_bins() {
  for bin in initdb pg_ctl postgres psql createdb; do
    if ! command -v "$bin" >/dev/null 2>&1; then
      echo "missing $bin; expected PostgreSQL binaries under $PG_BIN" >&2
      exit 1
    fi
  done
}

require_pgvector() {
  local control
  control="$(pg_config --sharedir 2>/dev/null)/extension/vector.control"
  if [[ ! -f "$control" ]]; then
    echo "pgvector is not installed for this PostgreSQL. Run: brew install pgvector" >&2
    exit 1
  fi
}

init_cluster() {
  require_bins
  require_pgvector
  mkdir -p "$(dirname "$PGDATA")" "$PGSOCKET"
  if [[ ! -d "$PGDATA/base" ]]; then
    initdb -D "$PGDATA" --encoding=UTF8 --locale=C
  fi
  cat > "$PGDATA/postgresql.conf" <<EOF
listen_addresses = ''
unix_socket_directories = '$PGSOCKET'
port = $PGPORT
max_connections = 32
shared_buffers = 64MB
work_mem = 4MB
maintenance_work_mem = 64MB
effective_cache_size = 512MB
autovacuum = on
autovacuum_max_workers = 1
max_worker_processes = 4
max_parallel_workers = 2
max_parallel_workers_per_gather = 1
synchronous_commit = off
checkpoint_timeout = 15min
log_min_messages = warning
EOF
  cat > "$PGDATA/pg_hba.conf" <<EOF
local all all trust
EOF
}

start_cluster() {
  init_cluster
  pg_ctl -D "$PGDATA" -l "$LOG_FILE" -o "-k '$PGSOCKET' -p $PGPORT" start
}

stop_cluster() {
  require_bins
  pg_ctl -D "$PGDATA" stop -m fast
}

status_cluster() {
  require_bins
  pg_ctl -D "$PGDATA" status
}

createdb_if_missing() {
  require_bins
  if ! psql -h "$PGSOCKET" -p "$PGPORT" -U "$PGUSER_NAME" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname = '$PGDATABASE'" | grep -q 1; then
    createdb -h "$PGSOCKET" -p "$PGPORT" -U "$PGUSER_NAME" "$PGDATABASE"
  fi
}

migrate_db() {
  createdb_if_missing
  for migration in migrations/*.sql; do
    psql -h "$PGSOCKET" -p "$PGPORT" -U "$PGUSER_NAME" -d "$PGDATABASE" -v ON_ERROR_STOP=1 -f "$migration"
  done
}

database_url() {
  printf 'postgresql://%s@localhost:%s/%s?host=%s\n' "$PGUSER_NAME" "$PGPORT" "$PGDATABASE" "$PGSOCKET"
}

cmd="${1:-}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$cmd" in
  init) init_cluster ;;
  start) start_cluster ;;
  stop) stop_cluster ;;
  restart) stop_cluster || true; start_cluster ;;
  status) status_cluster ;;
  createdb) createdb_if_missing ;;
  migrate) migrate_db ;;
  psql) createdb_if_missing; psql -h "$PGSOCKET" -p "$PGPORT" -U "$PGUSER_NAME" -d "$PGDATABASE" "$@" ;;
  url) database_url ;;
  *) usage; exit 2 ;;
esac
