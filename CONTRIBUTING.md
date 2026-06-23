# Contributing to dukememory

Thanks for considering a contribution. Dukememory is a local-first memory system
for coding agents, so changes should preserve project isolation, auditability,
and explicit memory lifecycle behavior.

## Development Setup

```bash
git clone https://github.com/dukedanya/dukememory.git
cd dukememory

brew install postgresql@17 pgvector
scripts/dukememory_postgres.sh start
scripts/dukememory_postgres.sh migrate
export DUKEMEMORY_DATABASE_URL="$(scripts/dukememory_postgres.sh url)"

cargo test
cargo fmt -- --check
```

## Local Checks

Run the focused checks first:

```bash
cargo test
cargo run -- doctor
cargo run -- mcp-smoke
```

For broader rollout confidence:

```bash
cargo run -- production-audit
cargo run -- codex-audit
cargo run -- codex-hook-audit
```

## Contribution Rules

- Agent-facing tools, commands, hooks, events, and documented actions must use
  the `dukememory_*` prefix.
- Memory access must default to the current project only. Cross-project lookup
  requires an explicit user request and an explicit project identifier.
- Automatic agent-written memories should normally start as `pending`.
- Prefer `superseded` or `archived` over destructive deletion.
- Do not add hosted telemetry, hosted memory storage, or cross-project defaults.
- Keep README short. Put detailed operational material in `docs/`.
- Update `docs/MCP_TOOLS.md` when adding, renaming, or changing MCP tools.

## Pull Requests

Please include:

- a short summary of the behavior change;
- verification commands and results;
- migration notes when schema changes are involved;
- safety or project-isolation implications.

The preferred PR check list is:

```bash
cargo fmt -- --check
cargo test
```

Use narrower tests for early iterations, but run the full suite before asking
for review.
