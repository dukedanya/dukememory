# Getting Started

This guide takes a fresh checkout from zero to a working local dukememory setup.

## 1. Install Requirements

```bash
brew install postgresql@17 pgvector
```

You also need:

- Rust with edition 2024 support;
- Ollama running locally or forwarded to `OLLAMA_BASE_URL`;
- optional `rust-analyzer` for deeper Rust code analysis.

## 2. Start The Local Database

```bash
scripts/dukememory_postgres.sh start
scripts/dukememory_postgres.sh migrate
export DUKEMEMORY_DATABASE_URL="$(scripts/dukememory_postgres.sh url)"
```

The local cluster lives under `~/.dukememory/postgres`, listens on a Unix
socket, and uses a low-resource laptop profile.

## 3. Run Health Checks

```bash
cargo run -- doctor
cargo run -- status --json
```

Use the deeper readiness gate when you want a rollout check:

```bash
cargo run -- doctor --deep
```

## 4. Store And Retrieve Memory

```bash
cargo run -- remember --kind decision "Use project_id for every memory lookup."
cargo run -- search "project memory isolation"
cargo run -- context "what should I know before editing retrieval"
```

Automatic agent-written memories should normally start as `pending`:

```bash
cargo run -- remember --status pending --kind project_rule "Review automatic memories before promotion."
cargo run -- list --status pending
cargo run -- promote <memory-id>
```

## 5. Run The MCP Server

```bash
cargo run -- mcp
```

For Codex integration snippets:

```bash
cargo run -- codex-config
cargo run -- codex-hooks
```

Install only after reviewing the generated output:

```bash
cargo run -- codex-config --install
cargo run -- codex-hooks --install
```

## 6. Validate Agent Flows

```bash
cargo run -- mcp-smoke
cargo run -- codex-audit
cargo run -- codex-hook-audit
```

For an isolated project rollout test:

```bash
cargo run -- production-audit
```

## 7. Open The Viewer

```bash
cargo run -- dukememory_app
```

The viewer opens a native local window for project-scoped memory and code graph
navigation.

## Troubleshooting

If semantic search fails, confirm that Ollama is reachable and the configured
embedding model is available. Keyword search, list, review, import/export, and
many operational commands continue to work without Ollama.

If project results look mixed, pass `project_path` explicitly through MCP tools
or `--project` through CLI commands.
