# dukememory

![dukememory banner](docs/assets/dukememory-banner.svg)

[![CI](https://github.com/dukedanya/dukememory/actions/workflows/ci.yml/badge.svg)](https://github.com/dukedanya/dukememory/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/Rust-2024-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![MCP](https://img.shields.io/badge/MCP-dukememory__*-2563EB)](#mcp-contract)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL%20%2B-pgvector-336791?logo=postgresql&logoColor=white)](#local-postgresql)
[![Ollama](https://img.shields.io/badge/Ollama-local%20models-10B981)](#embeddings)
[![Status](https://img.shields.io/badge/status-early%20local--first-F59E0B)](#status)

**Project-scoped memory, retrieval, and code-context infrastructure for Codex
and local AI agents.**

Dukememory is a local-first memory engine for agents that need durable project
knowledge without cross-repository leakage, blind automatic writes, or a hosted
memory service. It stores memories in PostgreSQL, retrieves with keyword +
semantic signals, indexes source code, and exposes everything through CLI and
MCP tools prefixed with `dukememory_*`.

## Why Dukememory

Agents become much more useful when they can remember decisions, architecture
constraints, task outcomes, and code facts. They also become risky when memory is
global, unreviewed, or impossible to audit. Dukememory is built around a stricter
contract:

- every lookup is scoped to the current project;
- automatic writes start as reviewable `pending` candidates;
- active memory is append-friendly and history-preserving;
- obvious secrets are blocked before storage;
- retrieval context is selected for the current task, not dumped wholesale;
- code and memory are connected through indexed symbols, graph facts, and audit
  events.

## Highlights

| Capability | What it gives an agent |
| --- | --- |
| Project-isolated memory | Decisions, rules, setup notes, and summaries never leak across repos by default |
| Hybrid retrieval | PostgreSQL full-text search plus local Ollama embeddings with Reciprocal Rank Fusion |
| Reviewed lifecycle | `pending`, `active`, `superseded`, and `archived` states instead of silent overwrites |
| Task context packs | Prompt-ready bundles with selected memories, fragments, graph facts, and code hits |
| Code graph index | Symbol search, outlines, approximate callers/callees, impact analysis, and code memories |
| Memory graph | Entities, facts, edges, provenance episodes, and temporal invalidation |
| Codex integration | MCP config generation, Stop/PreCompact extraction hooks, smoke tests, and audits |
| Native viewer | Local memory/code graph browser with project vaults, clusters, files, and bridge views |

## Contents

- [Quick Start](#quick-start)
- [Architecture At A Glance](#architecture-at-a-glance)
- [Agent Workflow](#agent-workflow)
- [Requirements](#requirements)
- [Documentation Map](#documentation-map)
- [Status](#status)
- [Local PostgreSQL](#local-postgresql)
- [Store And Search](#store-and-search)
- [MCP Contract](#mcp-contract)
- [Codex Integration](#codex-integration)
- [Backup And Portability](#backup-and-portability)
- [Embeddings](#embeddings)
- [Graph And Health](#graph-and-health)

## Quick Start

```bash
brew install postgresql@17 pgvector
scripts/dukememory_postgres.sh start
scripts/dukememory_postgres.sh migrate
export DUKEMEMORY_DATABASE_URL="$(scripts/dukememory_postgres.sh url)"

cargo run -- doctor
cargo run -- remember --kind decision "Use project_id as a mandatory filter for every memory lookup."
cargo run -- search "project filter memory"
cargo run -- context "what should I know before editing retrieval"
```

Run the MCP server:

```bash
cargo run -- mcp
```

Generate Codex configuration:

```bash
cargo run -- codex-config
cargo run -- codex-hooks
```

Run the native viewer:

```bash
cargo run -- dukememory_app
```

## Architecture At A Glance

```text
Codex / local agent
        |
        | MCP tools and CLI commands
        v
dukememory
  |-- project resolver and safety policy
  |-- memory lifecycle and review queue
  |-- context planner and retrieval packer
  |-- code index and code-memory links
  |-- memory graph and temporal facts
        |
        v
PostgreSQL + pgvector       Ollama
durable rows, FTS, vectors  embeddings, extraction, reranking
```

| Area | Purpose |
| --- | --- |
| Memory store | Project profiles, memory rows, lifecycle, deduplication, audit events, backups, export/import |
| Retrieval | PostgreSQL full-text search, Ollama embeddings, hybrid Reciprocal Rank Fusion, optional reranking |
| Context packing | Prompt-ready task context with selected memories, fragments, graph summaries, and code hits |
| Memory graph | Entities, facts, edges, episodes, temporal invalidation, and graph extraction |
| Code index | Multi-language symbol index, code search, outlines, approximate caller/callee/impact navigation |
| MCP server | Agent-facing `dukememory_*` tools, resources, and prompts over stdio or localhost HTTP |
| Codex integration | Config generation, Stop/PreCompact extraction hooks, audit commands, and hook smoke tests |
| GUI | Native project vault browser with memory, code, relationship, overview, cluster, and file views |

## Agent Workflow

1. Call `dukememory_prepare` or `dukememory_agent_before` with the active
   `project_path`.
2. Use the returned task-scoped memory fragments, graph hints, code hits, and
   code-neighborhood metadata while editing.
3. Store automatic observations as `pending` candidates through
   `dukememory_extract` or `dukememory_agent_after`.
4. Review candidates with `dukememory_review` and promote only trusted memories.
5. Use `dukememory_trace`, `dukememory_feedback`, and eval suites to turn bad
   retrievals into auditable regression cases.

## Requirements

- macOS or another Unix-like environment.
- Rust toolchain with edition 2024 support.
- PostgreSQL 17 and `pgvector`.
- Ollama for semantic embeddings, extraction, validation, and reranking.
- Optional `rust-analyzer` for deeper Rust diagnostics and LSIF-backed code
  indexing.

Default local model assumptions:

- Ollama base URL: `http://127.0.0.1:11435`
- Memory embedding model: `qwen3-embedding:8b`
- Fast code embedding model: `bge-m3`
- Extraction/validation model: `qwen3:14b`

Keyword retrieval, listing, review, export/import, and many operational commands
continue to work without Ollama. Explicit semantic search requires embeddings.

## Documentation Map

- [Architecture](docs/MEMORY_ARCHITECTURE.md) explains lifecycle, retrieval,
  graph storage, MCP surfaces, safety policy, schema evolution, and operational
  flows.
- [Migrations](migrations/) contain the ordered PostgreSQL schema.
- [Eval suite](eval/dukememory.json) contains retrieval and behavior regression
  cases.
- [LaunchAgent example](launchd/com.dukememory.tailscale-ollama-forwarder.plist)
  and [scripts](scripts/) cover local services and integration helpers.

## Status

This is an early local-first system under active development. The repository
contains the production PostgreSQL path, MCP/CLI surfaces, GUI, audits, and
evaluation harness, but APIs and schema details may still change before a stable
release.

## Local PostgreSQL

Dukememory's target production store is PostgreSQL plus `pgvector`, configured for a single local laptop user rather than a heavy network service. Do not use Docker on macOS for the local store; it adds VM overhead.

Bootstrap the local cluster:

```bash
brew install postgresql@17 pgvector
scripts/dukememory_postgres.sh start
scripts/dukememory_postgres.sh migrate
scripts/dukememory_postgres.sh url
```

The generated cluster lives under `~/.dukememory/postgres`, listens only on a Unix socket, uses port `55432`, and applies a low-resource profile: `max_connections=32`, `shared_buffers=64MB`, `work_mem=4MB`, `maintenance_work_mem=64MB`, and `autovacuum_max_workers=1`.

Use this environment variable for Codex and shell sessions:

```bash
export DUKEMEMORY_DATABASE_URL="$(scripts/dukememory_postgres.sh url)"
```

`DUKEMEMORY_DATABASE_URL` is the authoritative storage setting. `DUKEMEMORY_DATABASE_MARKER` is a local marker path used only for backup filenames and isolated temporary schema names. `DUKEMEMORY_DB` is still accepted as a legacy alias.

## Quick Check

```bash
cargo run -- doctor
cargo run -- doctor --json
cargo run -- doctor --deep
cargo run -- status --json
cargo run -- mcp-smoke
cargo run -- production-audit
cargo run -- codex-audit
cargo run -- codex-hook-audit
cargo run -- embed "inventory system memory test"
```

`doctor` checks the database, current project id, Ollama model roles, and local `rust-analyzer` availability. `doctor --json` and `status --json` print machine-readable reports for scripts. `doctor --deep` is the production readiness gate: it also verifies the MCP JSON-RPC smoke flow, isolated production audit, installed Codex config/hooks, embedding coverage, code index freshness, resolved code relations, built-in eval readiness, `rust-analyzer diagnostics`, and the loaded project-scoped LaunchAgent. `mcp-smoke` runs the isolated protocol check directly: initialize, tools/list, remember, search, context, status, and a negative cross-project search. `production-audit` creates a temporary Rust project and isolated PostgreSQL schema, then exercises indexing, project-scoped memory search, call graph lookup, backup, export/import, and maintenance dry-run. `codex-audit` reads `~/.codex/config.toml` and `~/.codex/hooks.json`, verifies the `dukememory` MCP server/env, verifies Stop/PreCompact hook wiring, and dry-runs the hook wrapper with empty stdin. `codex-hook-audit` sends a non-empty Codex-like JSON payload through the wrapper into an isolated PostgreSQL schema and verifies that pending memory was created.

## Store And Search

```bash
cargo run -- remember --kind decision "Use project_id as a mandatory filter for every memory lookup."
cargo run -- remember --deduplicate --kind decision "Use project_id as a mandatory filter for every memory lookup."
cargo run -- search "project filter memory"
cargo run -- search --mode semantic "where do we store stable project decisions"
cargo run -- context "what should I know before editing memory retrieval"
cargo run -- context --core-memory-limit 4 --memory-limit 10 --token-budget 3000 "what should I know before editing memory retrieval"
```

`context` returns a task-scoped prompt-ready bundle: task-selected core/project-rule fragments, task-selected memory fragments, graph facts/edges only around those selected memories, relevant indexed code symbols, and project/index counts. It does not load full project memory into the prompt. `core_memory_limit` caps query-matched core/project-rule fragments and does not consume `memory_limit`, which is reserved for task retrieval. Use `--token-budget` in CLI or `token_budget` over MCP to shrink or widen the packed context. MCP structured output is compact by default: `memories`, `memory_fragments`, `code`, `code_neighborhood`, and `graph_summary` are body-free metadata summaries. Pass `debug=true` only when you need full trace, full graph, verbose code-neighborhood relations, or fragment text in structuredContent.

Review workflow:

```bash
cargo run -- remember --status pending --kind project_rule "Use pending for automatic agent-written memories."
cargo run -- list --status pending
cargo run -- promote <memory-id>
cargo run -- review-apply --decision promote:<memory-id>:"stable project rule"
cargo run -- audit-log --limit 20
```

Native read-only memory viewer:

```bash
cargo run -- dukememory_app
cargo run -- dukememory_app --status any --path /path/to/project
cargo run -- dukememory_app --project my-game --kind decision
```

The viewer opens a local native window, not a browser. The left sidebar lists
projects as separate vaults; switching a vault clears the current map selection,
pan, and graph caches, then rebuilds the graph from that project only. The main
view uses an Obsidian-like dark map style and shows a clickable relationship
graph from project-scoped memories, `memory_entities`, `memory_facts`, and
`memory_edges`; when explicit graph data is not present yet, it falls back to
weak links from metadata such as tags, sources, and supersession. It uses
project-scoped `list`, keyword `search`, graph `search`, code graph, and
`status` reads, so it does not require Ollama to be available.

The map can switch between `Память`, `Код`, and `Связи`. `Обзор` shows a
project-level summary first, so large projects start from a stable spatial
layout of memory, entity, fact, code, file, and cluster areas instead of hundreds
of small nodes. Overview nodes act as navigation targets:
double-clicking `Память`, `Код`, `Файлы`, or an area node opens the matching
detailed graph view, and the right overview panel exposes the same jumps as
buttons. In memory/code domains, `Объекты`
shows concrete nodes and relations, `Типы` collapses them by `entity_type` or
symbol `kind`, and `Кластеры` groups large maps into semantic memory clusters or
code directories. In the code domain, `Файлы` collapses symbols into file-level
modules and shows cross-file code graph relations.
`Связи` is the memory-to-code bridge: it connects memories/entities to code
files or symbols when the memory text explicitly references them. Local
node/relation filters narrow the current map without changing stored data, and
code maps also support `kind` and file-path filters.

The map is interactive: dragging the canvas background pans the map, wheel or
trackpad scroll zooms around the cursor, dragging a node pins a manual local
position, the minimap jumps around the canvas, double-clicking the canvas or
pressing `Вписать` fits the graph into view, clicking nodes or relation labels
opens details, and `Назад`/`Вперед` navigates selection history. A compact
breadcrumb above the canvas shows the active domain/view and offers a quick
return to `Обзор`. In `Кластеры` mode, double-clicking a cluster opens that
cluster as a concrete subgraph;
`Кластеры` returns to the aggregate view. Code nodes/files can open their source
file, and `Экспорт JSON` writes the currently visible map to the temporary
directory. Visual settings, pinned nodes, and per-project pan/zoom state are
persisted in `gui-settings.json` next to the configured database marker. Large
maps use cached layout plus viewport rendering, color-coded directed edges,
parallel edge offsets, collision-aware card spacing, and circular ring layouts
for large maps where card overlap would otherwise make the graph unreadable.
Optional legend/minimap/quality overlays stay available; off-screen nodes and
relation labels are not painted or given hitboxes until they enter the visible
scroll area.

The right panel can run graph extraction directly from the GUI. `Предпросмотр`
asks the configured local Ollama model for proposed entities/facts/edges without
writing them, `Извлечь и применить` writes immediately, and
`Применить предпросмотр` stores a reviewed dry-run result.

To populate the explicit map from existing memories, first review proposed graph
items, then apply them:

```bash
cargo run -- dukememory_graph_extract --status active
cargo run -- dukememory_graph_extract --status active --apply
```

History-preserving replacement:

```bash
cargo run -- supersede --old-id <memory-id> --kind decision "New corrected memory text."
cargo run -- list --status superseded
```

Override the project when needed:

```bash
cargo run -- remember --project my-game --kind game_design "The inventory should support stackable resources."
cargo run -- search --project my-game "stackable resources"
```

New writes are checked by the safety policy. Obvious secret values such as API key assignments, private key markers, bearer tokens, and known token prefixes are blocked before PostgreSQL writes. Use `--allow-sensitive` only for explicit manual recovery cases.

Universal project profile:

```bash
cargo run -- profile
cargo run -- profile --project-type webapp --domain rust --domain api
cargo run -- ontology
```

`project_type` defaults to `generic`; domains are optional tags such as `rust`, `webapp`, `game`, `research`, or `ops`. Memory is still project-isolated first. `scope` is stored per memory and defaults to `project`; supported scopes are `project`, `session`, `module`, `user`, and `global`. `memory_tier` defaults to `archival`; use `core` for stable rules that should be prepended to context, and `conversation` for retained but normally compactable conversational material.

Retrieval eval:

```bash
cargo run -- eval --file eval/dukememory.json --mode keyword
cargo run -- eval --file eval/dukememory.json --mode rerank
cargo run -- eval --file eval/dukememory.json --mode keyword --compare-last --json
```

Eval suites are JSON files with `cases`, where each case has a `query` plus optional `target`, `expected_contains`, `forbidden_contains`, `expected_ids`, `min_results`, and `project_id`. Supported targets are `memory`, `code`, `context`, `semantic`, and `review_apply`; `memory` is the default. Semantic eval cases set `semantic_action` and may set `as_of` for temporal assertions. Reports include the target, top ids, missing expected ids, `recall_at_k`, `precision_at_k`, and `latency_ms`.
Each eval execution is also stored in `eval_runs` and prints a `run_id` so retrieval regressions can be audited over time. Suites may include a top-level `name`; Dukememory also stores a file hash so `--compare-last` can compare the current run to the previous run of the same suite/mode/hash.

## MCP Contract

All memory-facing MCP tools use the `dukememory_*` prefix.

Current tools:

- `dukememory_extract`
- `dukememory_models`
- `dukememory_project_profile`
- `dukememory_task_session`
- `dukememory_task_eval`
- `dukememory_test_plan`
- `dukememory_ontology`
- `dukememory_eval`
- `dukememory_prepare`
- `dukememory_agent_before`
- `dukememory_agent_task`
- `dukememory_devsystem`
- `dukememory_agent_after`
- `dukememory_context`
- `dukememory_context_plan`
- `dukememory_graph`
- `dukememory_episode`
- `dukememory_graph_extract`
- `dukememory_search`
- `dukememory_remember`
- `dukememory_remember_smart`
- `dukememory_get`
- `dukememory_list`
- `dukememory_review`
- `dukememory_review_apply`
- `dukememory_audit_log`
- `dukememory_validate_pending`
- `dukememory_promote`
- `dukememory_supersede`
- `dukememory_archive`
- `dukememory_prune_pending`
- `dukememory_compact`
- `dukememory_maintenance`
- `dukememory_ops_pipeline`
- `dukememory_status`
- `dukememory_health`
- `dukememory_cleanup_schemas`
- `dukememory_backup`
- `dukememory_export`
- `dukememory_import`
- `dukememory_code_index`
- `dukememory_code_lsif_index`
- `dukememory_code_status`
- `dukememory_code_files`
- `dukememory_code_outline`
- `dukememory_code_search`
- `dukememory_code_explore`
- `dukememory_code_memory`
- `dukememory_code_affected`
- `dukememory_code_patterns`
- `dukememory_code_duplicates`
- `dukememory_code_assist`
- `dukememory_code_review_plan`
- `dukememory_code_eval`
- `dukememory_code_brief`
- `dukememory_code_plan`
- `dukememory_code_risk`
- `dukememory_read_symbol`
- `dukememory_find_callers`
- `dukememory_find_callees`
- `dukememory_impact`
- `dukememory_semantic`
- `dukememory_dedupe`
- `dukememory_related`
- `dukememory_semantic_review`
- `dukememory_semantic_route`
- `dukememory_semantic_clusters`
- `dukememory_semantic_tags`
- `dukememory_stale_check`
- `dukememory_consistency_check`
- `dukememory_eval_generate`
- `dukememory_hard_negatives`
- `dukememory_embedding_health`
- `dukememory_model_migration`
- `dukememory_isolation_check`
- `dukememory_memory_hints`
- `dukememory_policy_decision`
- `dukememory_retrieval_quality`
- `dukememory_auto_eval`
- `dukememory_ab_compare`
- `dukememory_lifecycle_review`
- `dukememory_code_memory_suggest`
- `dukememory_verify_conflicts`
- `dukememory_topic_map`
- `dukememory_budget_optimize`
- `dukememory_feedback`
- `dukememory_self_heal`
- `dukememory_outcome_learn`
- `dukememory_conflict_graph`
- `dukememory_memory_compiler`
- `dukememory_policy_ab`
- `dukememory_context_policy`
- `dukememory_trace`
- `dukememory_task_replay`
- `dukememory_counterfactual_eval`
- `dukememory_code_causality`
- `dukememory_memory_impact`
- `dukememory_temporal_context`
- `dukememory_embed_missing`

MCP also exposes read-only resources:

- `dukememory://ontology`
- `dukememory://health`
- `dukememory://project/{project_id}/profile`

MCP `resources/templates/list` also exposes the project profile URI template for clients that support resource templates.

MCP prompt templates are available through `prompts/list` and `prompts/get`:

- `dukememory_agent_before`
- `dukememory_agent_after`
- `dukememory_memory_review`
- `dukememory_code_risk`

Run the stdio MCP server:

```bash
cargo run -- mcp
```

Run the experimental localhost HTTP MCP endpoint:

```bash
cargo run -- mcp-http --host 127.0.0.1 --port 8765
cargo run -- mcp-http-smoke
```

The HTTP endpoint accepts JSON-RPC `POST /mcp` requests and exposes an SSE-compatible `GET /mcp` endpoint announcement. It binds only to localhost, rejects non-local `Origin` headers, rejects unsupported methods, and enforces request-size/header-size limits. `mcp-http-smoke` runs an isolated localhost check covering initialize, SSE endpoint discovery, non-local origin rejection, wrong-method rejection, and oversized-body rejection. MCP initialization advertises tools, resources, prompts, and logging. Tool calls accept optional `roots` alongside `project_path`; when supplied, the project path must be inside one of those roots.

Run a non-mutating MCP smoke test against an isolated temporary PostgreSQL schema:

```bash
cargo run -- mcp-smoke
```

Run an isolated project rollout audit against a generated Rust project and temporary database:

```bash
cargo run -- production-audit
```

Verify installed Codex integration without modifying user config:

```bash
cargo run -- codex-audit
```

Verify the non-empty hook extraction path against a temporary database:

```bash
cargo run -- codex-hook-audit
```

For agent startup, prefer `dukememory_prepare`: it incrementally indexes the selected project, optionally embeds symbols from files changed in that run, and returns the same prompt-ready context shape as `dukememory_context`.

For a full agent loop, call `dukememory_agent_before` before the task and `dukememory_agent_after` after the task. They are MCP aliases for the tested prepare/extract workflow and keep all writes project-scoped.

`dukememory_devsystem` runs the `dukedevsystem` MCP advisory orchestrator for a task and touched files. It creates a task session, auto-indexes the project by default when `project_path` is supplied, emits a readiness-percent stage timeline for Planner, Memory, Architect, Coder, Test, Critic, Refactor, and final Memory, computes File Entropy Score reports, includes code-review-plan telemetry, recommends affected tests and executable fallback commands, reports stale or missing code-index/git/coverage signals, emits structured boundary repair plans for high-entropy files, and returns advisory quality gates with a gate summary. Optional `run_evidence=true` executes exact allowed recommended commands without a shell and records `quality_evidence_reports` with status, exit code, duration, and stdout/stderr excerpts; failed evidence becomes a blocker, timeout/skipped evidence needs human decision, and the default remains advisory `not_run`. By default it writes typed pending memory candidates: task intent, decision memory, per-file entropy observations, and an intent-graph candidate memory; `memory_writes` groups ids by category and includes structured graph candidates sourced to pending memory ids. MCP parameters include `auto_index`, `full_rebuild`, `embed_symbols`, `embed_symbol_limit`, `run_evidence`, `evidence_timeout_seconds`, `max_evidence_commands`, and `allowed_evidence_commands`; the structured report includes `telemetry.index_run` plus `telemetry.index_guard`. It supports an optional MCP `policy` object plus `[devsystem]` settings in `.dukememory.toml` for entropy thresholds, ignored/generated/static file patterns, required test commands, coverage threshold, and responsibility keywords. The key rule is: file size is a signal, not a verdict; responsibility density is the verdict.

Backend agent-intelligence tools:

- `dukememory_context_policy` learns a recommended memory/code/graph/token policy from retrieval events, feedback, and a live quality probe.
- `dukememory_self_heal` runs the autonomous backend memory self-healing loop over lifecycle review, outcome learning, conflict graph analysis, memory compilation, and audit.
- `dukememory_outcome_learn` turns completed or failed task sessions into helpful/unhelpful memory quality signals.
- `dukememory_conflict_graph` finds contradictory memories and graph facts; with `apply=true`, it invalidates weaker temporal graph facts.
- `dukememory_memory_compiler` promotes stable high-confidence rules to `core`, archives low-signal duplicates, and creates pending split candidates for long memories.
- `dukememory_policy_ab` compares live retrieval policies and records the recommended memory/code/mode policy.
- `dukememory_trace` and `dukememory_task_replay` build a flight-recorder view from a task session id, retrieval event id, or query. `dukememory_agent_task` links retrieval events to its task session so replay can inspect the exact context used by that run.
- `dukememory_counterfactual_eval` runs leave-one-out retrieval counterfactuals and returns hard-negative eval signals.
- `dukememory_code_causality` and `dukememory_memory_impact` connect selected memories to code symbols, code relations, impacted files, and affected tests.
- `dukememory_temporal_context` reads memory and graph context for a query at an `as_of` timestamp.
- `dukememory_feedback` applies typed quality and contradiction-risk signals to retrieved memories, records an audit event, and writes a regression eval case so bad retrievals become testable.

Print a Codex config snippet:

```bash
cargo run -- codex-config
```

Install it into `~/.codex/config.toml` with a backup:

```bash
cargo build
cargo run -- codex-config --install
```

Codex config example generated by the command:

```toml
[mcp_servers.dukememory]
command = "/path/to/dukememory/target/debug/dukememory"
args = ["mcp"]
startup_timeout_sec = 120

[mcp_servers.dukememory.env]
DUKEMEMORY_DATABASE_URL = "postgresql://USER@localhost:55432/dukememory?host=/Users/you/.dukememory/postgres-socket"
DUKEMEMORY_DATABASE_MARKER = "/Users/you/.dukememory/schema.marker"
DUKEMEMORY_DB = "/Users/you/.dukememory/schema.marker"
OLLAMA_BASE_URL = "http://127.0.0.1:11435"
DUKEMEMORY_EMBED_MODEL = "qwen3-embedding:8b"
DUKEMEMORY_FAST_EMBED_MODEL = "bge-m3"
DUKEMEMORY_EXTRACT_MODEL = "qwen3:14b"
DUKEMEMORY_VALIDATE_MODEL = "qwen3:14b"
DUKEMEMORY_FAST_CODE_MODEL = "qwen2.5-coder:14b"
DUKEMEMORY_DEEP_CODE_MODEL = "qwen3-coder:30b-a3b-q4_K_M"
DUKEMEMORY_AGENT_CODE_MODEL = "north-mini-code-1.0:q4_k_m"
DUKEMEMORY_EXPERIMENT_MODEL = "huihui-gemma4-12b-coder:q4_k_m"
```

When possible, pass `project_path` to tools. If omitted, the server falls back to its process cwd. Do not set `cwd` to the `dukememory` repo in a global MCP config unless you only want to manage this repo's own memory.

## Backup And Portability

Use `backup` for a full consistent PostgreSQL custom-format backup of the local database:

```bash
cargo run -- backup
cargo run -- backup --output /path/to/dukememory.pgdump
```

Use `export` and `import` when moving one project between databases or machines. Export JSON is project-scoped and does not include embedding blobs; rebuild vectors after import with `embed-missing`.

```bash
cargo run -- export --project my-game --output /tmp/my-game.dukememory.json
cargo run -- import /tmp/my-game.dukememory.json
cargo run -- import /tmp/my-game.dukememory.json --overwrite
cargo run -- embed-missing --project my-game --scope all --limit 500
```

Agent-facing equivalents are `dukememory_backup`, `dukememory_export`, and `dukememory_import`.

## Embeddings

New memories are embedded by default through Ollama. If Ollama is unavailable, the memory write still succeeds and the missing vector can be rebuilt later.

Memory vectors use `DUKEMEMORY_EMBED_MODEL`. Code symbol vectors use `DUKEMEMORY_FAST_EMBED_MODEL`, so rebuild code embeddings after changing that model.
Memory embedding text includes kind, tier, scope, tags, source, importance, confidence, and body text; code embedding text includes language, kind, file, line range, signature, docstring, and source snippet. This keeps semantic search aligned with the metadata agents actually use in context.

```bash
cargo run -- embed-missing --scope memories --limit 50
cargo run -- embed-missing --scope code --limit 50
```

Search modes:

- `keyword`: PostgreSQL full-text search only.
- `semantic`: Ollama query embedding plus cosine similarity over stored vectors.
- `hybrid`: Reciprocal Rank Fusion over keyword and semantic results.
- `rerank`: hybrid candidates reranked by `DUKEMEMORY_VALIDATE_MODEL`; falls back to hybrid order when reranking fails.

`context`, `prepare`, `dukememory_context`, and `dukememory_prepare` return a
project-scoped bundle with active memory hits, compact memory graph summaries,
and indexed code symbols. MCP `debug=true` expands structured trace data.

## Graph And Health

Memory graph extraction builds project-scoped entities, facts, and edges from
existing memories. It defaults to dry-run so you can inspect what will appear on
the native map:

```bash
cargo run -- dukememory_graph_extract --status active
cargo run -- dukememory_graph_extract --status active --query "code graph" --apply
```

Manual graph commands remain available for corrections:

```bash
cargo run -- graph-entity --type system Dukememory
cargo run -- graph-fact --entity-name Dukememory storage PostgreSQL
cargo run -- graph-edge --from-name Dukememory --to-name pgvector uses
cargo run -- graph-search Dukememory
```

Over MCP, `dukememory_graph` also supports `invalidate_fact` and `invalidate_edge`. These actions set `valid_to` and `invalidated_by` instead of deleting graph history, so temporal `as_of` graph reads remain auditable.

Operational commands:

```bash
cargo run -- health
cargo run -- health --json
cargo run -- audit-log --json --limit 50
cargo run -- cleanup-schemas
cargo run -- cleanup-schemas --apply
```

## Automatic Extraction

Extract durable pending memories from a transcript, summary, or hook payload:

```bash
cat session-summary.txt | cargo run -- dukememory-extract --source dukememory_manual
```

For hook runners, use the wrapper:

```bash
cat session-summary.txt | scripts/dukememory_codex_hook.sh
```

Print Codex hooks JSON:

```bash
cargo run -- codex-hooks
```

Install extraction hooks into `~/.codex/hooks.json` with a backup:

```bash
cargo run -- codex-hooks --install
```

Default events are `Stop` and `PreCompact`. Existing non-dukememory hooks are preserved. If a dukememory hook already exists, rerun with `--force` to replace it.

Useful environment variables:

- `DUKEMEMORY_BIN`: path to the compiled binary, defaults to `target/debug/dukememory`.
- `DUKEMEMORY_EVENT`: source event suffix, defaults to `codex_hook`.
- `DUKEMEMORY_PROJECT`: explicit project id when the hook runner cwd is not the project.
- `DUKEMEMORY_MAX_CANDIDATES`: extraction cap, defaults to `8`.

Extraction always writes `pending` memories. Review them with:

```bash
cargo run -- list --status pending
cargo run -- promote <memory-id>
```

`dukememory_extract`, `dukememory_agent_after`, and the CLI extraction path create a `memory_episodes` provenance row for the raw extraction input before candidate memories are written.

Extraction and `dukememory_remember` deduplicate against existing `pending` and `active` memories in the same project using a normalized body fingerprint. `archived` and `superseded` memories do not block a new write.

If the hook payload is JSON, extraction reads common fields such as `cwd`, `project_path`, `workspaceRoot`, `event`, `hookEventName`, `summary`, `transcript`, `messages`, `inputMessages`, `output`, `history`, `turns`, and `prompt`. Message content can be plain strings or multipart objects such as `input_text`, `output_text`, and `text`.

Maintenance:

```bash
cargo run -- review --limit 20
cargo run -- validate-pending --limit 20
cargo run -- validate-pending --limit 20 --apply
cargo run -- prune-pending --dry-run --max-confidence 0.5
cargo run -- prune-pending --max-confidence 0.5 --reason "low-confidence stale candidate"
cargo run -- compact --limit 40 --min-memories 20
cargo run -- compact --limit 40 --min-memories 20 --apply
cargo run -- maintenance
cargo run -- maintenance --json
cargo run -- maintenance --all --backup --apply
cargo run -- ops-report
cargo run -- ops-report --json
cargo run -- maintenance-launchd
cargo run -- maintenance-launchd --install --force --load
```

`compact` uses the extraction/summarization model to propose one `project_summary` from older active memories. Without `--apply` it only prints the proposal. With `--apply`, it stores the summary as active and archives the source memories with a `compacted into <summary-id>` reason.

`maintenance` orchestrates routine upkeep. With no step flags it performs a safe dry-run of pending validation and compaction checks. `--all` enables backup, validation, compaction, and missing-embedding checks. `--apply` applies validation, compaction, and embedding rebuild steps; backup writes a PostgreSQL custom-format dump when `--backup` or `--all` is enabled. `--json` emits the full maintenance report for scheduled jobs and monitoring.

`ops-report` prints a compact operational snapshot for the current project: schema/integrity, project memory counts, code-index freshness, health metrics, latest eval run, and recent audit events. Use `--json` for machine-readable reporting.

`maintenance-launchd` prints or installs a macOS LaunchAgent. The generated agent runs `dukememory maintenance --all --backup --apply` every 6 hours by default, pins the current project id unless `--project` is passed, and writes logs under `~/Library/Logs/dukememory/`. Use `--dry-run` to schedule non-mutating maintenance instead, and `--load` if you want the command to call `launchctl bootstrap` after writing the plist.

## Code Index

Current code indexing covers Rust, Python, JavaScript, JSX, TypeScript, TSX, Go, Java, Kotlin, and Swift symbols. The tree-sitter call graph extracts Rust/Python/JavaScript/TypeScript plus conservative generic calls for Go/Java/Kotlin/Swift when syntax is unambiguous; the optional LSIF layer remains Rust-specific:

```bash
cargo run -- code-index --path /path/to/project
cargo run -- code-index --path /path/to/project --embed --embed-limit 500
cargo run -- code-index --path /path/to/project --full
cargo run -- prepare --path /path/to/project --mode keyword "inventory system"
cargo run -- code-status
cargo run -- code-explore "how does project memory context get built"
cargo run -- code-search "project memory"
cargo run -- code-search --mode semantic "where is project memory indexed"
cargo run -- code-memory --action remember --symbol-id <symbol-id> "This symbol enforces the context packing invariant."
cargo run -- code-memory --action remember --symbol code_memory_summaries --file-path src/context_pack.rs --symbol-kind function "This symbol omits code-memory bodies from structured context."
cargo run -- code-memory --action search "context packing invariant"
cargo run -- code-memory --action search --symbol code_memory_summaries --file-path src/context_pack.rs --symbol-kind function
cargo run -- code-memory --action repair
cargo run -- code-memory --action repair --apply
cargo run -- code-affected src/context_pack.rs
cargo run -- code-lsif-index
cargo run -- code-brief <symbol-id-or-name>
cargo run -- code-plan "change memory validation policy"
cargo run -- code-risk "change code index relation extraction"
cargo run -- read-symbol <symbol-id-or-name>
cargo run -- find-callers <symbol-id-or-name>
cargo run -- find-callees <symbol-id-or-name>
cargo run -- impact <symbol-id-or-name>
```

Agent code navigation should prefer this project-scoped code graph before broad
filesystem search. For non-trivial code work, start with `dukememory_prepare`
or `dukememory_code_explore` / `code-explore`. `dukememory_code_explore` is the
CodeGraph-style one-call path: it returns relevant symbols, linked code
memories, route hints, caller/callee impact, and freshness warnings. Use
`dukememory_code_search`, `dukememory_read_symbol`, and caller/callee tools when
you need narrower follow-up navigation. Fall back to `rg` and direct file reads
for exact text checks, non-code files, stale files named by freshness warnings,
or gaps in the structural index.

`dukememory_code_memory` / `code-memory` stores durable notes about code linked
to a `symbol_id` or `file_path`: invariants, risks, usage rules, and test notes.
MCP writes default to `pending`; promote trusted notes with
`action=promote`. Task context loads only active code memories attached to
task-selected symbols/files, so code memory does not pull the whole project into
every prompt. Search/list results include `link_status` (`symbol`, `file`, or
`stale`) so agents can see when a remembered note points at code that moved or
disappeared. Symbol-linked notes also keep a snapshot of the symbol name, kind,
signature, and line range. Run `action=repair` as a dry-run after reindexing to
find stale notes that can be uniquely relinked to a moved/recreated symbol, then
rerun with `--apply` or MCP `apply=true` to update those links. `symbol` accepts
either an indexed symbol id or an exact unique symbol name; use `file_path` and
`symbol_kind` to disambiguate common names.

`dukememory_code_affected` / `code-affected` walks indexed relations from
changed files and returns likely affected test files. It is a test selection
hint, not a replacement for the compiler or full test suite.

`code-index` is incremental by default: unchanged Rust, Python, JavaScript, JSX, TypeScript, TSX, Go, Java, Kotlin, and Swift files are skipped, deleted files are removed from the index, and existing symbol embeddings are preserved for files whose hash did not change. Use `--full` when you need to delete and rebuild the whole code index for the project. Use `--embed` to build embeddings only for symbols in files indexed during that run.

After each indexing run, Dukememory resolves relation targets when they are unambiguous: same-file calls, module-qualified cross-file Rust calls, Python/JavaScript local calls, `use` targets, and external `mod foo;` declarations get `target_symbol_id` links. Qualified Rust calls such as `combat::apply()` keep their module path so duplicate function names in sibling modules can still resolve to the intended file. Common Rust standard-library, crate, and UI-builder calls are retained as `external_call` / `external_use` relations instead of project-quality unresolved edges. `code-status`, `code-index`, and MCP responses report total relation counts, project-quality resolved/unresolved counts, external relation counts, plus imported RA reference/call counts. `unresolved_relations` is therefore the remaining project-quality worklist, not every external library call seen by the parser.

`code-lsif-index` runs `rust-analyzer lsif` and imports definition/reference edges as `ra_reference` relations. References that are real invocations are also stored as `ra_call` relations for caller/callee tools. `dukememory_code_lsif_index` exposes the same layer over MCP, so Codex can refresh the Rust-analyzer-backed graph itself after `dukememory_code_index`.

## Operational Hardening

Use `doctor --deep`, `mcp-smoke`, `mcp-http-smoke`, `production-audit`, `codex-audit`, and `codex-hook-audit` after changing configuration, model roles, hooks, or MCP wiring. For large project rollouts, run `code-index`, `code-lsif-index`, `embed-missing --scope code`, `eval --file eval/dukememory.json --compare-last`, `ops-report`, and `doctor --deep` from each project root so every project gets its own scoped memory/code graph.

Install a release binary locally:

```bash
scripts/dukememory_install.sh
```
