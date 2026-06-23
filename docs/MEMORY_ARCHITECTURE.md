# dukememory Architecture

## Core Principles

- Project isolation is mandatory. Every memory row belongs to one `project_id`.
- Dukememory is generic-first. Game-specific memory is an optional domain profile, not the core schema.
- Every project has a profile: `project_type`, optional `domains`, optional description, and optional root path.
- Every memory has a `scope`; default retrieval remains project-local even when the scope label is `user` or `global`.
- Every memory has a `memory_tier`; default writes are `archival`, while `core` memories are prepended to context and `conversation` memories are retained for audit/compaction without becoming durable rules.
- Default retrieval reads only `active` memory.
- Automatic agent writes should use `pending` until accepted.
- Memory is append-friendly. Corrections use `superseded`, not destructive overwrite.
- Deletion is not part of the agent-facing contract. Use `archived` for removal from default retrieval.
- New memory writes pass a local safety policy before insertion.
- The production store is local lightweight PostgreSQL with `pgvector`; `DUKEMEMORY_DATABASE_MARKER` is only a local marker path for temporary schema isolation and backup naming. `DUKEMEMORY_DB` remains a legacy alias.
- PostgreSQL schema compatibility is explicit through SQL migrations in `migrations/`.
- All agent-facing tools use the `dukememory_*` prefix.

## Memory Statuses

| Status | Meaning | Default Retrieval |
| --- | --- | --- |
| `pending` | Candidate memory proposed by agent or hook | No |
| `active` | Trusted memory for normal project context | Yes |
| `superseded` | Replaced by a newer memory | No |
| `archived` | Retained for audit, excluded from normal retrieval | No |

## Universal Ontology

Core memory kinds are intended to work for any project:

- `decision`
- `project_rule`
- `constraint`
- `architecture`
- `code_fact`
- `bug_regression`
- `workflow`
- `setup`
- `external_service`
- `user_preference`
- `project_summary`
- `note`

Supported memory scopes are `project`, `session`, `module`, `user`, and `global`. These scopes are retrieval hints and lifecycle metadata; they do not bypass `project_id` isolation.

Supported memory tiers are `core`, `archival`, and `conversation`. `dukememory_remember`, `dukememory_search`, and `dukememory_list` expose tier controls over MCP; default writes use `archival`.

Domain-specific concepts should be expressed as profile domains and tags first. Examples: `game`, `webapp`, `library`, `research`, `ops`. Game concepts such as mechanics, assets, levels, balance, and lore are domain extensions, not required base kinds.

## Deduplication

Each memory has an internal normalized `body_hash`. Whitespace is collapsed and text is lowercased before hashing.

Agent-facing writes use deduplication by default:

- `dukememory_remember`
- `dukememory_extract`
- hook extraction through `scripts/dukememory_codex_hook.sh`

Duplicates are skipped when a matching `pending` or `active` memory already exists in the same project. `superseded` and `archived` memories do not block a new write.

## Current Retrieval

Retrieval is hybrid and project-scoped:

1. FTS/BM25.
2. Symbol and code graph matches.
3. Ollama embeddings with `qwen3-embedding:8b`.
4. Reciprocal Rank Fusion across signals.
5. Feedback quality signals (`quality_score`, `contradiction_risk`, and usage metadata) adjust memory ordering.
6. Optional `rerank` mode that asks `DUKEMEMORY_VALIDATE_MODEL` to reorder hybrid memory candidates.

Keyword search remains available so the system keeps working when Ollama is unavailable. Explicit `semantic` mode requires embeddings. Default `hybrid` mode may fall back to keyword results in MCP when embedding generation fails. `rerank` mode is intentionally a refinement layer, not a required retrieval dependency.

`dukememory_prepare` / `dukememory_agent_before` is the preferred first call for an agent task. It incrementally updates the project code index, optionally embeds symbols from files changed during that run, and returns a task-scoped prompt-ready bundle: query-matched core/project-rule fragments, task-selected memory fragments, memory graph context scoped to the selected memory ids, indexed code hits, project/index counts, and compact structured metadata. It does not place full project memory into the prompt. `core_memory_limit` caps query-matched core/project-rule context and does not consume `memory_limit`, which is reserved for task retrieval. `token_budget` controls the approximate packing budget and defaults low for MCP to reduce token use. MCP structured output keeps `memories`, `memory_fragments`, `code`, `code_neighborhood`, and `graph_summary` body-free by default; pass `debug=true` only when full trace, full graph, verbose code-neighborhood relations, or structured fragment text are required. `dukememory_agent_after` reuses the extraction pipeline after a task to store pending memories and optionally validate them. `dukememory_context` remains available when the caller explicitly wants retrieval without auto-indexing.

Retrieval events are audit records, not prompt context. `dukememory_agent_task` attaches retrieval events to its `task_session_id`, and `dukememory_trace` / `dukememory_task_replay` can replay those exact events before falling back to query-based lookup for older data. `dukememory_feedback` accepts a retrieval event id plus typed outcome/severity, updates retrieved memory quality and contradiction-risk signals, writes an audit event, and stores a regression eval case.

The autonomous self-healing loop is backend-only and audit-first. `dukememory_self_heal` composes lifecycle review, outcome learning, conflict graph analysis, and memory compilation. `dukememory_outcome_learn` converts completed/failed task-session outcomes into helpful/unhelpful memory quality signals. `dukememory_conflict_graph` finds contradictory memories and graph facts, and with `apply=true` invalidates only weaker temporal facts. `dukememory_memory_compiler` promotes stable high-confidence rules to `core`, archives low-signal duplicates, and writes pending split candidates for overlong memories. `dukememory_policy_ab` runs live retrieval policy trials and records the recommended policy. These tools default to dry-run planning; `apply=true` performs auditable mutations.

## Memory Graph

The graph layer is project-scoped and uses:

- `memory_entities` for named concepts, systems, modules, people, services, or domain objects.
- `memory_facts` for predicate/value statements optionally linked to a memory or episode.
- `memory_edges` for directed relations between entities, optionally linked to a memory or episode.
- `memory_episodes` for provenance records with source, summary, raw reference, raw JSON payload, and observation time.

Facts and edges can carry `valid_from`, `valid_to`, and `invalidated_by`; `dukememory_graph` search accepts `as_of` for temporal graph reads. `dukememory_graph` actions `invalidate_fact` and `invalidate_edge` close validity windows without deleting graph history. `dukememory_episode` adds or searches episode provenance records.

`dukememory_graph_extract` derives graph proposals from existing memories with the configured local LLM. It defaults to dry-run and writes only when `apply=true`, so agents can inspect proposed entities, facts, and edges before changing the graph.

CLI commands `graph-entity`, `graph-fact`, `graph-edge`, and `graph-search` remain available for manual corrections. MCP exposes manual graph edits/search as `dukememory_graph` and extraction as `dukememory_graph_extract`.

## MCP Tools

`mcp-smoke` is the protocol-level smoke check. It uses an isolated temporary PostgreSQL schema and sends JSON-RPC requests through the same MCP handler used by stdio: initialize, tools/list, `dukememory_remember`, `dukememory_search`, `dukememory_context`, and `dukememory_status`. It also searches the same token in another project id and fails if any result leaks across projects.

`production-audit` is the isolated rollout check. It creates a temporary Rust project and PostgreSQL schema, indexes the project, verifies code index freshness, writes and searches project-scoped memory, checks caller/callee graph lookups, creates a backup, exports/imports the project into another isolated schema, and runs maintenance in dry-run mode.

`codex-audit` is the installed-integration check. It reads `~/.codex/config.toml` and `~/.codex/hooks.json`, verifies the `dukememory` MCP server command, `args = ["mcp"]`, required model/database environment keys, Stop/PreCompact hook entries, script executability, and a dry-run of the hook wrapper with empty stdin.

`codex-hook-audit` is the non-empty hook path check. It sends a Codex-like JSON payload through `scripts/dukememory_codex_hook.sh` with an isolated `DUKEMEMORY_DB` marker and `DUKEMEMORY_PROJECT`, then verifies that the wrapper, `dukememory-extract`, model extraction, safety policy, PostgreSQL write, and pending-memory lookup all completed.

`doctor --deep` includes a built-in isolated eval readiness check. File-based retrieval suites can be run with `eval --file`; each run stores suite name, suite hash, mode, per-case details, and optional comparison data against the previous run of the same suite/mode/hash. Eval cases may target `memory`, `code`, `context`, `semantic`, or dry-run `review_apply`, so retrieval regressions can cover memory lookup, code search, combined context assembly, agent-intelligence operations, and review-decision wiring.

`dukememory_devsystem` is the MCP-facing `dukedevsystem` quality loop. It auto-indexes the project by default when `project_path` is supplied, supports `auto_index`, `full_rebuild`, `embed_symbols`, and `embed_symbol_limit`, and returns both `telemetry.index_run` and `telemetry.index_guard`. It returns structured readiness progress, stage reports for the advisory agent roles, File Entropy Score reports, code-review-plan telemetry, test commands, missing telemetry signals, effective policy, structured boundary repair plans, advisory quality gates, gate summary, optional `quality_evidence_reports`, and typed pending memory write ids. Evidence execution is opt-in with `run_evidence=true`; commands are parsed into argv and run without a shell, can be narrowed with exact `allowed_evidence_commands`, respect `evidence_timeout_seconds` and `max_evidence_commands`, and store only bounded stdout/stderr excerpts. The Memory Agent writes deduplicated pending task-intent, decision, per-file entropy, and intent-graph candidate memories when `write_memory=true`; `write_memory=false` leaves `memory_writes.status=disabled` and writes nothing. Executed evidence is summarized in pending decision memory and intent-graph candidates. Intent graph links are represented as pending memory candidates with source memory ids, not automatically promoted graph edges. Policy comes from defaults, optional `[devsystem]` values in `.dukememory.toml`, and optional per-call MCP `policy` overrides. It is advisory and does not auto-apply boundary repairs.

Current tools:

- `dukememory_extract`
- `dukememory_models`
- `dukememory_project_profile`
- `dukememory_ontology`
- `dukememory_eval`
- `dukememory_prepare`
- `dukememory_agent_before`
- `dukememory_devsystem`
- `dukememory_agent_after`
- `dukememory_context`
- `dukememory_graph`
- `dukememory_episode`
- `dukememory_graph_extract`
- `dukememory_search`
- `dukememory_remember`
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

Read-only MCP resources are available for clients that prefer resource reads over tool calls:

- `dukememory://ontology`
- `dukememory://health`
- `dukememory://project/{project_id}/profile`

MCP resource templates expose `dukememory://project/{project_id}/profile` through `resources/templates/list`.

MCP prompts expose ready task instructions through `prompts/list` / `prompts/get`:

- `dukememory_agent_before`
- `dukememory_agent_after`
- `dukememory_memory_review`
- `dukememory_code_risk`

The stdio server is the default transport. `mcp-http` provides an experimental localhost HTTP endpoint with JSON-RPC `POST /mcp` and an SSE-compatible endpoint announcement on `GET /mcp`. It binds only to localhost, rejects non-local `Origin` headers, rejects unsupported methods, and enforces request-size/header-size limits. `mcp-http-smoke` exercises initialize, endpoint discovery, non-local origin rejection, wrong-method rejection, and oversized-body rejection against an isolated temporary database. Tool schemas that accept `project_path` also accept optional `roots`; when roots are supplied, `project_path` must resolve inside one root.

## Codex Integration

`cargo run -- codex-config` prints the MCP config snippet for Codex.

`cargo run -- codex-config --install` appends the snippet to `~/.codex/config.toml` and writes a timestamped backup first. If `[mcp_servers.dukememory]` already exists, installation fails unless `--force` is used.

`cargo run -- codex-hooks` prints the hook JSON for pending memory extraction.

`cargo run -- codex-hooks --install` appends extraction hooks to `~/.codex/hooks.json` and writes a timestamped backup first. Default events are `Stop` and `PreCompact`. Existing non-dukememory hooks are preserved; existing dukememory hooks require `--force` to replace.

## Embedding Storage

Embeddings are stored separately from durable facts:

- memory vectors live in `memory_embeddings_4096`
- code symbol vectors live in `code_symbol_embeddings_1024`
- rows include `project_id`, model name, dimensions, and a little-endian `f32` vector blob

This keeps memories auditable text-first while allowing vectors to be rebuilt when `DUKEMEMORY_EMBED_MODEL` or `DUKEMEMORY_FAST_EMBED_MODEL` changes. Memory vectors use `DUKEMEMORY_EMBED_MODEL`; code symbol vectors use `DUKEMEMORY_FAST_EMBED_MODEL`. Memory embedding text includes kind, tier, scope, tags, source, importance, confidence, and body. Code symbol embedding text includes language, kind, file path, line range, signature, docstring, and source snippet.

## Safety Policy

New writes through `remember`, `supersede`, `dukememory_remember`, `dukememory_supersede`, and automatic extraction are inspected before insertion. The policy blocks obvious secret values, including API key assignments, bearer tokens, private key markers, known token prefixes, and labeled high-entropy credential-like strings.

Possible personal data such as email-like or phone-like text is reported as a warning-level finding but is not blocked by default. Manual CLI/MCP writes can set `allow_sensitive`, but automatic extraction and compaction do not use that override.

## Schema And Portability

The target PostgreSQL schema is managed by ordered SQL migrations in `migrations/`. The local laptop profile runs PostgreSQL through `scripts/dukememory_postgres.sh`, listens only on a Unix socket, and keeps resource limits low (`max_connections=32`, `shared_buffers=64MB`, `work_mem=4MB`, `maintenance_work_mem=64MB`, `autovacuum_max_workers=1`).

Schema v1 adds the PostgreSQL baseline plus universal project profile fields and memory scope:

- `projects.project_type`
- `projects.description`
- `projects.domains`
- `projects.updated_at`
- `memories.scope`

Schema v2 adds context-layer and temporal provenance fields:

- `memories.memory_tier`
- `memory_episodes.raw_payload`
- temporal graph indexes for fact/edge validity windows
- `memory_facts.invalidated_by`
- `memory_edges.invalidated_by`

Schema v3 adds eval run and graph-operation hardening:

- `eval_cases.expected_ids`
- `idx_eval_runs_project_created`

Schema v4 adds operational audit and eval baseline metadata:

- `dukememory_audit_events`
- `eval_runs.suite_name`
- `eval_runs.suite_hash`
- `idx_eval_runs_project_suite_mode_created`

There are two portability paths:

- PostgreSQL logical backups use `pg_dump` against `DUKEMEMORY_DATABASE_URL`.
- `dukememory_export` / `export` writes one project to JSON. It includes memories, code files, symbols, and relations, but intentionally excludes embedding blobs. After `dukememory_import` / `import`, rebuild vectors with `dukememory_embed_missing` or `embed-missing`.

Import is non-destructive by default: existing row ids are skipped. With `overwrite`, only the exported `project_id` is cleared and replaced; other projects remain untouched.

## Model Roles

- `DUKEMEMORY_EMBED_MODEL`: durable memory embeddings and memory semantic search.
- `DUKEMEMORY_FAST_EMBED_MODEL`: indexed code symbol embeddings and code semantic search.
- `DUKEMEMORY_EXTRACT_MODEL`: transcript/hook memory extraction.
- `DUKEMEMORY_VALIDATE_MODEL`: reserved for memory validation/review.
- `DUKEMEMORY_FAST_CODE_MODEL`: reserved for quick code reasoning.
- `DUKEMEMORY_DEEP_CODE_MODEL`: reserved for deeper code analysis.
- `DUKEMEMORY_AGENT_CODE_MODEL`: reserved for agent-facing code work.
- `DUKEMEMORY_EXPERIMENT_MODEL`: reserved for experimental code/model flows.

`doctor` reports every configured role against the local Ollama model list and also reports local `rust-analyzer` availability for Rust code-analysis readiness. `doctor --deep` is the strict readiness gate: it fails when required model roles are missing, the MCP JSON-RPC smoke flow fails, the isolated production audit fails, installed Codex config/hooks fail audit, memory/code embeddings are incomplete, the saved code index no longer matches files on disk, no code relation targets are resolved, no rust-analyzer references or call edges are imported for indexed code, `rust-analyzer diagnostics` fails, or the macOS LaunchAgent is not loaded for the current project.

## Automatic Memory Extraction

Automatic extraction is implemented through `dukememory_extract` and the CLI command `dukememory-extract`. The extraction path creates a `memory_episodes` provenance row for the raw input before writing pending candidates.

Extraction is conservative:

1. Read a transcript or summary from a Codex hook.
2. Extract only stable facts, project decisions, project rules, recurring pitfalls, and useful implementation lessons.
3. Write candidates as `pending` with `kind`, `source`, `tags`, `importance`, `confidence`, and `reason`.
4. Promote candidates only after review.

The extractor should reject:

- secrets, tokens, passwords, private keys
- temporary task chatter
- unverified guesses
- raw logs unless they encode a durable lesson
- cross-project facts unless the project is explicit

Suggested candidate kinds:

- `decision`
- `project_rule`
- `constraint`
- `architecture`
- `game_design`
- `code_fact`
- `bug_regression`
- `asset_pipeline`
- `user_preference`

Hook runners can pipe transcript or summary text into `scripts/dukememory_codex_hook.sh`. The wrapper exits without writing memory when stdin is empty. When input is present, it calls `dukememory-extract` and uses `dukememory_hook:<event>` as the source.

When hook input is JSON, extraction inspects common payload fields including `cwd`, `project_path`, `workspaceRoot`, `event`, `hookEventName`, `summary`, `transcript`, `messages`, `inputMessages`, `output`, `history`, `turns`, and `prompt`. Message content can be plain strings or multipart objects such as `input_text`, `output_text`, and `text`.

## Maintenance

Pending memory review and cleanup are agent-facing operations:

- `dukememory_review` lists pending candidates for promotion/archive decisions.
- `dukememory_review_apply` applies batch `promote`, `archive`, or `keep` decisions and records audit events. It defaults to dry-run.
- `dukememory_validate_pending` uses `DUKEMEMORY_VALIDATE_MODEL` to prepare or apply promote/archive/keep decisions.
- `dukememory_prune_pending` archives pending candidates in bulk and defaults to `dry_run`.
- `dukememory_compact` proposes or applies a `project_summary` for older active memories. It defaults to dry-run, excludes existing `project_summary` rows unless a kind filter is explicit, and archives source memories only when `apply=true`.
- `dukememory_maintenance` orchestrates backup, validation, compaction, and missing-embedding checks. With no step flags it runs a safe validation/compaction dry-run. With `all=true`, it enables every maintenance step; `apply=true` applies validation, compaction, and embedding rebuilds. CLI maintenance runs can emit JSON and are written to the audit log.

`maintenance-launchd` is the OS-level scheduling wrapper for macOS. It generates or installs `~/Library/LaunchAgents/com.dukememory.maintenance.plist`, includes the configured model/database environment, pins the current project id unless `--project` is passed, and defaults to running `maintenance --all --backup --apply` every 6 hours. Loading and unloading are explicit operations so installing a plist can be reviewed before launchctl changes runtime state.

## Code Index

Current code tools are implemented as a Rust/Python/JavaScript/TypeScript plus generic Go/Java/Kotlin/Swift structural index:

- file walking respects ignore rules
- Rust, Python, JavaScript, JSX, TypeScript, TSX, Go, Java, Kotlin, and Swift source are parsed with tree-sitter
- symbols are stored in `code_symbols`
- imports and calls are stored in `code_relations`
- symbol FTS is stored in `code_symbol_fts`
- optional symbol embeddings are stored in `code_symbol_embeddings_1024`
- durable code notes are stored in `code_memories` and linked to `symbol_id` or `file_path`
- `Cargo.toml` package metadata and external `mod foo;` declarations are indexed as code relations.
- `rust-analyzer lsif` definitions/references are imported as `ra_reference` relations by `code-lsif-index` / `dukememory_code_lsif_index`.
- LSIF references that are actual invocations are also imported as `ra_call` relations, so caller/callee tools can use Rust-analyzer-backed call edges without treating every reference as a call.

`dukememory_code_explore` is the preferred one-call code navigation tool for agents. It combines symbol search, selected symbol/file code memories, route hints, caller/callee impact, and optional freshness checks into one response. Lower-level tools remain available for precise follow-up queries.

`code_memories` is intentionally separate from normal project memory. Normal memories describe durable project knowledge; code memories describe local symbol/file facts such as invariants, risks, usage constraints, and test notes. Task context retrieves only active code memories attached to selected code hits, preserving task-scoped context packing. Code memories have their own lifecycle: `pending` for proposed notes, `active` for trusted notes used by default retrieval, and `archived` for retired notes kept for audit. Search/list results include `link_status`: `symbol`, `file`, or `stale` when the indexed symbol/file no longer exists. Symbol-linked notes store a symbol snapshot (`symbol_name`, `symbol_kind`, signature, and line range), so `dukememory_code_memory action=repair` can dry-run and then apply unambiguous relinks after reindexing. `symbol` inputs accept either an indexed symbol id or an exact unique symbol name; callers should pass `file_path` and/or `symbol_kind` when a name is ambiguous.

`dukememory_code_affected` uses indexed code relations to return likely test files affected by changed source files. This is a planning and test-selection hint, not authoritative compiler analysis.

Indexing is incremental by default. `dukememory_code_index`, `dukememory_devsystem` auto-indexing, and `code-index` skip Rust, Python, JavaScript, JSX, TypeScript, TSX, Go, Java, Kotlin, and Swift files whose stored hash did not change, remove files that disappeared, and preserve symbol embeddings for unchanged files. Use `full_rebuild` or `code-index --full` to delete and rebuild the whole project code index. When `embed_symbols` or `code-index --embed` is enabled, only symbols from files indexed during that run are embedded.

After indexing, relation targets are resolved when the project-local match is unambiguous. This fills `code_relations.target_symbol_id` for same-file calls, module-qualified cross-file Rust calls, Python/JavaScript local calls, `use` targets, and external module declarations. Qualified Rust calls keep their module path, so `combat::apply()` can resolve to `src/combat.rs` even when another module also defines `apply`. Common Rust standard-library, crate, and UI-builder calls are retained as `external_call` / `external_use` relations instead of project-quality unresolved edges. Status responses expose total relations, project-quality resolved/unresolved counts, external relation counts, imported RA references, and imported RA calls. `unresolved_relations` is the remaining project-quality worklist, not every external library call seen by the parser.

The tree-sitter call graph is intentionally conservative for unresolved or ambiguous targets. The LSIF layer adds Rust-analyzer-backed definition/reference edges, including references and call edges that tree-sitter cannot resolve by syntax alone.

## Project Resolution

Resolution order:

1. Explicit `project_id`.
2. Explicit `project_path`, normalized to the nearest project root.
3. Process current directory fallback.

Project roots are detected by:

- `.git`
- `.dukememory.toml`
- `.codegraph`

The fallback intentionally stops at the user home directory to avoid accidentally using a home-folder-wide project id.
