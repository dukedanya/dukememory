# AGENTS.md

## Naming

All memory-facing tools, commands, hooks, events, tables exposed to agents, and documented agent actions must use the `dukememory_*` prefix.

Use:

- `dukememory_extract`
- `dukememory_models`
- `dukememory_project_profile`
- `dukememory_task_session`
- `dukememory_task_eval`
- `dukememory_test_plan`
- `dukememory_devsystem`
- `dukememory_ontology`
- `dukememory_eval`
- `dukememory_prepare`
- `dukememory_agent_before`
- `dukememory_agent_task`
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
- `dukememory_read_symbol`
- `dukememory_code_brief`
- `dukememory_code_plan`
- `dukememory_code_risk`
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
- `dukememory_context_policy`
- `dukememory_trace`
- `dukememory_task_replay`
- `dukememory_counterfactual_eval`
- `dukememory_code_causality`
- `dukememory_memory_impact`
- `dukememory_temporal_context`
- `dukememory_embed_missing`

Do not introduce generic agent-facing names such as `memory_search`, `remember`, `read_memory`, or `project_memory_search` for new MCP tools or agent-facing memory APIs.

## Project Isolation

Memory access must default to the current project only. Cross-project memory lookup requires an explicit user request and an explicit project identifier.

When exposing tools through MCP, accept `project_path` so the client can pass the active workspace root. If neither `project_id` nor `project_path` is supplied, falling back to process cwd is allowed.

Project isolation applies to keyword search, semantic vector search, hybrid search, code index queries, and embedding backfill.

## Memory Lifecycle

Automatic agent-written memories should normally start as `pending`.

Use `dukememory_extract` for transcript, summary, or hook-payload extraction. It must write pending candidates, not active memories.

Use `cargo run -- codex-hooks` to print Codex hook JSON and `cargo run -- codex-hooks --install` to install extraction hooks with backup. Hook installation must preserve unrelated hooks.

Agent-facing writes should deduplicate by normalized body within the same project. A duplicate skip is a successful no-op, not a failure.

Use `dukememory_review` before promoting extracted memories. Use `dukememory_prune_pending` with `dry_run=true` before bulk archiving pending candidates.

Use:

- `pending` for proposed memory that needs review.
- `active` for trusted memory used in default retrieval.
- `superseded` for old facts replaced by a newer memory.
- `archived` for memory that should not be retrieved by default but should remain auditable.

Prefer `dukememory_supersede` or `dukememory_archive` over physical deletion.

## Retrieval

At the start of a non-trivial task, prefer `dukememory_context` with the current task query. The context bundle includes only query-selected memory fragments, graph entities/facts/edges around those selected memories, and indexed code hits. Use narrower `dukememory_search`, `dukememory_graph`, or `dukememory_code_search` calls only when the context bundle is not enough.

Default search mode should be `hybrid`: keyword FTS plus semantic embeddings, merged with Reciprocal Rank Fusion. If Ollama embeddings are unavailable, MCP tools may fall back to keyword results for `hybrid`, but explicit `semantic` mode should report the embedding error.

Use `dukememory_embed_missing` to backfill vectors for existing memories or indexed code symbols.

Keep context calls compact by default. Use `debug=true` on `dukememory_prepare` or `dukememory_context` only when diagnosing retrieval/scoring internals, because it expands structuredContent with full trace, graph, code-neighborhood relations, and fragment text.

## Memory Graph

Use `dukememory_graph_extract` to build project-scoped graph entities, facts, and edges from existing memories. It defaults to dry-run; only set `apply=true` after reviewing the proposed graph items.

Use `dukememory_graph` for manual entity/fact/edge edits or graph search. Keep relations broad and durable so large projects do not become a dense type-level tangle. Prefer meaningful edges such as `uses`, `depends_on`, `enforces`, `replaces`, `documents`, `runs_before`, or `belongs_to`.

## Code Index

For code understanding and navigation, agents should use Dukememory's project-scoped code graph before falling back to filesystem search or broad file reads.

Preferred workflow:

- Start non-trivial code tasks with `dukememory_prepare` when indexing freshness matters; pass `project_path`.
- Use `dukememory_code_explore` as the first follow-up for structural code questions, implementation work, and code-flow questions; it returns relevant symbols, linked code memories, route hints, impact, and freshness warnings in one response.
- Use `dukememory_code_search` to locate relevant symbols instead of grepping first.
- Use `dukememory_code_files` and `dukememory_code_outline` when the task is file-oriented.
- Use `dukememory_read_symbol` to inspect an exact symbol body.
- Use `dukememory_find_callers`, `dukememory_find_callees`, or `dukememory_impact` to navigate call relationships.
- Use `dukememory_code_memory` for durable symbol/file notes. Automatic proposed code memories should start as `pending`; promote trusted notes before relying on them in default context. Treat `link_status=stale` as a signal to refresh or archive the note.
- `dukememory_code_memory` `symbol` inputs may use an indexed symbol id or an exact unique symbol name; pass `file_path` and/or `symbol_kind` to disambiguate common names before writing a symbol-linked note.
- Symbol-linked code memories store a symbol snapshot. Use `dukememory_code_memory action=repair` as a dry-run after reindexing to find stale notes with a unique replacement symbol; apply only those unambiguous relinks.
- Use `dukememory_code_status` to check index coverage and `dukememory_code_index` / `dukememory_code_lsif_index` to refresh the graph when it is stale.

Plain `rg`/file reads are still appropriate for non-code assets, exact text checks, generated files not represented in the index, or when the code graph returns no useful hit.

The first code-index layer is multi-language and approximate: it indexes Rust, Python, JavaScript/JSX, TypeScript/TSX, Go, Java, Kotlin, and Swift symbols, with tree-sitter call edges where available and conservative generic call edges for newer language layers. Treat `dukememory_find_callers`, `dukememory_find_callees`, and `dukememory_impact` as useful navigation hints, not as authoritative compiler-grade name resolution. The LSIF layer adds rust-analyzer-backed references and call edges when `dukememory_code_lsif_index` has been run.
