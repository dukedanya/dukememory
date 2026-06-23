# MCP Tools

Dukememory exposes agent-facing tools with the `dukememory_*` prefix. Tools
default to project-scoped access. When available, pass `project_path` so the MCP
client can bind memory and code lookup to the active workspace.

## Recommended Agent Loop

| Phase | Tool | Purpose |
| --- | --- | --- |
| Before work | `dukememory_prepare` or `dukememory_agent_before` | Refresh code index and return task-scoped memory/code context |
| During work | `dukememory_code_explore`, `dukememory_read_symbol`, `dukememory_search` | Navigate code and memory without broad prompt loading |
| After work | `dukememory_agent_after` or `dukememory_extract` | Extract pending memory candidates from task output |
| Review | `dukememory_review`, `dukememory_promote`, `dukememory_archive` | Promote trusted memories and retire bad ones |
| Feedback | `dukememory_feedback`, `dukememory_trace`, `dukememory_task_replay` | Turn retrieval quality issues into auditable signals |

## Core Memory Tools

| Tool | Mutates | Purpose |
| --- | --- | --- |
| `dukememory_remember` | Yes | Write one memory, normally pending or active by explicit request |
| `dukememory_remember_smart` | Optional | Run policy guidance before writing a pending candidate |
| `dukememory_search` | No | Search active project memory |
| `dukememory_context` | No | Build compact task-scoped memory/code/graph context |
| `dukememory_get` | No | Read one memory by id |
| `dukememory_list` | No | List memories by status, kind, tier, or project |
| `dukememory_review` | No | Review pending memory candidates |
| `dukememory_review_apply` | Optional | Apply structured review decisions |
| `dukememory_validate_pending` | Optional | Validate pending memory quality |
| `dukememory_promote` | Yes | Promote a pending memory to active |
| `dukememory_supersede` | Yes | Replace an old fact while preserving history |
| `dukememory_archive` | Yes | Remove a memory from default retrieval without deleting audit history |
| `dukememory_prune_pending` | Optional | Dry-run or archive stale pending candidates |
| `dukememory_compact` | Optional | Summarize compactable conversation memory |

## Agent Task Tools

| Tool | Mutates | Purpose |
| --- | --- | --- |
| `dukememory_prepare` | Optional | Incrementally index code and return task context |
| `dukememory_agent_before` | Optional | Alias for task-intake context preparation |
| `dukememory_agent_task` | Optional | Create a task session, collect context, code assist, and artifacts |
| `dukememory_agent_after` | Yes | Extract pending memory candidates after a task |
| `dukememory_task_session` | Yes | Track task session status, phase, progress, and artifacts |
| `dukememory_task_eval` | Optional | Build and run an eval case from a task session |
| `dukememory_test_plan` | No | Recommend affected tests from session files |
| `dukememory_devsystem` | Optional | Run advisory quality gates, file entropy, boundary repair, and pending memory writes |

## Code Intelligence Tools

| Tool | Mutates | Purpose |
| --- | --- | --- |
| `dukememory_code_index` | Yes | Index project source symbols and approximate call relations |
| `dukememory_code_lsif_index` | Yes | Import rust-analyzer-backed references when available |
| `dukememory_code_status` | No | Report code index coverage and freshness |
| `dukememory_code_files` | No | List indexed files |
| `dukememory_code_outline` | No | Read file-level code outline |
| `dukememory_code_search` | No | Search indexed code symbols |
| `dukememory_code_explore` | No | Return relevant symbols, relations, code memories, and route hints |
| `dukememory_read_symbol` | No | Read an exact indexed symbol |
| `dukememory_find_callers` | No | Find approximate callers |
| `dukememory_find_callees` | No | Find approximate callees |
| `dukememory_impact` | No | Estimate impacted files and tests |
| `dukememory_code_memory` | Optional | Manage pending/active code notes linked to symbols or files |
| `dukememory_code_affected` | No | Find affected symbols/files from changed paths |
| `dukememory_code_patterns` | No | Report project code patterns |
| `dukememory_code_duplicates` | No | Find likely duplicate code symbols |
| `dukememory_code_assist` | No | Return compact code-assist signals for a task |
| `dukememory_code_review_plan` | No | Build a review plan from code telemetry |
| `dukememory_code_eval` | Optional | Run code-oriented eval cases |
| `dukememory_code_brief` | No | Summarize relevant code context |
| `dukememory_code_plan` | No | Produce code-change planning hints |
| `dukememory_code_risk` | No | Identify code risk signals |
| `dukememory_embed_missing` | Optional | Backfill missing memory or code embeddings |

## Graph And Semantic Tools

| Tool | Mutates | Purpose |
| --- | --- | --- |
| `dukememory_graph` | Optional | Search or edit memory entities, facts, and edges |
| `dukememory_episode` | Optional | Add or search provenance episodes |
| `dukememory_graph_extract` | Optional | Propose or apply graph extraction from memories |
| `dukememory_semantic` | Depends on action | Generic semantic operation entry point |
| `dukememory_dedupe` | No | Find likely duplicate memories |
| `dukememory_related` | No | Find related memories and code hits |
| `dukememory_semantic_review` | No | Review semantic conflicts and supersession candidates |
| `dukememory_semantic_route` | No | Route task intent to retrieval/code/graph strategy |
| `dukememory_semantic_clusters` | No | Cluster project memory |
| `dukememory_semantic_tags` | No | Suggest tags |
| `dukememory_stale_check` | No | Find stale memory candidates |
| `dukememory_consistency_check` | No | Check memory consistency |
| `dukememory_eval_generate` | Optional | Generate eval cases |
| `dukememory_hard_negatives` | No | Build hard-negative retrieval signals |
| `dukememory_embedding_health` | No | Report embedding coverage |
| `dukememory_model_migration` | No | Plan model migration/backfill |
| `dukememory_isolation_check` | No | Check project isolation behavior |
| `dukememory_memory_hints` | No | Return compact memory hints |
| `dukememory_policy_decision` | No | Recommend memory policy for a write/query |
| `dukememory_retrieval_quality` | No | Probe retrieval quality |
| `dukememory_auto_eval` | Optional | Build automatic eval signals |
| `dukememory_ab_compare` | No | Compare retrieval models or policies |
| `dukememory_lifecycle_review` | No | Review lifecycle health |
| `dukememory_code_memory_suggest` | Optional | Suggest code-memory notes |
| `dukememory_verify_conflicts` | No | Verify contradiction candidates |
| `dukememory_topic_map` | No | Summarize memory topics |
| `dukememory_budget_optimize` | No | Recommend context budget |
| `dukememory_feedback` | Yes | Apply typed retrieval feedback and create regression evals |
| `dukememory_self_heal` | Optional | Compose lifecycle, outcome, conflict, and compiler signals |
| `dukememory_outcome_learn` | Optional | Learn from completed/failed task sessions |
| `dukememory_conflict_graph` | Optional | Invalidate weaker temporal graph facts when applied |
| `dukememory_memory_compiler` | Optional | Promote stable rules, archive low-signal duplicates, split long memory |
| `dukememory_policy_ab` | No | Run live retrieval policy trials |
| `dukememory_context_policy` | No | Learn recommended context policy |
| `dukememory_trace` | No | Build flight-recorder trace |
| `dukememory_task_replay` | No | Replay context used by a task |
| `dukememory_counterfactual_eval` | Optional | Run leave-one-out retrieval counterfactuals |
| `dukememory_code_causality` | No | Connect memory to code causality and affected tests |
| `dukememory_memory_impact` | No | Alias focused on memory impact over code |
| `dukememory_temporal_context` | No | Read context as of a timestamp |

## Operations Tools

| Tool | Mutates | Purpose |
| --- | --- | --- |
| `dukememory_models` | No | Report configured Ollama model roles |
| `dukememory_project_profile` | Optional | Read or update project profile metadata |
| `dukememory_ontology` | No | Return memory ontology |
| `dukememory_eval` | Optional | Run retrieval eval suites |
| `dukememory_audit_log` | No | Read audit events |
| `dukememory_maintenance` | Optional | Run maintenance checks or apply selected maintenance |
| `dukememory_ops_pipeline` | Optional | Run operational health pipeline |
| `dukememory_status` | No | Report project status |
| `dukememory_health` | No | Report store/system health |
| `dukememory_cleanup_schemas` | Optional | Clean temporary schemas |
| `dukememory_backup` | Yes | Create PostgreSQL backup |
| `dukememory_export` | No | Export one project to JSON |
| `dukememory_import` | Yes | Import a project JSON export |

## Resources And Prompts

Read-only resources:

- `dukememory://ontology`
- `dukememory://health`
- `dukememory://project/{project_id}/profile`

Prompt templates:

- `dukememory_agent_before`
- `dukememory_agent_after`
- `dukememory_memory_review`
- `dukememory_code_risk`
