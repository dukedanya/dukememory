# dukememory

![dukememory banner](docs/assets/dukememory-banner.svg)

[English](README.md) | [Русский](README.ru.md)

[![CI](https://github.com/dukedanya/dukememory/actions/workflows/ci.yml/badge.svg)](https://github.com/dukedanya/dukememory/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/Rust-2024-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![MCP Server](https://img.shields.io/badge/MCP-server-2563EB)](#mcp-сервер-для-ai-агентов)
[![PostgreSQL + pgvector](https://img.shields.io/badge/PostgreSQL%20%2B-pgvector-336791?logo=postgresql&logoColor=white)](#быстрый-старт)
[![Ollama](https://img.shields.io/badge/Ollama-local%20AI%20models-10B981)](#требования)

**Локальная память для AI-агентов, MCP-сервер, семантический поиск и code graph
context для Codex и developer agents.**

Dukememory дает локальным AI-агентам для программирования проектную долговременную
память. Система хранит решения, архитектурные правила, результаты задач, факты о
коде и feedback retrieval-качества в PostgreSQL + pgvector; использует Ollama для
локальных embeddings и extraction; индексирует исходный код; и открывает все
возможности через CLI и MCP-инструменты с префиксом `dukememory_*`.

Проект создан для разработчиков, которым нужна persistent agent memory без
облачного хранилища, утечек между репозиториями и автоматических записей без
review.

## Зачем Это Нужно

У большинства систем памяти для AI-агентов одни и те же проблемы: память
глобальная, плохо аудируется, легко загрязняется или не связана с кодом.
Dukememory использует более строгую модель:

- память изолирована по `project_id` и по умолчанию привязана к текущему
  репозиторию;
- автоматические наблюдения сначала попадают в `pending` и не участвуют в
  retrieval без review;
- старые факты переводятся в `superseded` или `archived`, а не перезаписываются;
- очевидные секреты блокируются до записи;
- context packs включают только релевантную задаче память, graph facts и code
  hits;
- все MCP tools, CLI commands, hooks, events и agent-facing API используют
  префикс `dukememory_*`.

## Возможности

| Возможность | Что делает |
| --- | --- |
| Память AI-агента | Хранит проектные решения, правила, setup notes, summaries и outcomes задач |
| MCP-сервер | Экспортирует инструменты memory, search, code context, graph, audit, backup, eval и maintenance |
| Hybrid retrieval | Объединяет PostgreSQL full-text search, pgvector semantic search и Reciprocal Rank Fusion |
| Локальные embeddings | Работает с Ollama-моделями вроде `qwen3-embedding:8b` и `qwen3:14b` |
| Review workflow | Держит agent-written memory candidates в `pending` до promotion |
| Code graph context | Индексирует symbols в Rust, Python, JavaScript, TypeScript, Go, Java, Kotlin и Swift |
| Memory graph | Хранит entities, facts, edges, provenance episodes и temporal invalidation |
| Интеграция с Codex | Генерирует MCP config и Stop/PreCompact extraction hooks |
| Native viewer | Открывает локальный memory/code graph browser для project vaults |

## Быстрый Старт

```bash
git clone https://github.com/dukedanya/dukememory.git
cd dukememory

brew install postgresql@17 pgvector
scripts/dukememory_postgres.sh start
scripts/dukememory_postgres.sh migrate
export DUKEMEMORY_DATABASE_URL="$(scripts/dukememory_postgres.sh url)"

cargo run -- doctor
cargo run -- remember --kind decision "Use project_id for every memory lookup."
cargo run -- search "project memory isolation"
cargo run -- context "what should I know before editing retrieval"
```

Запустить MCP-сервер:

```bash
cargo run -- mcp
```

Сгенерировать Codex config и hooks:

```bash
cargo run -- codex-config
cargo run -- codex-hooks
```

Открыть native memory viewer:

```bash
cargo run -- dukememory_app
```

## MCP-Сервер Для AI-Агентов

Dukememory в первую очередь спроектирован как MCP-сервер для локальных coding
agents. Основные семейства инструментов:

| Семейство | Примеры |
| --- | --- |
| Task context | `dukememory_prepare`, `dukememory_context`, `dukememory_agent_before` |
| Memory writes | `dukememory_remember`, `dukememory_extract`, `dukememory_agent_after` |
| Review lifecycle | `dukememory_review`, `dukememory_promote`, `dukememory_supersede`, `dukememory_archive` |
| Code intelligence | `dukememory_code_search`, `dukememory_code_explore`, `dukememory_read_symbol`, `dukememory_impact` |
| Graph и semantic ops | `dukememory_graph`, `dukememory_graph_extract`, `dukememory_trace`, `dukememory_feedback` |
| Operations | `dukememory_status`, `dukememory_health`, `dukememory_backup`, `dukememory_export`, `dukememory_import` |

Для нетривиальных agent tasks начинайте с `dukememory_prepare` и передавайте
`project_path`. Инструмент обновляет code index и возвращает компактный
task-scoped context bundle, а не выгружает всю проектную память в prompt.

## Архитектура

```text
Codex / local AI agent
        |
        | MCP tools or CLI commands
        v
dukememory
  |-- project isolation and safety policy
  |-- memory lifecycle and review queue
  |-- hybrid retrieval and context packing
  |-- code symbol index and code memories
  |-- memory graph and audit trail
        |
        +--> PostgreSQL + pgvector
        +--> Ollama embeddings and local LLM extraction
```

## Требования

- Rust toolchain с поддержкой edition 2024.
- PostgreSQL 17 с `pgvector`.
- Ollama для semantic embeddings, extraction, validation и optional rerank.
- macOS или другая Unix-like среда.
- Опционально `rust-analyzer` для более глубокого анализа Rust-кода.

Локальные модели по умолчанию:

| Роль | Значение |
| --- | --- |
| Ollama base URL | `http://127.0.0.1:11435` |
| Memory embeddings | `qwen3-embedding:8b` |
| Fast code embeddings | `bge-m3` |
| Extraction and validation | `qwen3:14b` |

Keyword search, listing, review, export/import и многие operational commands
работают без Ollama. Явный semantic search требует embeddings.

## Типовые Сценарии

- Дать Codex или другому coding agent долговременную память проекта.
- Построить local-first RAG layer для software engineering tasks.
- Хранить architecture decisions и project rules с поиском по репозиторию.
- Получать task-scoped context из memory, code symbols и graph facts.
- Аудировать, review, promote, supersede, archive, backup, export и import
  agent memory.
- Запускать локальный semantic search по проектной памяти через PostgreSQL,
  pgvector и Ollama.

## Документация

| Документ | Что внутри |
| --- | --- |
| [Architecture](docs/MEMORY_ARCHITECTURE.md) | Memory lifecycle, retrieval, graph storage, MCP surfaces, safety policy, schema evolution |
| [Migrations](migrations/) | Ordered PostgreSQL schema migrations |
| [Eval suite](eval/dukememory.json) | Retrieval и behavior regression cases |
| [Scripts](scripts/) | Local PostgreSQL, Codex hook, install и Ollama forwarding helpers |
| [LaunchAgent example](launchd/com.dukememory.tailscale-ollama-forwarder.plist) | Пример macOS service для local Ollama forwarding |

## Статус Проекта

Dukememory находится на ранней стадии активной разработки. PostgreSQL store,
MCP server, CLI, code index, graph layer, audits, evals и native viewer уже
реализованы, но API и schema details могут измениться до стабильного релиза.

## Ключевые Слова Для Поиска

память AI-агента, память Codex, MCP сервер, local-first RAG, developer agent
memory, semantic search, vector search, pgvector, PostgreSQL, Ollama embeddings,
code graph, code intelligence, project memory, долговременная память для
AI-агентов.
