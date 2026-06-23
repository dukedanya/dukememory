# Security Policy

Dukememory is local-first software for storing agent memory, retrieval metadata,
code context, and audit events. Treat it as part of your local development
environment, not as a hardened multi-tenant hosted service.

## Supported Versions

| Version | Supported |
| --- | --- |
| `main` | Best effort |
| `0.1.x` | Best effort |

## Reporting a Vulnerability

Open a private GitHub security advisory if available, or contact the repository
owner through GitHub. Please include:

- affected commit or release;
- reproduction steps;
- expected impact;
- whether local files, memory rows, embeddings, hooks, or MCP transport are
  involved.

Do not publish exploit details before there is time to assess and patch.

## Security Model

Dukememory assumes:

- a single-user local workstation by default;
- local PostgreSQL with `pgvector`;
- local Ollama models;
- project-scoped memory lookup;
- explicit user intent for cross-project access.

Important safety properties:

- new memory writes pass a local safety policy before insertion;
- obvious secret values are blocked before memory writes;
- possible personal data can be reported as warning-level findings;
- automatic extraction writes `pending` memories by default;
- MCP tools accept `project_path` so clients can bind reads to the active
  workspace root;
- temporal graph facts can be invalidated without deleting audit history.

## Non-Goals

Dukememory does not currently claim:

- multi-tenant isolation;
- hardened network exposure;
- complete secret detection;
- cryptographic protection of local database contents;
- safety guarantees for untrusted model output.

Keep the database, generated backups, exported JSON files, and Codex hook
configuration under the same access controls as your source code.
