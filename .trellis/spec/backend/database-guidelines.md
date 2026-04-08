# Database Guidelines

## Overview

The project uses SQLAlchemy 2 async ORM with Alembic migrations and PostgreSQL extensions, including `pgvector`.

## Query Patterns

- Always scope user-owned records by `user_id`.
- Prefer batch reads over per-row ORM lookups in loops.
- For semantic search, keep database ordering on `embedding.cosine_distance(...)`; do not fetch all embeddings into Python for ranking.
- Snapshot-oriented reads should query the active snapshot tables instead of reconstructing graph state ad hoc.

## Transactions And Task State

- Background jobs must persist status transitions (`pending`, `running`, `completed`, `failed`, `partial_failed`) in the database.
- Launch serialization uses PostgreSQL advisory locks for full refresh creation, pipeline creation, and scheduler ticks.
- If an async workflow fails after partially updating state, the terminal error must still be committed so the UI can report it.

## Migrations

- Migrations live in `alembic/versions`.
- New migrations must be additive and idempotent where practical.
- pgvector schema changes should explicitly preserve the `vector` extension and any ANN index requirements.
- Semantic-search-related migrations must mention whether they change runtime query assumptions.

## Naming Conventions

- Table/index/constraint naming follows the SQLAlchemy naming convention in [`src/nebula/db/models.py`](/Users/simonsun/.codex/worktrees/bcde/MyNebula/src/nebula/db/models.py).
- Composite user scoping indexes should start with `ix_<table>_user_...`.
- ANN indexes should include the embedding column and distance family in the name, for example `ix_starred_repos_embedding_cosine_ann`.

## Common Mistakes To Avoid

- Do not return unscoped records and filter them in Python.
- Do not materialize full candidate sets in memory when pgvector ordering can do the first-stage narrowing in SQL.
- Do not add migrations without a matching test or assertion for the expected query/index contract.
