# Taxonomy v1 Implementation Plan

Date: 2026-02-08
Branch: codex/taxonomy-plan

## Goal
Build a data-driven taxonomy pipeline that improves clustering and naming consistency across users while keeping online clustering stable.

## Task 1 - Add taxonomy data models and migration
- Add SQLAlchemy models for taxonomy versions, terms, mappings, candidates, and user overrides.
- Export new models in `src/nebula/db/__init__.py`.
- Create Alembic migration for all new tables and indexes.
- Verification:
  - `python3 -m py_compile src/nebula/db/models.py`
  - migration file imports compile cleanly.

## Task 2 - Add taxonomy service and candidate generation primitives
- Create `src/nebula/core/taxonomy.py` with:
  - token normalization
  - pairwise candidate scoring (co-occurrence + lexical + embedding similarity hooks)
  - confidence bands (high/mid/low)
  - materialize candidate rows for persistence
- Keep implementation deterministic and provider-agnostic.
- Verification:
  - `python3 -m py_compile src/nebula/core/taxonomy.py`

## Task 3 - Integrate taxonomy normalization into embedding + clustering flow
- Add normalization lookup utility that maps raw tags using active taxonomy mappings.
- Apply it in embedding text construction and cluster naming input path.
- Keep fallback behavior if no taxonomy version exists.
- Verification:
  - `python3 -m py_compile src/nebula/core/embedding.py src/nebula/api/sync.py src/nebula/core/clustering.py`

## Task 4 - Add offline taxonomy refresh task
- Add background task that:
  - scans repo tags for a user
  - generates candidates
  - stores a draft taxonomy version + candidate records
- Do not auto-publish user overrides.
- Verification:
  - `python3 -m py_compile src/nebula/api/sync.py src/nebula/core/taxonomy.py`

## Task 5 - Add API endpoints for review/publish
- Add endpoints:
  - list taxonomy versions
  - list candidates by confidence
  - publish version
- Enforce safe defaults and idempotent publish behavior.
- Verification:
  - `python3 -m py_compile src/nebula/api/sync.py`

## Task 6 - Add tests for taxonomy logic
- Add unit tests for normalization, confidence tiering, and candidate deduplication.
- Verification:
  - `pytest tests/test_taxonomy.py` (or compile fallback if pytest unavailable)

## Task 7 - Add observability fields
- Add run statistics for candidate generation and publish outcomes.
- Log version id, candidate counts, and confidence distribution.
- Verification:
  - `python3 -m py_compile src/nebula/core/taxonomy.py src/nebula/api/sync.py`

## Task 8 - Documentation
- Add operator notes in `doc/ENV_VARS.md` and create `doc/TAXONOMY.md`.
- Verification:
  - manual review for clarity and consistency.
