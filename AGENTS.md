# Project Agent Entrypoint

This file is the shared root entrypoint for AI assistants in MyNebula.

## Working rules

- Keep changes minimal, explicit, and verifiable.
- Preserve snapshot-backed reads, persisted pipeline state, and single-user
  runtime assumptions unless a task explicitly changes them end to end.
- Visual language is Vercel Geist in `DESIGN.md` / `frontend/DESIGN.md`.
  In-app chrome uses 6px squares and hairlines, not marketing pills.

## Commands

```bash
uv run pytest -q -m "not integration"
uv run ruff check src tests scripts alembic
pnpm --prefix frontend test
pnpm --prefix frontend lint
pnpm --prefix frontend exec tsc --noEmit
```

Frontend package manager is pnpm (`frontend/pnpm-lock.yaml`).
Backend package manager is uv.

Internet-facing flags (`FORCE_SECURE_COOKIES`, `HTTPS_REDIRECT`, `TRUSTED_HOSTS`)
refuse `READ_ACCESS_MODE=demo` unless `ALLOW_ANONYMOUS_DEMO=true`.
