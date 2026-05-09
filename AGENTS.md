# Project Agent Entrypoint

This file is the shared root entrypoint for AI assistants in MyNebula.

## Read order

1. Start at [.trellis/spec/README.md](.trellis/spec/README.md)
2. Use [.trellis/spec/shared/index.md](.trellis/spec/shared/index.md) for repository-wide facts
3. Use [.trellis/spec/backend/index.md](.trellis/spec/backend/index.md) before backend work
4. Use [.trellis/spec/frontend/index.md](.trellis/spec/frontend/index.md) before frontend work
5. Use [.trellis/spec/shared/verification.md](.trellis/spec/shared/verification.md) before claiming completion

## Working rules

- Treat `.trellis/spec/` as the only detailed AI-facing project contract.
- Keep changes minimal, explicit, and verifiable.
- Preserve snapshot-backed reads, persisted pipeline state, and single-user
  runtime assumptions unless a task explicitly changes them end to end.
- Update Trellis specs whenever behavior, structure, scripts, public APIs, or
  verification commands change.
