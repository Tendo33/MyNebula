# Project Agent Entrypoint

This file is the shared root entrypoint for AI assistants in MyNebula.

## Working rules

- Keep changes minimal, explicit, and verifiable.
- Preserve snapshot-backed reads, persisted pipeline state, and single-user
  runtime assumptions unless a task explicitly changes them end to end.
