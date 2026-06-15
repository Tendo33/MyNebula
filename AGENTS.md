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

<!-- TRELLIS:START -->
# Trellis Instructions

These instructions are for AI assistants working in this project.

This project is managed by Trellis. The working knowledge you need lives under `.trellis/`:

- `.trellis/workflow.md` — development phases, when to create tasks, skill routing
- `.trellis/spec/` — package- and layer-scoped coding guidelines (read before writing code in a given layer)
- `.trellis/workspace/` — per-developer journals and session traces
- `.trellis/tasks/` — active and archived tasks (PRDs, research, jsonl context)

If a Trellis command is available on your platform (e.g. `/trellis:finish-work`, `/trellis:continue`), prefer it over manual steps. Not every platform exposes every command.

If you're using Codex or another agent-capable tool, additional project-scoped helpers may live in:
- `.agents/skills/` — reusable Trellis skills
- `.codex/agents/` — optional custom subagents

Managed by Trellis. Edits outside this block are preserved; edits inside may be overwritten by a future `trellis update`.

<!-- TRELLIS:END -->
