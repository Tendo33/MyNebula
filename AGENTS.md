# Project Agent Entrypoint

This file is the shared root entrypoint for AI assistants in this repository.

## Read order

1. Start at [ai_docs/START_HERE.md](ai_docs/START_HERE.md)
2. Use [ai_docs/INDEX.md](ai_docs/INDEX.md) to choose the right task path
3. Use [ai_docs/reference/verification.md](ai_docs/reference/verification.md) before claiming completion

## Working rules

- Treat `ai_docs/` as the only detailed project documentation source of truth
- Read `current/` before `standards/`, and use `reference/` for shared commands or paths
- Keep changes minimal, explicit, and verifiable
- Update docs whenever behavior, structure, scripts, adapters, or public APIs change

<!-- TRELLIS:START -->
# Trellis Instructions

These instructions are for AI assistants working in this project.

Use the `/trellis:start` command when starting a new session to:
- Initialize your developer identity
- Understand current project context
- Read relevant guidelines

Use `@/.trellis/` to learn:
- Development workflow (`workflow.md`)
- Project structure guidelines (`spec/`)
- Developer workspace (`workspace/`)

Keep this managed block so 'trellis update' can refresh the instructions.

<!-- TRELLIS:END -->
