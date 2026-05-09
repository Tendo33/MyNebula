# Project Docs

MyNebula uses `.trellis/spec/` as the only detailed AI-facing project contract.

## Rules

- Current implementation facts live in the relevant backend/frontend/shared
  spec file.
- Verification commands and hotspot checks live in `shared/verification.md`.
- Root files such as `AGENTS.md` and `CLAUDE.md` stay thin and point into
  `.trellis/spec/`.
- README files may summarize project usage, but AI-facing implementation rules
  belong in `.trellis/spec/`.
- Do not recreate a parallel documentation system.
