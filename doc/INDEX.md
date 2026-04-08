# Documentation Map

This directory is the operational documentation hub for MyNebula.

## Start Here

- [README.md](/Users/simonsun/.codex/worktrees/bcde/MyNebula/README.md): product overview, quick start, API overview, local development
- [README.zh.md](/Users/simonsun/.codex/worktrees/bcde/MyNebula/README.zh.md): Chinese overview and quick start
- [ENV_VARS.md](/Users/simonsun/.codex/worktrees/bcde/MyNebula/doc/ENV_VARS.md): environment variables and deployment-sensitive settings
- [QUALITY_GATES.md](/Users/simonsun/.codex/worktrees/bcde/MyNebula/doc/QUALITY_GATES.md): local/CI verification gates and when to run offline evals

## Deployment And Operations

- [DOCKER_DEPLOY.md](/Users/simonsun/.codex/worktrees/bcde/MyNebula/doc/DOCKER_DEPLOY.md): Docker Compose deployment, migration behavior, and proxy/auth notes
- [RESET_GUIDE.md](/Users/simonsun/.codex/worktrees/bcde/MyNebula/doc/RESET_GUIDE.md): database reset and recovery flows
- [PRE_COMMIT_GUIDE.md](/Users/simonsun/.codex/worktrees/bcde/MyNebula/doc/PRE_COMMIT_GUIDE.md): lightweight git hook checks

## Architecture And Data Contracts

- [MODELS_GUIDE.md](/Users/simonsun/.codex/worktrees/bcde/MyNebula/doc/MODELS_GUIDE.md): ORM models, API schemas, and migration boundaries
- [.trellis/spec/backend/index.md](/Users/simonsun/.codex/worktrees/bcde/MyNebula/.trellis/spec/backend/index.md): backend conventions
- [.trellis/spec/frontend/index.md](/Users/simonsun/.codex/worktrees/bcde/MyNebula/.trellis/spec/frontend/index.md): frontend conventions
- [.trellis/spec/frontend/query-and-filtering.md](/Users/simonsun/.codex/worktrees/bcde/MyNebula/.trellis/spec/frontend/query-and-filtering.md): shared query/search/filtering rules

## Programmatic Usage And Remediation History

- [SDK_USAGE.md](/Users/simonsun/.codex/worktrees/bcde/MyNebula/doc/SDK_USAGE.md): supported Python-level entry points
- [REMEDIATION_BACKLOG.md](/Users/simonsun/.codex/worktrees/bcde/MyNebula/doc/REMEDIATION_BACKLOG.md): completed remediation tasks and verification commands

## Current State Snapshot

The current codebase reflects these project-level decisions:

- admin auth and read-mode boundaries are active and proxy-aware
- pipeline/full-refresh/scheduler states are unified around `pending/running/completed/failed/partial_failed`
- schedule timestamps are timezone-aware UTC on the backend
- Data and Dashboard avoid redundant graph snapshot fetches where lighter contracts are enough
- Graph, Data, and Command Palette share the same literal search semantics
