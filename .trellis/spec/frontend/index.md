# Frontend Development Guidelines

## Overview

MyNebula frontend is a React + TypeScript SPA built with Vite. The current app shape has four important data surfaces:

- Dashboard: summary and trend views
- Data: tabular repository exploration
- Graph: snapshot-backed visualization with progressive edge loading
- Settings: admin-only operational controls

## Current Conventions

- Query data comes from `frontend/src/api/v2/*` and is consumed through feature hooks.
- Graph page owns the active snapshot view through `GraphContext`; other pages should not fetch graph payloads just to recover lightweight metadata.
- Search semantics for Graph, Data, and Command Palette must stay aligned through shared utilities, not hand-written per component.
- Settings polling must be abortable and tied to component lifecycle or auth/session changes.

## Non-Negotiable Rules

- Do not add duplicate snapshot fetches to Data or Dashboard when a lighter API contract can carry the same metadata.
- Do not fork search matching rules across pages. Shared normalization and field matching live in `frontend/src/utils/search.ts`.
- Do not put expensive graph filtering logic directly inside render-heavy components; keep it in memoized context helpers or pure utility modules.
- Do not change `GraphContext` public shape casually. Optimize internals first to avoid wide component churn.

## Source Of Truth

- Dashboard query shaping: [`frontend/src/features/dashboard/hooks/useDashboardQuery.ts`](/Users/simonsun/.codex/worktrees/bcde/MyNebula/frontend/src/features/dashboard/hooks/useDashboardQuery.ts)
- Data page repository query: [`frontend/src/features/data/hooks/useDataReposQuery.ts`](/Users/simonsun/.codex/worktrees/bcde/MyNebula/frontend/src/features/data/hooks/useDataReposQuery.ts)
- Shared search semantics: [`frontend/src/utils/search.ts`](/Users/simonsun/.codex/worktrees/bcde/MyNebula/frontend/src/utils/search.ts)
- Graph filter derivation: [`frontend/src/contexts/graphFiltering.ts`](/Users/simonsun/.codex/worktrees/bcde/MyNebula/frontend/src/contexts/graphFiltering.ts)
- Settings polling lifecycle: [`frontend/src/pages/settings/polling.ts`](/Users/simonsun/.codex/worktrees/bcde/MyNebula/frontend/src/pages/settings/polling.ts)
