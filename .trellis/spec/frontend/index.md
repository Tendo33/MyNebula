# Frontend Development Guidelines

## Overview

MyNebula frontend is a React + TypeScript SPA built with Vite. The current app
shape has four important data surfaces:

- Dashboard: summary and trend views.
- Data: tabular repository exploration.
- Graph: snapshot-backed visualization with progressive edge loading, timeline,
  and detail sidebar.
- Settings: admin login, sync, scheduled jobs, and operational controls.

## Current Organization

- `frontend/src/api`: HTTP clients and v2 API adapters.
- `frontend/src/components`: graph, layout, and UI components.
- `frontend/src/contexts`: auth, graph context, and graph filtering.
- `frontend/src/features`: business query hooks.
- `frontend/src/pages`: page-level containers.
- `frontend/src/stores`: client state.
- `frontend/src/utils`: search, formatting, and shared utilities.
- `frontend/src/types`: API and graph types.

## Current Conventions

- Query data comes from `frontend/src/api/v2/*` and is consumed through feature
  hooks.
- Graph page owns the active snapshot view through `GraphContext`; other pages
  should not fetch graph payloads just to recover lightweight metadata.
- Search semantics for Graph, Data, and Command Palette must stay aligned
  through `frontend/src/utils/search.ts`.
- Graph/Data page search uses shared local filtering; Command Palette may call
  backend semantic search when local repo results are missing.
- Settings polling must be abortable and tied to component lifecycle and
  auth/session changes.
- Progressive edge loading should switch from automatic prefetch to explicit
  user continuation after the threshold.
- Settings full refresh `partial_failed` must preserve warning semantics; do not
  render it as a complete success.
- Settings step mapping belongs in `frontend/src/pages/settings/progress.ts`.
- Frontend package management is pnpm with `frontend/pnpm-lock.yaml`.

## Non-Negotiable Rules

- Do not add duplicate snapshot fetches to Data or Dashboard when a lighter API
  contract can carry the same metadata.
- Do not fork search matching rules across pages.
- Do not put expensive graph filtering logic directly inside render-heavy
  components; keep it in memoized context helpers or pure utility modules.
- Do not change `GraphContext` public shape casually. Optimize internals first
  to avoid wide component churn.

## Source Of Truth

- Dashboard query shaping: `frontend/src/features/dashboard/hooks/useDashboardQuery.ts`
- Data page repository query: `frontend/src/features/data/hooks/useDataReposQuery.ts`
- Shared search semantics: `frontend/src/utils/search.ts`
- Graph filter derivation: `frontend/src/contexts/graphFiltering.ts`
- Settings polling lifecycle: `frontend/src/pages/settings/polling.ts`
