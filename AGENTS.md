# Project Agent Entrypoint

This file is the shared root entrypoint for AI assistants in MyNebula.

## Shape

A static site plus an offline pipeline, one TypeScript package on Node 24 (pnpm).

- `pipeline/` runs directly with `node` (type stripping, no build, no runtime
  deps). `pnpm sync` fetches stars, writes optional LLM summaries/tags and
  embeddings for new or changed repos, links related repos, groups them as
  Louvain communities (ids matched to the previous run), and writes
  `frontend/public/data/graph.json` + `related.json`. State lives in
  `data/cache.json.gz`; reruns must stay incremental.
- `frontend/` (Vite, vanilla TypeScript, d3-force): Graph (home), Data,
  Settings, and the Cmd/Ctrl+K palette. Hash routing so any static host works.
  The canvas is `frontend/src/graph/canvas.ts`: layout is a live force
  simulation on the edges with a gentle drift (paused in hidden tabs, off under
  reduced motion). Colour and visibility rules are pure functions in
  `graph/style.ts`.
- `.github/workflows/pages.yml` runs the sync daily and deploys to Pages.

## Working rules

- Keep changes minimal, explicit, and verifiable.
- Pipeline code must stay erasable TypeScript (no enums, namespaces, or
  parameter properties) because Node strips types instead of compiling.
- Keep the frontend dependency-light: no UI framework or component library.
- Templates go through `html` in `frontend/src/html.ts`, which escapes every
  interpolation. Write conditional markup as `cond ? html`…` : ''`, never
  `cond && …`, because booleans render as "true"/"false".
- Graph, Data page, and palette search share `frontend/src/search.ts`.
- Visual language is Vercel Geist in `DESIGN.md`:
  hairlines over cards, 6px controls, one ink plus cluster hues.

## Commands

```bash
pnpm lint    # tsc over pipeline + frontend
pnpm test    # vitest
pnpm build
```
