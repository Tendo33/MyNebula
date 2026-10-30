# Claude Code Project Instructions

This file is Claude Code's root entrypoint for MyNebula.

## Read order

1. Start at [AGENTS.md](AGENTS.md)

## Project guardrails

- MyNebula turns GitHub Stars into a semantic knowledge graph, published as a
  static site built by an offline TypeScript pipeline.
- Keep the pipeline incremental: summaries and embeddings are cached in
  `data/cache.json.gz` and only new or changed repos may hit paid APIs.
- Keep the JSON contract (`graph.json`, `related.json`) in sync between
  `pipeline/graph.ts` and `frontend/src/types.ts`.
- Frontend work must preserve Graph, Data, and Settings flows, URL state on
  Graph and Data, the command palette, and en/zh locales.
- One package, pnpm, `pnpm-lock.yaml` at the repo root.

## Commands

```bash
pnpm lint
pnpm test
pnpm build
```

## Claude execution style

- State assumptions explicitly when they shape the solution.
- Keep diffs tightly scoped to the task.
- Match existing style even when you would normally choose differently.
- Visual work follows DESIGN.md and PRODUCT.md; do not introduce a second palette.
