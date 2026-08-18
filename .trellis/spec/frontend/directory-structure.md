# Frontend Directory Structure

Use this when adding frontend files under `frontend/`.

## Current Shape

```text
frontend/src/
├── api/          # HTTP clients and v2 adapters
├── components/   # Graph, layout, and UI components
├── contexts/     # Auth, graph, and filtering context
├── features/     # Business query hooks
├── hooks/        # Shared hooks
├── lib/          # Shared frontend utilities
├── locales/      # English and Chinese resources
├── pages/        # Dashboard, Data, Graph, Settings containers
├── stores/       # Client state
├── test/         # Test setup
├── types/        # API and graph types
└── utils/        # Search, formatting, and shared helpers
```

## Placement Rules

| New thing | Default location |
| --- | --- |
| API helper or v2 adapter | `frontend/src/api/` |
| Reusable UI primitive | `frontend/src/components/ui/` |
| Graph component | `frontend/src/components/graph/` |
| Page container | `frontend/src/pages/` |

## Component Decomposition

Page containers and canvas components are wiring. Keep each under 400 lines by
extracting along real seams, not by slicing at a line count:

- **Pure computation** into a sibling module. Every dependency becomes an
  explicit parameter — a `useCallback` that closes over component state is not
  pure just because it moved file.
- **Stateful orchestration** into a `hooks/` module next to its consumer.
- **Markup** into a presentational subcomponent.

A pure function extracted from a component **must** get a direct unit test.
That is the point of extracting it; an untested extraction has only relabelled
the debt.

Current shape of the decomposed surfaces:

```text
components/graph/
├── Graph2D.tsx             # wiring only
├── graph2dTypes.ts         # shared types and layout constants
├── graph2dLayout.ts        # pure: snapshot payload -> force-graph input
├── graph2dStyles.ts        # pure: node/link colour and width rules
├── graph2dPainters.ts      # pure: canvas painting and hull drawing
├── graph2dUtils.ts         # pure: geometry and node radius
├── GraphHoverCard.tsx      # presentational hover overlay
└── hooks/                  # avatar cache, viewport, forces

components/ui/
├── CommandPalette.tsx          # keyboard nav and markup
├── CommandPaletteResultList.tsx
├── commandPaletteTypes.ts
├── commandPaletteFacets.ts     # pure: language/tag facet counting
└── hooks/                      # results, recent searches, dialog focus trap

pages/settings/
├── hooks/                  # auth, schedule, sync controls
├── polling.ts
└── progress.ts

pages/data/
├── DataRepoTable.tsx
├── DataTableParts.tsx
├── dataPageFilters.ts      # pure: URL param parse/build, sort, formatting
└── hooks/useDataPageUrlState.ts
```

When moving a `useCallback` or `useEffect` into a hook, keep its dependency
array byte-identical. In `Graph2D` a changed dependency array retriggers the
force simulation and visibly re-lays-out the graph, which is a behaviour change
rather than a refactor.

`useDialogFocusTrap` is the shared focus trap. Use it rather than
reimplementing a Tab trap per dialog; the accessibility baseline test accepts it
in place of an inline trap, and its behaviour is covered by
`components/ui/hooks/__tests__/useDialogFocusTrap.test.tsx`.
| Business query hook | `frontend/src/features/<feature>/hooks/` |
| Auth/graph shared context | `frontend/src/contexts/` |
| Shared search/format helper | `frontend/src/utils/` |
| Locale text | `frontend/src/locales/` |

## Rules

- Keep Dashboard, Data, Graph, and Settings boundaries clear.
- Keep graph filtering helpers out of render-heavy components.
- Do not create duplicate API clients for the same v2 contract.
- Keep URL-state helpers close to pages that own query params.
