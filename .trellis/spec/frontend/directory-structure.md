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
| Business query hook | `frontend/src/features/<feature>/hooks/` |
| Auth/graph shared context | `frontend/src/contexts/` |
| Shared search/format helper | `frontend/src/utils/` |
| Locale text | `frontend/src/locales/` |

## Rules

- Keep Dashboard, Data, Graph, and Settings boundaries clear.
- Keep graph filtering helpers out of render-heavy components.
- Do not create duplicate API clients for the same v2 contract.
- Keep URL-state helpers close to pages that own query params.
