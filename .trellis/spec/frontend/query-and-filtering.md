# Query and Filtering Rules

## Page-Level Data Contracts

- Dashboard should prefer `/api/v2/dashboard` for summary cards, top languages, top topics, and top clusters.
- Dashboard may independently load `/api/v2/graph/timeline` for activity charts, but should not fetch full graph payload just to rebuild language/topic aggregates.
- Data page should use `/api/v2/data/repos` as its single source of repository rows and cluster badge/filter metadata.
- Graph page remains the only place that owns the full snapshot payload and paged edge loading lifecycle.

## Shared Search Semantics

- Normalize search with lowercase + trim before matching or sending `q`.
- Shared repo search fields:
  - `name`
  - `full_name`
  - `description`
  - `ai_summary`
  - `language`
  - `ai_tags`
  - `topics`
- `stars:>N` is a first-class query and must be handled consistently in Graph filtering, Command Palette repo results, and Data page backend filtering.
- Cluster search matches `name`, `description`, and `keywords`.

## Graph Filtering

- Build graph filter indexes only when raw snapshot data changes.
- Derive filtered graph state in stages:
  - visible nodes
  - visible node id set
  - visible edges
  - visible clusters
- Preserve ghost-node behavior and existing URL-driven node/cluster selection semantics while optimizing internals.
