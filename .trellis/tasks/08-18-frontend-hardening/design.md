# Design — Frontend decomposition, graph perf, i18n and a11y baseline

## Scope boundary

`frontend/src/` only, plus `.trellis/spec/frontend/`. No backend file, no API
contract, no Tailwind config, no design token change.

## Guiding rule for E1

Split along the seam that already exists in the code, not along line counts. The
four files are overweight for three different reasons, so they get three
different treatments.

### `Graph2D.tsx` (910 → target < 400)

Existing seams, read from the file's own structure:

| Concern | Current location | Destination |
| --- | --- | --- |
| Pure geometry and layout math (`clusterLayoutData`, hull signature, position scaling) | inline `useMemo`s, lines 135–225 | extend `graph2dUtils.ts` (already exists, already tested) |
| Canvas painting (`paintNode`, `paintNodeArea`, `drawClusterHulls`) | lines 446–663 | `graph2dPainters.ts` — pure functions taking `(ctx, node, scale, theme)` |
| Colour resolution (`getNodeColor`, `getLinkColor`, `getLinkWidth`) | lines 360–445 | `graph2dStyles.ts` — pure, directly testable |
| Avatar image cache and redraw throttling | `imageCache` ref, `triggerAvatarRedraw`, lines 93–133 | `useAvatarImageCache.ts` hook |
| Auto-fit, focus-by-id, user-interaction tracking | lines 334–358, 664–732 | `useGraphViewport.ts` hook |
| Force simulation configuration | lines 226–333 | `useGraphForces.ts` hook |
| Remaining: JSX, hover/click handlers, wiring | — | stays in `Graph2D.tsx` |

The painting and colour functions are the highest-value extraction: they are
pure, currently untestable because they close over component state, and they are
where graph visual regressions originate.

A `useCallback` that closes over component state does not become pure by moving
files. Each extraction must take its dependencies as explicit parameters. If a
function cannot be made parameter-explicit without contortion, it stays.

### `Settings.tsx` (821 → target < 400)

Presentational extraction already happened — `pages/settings/` holds
`SettingsLoginForm`, `SettingsAppearance`, `SettingsSchedule`,
`SettingsDataSection`, plus `polling.ts` and `progress.ts`. What remains is 17
`useState` hooks and the orchestration between them.

The fix is therefore hook extraction, not further component extraction:

| Concern | Destination |
| --- | --- |
| Login form state, submit, error, `adminAuthConfigured` probing | `useSettingsAuth.ts` |
| Schedule load/save state | `useSettingsSchedule.ts` |
| Sync/refresh/recluster triggering, confirm dialog, progress steps, polling lifecycle | `useSettingsSyncControls.ts` |

The polling lifecycle is the delicate part: `pages/__tests__/settings.polling.test.tsx`
and `Settings.partial-failed.test.tsx` assert on it. Those tests are the
contract. If either needs an assertion change, the refactor broke behaviour.

### `CommandPalette.tsx` (784 → target < 400)

| Concern | Destination |
| --- | --- |
| Local filtering plus semantic-search fallback orchestration | `useCommandPaletteResults.ts` |
| Keyboard navigation and focus management | `useCommandPaletteKeyboard.ts` |
| Result row rendering per result kind | `CommandPaletteResultList.tsx` |

`utils/search.ts` is already shared with the Data page and must stay shared — the
extraction must not fork the search semantics that
`.trellis/spec/frontend/query-and-filtering.md` treats as a cross-page contract.

### `DataPage.tsx` (735 → target < 400)

| Concern | Destination |
| --- | --- |
| URL search-param state sync | `useDataPageUrlState.ts` |
| Filter and sort state | `useDataPageFilters.ts` |
| Table rendering | `DataRepoTable.tsx` |

`pages/__tests__/DataPage.url-state.test.tsx` pins the URL-state behaviour and
must pass unmodified.

## E2 — Adjacency index placement

### Current

```tsx
export const useNodeNeighbors = (nodeId) => {
  const { rawData } = useGraph();
  const adjacencyIndex = useMemo(() => { /* iterate every edge */ }, [rawData]);
  return useMemo(() => new Set(adjacencyIndex.get(nodeId) ?? []), [adjacencyIndex, nodeId]);
};
```

The `useMemo` lives in whichever component calls the hook. Three consequences:

1. N consumers means N copies of the same map.
2. Progressive edge loading changes `rawData` on every page, so each copy is
   rebuilt per page, per consumer.
3. The memo dependency is `rawData`, which also changes when nodes change, so
   node-only updates rebuild the edge index unnecessarily.

### Target

Build once in `GraphProvider`, alongside the existing `edgeFilterIndexes`
memo — which already iterates `stagedEdges`, so the traversal cost may be
shareable:

```tsx
const adjacencyIndex = useMemo(() => buildAdjacencyIndex(stagedEdges), [stagedEdges]);
```

Expose it on `GraphContextValue`, and reduce the hook to:

```tsx
export const useNodeNeighbors = (nodeId) => {
  const { adjacencyIndex } = useGraph();
  return useMemo(() => new Set(adjacencyIndex.get(nodeId) ?? []), [adjacencyIndex, nodeId]);
};
```

`buildAdjacencyIndex` goes in `contexts/graphFiltering.ts` next to the existing
`buildGraphEdgeIndex`, and gets a direct unit test.

Note the dependency narrows from `rawData` to `stagedEdges`, so node-only
changes no longer invalidate it. That is a behaviour-preserving improvement:
adjacency is a function of edges alone.

The edge source/target normalisation (`typeof edge.source === 'object' ?
edge.source.id : edge.source`) must be preserved — `react-force-graph` mutates
edge endpoints from ids into node objects after simulation, and dropping that
branch silently breaks neighbour lookup after the first render.

`GraphContextValue`'s existing giant `useMemo` dependency array gains one entry.

## E3 — i18n parity guard

Add `repoDetails.similar` to `en`. Then extend
`locales/zh/translation.test.ts` with a symmetric key-set comparison that
flattens both bundles and asserts set equality, reporting the offending keys in
the failure message. A test that only says "bundles differ" costs the next
person a manual diff.

## E4 — Accessibility baseline

### What the baseline is

Documented in `.trellis/spec/frontend/`, scoped to what this codebase can hold:

1. Every interactive control has an accessible name — visible text, or
   `aria-label` when icon-only.
2. Modal surfaces have `role="dialog"`, `aria-modal="true"`, an accessible name,
   focus moved in on open and restored on close, focus trapped while open, and
   Escape to close.
3. Async status regions that already exist — `SyncProgress`, error and loading
   states — announce via `aria-live`.
4. Icon-only buttons never rely on colour or position alone to convey meaning.

Deliberately out of scope, and recorded as such so the omission reads as a
decision: full colour-contrast auditing, screen-reader transcript testing, and
`react-force-graph`'s canvas — a canvas-rendered force graph has no accessible
node tree, and giving it one is a separate product-level project.

### Automation

Prefer `jest-axe` (or `vitest-axe`) over hand-written assertions: it is dev-only,
integrates with the existing Vitest + Testing Library setup, and encodes rules
rather than one-off checks. Smoke a11y assertions go on the top-level rendered
surfaces already covered by smoke tests — `dashboard.v2.smoke`, `data.v2.smoke`
— plus the two modal surfaces.

Automated rules catch roughly a third of real accessibility problems. The
baseline document must say so, or the green check will be read as a guarantee it
is not.

## Ordering within the task

E3 first — it is a two-line fix plus a test, and gets a guard in place early.
Then E2, which is small, self-contained, and touches the file E1's graph work
depends on. Then E4's modal semantics, since `CommandPalette` gets focus
management before it is split. Then E1, the largest and riskiest, informed by
the tests the earlier steps added.

## Risks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Extracting `useCallback`s from `Graph2D` changes render identity and retriggers the force simulation | High | Graph visibly re-lays-out or flickers | Extract pure functions with explicit parameters first; move hooks only where dependency arrays can be preserved exactly; verify against the Playwright graph flow, not only unit tests |
| Settings polling lifecycle breaks during hook extraction | High | Sync progress stalls or double-polls | `settings.polling.test.tsx` and `Settings.partial-failed.test.tsx` must pass unmodified; treat any needed assertion change as a regression |
| Narrowing the adjacency memo to `stagedEdges` misses a case where nodes matter | Low | Stale neighbours | Adjacency is a pure function of edges; covered by a direct unit test on `buildAdjacencyIndex` |
| Losing the object-or-id edge endpoint normalisation | Medium | Neighbour highlight breaks after simulation starts | Explicitly called out above; unit test covers both endpoint shapes |
| Focus trapping conflicts with the existing `useCommandPalette` open/close hook | Medium | Keyboard trap or unfocusable modal | Add focus management before splitting the component, so one change is verified at a time |
| A file lands just under 400 LOC via cosmetic moves | Medium | Debt relabelled rather than repaid | Acceptance requires extracted pure functions to have direct tests; untested extraction does not count |
