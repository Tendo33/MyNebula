# Frontend decomposition, graph perf, i18n and a11y baseline

## Goal

Reduce the maintenance risk concentrated in four oversized frontend modules,
remove a repeated O(E) computation from the graph render path, and establish the
i18n and accessibility baselines the project currently lacks.

This is finding group **E** of parent task
`.trellis/tasks/08-18-comprehensive-hardening`.

## Background

The frontend baseline is good: ESLint clean, `tsc --noEmit` clean, zero `any`,
zero `onClick` on non-interactive elements, feature-scoped React Query hooks, a
shared `graphFiltering` module, and 19 test files. The debt is structural, not
qualitative.

The 2026-05-09 health check flagged the oversized components and explicitly
deferred splitting `Graph2D.tsx` as out of scope for an automated fix. It has
not been picked up since.

## Requirements

### E1 — Decompose the four oversized modules

| File | LOC | Character of the bulk |
| --- | --- | --- |
| `components/graph/Graph2D.tsx` | 910 | canvas painting, hull geometry, avatar image cache, force config, auto-fit, focus, hover, click |
| `pages/Settings.tsx` | 821 | 17 `useState` hooks plus polling, login, schedule, refresh, and recluster orchestration |
| `components/ui/CommandPalette.tsx` | 784 | local filtering, semantic search fallback, keyboard navigation, result rendering |
| `pages/DataPage.tsx` | 735 | URL state, filters, sorting, table rendering |

These are not uniformly overweight. `Settings.tsx` has already had its
presentational parts extracted into `pages/settings/*`; what remains is state
and orchestration, so its fix is hook extraction, not more components.
`Graph2D.tsx` mixes pure computation, pure painting, and interaction, so its
seams are different.

Requirements:

- Each file drops below 400 LOC.
- Extractions must be along real seams — pure helpers into modules, stateful
  orchestration into hooks, rendering into subcomponents — not arbitrary
  line-count slicing.
- Rendered output and user-visible behaviour must be unchanged. This is a
  refactor, not a redesign.
- Newly extracted pure functions must get direct unit tests, which is the point
  of extracting them.

### E2 — Build the graph adjacency index once

`useNodeNeighbors` (`contexts/GraphContext.tsx:285`) builds a full
`Map<number, Set<number>>` over every edge, inside a `useMemo` that lives in the
**consuming component**. Every component calling the hook maintains its own copy,
and each rebuilds on every `rawData` change — which, with progressive edge
loading, is every edge page.

Requirement: compute the adjacency index once in `GraphProvider`, expose it
through context, and reduce `useNodeNeighbors` to a lookup. Neighbour results
must be identical.

### E3 — Close the i18n gap and prevent recurrence

`repoDetails.similar` exists in `locales/zh/translation.json` and not in
`locales/en/translation.json` — 206 keys versus 205.

Requirements:

- Add the missing English key.
- Add a test that fails when the two bundles' key sets diverge, so the next drift
  is caught at test time. `locales/zh/translation.test.ts` already exists and is
  the natural home.

### E4 — Establish an accessibility baseline

There are 70 `<button>` elements and sparse `aria-*` usage, concentrated in a few
files (`GraphPage.tsx` 10, `SyncProgress.tsx` 8, `Sidebar.tsx` 7) with several
interactive components at zero. `App.tsx` already ships a skip link and
`#main-content` target, so the foundation exists.

Requirements:

- Establish, and document, what "accessible enough" means for this project.
- Every interactive control must have an accessible name.
- Modal surfaces — `CommandPalette` and the Settings confirmation dialog — must
  have correct dialog semantics, focus trapping, and Escape handling.
- Add an automated check so the baseline holds, rather than a one-time sweep.

## Constraints

- Independent of children A–D; may proceed in parallel but is sequenced last to
  keep review linear.
- Preserve Dashboard, Data, Graph, and Settings flows, including React Query
  usage, `GraphContext`, shared search utilities, progressive edge loading, and
  the Settings polling lifecycle.
- No `any` in new TypeScript. No `@ts-expect-error` added to make a split
  typecheck.
- Frontend package management stays pnpm with `frontend/pnpm-lock.yaml`.
- No new runtime dependency for E1 or E2. E4 may add a dev-only a11y testing
  dependency if justified in `design.md`.
- No visual redesign. Tailwind class strings move with their markup; they are
  not rewritten.

## Non-goals

- Not migrating away from `react-force-graph-2d`, Zustand, or React Query.
- Not changing the graph's visual design, layout algorithm, or colour system.
- Not adding a component library or design-token system.
- Not a full WCAG 2.1 AA certification effort. E4 establishes a baseline and a
  guard, not an audit programme.
- Not touching backend contracts.

## Acceptance criteria

- [ ] `Graph2D.tsx`, `Settings.tsx`, `CommandPalette.tsx`, and `DataPage.tsx`
      are each under 400 LOC.
- [ ] Every function extracted as pure has at least one direct unit test.
- [ ] The adjacency index is built in `GraphProvider`; `useNodeNeighbors`
      performs a lookup only. A test asserts neighbour sets are unchanged.
- [ ] A test asserts the index is not rebuilt per consumer — for example, two
      components using the hook share one computation.
- [ ] `en` and `zh` bundles have identical key sets.
- [ ] A test fails when a key exists in one bundle and not the other.
- [ ] Every interactive control has an accessible name.
- [ ] `CommandPalette` and the Settings confirmation dialog have dialog
      semantics, focus trapping, and Escape-to-close, each covered by a test.
- [ ] An automated a11y check runs as part of `pnpm run test`.
- [ ] `pnpm --prefix frontend run lint`,
      `pnpm --prefix frontend exec tsc --noEmit -p tsconfig.json`,
      `pnpm --prefix frontend run test`, and
      `pnpm --prefix frontend run build` all pass.
- [ ] The existing 19 frontend test files pass **without modification**, except
      where a test imports a moved module and needs only an import-path change.
      Any test whose assertions must change is a behaviour regression and must be
      justified in the task notes.
- [ ] `.trellis/spec/frontend/` documents the new module boundaries and the a11y
      baseline.

## Verification

Frontend scope, plus the Graph / Search hotspot matrix from
`.trellis/spec/shared/verification.md`, plus the documentation link check
because spec files change. The Playwright `graph-sync-flow` spec must also pass,
since E1 and E2 touch the graph render path.

## Progress notes (2026-08-18)

### E2, E3, E4 complete. E1 not started.

**E3 — i18n.** The scan's "one key drift" understated the problem. A full
audit of `t()` call sites against both bundles found **26 keys used in code but
absent from `en`, 25 absent from `zh`**. Every one had an inline
`t('key', 'Fallback')` default, so English looked fine while the **Chinese UI
silently rendered English** for 25 strings — a real user-facing defect for a
bilingual product. All 26 added to both bundles with proper translations.
`locales/zh/translation.test.ts` now enforces bundle parity *and* scans `src/`
for keys missing from either bundle; verified to fail, with the offending key
named, when a key is removed.

**E2 — adjacency index.** `buildAdjacencyIndex` added to `graphFiltering.ts`
with 8 unit tests (id endpoints, object endpoints, mixed, self-referential,
duplicate edges). Built once in `GraphProvider`, memoized on `stagedEdges`
rather than `rawData`, and exposed on context; `useNodeNeighbors` reduced to a
lookup. A rendering test asserts two consumers observe the same index instance.

**E4 — accessibility. The PRD overstated this finding.** The `aria-` grep that
produced "sparse aria coverage" counted occurrences per file, not compliance.
Verified state before any change:

- `CommandPalette`: already had `role="dialog"`, `aria-modal`,
  `aria-labelledby`, and Escape. Missing: Tab focus trap and focus restore.
- Settings confirm dialog: already complete — focus on open, Tab trap, focus
  restore. It was the in-repo precedent the CommandPalette fix follows.
- Exactly **one** button in the codebase lacked an accessible name
  (`RepoDetailsPanel.tsx:312`), not a broad gap.
- `aria-live` already present on `SyncProgress`; `role="alert"` on
  `ErrorFallback`.

Changes made: Tab focus trap plus focus restore in `CommandPalette`;
`aria-label` on the `RepoDetailsPanel` close button; Escape handling for the
Settings confirm dialog moved out of `Settings.tsx` into
`SettingsDataSection.tsx` so the dialog owns its own dismissal alongside its
focus trap. Added `src/__tests__/accessibility.baseline.test.ts`, a static guard
over every `.tsx` (accessible names, dialog semantics, Escape, Tab trap),
verified to fail with the exact `file:line` when a label is removed.

No a11y dependency was added: the guards needed are structural, and a
dependency would have to earn its place.

**E1 — decomposition: not started.** `Graph2D.tsx` (910), `Settings.tsx` (818),
`CommandPalette.tsx` (821), `DataPage.tsx` (735) are unchanged in structure.
This is the largest and highest-risk item in the epic — roughly 3,300 LOC to be
split into ~15 modules while keeping rendered output identical — and it was not
attempted rather than half-done. `design.md` records the seams per file.

Frontend suite: 62 → 78 tests. `lint`, `tsc --noEmit`, and `build` all clean.

## E1 complete (2026-08-18)

All four targets are under 400 lines with rendered output unchanged.

| File | Before | After |
| --- | --- | --- |
| `Graph2D.tsx` | 910 | **296** |
| `CommandPalette.tsx` | 784 | **376** |
| `Settings.tsx` | 821 | **371** |
| `DataPage.tsx` | 735 | **341** |

Modules extracted, each along an existing seam:

- Graph: `graph2dTypes`, `graph2dLayout`, `graph2dStyles`, `graph2dPainters`,
  `GraphHoverCard`, `hooks/useAvatarImageCache`, `hooks/useGraphForces`,
  `hooks/useGraphViewport`.
- Command palette: `commandPaletteTypes`, `commandPaletteFacets`,
  `CommandPaletteResultList`, `hooks/useCommandPaletteResults`,
  `hooks/useRecentSearches`, `hooks/useDialogFocusTrap`.
- Settings: `hooks/useSettingsAuth`, `hooks/useSettingsSchedule`,
  `hooks/useSettingsSyncControls`.
- Data: `dataPageFilters`, `DataTableParts`, `DataRepoTable`,
  `hooks/useDataPageUrlState`.

Frontend suite: 62 → **164 tests**, 34 files. New direct unit tests cover
`graph2dStyles` (21), `graph2dPainters` (21), `graph2dLayout` (12),
`commandPaletteFacets` (10), `dataPageFilters` (17), `useDialogFocusTrap` (5),
`buildAdjacencyIndex` (8).

**Every pre-existing test file passed without modification**, verified by
`git diff --stat` over the pinned tests: `settings.polling.test.tsx`,
`Settings.partial-failed.test.tsx`, `DataPage.url-state.test.tsx`,
`CommandPalette.test.tsx`.

Two design decisions worth recording:

- `useDataPageUrlState` cannot own the page-clamp effect: `totalPages` derives
  from the query that the hook's own state drives, so the hook exposes
  `clampPage(totalPages)` and the page calls it. Taking `totalPages` as a hook
  argument would have been a circular dependency.
- Moving `CommandPalette`'s focus trap into the shared `useDialogFocusTrap`
  broke the static a11y guard, which matched the literal `'Tab'` per file. The
  guard now also accepts delegation to that hook, and the hook gained five
  behavioural tests so the guarantee no longer rests on a string match.
