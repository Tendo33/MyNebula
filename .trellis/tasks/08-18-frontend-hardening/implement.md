# Implementation plan — Frontend decomposition, graph perf, i18n and a11y baseline

Ordering: child **5 of 5** in `.trellis/tasks/08-18-comprehensive-hardening`.
Independent of A–D; sequenced last to keep the review queue linear.

Internal order is smallest-and-safest first, so that by the time the risky
`Graph2D` split starts, the graph path already has more tests around it.

## Step 0 — Baseline

- [ ] Record current LOC for the four targets:
      `wc -l frontend/src/components/graph/Graph2D.tsx frontend/src/pages/Settings.tsx frontend/src/components/ui/CommandPalette.tsx frontend/src/pages/DataPage.tsx`
- [ ] Record the current frontend test count: `pnpm --prefix frontend run test`
- [ ] Confirm the Playwright graph flow passes before any change, so a later
      failure is attributable.

## Step 1 — E3: i18n parity

- [ ] Add `repoDetails.similar` to `frontend/src/locales/en/translation.json`,
      matching the zh entry's meaning.
- [ ] Extend `frontend/src/locales/zh/translation.test.ts` with a flatten-and-
      compare test asserting `en` and `zh` key sets are equal, naming the
      offending keys in the failure message.
- [ ] Verify the guard actually guards: temporarily delete a key, confirm the
      test fails with a useful message, restore it.

**Gate 1:** parity test passes and demonstrably fails on drift.

## Step 2 — E2: adjacency index in the provider

- [ ] Add `buildAdjacencyIndex(edges)` to `frontend/src/contexts/graphFiltering.ts`,
      preserving the `typeof edge.source === 'object' ? edge.source.id : edge.source`
      normalisation for both endpoints.
- [ ] Unit-test it in `graphFiltering.test.ts`: id endpoints, object endpoints,
      mixed, empty, and self-referential edges.
- [ ] In `GraphProvider`, memoise it on `stagedEdges` and add it to
      `GraphContextValue` and to the value `useMemo` dependency array.
- [ ] Reduce `useNodeNeighbors` to a context lookup plus the existing per-node
      `useMemo`.
- [ ] Add a test asserting two components using the hook observe the same index
      instance, proving it is not rebuilt per consumer.
- [ ] Add a test asserting neighbour sets are unchanged from the previous
      implementation for a fixture graph.

**Gate 2:** `pnpm --prefix frontend run test` green; graph-related tests pass
unmodified.

## Step 3 — E4: modal semantics and a11y automation

- [ ] Add the a11y testing dependency as a dev dependency via pnpm; record the
      choice and its justification in the task notes.
- [ ] `CommandPalette`: `role="dialog"`, `aria-modal="true"`, accessible name,
      focus moved in on open, focus restored to the prior element on close,
      focus trapped while open, Escape closes. Coordinate with the existing
      `useCommandPalette` open/close hook rather than duplicating its state.
- [ ] Settings confirmation dialog: same treatment.
- [ ] Sweep interactive controls for accessible names; add `aria-label` to
      icon-only buttons. Start with the files at zero aria coverage.
- [ ] Add `aria-live` to `SyncProgress` and to loading/error regions that
      announce async state.
- [ ] Add a11y smoke assertions to `dashboard.v2.smoke.test.tsx`,
      `data.v2.smoke.test.tsx`, and both modal surfaces.
- [ ] Write the baseline into `.trellis/spec/frontend/`, including the explicit
      out-of-scope list (contrast auditing, screen-reader transcripts, the
      force-graph canvas) and the note that automated rules catch only part of
      real accessibility problems.

**Gate 3:** a11y checks run inside `pnpm run test`; both modals pass focus-trap
and Escape tests.

## Step 4 — E1: decompose, one file per sub-step

Each file is a separate commit. Do not start the next until the previous
verifies.

### 4a — `Graph2D.tsx`

- [ ] Extract colour resolution to `graph2dStyles.ts` with explicit parameters.
      Unit-test each function.
- [ ] Extract painting to `graph2dPainters.ts`, taking
      `(ctx, node/link, scale, style inputs)` explicitly. Unit-test with a fake
      2D context asserting the call sequence.
- [ ] Move remaining pure geometry into the existing `graph2dUtils.ts`.
- [ ] Extract `useAvatarImageCache`, preserving the unmount timer cleanup added
      by the 2026-05-09 health fix and the `Number.isFinite` coordinate guard.
- [ ] Extract `useGraphViewport` (auto-fit, focus-by-id, interaction tracking).
- [ ] Extract `useGraphForces`.
- [ ] Keep every hook's dependency array byte-identical through the move. A
      changed dependency array is a behaviour change, not a refactor.

**Gate 4a:** under 400 LOC; `pnpm run test` green; Playwright graph flow passes;
manual check that the graph does not visibly re-lay-out or flicker on load.

### 4b — `Settings.tsx`

- [ ] Extract `useSettingsAuth`, `useSettingsSchedule`,
      `useSettingsSyncControls`.
- [ ] `settings.polling.test.tsx` and `Settings.partial-failed.test.tsx` must
      pass **unmodified**. If either needs an assertion change, stop: the
      polling lifecycle regressed.

**Gate 4b:** under 400 LOC; both polling tests pass unmodified.

### 4c — `CommandPalette.tsx`

- [ ] Extract `useCommandPaletteResults` and `useCommandPaletteKeyboard`;
      move result rendering to `CommandPaletteResultList.tsx`.
- [ ] `utils/search.ts` stays shared with the Data page; do not fork search
      semantics.
- [ ] `CommandPalette.test.tsx` passes; step 3's focus-trap tests still pass.

**Gate 4c:** under 400 LOC; search behaviour identical on both Graph and Data.

### 4d — `DataPage.tsx`

- [ ] Extract `useDataPageUrlState`, `useDataPageFilters`, and
      `DataRepoTable.tsx`.
- [ ] `DataPage.url-state.test.tsx` passes unmodified.

**Gate 4d:** under 400 LOC.

## Step 5 — Spec

- [ ] `.trellis/spec/frontend/directory-structure.md` and `components.md`:
      record the new module boundaries — where pure graph helpers live, where
      page-level hooks live, and the rule that pure functions extracted from
      components must be tested directly.
- [ ] Confirm the a11y baseline written in step 3 is linked from
      `.trellis/spec/frontend/index.md`.

## Step 6 — Verify

```bash
pnpm --prefix frontend run lint
pnpm --prefix frontend exec tsc --noEmit -p tsconfig.json
pnpm --prefix frontend run test
pnpm --prefix frontend run build
pnpm --prefix frontend run test:e2e
wc -l frontend/src/components/graph/Graph2D.tsx frontend/src/pages/Settings.tsx frontend/src/components/ui/CommandPalette.tsx frontend/src/pages/DataPage.tsx
```

Then the Graph / Search hotspot matrix from
`.trellis/spec/shared/verification.md`:

```bash
pnpm --prefix frontend run test -- src/components/ui/__tests__/CommandPalette.test.tsx
pnpm --prefix frontend run test -- src/pages/__tests__/GraphPage.url-state.test.tsx
pnpm --prefix frontend run test -- src/features/data/hooks/useDataReposQuery.test.tsx
```

Plus the documentation link check.

Report the before/after LOC table and the list of pre-existing tests that
required changes. An empty list is the expected result; a non-empty one needs a
justification per entry.

## Rollback points

Every sub-step of step 4 is its own commit against a specific file, so any single
decomposition can be reverted without disturbing the others. Steps 1–3 are
additive and independently revertable. No backend state, schema, or
configuration is involved, so rollback is immediate and complete.
