# Delivery hardening: runtime image and CI gate parity

## Goal

Make the shipped container image contain only what it needs to run, and make the
CI gate actually enforce what `.trellis/spec/shared/verification.md` declares to
be the project's verification contract.

This is finding group **D** of parent task
`.trellis/tasks/08-18-comprehensive-hardening`.

## Requirements

### D1 — Remove the build toolchain from the runtime image

`Dockerfile` stage 2 runs:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*
```

`gcc`, `g++`, and `python3-dev` are needed to build wheels for the numeric stack
(`numpy`, `scikit-learn`, `numba`, `psycopg2-binary`). They are never removed, so
they ship to production. `curl` is genuinely needed at runtime by the
`HEALTHCHECK` and by the compose healthcheck, and stays.

The frontend is already correctly split into a `frontend-builder` stage. The
Python side is not.

Requirements:

- Compilers and development headers must not be present in the final image.
- Build them in a dedicated Python builder stage and copy only the resolved
  environment into the runtime stage.
- `curl` stays, because both `Dockerfile` and `docker-compose.yml` healthchecks
  invoke it.
- The image must still run as the non-root `appuser`.
- `CMD` and `EXPOSE` behaviour is unchanged.

### D2 — Align CI lint scope with the verification contract

`.trellis/spec/shared/verification.md` declares:

```bash
uv run ruff check src tests scripts alembic
uv run ruff format --check src tests scripts alembic
```

`.github/workflows/ci.yml` `backend-lint` runs:

```bash
uv run ruff check src/
uv run ruff format --check src/
```

`tests/`, `scripts/`, and `alembic/` are unlinted in CI. They pass locally today,
so this is drift rather than active breakage — but it means a contributor can
merge unformatted test or migration code, and it means the canonical
verification document is describing a gate that does not exist.

Requirements:

- CI lint scope must equal the scope in
  `.trellis/spec/shared/verification.md`.
- If the two must differ, the spec is the authority and CI changes to match, not
  the reverse.

### D3 — Run the end-to-end test that already exists

`frontend/playwright.config.ts` and
`frontend/e2e/graph-sync-flow.spec.ts` are committed. `package.json` exposes
`pnpm run test:e2e`. No CI job runs any of it, so the E2E suite has no enforced
signal and can rot silently.

Requirements:

- CI must run the Playwright suite.
- The job must install browsers deterministically and produce a retrievable
  artifact on failure.
- Because E2E needs a running stack, the job must define how the application
  under test is brought up, or the suite must be scoped to what can run without
  one. Whichever is chosen must be explicit, not implicit.

### D4 — Wire the integration test tier into CI

Sibling task `08-18-pipeline-test-coverage` (C) introduces a PostgreSQL-backed
integration tier that auto-skips when no database is reachable. CI already
provisions `pgvector/pgvector:pg16` and runs `alembic upgrade head`.

Requirements:

- The `backend-test` job must run the integration tier explicitly.
- A run that collects zero integration tests must fail the job. Auto-skip is a
  developer convenience; in CI a silent skip is a false green and is the single
  most likely way this epic's work quietly stops being enforced.
- The invocation must match the command string recorded in
  `.trellis/spec/shared/verification.md` by task C.

## Constraints

- Runs after `08-18-pipeline-test-coverage`, which defines the integration
  invocation D4 consumes.
- `docker-compose.yml` requires `MYNEBULA_IMAGE` to name a published immutable
  tag or digest. Do not weaken that.
- Preserve the existing job graph shape: `backend-lint` → `backend-test`,
  `frontend-lint` → `frontend-test` / `frontend-build`, and
  `container-build` depending on all three.
- No change to application source behaviour.
- pnpm stays at 10.26.1 and Node at 20, matching `packageManager`.

## Non-goals

- Not adding image publishing, registry pushes, or release automation.
  `release.yml` owns that and is out of scope.
- Not introducing a different container base image or switching off `python:3.12-slim`.
- Not adding new lint tools (mypy, bandit, semgrep). Scope is aligning the
  existing gate, not expanding it.
- Not rewriting the Playwright specs themselves.

## Acceptance criteria

- [ ] `docker build --tag mynebula:ci .` succeeds.
- [ ] `docker run --rm mynebula:ci sh -c "command -v gcc || echo absent"` reports
      `absent`, and the same for `g++`.
- [ ] `curl` is still present and the `HEALTHCHECK` still passes.
- [ ] The built image runs as `appuser`, not root.
- [ ] Final image size is smaller than the pre-change image; both numbers are
      recorded in the task notes.
- [ ] `backend-lint` in CI runs ruff over `src tests scripts alembic` for both
      `check` and `format --check`.
- [ ] `backend-test` runs the integration tier and fails if zero integration
      tests are collected.
- [ ] A CI job runs the Playwright suite, with browser installation pinned and a
      failure artifact uploaded.
- [ ] `.trellis/spec/shared/verification.md`'s "CI Gate" section matches the
      workflow file exactly.
- [ ] Full-stack verification passes.

## Verification

Full stack, because this task touches CI, the container image, and spec docs.
The container assertions must be run against a locally built image, not
inferred from the Dockerfile.

## D1–D4 complete (2026-08-18)

### D1 — runtime image, measured against built images

Split into `frontend-builder` → `python-builder` → runtime. Compilers live only
in the builder; the runtime stage installs `curl` alone and receives the
resolved `/app/.venv`. `uv` is no longer in the runtime image, so `CMD` invokes
the venv's `uvicorn` directly via `PATH`.

| | Before | After |
| --- | --- | --- |
| Image size | 455 MB | **330 MB** (−27%) |
| `gcc` / `g++` | `/usr/bin/gcc`, `/usr/bin/g++` | **absent** |
| `curl` | present | present |
| User | appuser | appuser |
| `import nebula.main` | ok | ok |

Assertions were run against `docker run`, not read off the Dockerfile. The
image was additionally started against a pgvector container: `/health` returned
`{"status":"healthy","database":"connected","scheduler":{"status":"running"}}`,
`GET /` and `GET /graph` returned 200, and Docker's own HEALTHCHECK reached
`healthy`.

### D3 — the E2E suite could not run at all

The finding was recorded as "CI never runs it". The deeper cause is that it was
broken: `playwright.config.ts` started its server with
`pnpm run dev -- --host 127.0.0.1 --port 4173`, and the literal `--` makes vite
ignore every following flag. Vite started on the default 5173 while Playwright
waited on 4173, so the run always died with
`Timed out waiting 60000ms from config.webServer`.

Fixed by invoking vite directly with `--strictPort`, so a busy port fails loudly
instead of silently relocating. Shape A confirmed from the spec itself: it mocks
its own network payloads and is gated behind `RUN_E2E=1`, so no backend is
needed.

### D2 / D4 — CI

- `backend-lint` now lints `src tests scripts alembic`, matching the spec.
- `backend-test` gained a **separate** integration step. Separate on purpose:
  the tier auto-skips without a database, and a skip folded into `pytest -q`
  reads as a pass, whereas a standalone step fails on pytest's exit code 5.
- New `e2e` job: chromium only, `RUN_E2E=1`, report and traces uploaded with
  `if: always()`. Depends on `frontend-build`, and deliberately not wired into
  `container-build`'s `needs` so an E2E failure does not block the image signal.
- `container-build` now asserts on the built image (no gcc/g++, curl present,
  runs as appuser, `import nebula.main` succeeds).

`.trellis/spec/shared/verification.md` "CI Gate" was rewritten to match
`ci.yml` command for command, and gained Integration tier, End-to-end, and
Container sections.

### D3 verification status — partial, and here is exactly what is unverified

**Verified:** the `webServer` fix works. Before it, the run died with
`Timed out waiting 60000ms from config.webServer` — Vite was on 5173 while
Playwright waited on 4173. After it, that error is gone and the run proceeds to
browser launch, which is direct evidence the dev server now comes up on the
awaited port and Playwright connects to it.

**Not verified:** the three specs have not been executed to green locally. The
chromium build `@playwright/test@1.58.2` requires (revision 1208) will not
download on this machine — four attempts across ~40 minutes each stalled at
428 KB, twice leaving a directory containing only `ABOUT` and `LICENSE` and no
binary. Revision 1234 downloads fine, so it is specific to that CDN artifact,
not a blanket network block.

A second, real problem surfaced while diagnosing: `pnpm exec playwright`
resolved to a *different* `playwright-core` than the one `@playwright/test`
uses, and installed revision 1234 while the test runner wanted 1208. The
conventional `pnpm exec playwright install` command would therefore install the
wrong browser. Both CI and `verification.md` now invoke the project's own CLI:

```bash
node node_modules/@playwright/test/cli.js install --with-deps chromium
```

On a clean CI runner there is no global pnpm store to mis-resolve against and no
partial cache, so the job is expected to work — but that expectation has not
been observed. **The first real CI run is the verification.** If it fails, the
cause is browser provisioning, not the config, which is the part that was
proven here.
