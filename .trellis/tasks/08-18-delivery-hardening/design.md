# Design — Delivery hardening

## Scope boundary

Three files: `Dockerfile`, `.github/workflows/ci.yml`, and
`.trellis/spec/shared/verification.md`. No application source changes.

## D1 — Python builder stage

### Current shape

```
Stage 1: node:20-alpine        -> frontend/dist
Stage 2: python:3.12-slim      -> apt-get gcc g++ python3-dev curl
                                  uv sync (compiles wheels)
                                  COPY src, alembic
                                  uv sync again
                                  COPY --from=frontend-builder dist
                                  useradd appuser; USER appuser
```

Compilers stay resident for the life of the image.

### Target shape

```
Stage 1: node:20-alpine        -> frontend/dist            (unchanged)
Stage 2: python:3.12-slim AS python-builder
           apt-get gcc g++ python3-dev
           uv sync --frozen --no-dev   -> /app/.venv
Stage 3: python:3.12-slim
           apt-get curl only
           COPY --from=python-builder /app/.venv /app/.venv
           COPY src, alembic, alembic.ini
           COPY --from=frontend-builder /app/frontend/dist
           useradd appuser; chown; USER appuser
```

### Decisions

**Copy the virtualenv, not site-packages.** `uv sync` materialises `/app/.venv`.
Copying the whole venv keeps console scripts and `pyvenv.cfg` consistent. The
runtime stage must use the same Python minor version as the builder for this to
be valid — both are `python:3.12-slim` from the same base, pinned in one place.

**The project itself must still be installed.** The current Dockerfile does
`uv sync --no-install-project` for caching, copies source, then `uv sync` again
to install the project. In the split, the builder handles dependency resolution
and compilation; the project install must still happen so that
`nebula.main:app` is importable. Two viable shapes:

1. Copy source into the builder before the final `uv sync`, then copy the venv.
2. Copy the venv into the runtime stage and rely on `PYTHONPATH=/app/src`.

Option 1 is chosen: it keeps the runtime stage free of any path trickery and
preserves the existing `uv run uvicorn` invocation. The cost is that a source
change invalidates the builder's final layer, but the dependency layer above it
still caches, which is where the compilation time lives.

**Keep `curl`.** Both `Dockerfile` `HEALTHCHECK` and `docker-compose.yml`'s
`api` healthcheck shell out to `curl -f http://localhost:8000/health`. Removing
it would break both. Its CVE profile is not the concern here; toolchains are.

**Do not switch base images.** Moving to `-alpine` would change the libc and
force source builds of the numeric stack; moving to `distroless` would break the
`curl` healthcheck and the `uv run` entrypoint. Out of scope.

### Verification approach

Assertions run against a built image, not read off the Dockerfile:

```bash
docker build --tag mynebula:ci .
docker run --rm mynebula:ci sh -c 'command -v gcc || echo absent'
docker run --rm mynebula:ci sh -c 'command -v g++ || echo absent'
docker run --rm mynebula:ci sh -c 'command -v curl'
docker run --rm mynebula:ci id -un
docker image inspect mynebula:ci --format '{{.Size}}'
```

Record before and after sizes in the task notes; "smaller" is an acceptance
criterion and needs a number behind it.

## D2 — Lint scope parity

One-line change in `backend-lint`:

```yaml
- name: Ruff check
  run: uv run ruff check src tests scripts alembic

- name: Ruff format check
  run: uv run ruff format --check src tests scripts alembic
```

The spec is the authority. Both scopes already pass locally as of the
2026-08-18 scan, so this is expected to be a no-op on signal and a real change
on enforcement.

## D3 — Playwright in CI

### The real question: what does the E2E suite need

`frontend/e2e/graph-sync-flow.spec.ts` is a graph and sync flow test. Whether it
needs a live backend or runs against mocked network is the deciding factor, and
`playwright.config.ts` (`webServer`, `baseURL`) answers it. **Read both before
choosing a shape** — this is the one open question in this task, and picking
wrong produces either a flaky job or a job that tests nothing.

Two shapes:

**Shape A — frontend only.** If the spec mocks its network layer and the config
declares a `webServer` running Vite preview, the job is: install deps, build,
`pnpm exec playwright install --with-deps chromium`, `pnpm run test:e2e`. Cheap
and stable.

**Shape B — full stack.** If the spec needs a real API, the job adds the
`pgvector/pgvector:pg16` service, runs `alembic upgrade head`, starts uvicorn,
waits on `/health`, then runs Playwright against it. More faithful, more moving
parts, needs an explicit readiness wait rather than a sleep.

Whichever shape holds, it is written down in the spec so the next reader does
not have to re-derive it.

### Common requirements

- Browser install pinned via `pnpm exec playwright install --with-deps chromium`
  — do not install all browsers.
- `actions/upload-artifact` for `playwright-report/` and `test-results/`, with
  `if: always()`, so a failure is diagnosable without a rerun.
- The job depends on `frontend-build`, since it needs built assets.
- Do not add it to `container-build`'s `needs` — E2E failure should not block the
  image build signal, and coupling them lengthens the critical path.

## D4 — Integration tier in CI

Added to `backend-test`, after `alembic upgrade head`:

```yaml
- name: Run integration tests
  run: uv run pytest -q -m integration
```

### Guarding against a silent skip

The integration tier auto-skips when no database is reachable. In CI that would
be a false green. Two mechanisms, both applied:

1. The CI database is reachable by construction — the service container and
   `DATABASE_URL` are already configured, and `alembic upgrade head` in the
   preceding step proves connectivity before the test step runs.
2. The test step must fail on zero collection. `pytest` exits with code 5 on
   "no tests collected", which fails the step by default — so the guard is to
   run integration as its **own step** rather than folding it into the existing
   `pytest -q`, where a skip would vanish into the summary.

This is why D4 adds a separate step instead of relying on the existing
`uv run pytest -q` step picking the tier up.

### Coupling to task C

The invocation string is owned by
`.trellis/spec/shared/verification.md`, written by task C. This task copies it
verbatim. If C's command differs from the placeholder above, C wins.

## Spec synchronisation

`.trellis/spec/shared/verification.md`'s "CI Gate" section currently claims to
be "the local equivalent" of GitHub Actions. After this task it must actually be
that, including:

- the widened ruff scope,
- the integration test step,
- the Playwright job and its chosen shape.

A stale CI Gate section is worse than no section, because it is read as
authoritative.

## Risks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Copied venv breaks because builder and runtime Python differ | Low | Image fails to start | Both stages pin `python:3.12-slim`; the smoke assertion runs the built image, not just builds it |
| Removing `python3-dev` breaks a runtime C extension load | Low | Import error at startup | `docker run` the image and import `nebula.main` as part of verification, not just `docker build` |
| Playwright job is flaky and gets ignored | Medium | Gate decays to noise | Pin browser install, upload artifacts, choose the simplest shape the spec actually needs |
| Widened ruff scope surfaces pre-existing violations | Low | CI red on unrelated files | Both scopes verified passing in the 2026-08-18 scan; if new violations appear, fix them in this task rather than narrowing the scope back |
| Integration step silently skips | Medium | False green, epic's coverage work stops being enforced | Separate CI step; `alembic upgrade head` proves connectivity first; zero-collection exits non-zero |
