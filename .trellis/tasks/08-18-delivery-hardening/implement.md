# Implementation plan — Delivery hardening

Ordering: child **4 of 5** in `.trellis/tasks/08-18-comprehensive-hardening`.
Requires `08-18-pipeline-test-coverage` to be merged first, because D4 consumes
the integration invocation that task records in
`.trellis/spec/shared/verification.md`.

## Step 0 — Baseline

- [ ] Build the current image and record its size:
      `docker build --tag mynebula:before . && docker image inspect mynebula:before --format '{{.Size}}'`
- [ ] Confirm the current image contains the toolchain:
      `docker run --rm mynebula:before sh -c 'command -v gcc; command -v g++'`
- [ ] Read `frontend/playwright.config.ts` and
      `frontend/e2e/graph-sync-flow.spec.ts` and decide Shape A or Shape B from
      `design.md`. Record the decision and its evidence in the task notes before
      writing any workflow YAML.

**Gate 0:** the Playwright shape is decided from the config and spec, not
assumed. Blocking for step 3 only; steps 1 and 2 may proceed.

## Step 1 — D2: widen CI lint scope

Smallest, lowest-risk change; do it first so the rest of the task is verified
under the corrected gate.

- [ ] `.github/workflows/ci.yml`, `backend-lint`:
      `uv run ruff check src tests scripts alembic`
- [ ] Same job: `uv run ruff format --check src tests scripts alembic`
- [ ] Run both locally to confirm they pass before pushing.

## Step 2 — D1: split the Python builder stage

- [ ] Rename current stage 2 to `python-builder`; keep its
      `gcc g++ python3-dev` install and its `uv sync` layers.
- [ ] Move `COPY src/ ./src/`, `COPY alembic/ ./alembic/`, `COPY alembic.ini ./`
      and the project-installing `uv sync --frozen --no-dev` into the builder,
      per decision "Option 1" in `design.md`.
- [ ] Add a new final stage from `python:3.12-slim` that installs **only**
      `curl`.
- [ ] `COPY --from=python-builder /app/.venv /app/.venv`, plus `src/`,
      `alembic/`, `alembic.ini`.
- [ ] `COPY --from=frontend-builder /app/frontend/dist ./frontend/dist`
- [ ] Recreate `appuser`, `chown -R appuser:appuser /app`, `USER appuser`.
- [ ] Keep `ENV`, `EXPOSE 8000`, `HEALTHCHECK`, and the `CMD` line unchanged.
- [ ] Keep `uv` available in the runtime stage if `CMD` still uses `uv run`;
      otherwise invoke the venv's `uvicorn` directly. Pick one and make the
      `CMD` consistent with it.

**Verification — run against the built image, not read off the file:**

```bash
docker build --tag mynebula:after .
docker run --rm mynebula:after sh -c 'command -v gcc || echo absent'
docker run --rm mynebula:after sh -c 'command -v g++ || echo absent'
docker run --rm mynebula:after sh -c 'command -v curl'
docker run --rm mynebula:after id -un
docker run --rm mynebula:after python -c "import nebula.main; print('import ok')"
docker image inspect mynebula:after --format '{{.Size}}'
```

- [ ] `gcc` and `g++` report `absent`.
- [ ] `curl` resolves.
- [ ] `id -un` prints `appuser`.
- [ ] `import nebula.main` succeeds — this is the real guard against a broken
      copied virtualenv, and `docker build` succeeding does not prove it.
- [ ] Record before/after sizes in the task notes.

**Gate 2:** all six assertions above pass. Blocking.

## Step 3 — D3: Playwright job

Implement the shape chosen at step 0.

- [ ] Add an `e2e` job to `ci.yml` with `needs: [frontend-build]`.
- [ ] pnpm 10.26.1 and Node 20, matching the other frontend jobs.
- [ ] `pnpm exec playwright install --with-deps chromium` — chromium only.
- [ ] If Shape B: add the `pgvector/pgvector:pg16` service, run
      `alembic upgrade head`, start uvicorn, and poll `/health` until ready.
      Use an explicit readiness loop with a timeout, not `sleep`.
- [ ] `pnpm run test:e2e`
- [ ] `actions/upload-artifact` with `if: always()` for `playwright-report/` and
      `test-results/`.
- [ ] Do not add `e2e` to `container-build`'s `needs`.

**Gate 3:** the job passes on a real CI run, not just YAML that parses. If the
existing spec is broken, fix the job configuration; do not modify the spec's
assertions to make it pass — that would be scope creep into E2E rewriting, which
this task's PRD excludes.

## Step 4 — D4: integration tier step

- [ ] Read the exact invocation from
      `.trellis/spec/shared/verification.md` as written by
      `08-18-pipeline-test-coverage`.
- [ ] Add it to `backend-test` as its **own step**, placed after
      `alembic upgrade head` and after the existing `pytest -q` step.
- [ ] Do not merge it into the existing `pytest -q` step; a separate step is
      what makes a zero-collection exit visible.
- [ ] Confirm on a real CI run that the step reports a non-zero collected count.

**Gate 4:** CI logs show integration tests actually ran, with a count. A green
job whose integration step reports "no tests ran" fails this gate.

## Step 5 — Spec synchronisation

- [ ] Rewrite the "CI Gate" section of
      `.trellis/spec/shared/verification.md` so it matches `ci.yml`
      command for command: widened ruff scope, integration step, Playwright job
      and its chosen shape.
- [ ] If the container assertions from step 2 are worth re-running by hand, add
      them to the verification document as a container section.

## Step 6 — Verify

Full stack, since CI, the image, and spec docs all change.

```bash
uv run ruff check src tests scripts alembic
uv run ruff format --check src tests scripts alembic
uv run pytest -q
uv run pytest -q -m integration
pnpm --prefix frontend run lint
pnpm --prefix frontend exec tsc --noEmit -p tsconfig.json
pnpm --prefix frontend run test
pnpm --prefix frontend run build
docker build --tag mynebula:ci .
```

Plus the container assertions from step 2 and the documentation link check.

Report the real CI run result, not only local results. This task's deliverable
*is* the CI gate, so a local-only report does not demonstrate completion.

## Rollback points

- Step 1: one-line revert per command; no coupling.
- Step 2: revert `Dockerfile` alone. Nothing else references its internals, and
  `docker-compose.yml` consumes only the published image tag.
- Step 3: delete the `e2e` job; no other job depends on it by design.
- Step 4: delete the integration step; the tier still runs locally and
  auto-skips elsewhere.
- Step 5: documentation only.

Every step is independently revertable, and no step changes application
behaviour, so there is no runtime rollback risk.
