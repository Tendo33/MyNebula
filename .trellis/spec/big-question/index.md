# Common Issues And Solutions

Documented pitfalls for MyNebula.

## Severity Levels

| Level | Description |
| --- | --- |
| Critical | App fails to start, build fails, or API/static routing is broken |
| Warning | Feature broken, tests misleading, or deployment behavior wrong |
| Info | Degraded developer experience with a known workaround |

## Issue Index

| Issue | Category | Severity |
| --- | --- | --- |
| [Generated Frontend Assets](./generated-frontend-assets.md) | Build / Version Control | Warning |
| [Static Fallback vs API Routes](./static-fallback-vs-api-routes.md) | Routing | Critical |
| [Settings Cache In Tests](./settings-cache-in-tests.md) | Configuration / Testing | Warning |

## Quick Debugging Checklist

### Frontend Build Works But Backend Serves Old UI

1. Check whether `frontend/dist/` is generated and ignored.
2. Confirm deploy/build step runs `pnpm --prefix frontend run build`.
3. Confirm backend static path points to the generated output.

### API Requests Return HTML

1. Check API routes are matched before static fallback.
2. Confirm unknown API paths return JSON errors, not `index.html`.
3. Add a regression test for routing precedence.

### Config Tests Pass Alone But Fail In Suite

1. Check whether settings are cached.
2. Use `reload_settings()` or the target project's explicit reset helper.
3. Keep environment-variable mutation isolated per test.
