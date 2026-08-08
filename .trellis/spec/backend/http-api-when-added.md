# HTTP API

MyNebula exposes its application API under `/api/v2`; `/api` is only the
outer mount prefix. Retired unversioned routes are not compatibility surfaces.

## Rules

- Keep route handlers thin.
- Validate auth/access at the API boundary.
- Admin writes require a valid admin session and CSRF validation.
- `READ_ACCESS_MODE=demo` may allow anonymous read paths for Data, Graph, and
  semantic search; authenticated mode requires admin session.
- Do not trust forwarded headers unless explicitly enabled and source IP is
  trusted.
- Use explicit errors for missing graph versions or invalid state transitions.
- Snapshot hydration/unavailability maps to a bounded `503` with a request ID
  and `Retry-After`; detailed exceptions stay in server logs.
- `interrupted` is terminal and retryable for pipeline/full-refresh status APIs.
- Public health output exposes component state only, never raw scheduler errors.

## Current Stable Entrypoints

- ASGI app: `nebula.main:create_app`
- Run command: `mynebula` or `uv run uvicorn nebula.main:app`
- Config: `nebula.core.config`
- Preferred external integration surface: HTTP API, not internal service imports.

## Contract Notes

- Graph, Dashboard, and Data views should prefer snapshot-backed or lightweight
  aggregate responses.
- Settings/Sync write routes should resolve the user at the API boundary and
  pass it to services.
- Partial failures must remain visible in API responses so the frontend can
  render warning states.
- Login throttling reserves attempts transactionally across IP and username
  buckets. Opportunistic stale-row cleanup is limited to those two locked
  buckets, so one login request does not globally delete unrelated rate-limit
  history. Logout increments the server-side session version and revokes all
  earlier admin cookies.
