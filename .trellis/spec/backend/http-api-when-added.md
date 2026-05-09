# HTTP API

MyNebula already has FastAPI routes under `/api` and `/api/v2`.

## Rules

- Keep route handlers thin.
- Validate auth/access at the API boundary.
- Admin writes require a valid admin session and CSRF validation.
- `READ_ACCESS_MODE=demo` may allow anonymous read paths for Data, Graph, and
  semantic search; authenticated mode requires admin session.
- Do not trust forwarded headers unless explicitly enabled and source IP is
  trusted.
- Use explicit errors for missing graph versions or invalid state transitions.

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
