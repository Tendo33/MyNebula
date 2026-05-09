# Configuration And Logging

## Configuration

- Configuration should live behind a small settings module.
- Use `pydantic-settings`.
- Keep defaults suitable for local development.
- Use `reload_settings()` or an equivalent explicit helper for tests that need
  to change environment variables.
- Add validation only for values that can actually be invalid in the current
  project.

## Logging

- Use the project's logging entrypoint instead of configuring loggers ad hoc.
- Prefer structured fields for request IDs, task IDs, file paths, and external
  operation names.
- Do not log secrets, API tokens, passwords, raw auth headers, or sensitive
  personal data.
- When a frontend request crosses into backend code, preserve or generate a
  request identifier and include it in backend logs.

## Static Frontend Mount

If the backend serves the Vite build:

- Treat `frontend/dist/` as generated output.
- Keep the build command in verification.
- Serve static assets with cache headers appropriate for hashed files.
- Let client-side routing fall back to `index.html` only for non-API paths.
- Keep API routes and static fallback rules visibly separate.
