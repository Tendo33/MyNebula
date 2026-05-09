# Generated Frontend Assets

## Problem

The frontend works locally, but deployed or backend-served pages show stale UI,
missing assets, or mismatched hashed filenames.

## Root Cause

Vite production output is generated. The source of truth remains
`frontend/src` plus the pnpm build command.

## Solution

- Keep `frontend/dist/` out of git unless explicitly documented.
- Make deployment or container build run `pnpm --prefix frontend run build`.
- Keep backend static path configurable or clearly documented.
- Test that the backend serves the built asset directory when static serving is
  part of the app.

## Prevention Checklist

- [ ] Frontend build command is in verification.
- [ ] Static output is ignored or intentionally documented.
- [ ] Backend static path is documented.
- [ ] Deploy docs mention frontend build order.
