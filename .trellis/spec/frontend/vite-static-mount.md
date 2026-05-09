# Vite Static Mount

MyNebula deployment serves frontend build assets through the API image.

## Build Boundary

- Source lives in `frontend/src`.
- Production assets are produced by `pnpm --prefix frontend run build`.
- The backend may serve `frontend/dist` as static files.
- Do not make Python import frontend source files.
- Do not make frontend code depend on backend internals.

## Routing

- API routes must win before static fallback.
- Static asset paths should be served directly.
- Client-side app routes may fall back to `index.html`.
- Unknown API routes should return API errors, not the frontend shell.

## Verification

For static mount changes, run frontend build and the backend/static-serving tests
that cover routing precedence.
