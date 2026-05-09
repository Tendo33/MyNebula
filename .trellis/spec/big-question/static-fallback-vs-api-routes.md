# Static Fallback vs API Routes

## Problem

An API request returns the frontend shell HTML instead of a JSON error.

## Impact

Frontend callers may try to parse HTML as JSON. Monitoring also becomes noisy
because API failures look like successful page responses.

## Root Cause

The static frontend fallback is registered before API route handling, or the
fallback does not exclude API prefixes.

## Solution

- Match API routes before static assets and client-app fallback.
- Make unknown API routes return API-shaped errors.
- Use fallback to `index.html` only for non-API paths.
- Add tests for one known API route, one unknown API route, and one client route.

## Prevention Checklist

- [ ] API route registration happens before static fallback.
- [ ] API prefix is explicitly excluded from SPA fallback.
- [ ] Unknown API path has a regression test.
- [ ] Client route fallback has a regression test.
