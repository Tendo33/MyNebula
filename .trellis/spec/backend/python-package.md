# Python Package Rules

## Package Shape

Use `src/nebula/` for importable Python code and `tests/` for tests.

Keep modules boring and discoverable. The actual layout is:

```text
src/nebula/
├── api/
├── application/
├── core/
├── db/
├── domain/
├── infrastructure/
├── jobs/
├── schemas/
└── utils/
```

Place new code by the current boundary instead of creating parallel layers. See
[directory-structure.md](./directory-structure.md) for what belongs where.

## Typing

- Prefer explicit return types on public functions.
- Avoid `Any` unless the boundary is genuinely dynamic and documented.
- Use `Protocol` for behavior contracts when it avoids coupling.
- There is no `py.typed` marker. MyNebula ships as an application, not a typed
  library, so downstream type distribution is not a current concern. Add the
  marker only alongside a real decision to publish the package for import.

## Public API Changes

When adding or moving public symbols:

1. Update the relevant `__init__.py` export.
2. Update or add tests that import from the public surface.
3. Update docs that mention stable imports.
4. Run backend verification.
