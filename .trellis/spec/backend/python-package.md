# Python Package Rules

## Package Shape

Use `src/nebula/` for importable Python code and `tests/` for tests.

Keep modules boring and discoverable:

```text
src/nebula/
├── config/
├── contracts/
├── core/
├── models/
├── observability/
└── utils/
```

`api/`, `application/`, `core/`, `db/`, `domain/`, `infrastructure/`, and
`schemas/` already exist. Place new code by the current boundary instead of
creating parallel layers.

## Typing

- Keep `py.typed` for typed package distribution.
- Prefer explicit return types on public functions.
- Avoid `Any` unless the boundary is genuinely dynamic and documented.
- Use `Protocol` for behavior contracts when it avoids coupling.

## Public API Changes

When adding or moving public symbols:

1. Update the relevant `__init__.py` export.
2. Update or add tests that import from the public surface.
3. Update docs that mention stable imports.
4. Run backend verification.
