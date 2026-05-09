# Type Safety

## Python Rules

- Keep `py.typed` in typed packages.
- Public functions need explicit return types.
- Prefer Pydantic models for external data shapes.
- Prefer `Protocol` when a behavior contract is needed without tying callers to
  a concrete class.
- Avoid `Any`; use explicit models, `TypedDict`, `Protocol`, or `object` plus
  narrowing.

## Pydantic

- Use Pydantic v2 APIs.
- Keep validation at the boundary.
- Do not duplicate response/request shapes by hand when a Pydantic model is the
  source of truth.
- Export reusable models from the package's model namespace.

## Frontend Boundary

If the frontend consumes backend JSON:

- Document request, success response, and error response shapes.
- Keep field names stable.
- Add tests for serialization-sensitive behavior such as dates, enums, and
  optional fields.
