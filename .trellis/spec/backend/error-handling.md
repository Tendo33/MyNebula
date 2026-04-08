# Error Handling

## Overview

MyNebula distinguishes between:

- HTTP contract errors returned directly from route handlers with `HTTPException`
- service-level operational failures recorded into `SyncTask` / `PipelineRun`
- global unexpected exceptions, which are logged server-side and returned as generic `500`

## Error Handling Patterns

- Route handlers should use `HTTPException` for request validation, auth failures, missing resources, and conflicts.
- Background tasks should catch exceptions at the orchestration boundary, persist terminal failure state, and log the exception once.
- If a subtask fails inside a larger workflow, the parent workflow must not report success. Persist the failing phase in task metadata.
- Partial failures are first-class results, not silent warnings. Use `PipelineStatus.partial_failed` and keep a human-readable `last_error`.

## API Error Responses

- Auth/configuration failures: `401`, `403`, `409`, or `503` as appropriate
- Unknown server failures: generic `{"detail": "Internal server error"}` from the global exception handler
- Admin write endpoints must not leak stack traces to clients

## Common Mistakes To Avoid

- Marking parent jobs as `completed` after a child task wrote `failed`
- Swallowing scheduler launch failures and leaving schedule state stuck on `running`
- Returning raw exception messages from global exception handlers
- Clearing `last_error` too early when the terminal status is still `partial_failed` or `failed`
