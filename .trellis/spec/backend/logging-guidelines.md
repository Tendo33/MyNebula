# Logging Guidelines

## Overview

The backend uses the shared logger utility (`nebula.utils.get_logger`) backed by Loguru. Logging should help operators understand pipeline progress and failure boundaries without leaking secrets or request-derived sensitive data.

## Log Levels

- `info`: lifecycle milestones, sync phase transitions, scheduler launches, successful admin login
- `warning`: recoverable failures, skipped work, secondary system degradation
- `error`: request/service failures that terminate the current operation
- `exception`: same as error, but when a traceback materially helps diagnose the issue

## What To Log

- Background job identifiers: `run_id`, `task_id`, `user_id`, `phase`
- Scheduler launch decisions and skip reasons
- Auth success/failure metadata that is safe to expose, such as masked username and coarse client IP
- Database/task state transitions that affect visible UI state

## What Not To Log

- Admin passwords, session secrets, API keys, CSRF token values
- Full cookie contents or signed session payloads
- Large embedding vectors or raw README bodies
- Untrusted forwarded header chains unless they have already passed proxy validation

## Format Guidance

- Prefer short structured key-value fragments inside message strings over long prose.
- Include the failure reason once; do not duplicate the same exception across nested layers unless ownership changes.
- When an operation ends in `partial_failed`, log both the terminal status and the summarized partial error text.
