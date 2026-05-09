# Settings Cache In Tests

## Problem

Configuration tests pass individually but fail when the whole test suite runs.

## Root Cause

The settings loader caches values. A test mutates environment variables after
settings were already loaded, then reads stale config.

## Solution

- Use the project's explicit reset helper, such as `reload_settings()`.
- Isolate environment variable mutation with test fixtures.
- Avoid importing modules that load settings at import time.

## Prevention Checklist

- [ ] Tests that mutate env vars reset settings afterward.
- [ ] Config modules avoid side effects at import time.
- [ ] New settings fields have tests for default and overridden values.
