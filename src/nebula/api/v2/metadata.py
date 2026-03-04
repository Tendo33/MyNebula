"""Shared helpers for v2 response metadata fields."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone


def build_v2_metadata(
    *,
    version: str | None = None,
    generated_at: str | None = None,
    request_id: str | None = None,
) -> dict[str, str | None]:
    return {
        "version": version,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "request_id": request_id or str(uuid.uuid4()),
    }
