"""Domain constants for snapshot and pipeline lifecycle."""

from enum import StrEnum


class SnapshotStatus(StrEnum):
    building = "building"
    ready = "ready"
    active = "active"
    failed = "failed"


class PipelineStatus(StrEnum):
    pending = "pending"
    running = "running"
    partial_failed = "partial_failed"
    completed = "completed"
    failed = "failed"


class PipelinePhase(StrEnum):
    pending = "pending"
    stars = "stars"
    embedding = "embedding"
    clustering = "clustering"
    snapshot = "snapshot"
    completed = "completed"
